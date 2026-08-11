"""Gate semantics — how a MoE router turns hidden states into expert weights.

The pipeline's original assumption was `softmax(logits)` over all experts, which is
right for OLMoE / Mixtral / Qwen / Phi-MoE and wrong for DeepSeekMoE. This module makes
the gate pluggable so one implementation serves every family, and so the two quantities
a biased gate produces can never be confused:

    scores      = f(W h)                    bias-FREE affinity  → sets HOW MUCH  an
                                            expert contributes (the gating weight)
    sel_scores  = scores + bias             → sets WHICH experts fire (selection only)

DeepSeek's auxiliary-loss-free load balancing (Wang et al. 2024a) adds `bias` to the
selection score but computes the gating weight from the bias-free score. A routing
"flip" caused by the bias term is a training-time load-balancing artifact, not a
semantic routing change — so studies must say which tensor they used. `RouteResult`
carries both.

Supported primitives include softmax, sigmoid, and sqrt-softplus affinities; flat or
group-limited top-k; selection-only bias; score renormalization; and optional hash-routed
leading layers. Gates that do not expose pre-selection scores can be recomputed from
their input and weight matrix by `hooks.capture_routing`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

SCORING_FUNCS = ("softmax", "sigmoid", "sqrtsoftplus")

# Layer routing kinds returned by `routing_kind`.
DENSE, HASH, LEARNED = "dense", "hash", "learned"


@dataclass(frozen=True)
class GateSpec:
    """Everything about a family's gate that isn't module layout (that's ArchSpec).

    Layer prefixes (`first_k_dense_replace`, `num_hash_layers`) are counted from layer
    0 in that order: the dense prefix comes first (those layers have no gate at all),
    then the hash-routed MoE layers, then learned routing for the rest.
    """

    scoring_func: str = "softmax"
    top_k: int = 8
    use_bias: bool = False  # e_score_correction_bias — SELECTION ONLY
    n_group: int = 1  # >1 enables node-limited routing (V2/V3)
    topk_group: int = 0
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    num_hash_layers: int = 0  # leading MoE layers routed by token id
    first_k_dense_replace: int = 0  # leading layers with a dense MLP (no gate)

    def __post_init__(self):
        if self.scoring_func not in SCORING_FUNCS:
            raise ValueError(
                f"scoring_func={self.scoring_func!r} not in {SCORING_FUNCS}. "
                f"Expected one of the registered, architecture-neutral scoring functions."
            )

    @property
    def grouped(self) -> bool:
        return self.n_group > 1 and self.topk_group > 0

    @property
    def is_plain_topk(self) -> bool:
        """True when expert selection is exactly `logits.topk(k)`.

        This is the condition under which reading the gate's raw logit output and
        top-k-ing it is *correct* — everything the pipeline did before this module
        existed. Any other gate needs `hooks.capture_expert_selection`, because
        `topk(logits)` would ignore the balancing bias, the group mask, or the fact that
        the gate emits no logit tensor at all.
        """
        return (
            self.scoring_func == "softmax"
            and not self.use_bias
            and not self.grouped
            and self.num_hash_layers == 0
        )

    @classmethod
    def from_config(cls, model_ns) -> GateSpec:
        """Build from a config `model` namespace, reading its optional `routing:` block.

        With no `routing:` block this returns the flat-softmax default, which is the
        behavior every currently-supported family already has.
        """
        r = getattr(model_ns, "routing", None)
        g = (lambda k, d: getattr(r, k, d)) if r is not None else (lambda k, d: d)
        top_k = int(g("top_k", 0) or getattr(model_ns, "top_k", 0) or 0)
        return cls(
            scoring_func=str(g("scoring_func", "softmax")),
            top_k=top_k,
            use_bias=bool(g("use_bias", False)),
            n_group=int(g("n_group", 1) or 1),
            topk_group=int(g("topk_group", 0) or 0),
            norm_topk_prob=bool(g("norm_topk_prob", True)),
            routed_scaling_factor=float(g("routed_scaling_factor", 1.0)),
            num_hash_layers=int(g("num_hash_layers", 0) or 0),
            first_k_dense_replace=int(g("first_k_dense_replace", 0) or 0),
        )


@dataclass
class RouteResult:
    """The four tensors a routing decision produces, kept separate on purpose.

    scores       (T, E)      bias-free affinity — the CONTRIBUTION quantity
    sel_scores   (T, E)      scores + bias, group-masked — the SELECTION quantity
    indices      (T, top_k)  which experts fire
    weights      (T, top_k)  gathered, renormalized, scaled gating weights
    dense        (T, E)      `weights` scattered back to expert positions (0 elsewhere)
    """

    scores: torch.Tensor
    sel_scores: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor
    dense: torch.Tensor
    #: (T, E) bool — experts node-limited routing leaves selectable. None on a flat gate,
    #: where every expert is always eligible.
    eligible: torch.Tensor | None = None
    # The router's own returned values, when its forward exposes them. Keeping these
    # separate lets fixtures compare RouteAudit with the shipped implementation on the
    # same device instead of mistaking a CPU/GPU rounding difference for model drift.
    official_weights: torch.Tensor | None = None
    official_indices: torch.Tensor | None = None


# ─────────────────────────── scoring ───────────────────────────


def affinity(logits: torch.Tensor, scoring_func: str = "softmax") -> torch.Tensor:
    """Router logits → per-expert affinity scores.

    `sqrtsoftplus` and `sigmoid` are elementwise and unnormalized
    — the scores across experts do NOT sum to 1, so they are not a distribution and
    entropy-style statistics computed on them are not comparable to softmax gates.
    """
    if scoring_func == "softmax":
        return logits.softmax(dim=-1)
    if scoring_func == "sigmoid":
        return logits.sigmoid()
    if scoring_func == "sqrtsoftplus":
        return F.softplus(logits).sqrt()
    raise ValueError(f"unknown scoring_func {scoring_func!r} (expected one of {SCORING_FUNCS})")


# ─────────────────────────── selection ───────────────────────────


def group_mask(sel_scores: torch.Tensor, gs: GateSpec) -> torch.Tensor:
    """Node-limited routing mask (DeepSeek-V2/V3): keep only the best `topk_group`
    expert groups per token, scored by the sum of each group's top-2 experts.

    Returns a bool (T, E) mask of experts still eligible. All-True when the spec is
    flat or the group params don't divide the expert count.
    """
    T, E = sel_scores.shape
    if not gs.grouped or E % gs.n_group != 0:
        return torch.ones_like(sel_scores, dtype=torch.bool)
    per = E // gs.n_group
    grouped = sel_scores.view(T, gs.n_group, per)
    group_score = grouped.topk(min(2, per), dim=-1).values.sum(dim=-1)  # (T, n_group)
    keep = group_score.topk(min(gs.topk_group, gs.n_group), dim=-1).indices
    m = torch.zeros(T, gs.n_group, dtype=torch.bool, device=sel_scores.device)
    m.scatter_(1, keep, True)
    return m.unsqueeze(-1).expand(T, gs.n_group, per).reshape(T, E)


def selection_scores(scores: torch.Tensor, bias: torch.Tensor | None, gs: GateSpec) -> torch.Tensor:
    """scores + bias, with ineligible groups masked. The tensor top-k actually runs on.

    Ineligible groups are filled with negative infinity, matching current Transformers
    routers and ensuring excluded experts cannot enter the final top-k.
    """
    sel = scores
    if gs.use_bias and bias is not None:
        sel = sel + bias.to(sel.dtype).view(1, -1)
    if gs.grouped:
        sel = sel.masked_fill(~group_mask(sel, gs), float("-inf"))
    return sel


def eligible_mask(scores: torch.Tensor, bias: torch.Tensor | None, gs: GateSpec) -> torch.Tensor | None:
    """Which experts node-limited routing leaves selectable, or None for a flat gate.

    Kept separate from `selection_scores` so margin analysis can explicitly identify an
    expert as unreachable instead of inferring eligibility from a sentinel score.
    """
    if not gs.grouped:
        return None
    sel = scores
    if gs.use_bias and bias is not None:
        sel = sel + bias.to(sel.dtype).view(1, -1)
    return group_mask(sel, gs)


def select(scores: torch.Tensor, bias: torch.Tensor | None, gs: GateSpec) -> torch.Tensor:
    """→ (T, top_k) indices of the experts that fire."""
    return selection_scores(scores, bias, gs).topk(gs.top_k, dim=-1).indices


# ─────────────────────────── weighting ───────────────────────────


def gate_weights(scores: torch.Tensor, indices: torch.Tensor, gs: GateSpec) -> torch.Tensor:
    """→ (T, top_k) gating weights, from the BIAS-FREE scores.

    This is the half of the gate the balancing bias must not touch: it decides how much
    each selected expert contributes, and is what any contribution/ablation study wants.
    """
    w = scores.gather(-1, indices)
    if gs.norm_topk_prob:
        # Use an additive epsilon rather than clamp_min; they differ whenever the sum
        # is not tiny.
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w * gs.routed_scaling_factor


def route(
    logits: torch.Tensor, bias: torch.Tensor | None, gs: GateSpec, *, dtype: torch.dtype = torch.float32
) -> RouteResult:
    """Full gate: router logits (T, E) → RouteResult.

    Computed in float32 by default. The model's own gate runs in its native dtype; we
    upcast because bf16 ties make top-k selection order unstable, which shows up as
    spurious "routing flips" in analysis. Pass `dtype=logits.dtype` to reproduce the
    model's own arithmetic exactly (e.g. for a bit-for-bit fixture check).
    """
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    logits = logits.to(dtype)
    scores = affinity(logits, gs.scoring_func)
    sel = selection_scores(scores, bias, gs)
    # `sorted=False` matches both released routers. Selection is a set; the order within
    # it is an implementation detail, so never compare index tensors positionally.
    idx = sel.topk(gs.top_k, dim=-1, sorted=False).indices
    w = gate_weights(scores, idx, gs)
    dense = torch.zeros_like(scores).scatter_(-1, idx, w)
    return RouteResult(
        scores=scores,
        sel_scores=sel,
        indices=idx,
        weights=w,
        dense=dense,
        eligible=eligible_mask(scores, bias, gs),
    )


# ─────────────────────────── margins ───────────────────────────


def selection_margin(
    sel_scores: torch.Tensor,
    gs: GateSpec,
    expert_ids: torch.Tensor | None = None,
    eligible: torch.Tensor | None = None,
) -> torch.Tensor:
    """How far each expert is from flipping in or out of the top-k, in selection-score
    units. This is the quantity an input perturbation has to overcome.

        margin > 0   expert is IN;  it can lose this much before dropping out
        margin < 0   expert is OUT; it must gain this much to get in
        margin = 0   exactly at the boundary

    Concretely: an in-expert is measured against the best excluded competitor (the
    (k+1)-th score), an out-expert against the weakest included one (the k-th score).

    Returns (T, E), or (T, len(expert_ids)) when `expert_ids` selects a subset — e.g.
    the safety experts found by the harvest phase.

    For a grouped gate (V2/V3), `eligible` (from `RouteResult.eligible`) documents which
    groups were excluded and guarantees their margin is -inf: no change to an expert's
    own score alone can select it. The current reference already uses -inf as its mask
    sentinel, so passing the mask is explicit but numerically idempotent. Flat gates
    leave every expert eligible.
    """
    k = gs.top_k
    top = sel_scores.topk(k + 1, dim=-1).values  # (T, k+1), descending
    kth = top[:, k - 1 : k]  # weakest selected
    kth_plus_1 = top[:, k : k + 1]  # best excluded
    is_in = sel_scores >= kth
    margin = torch.where(is_in, sel_scores - kth_plus_1, sel_scores - kth)
    if eligible is not None:
        margin = margin.masked_fill(~eligible, float("-inf"))
    if expert_ids is not None:
        ids = torch.as_tensor(expert_ids, dtype=torch.long, device=sel_scores.device)
        margin = margin.index_select(-1, ids)
    return margin


# ─────────────────────────── layer classification ───────────────────────────


def routing_kind(layer_idx: int, gs: GateSpec) -> str:
    """Classify a layer's routing: "dense" | "hash" | "learned".

    Hash-routed layers pick experts from a predefined
    `tid2eid[token_id]` table. That routing is content-independent and non-differentiable,
    so those layers carry no steerable safety signal: exclude them from harvest, from
    routing-shift metrics, and from any attack loss. Gradients through them are
    structurally zero, not small — including them silently dilutes every per-layer
    statistic with layers that cannot move.
    """
    if layer_idx < gs.first_k_dense_replace:
        return DENSE
    if layer_idx < gs.first_k_dense_replace + gs.num_hash_layers:
        return HASH
    return LEARNED


def learned_router_layers(n_layers: int, gs: GateSpec) -> list[int]:
    """The layers whose routing responds to content — the only valid targets for
    content-based routing analysis or an input-space attack."""
    return [i for i in range(n_layers) if routing_kind(i, gs) == LEARNED]


def hash_route(token_ids: torch.Tensor, tid2eid: torch.Tensor) -> torch.Tensor:
    """Hash routing: token id → experts, straight off the static table.

    Trivial by construction, and that is the point — layers 0..num_hash_layers-1 are the
    only ones whose routing ground truth is perfectly known and context-independent, so
    they make a free correctness oracle: run any routing-capture pipeline over them and
    it must reproduce this exactly before its output on learned layers can be trusted.
    """
    return tid2eid[token_ids]
