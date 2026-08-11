"""Forward hooks for MoE models.

Four quantities matter to this pipeline:

  - router_logits  : per-layer, pre-top-k. Site of RouteAudit's suffix search (capture
                     AND mutate — the router mutator is how a defense could steer
                     routing back at eval time).
  - gate_input     : the gate's own input (T, d_model), independent of the model's
                     internal residual representation.
  - routing        : faithful per-layer routing recomputed from gate_input through a
                     :class:`~routeaudit.model.gate_math.GateSpec`. Needed for gates that
                     never expose a pre-selection score tensor — DeepSeek's Gate returns
                     only `(weights, indices)`, so there is nothing to read off the output.
  - residual       : per-layer decoder-layer output, retained without reshaping.

Which attributes hold the MoE block, router, and experts is described by an
:class:`~routeaudit.model.archspec.ArchSpec` (presets for OLMoE, Mixtral, Qwen, Phi-MoE,
DeepSeek). We attach hooks on:

  - block.<router_attr>      : capture logits / gate input / routing, and optionally
                               mutate the logits.
  - layer (forward post)     : capture the residual-stream output.

We deliberately do not patch internals beyond hooks — keeps the loader pluggable.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from . import gate_math

# ─────────────────────────── Capture container ───────────────────────────


@dataclass
class HookCapture:
    """Holds activations captured during a forward pass.

    All entries are keyed by layer index. Tensors are detached and kept on the
    model's device unless `to_cpu=True` was set.
    """

    router_logits: dict[int, torch.Tensor] = field(default_factory=dict)
    residual: dict[int, torch.Tensor] = field(default_factory=dict)
    gate_input: dict[int, torch.Tensor] = field(default_factory=dict)
    routing: dict[int, "gate_math.RouteResult"] = field(default_factory=dict)

    #: layer index → (T, top_k) selected expert ids. The lightweight alternative to
    #: `routing` for sweeps over many tokens, where keeping five (T, E) tensors per
    #: layer would cost gigabytes.
    expert_indices: dict[int, torch.Tensor] = field(default_factory=dict)

    #: layer index → number of streams in `residual[layer]` (1 = standard).
    residual_streams: dict[int, int] = field(default_factory=dict)

    def clear(self) -> None:
        self.router_logits.clear()
        self.residual.clear()
        self.gate_input.clear()
        self.routing.clear()
        self.expert_indices.clear()
        self.residual_streams.clear()


# ─────────────────────────── Mutators ───────────────────────────
#
# A router mutator is a callable that takes the pre-truncation logit tensor
# (T, n_experts) and returns a (possibly modified) tensor.


RouterMutator = Callable[[torch.Tensor, int, int], torch.Tensor]
"""(logits[T, n_experts], layer_idx, step_idx) -> logits"""


# ─────────────────────────── Manager ───────────────────────────


class MoEHookManager:
    """Owns the lifecycle of forward hooks on a MoE model.

    Use as a context manager so hooks are always removed:

        with MoEHookManager(model, spec) as hm:
            hm.capture_router_logits()
            out = model(**batch)
            # hm.capture.router_logits is populated

    `spec` is an :class:`ArchSpec`; when omitted the OLMoE preset is used, which
    reproduces the original hardcoded behavior.
    """

    def __init__(self, model: torch.nn.Module, spec=None):
        from .archspec import ArchSpec
        self.model = model
        self.spec = spec or ArchSpec()
        self.capture = HookCapture()

        # Mutator (None = pass-through).
        self._router_mutator: Optional[RouterMutator] = None

        # Per-call step counter. Bumped externally by the caller (one bump per
        # generated token). Used by mutators that depend on decoding step.
        self.step_idx: int = 0

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._capture_router = False
        self._detach_router = True
        self._capture_residual = False
        self._capture_gate_input = False
        self._gate_spec: Optional[gate_math.GateSpec] = None
        self._selection_only = False
        self._capture_hash = False
        self._token_ids: Optional[torch.Tensor] = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def __enter__(self) -> "MoEHookManager":
        self._install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ── capture switches (set before entering the context, or before forward) ──

    def capture_router_logits(self, detach: bool = True) -> "MoEHookManager":
        """Capture pre-top-k router logits in `capture.router_logits`.

        `detach=True` (default) stores a detached tensor — right for analysis, and it
        keeps the autograd graph from being pinned alive across a whole forward.

        **`detach=False` is required for any loss that must produce a gradient.** A
        detached tensor has `requires_grad=False` and no `grad_fn`, so a routing loss
        built on it is a constant: it still evaluates to the right number, and it still
        works for scoring or ranking candidates, but it contributes exactly zero to
        `backward()`. That failure is silent whenever the total loss has another term
        that does carry gradient — the optimization runs, the loss goes down, and the
        routing objective is steering nothing.
        """
        self._capture_router = True
        self._detach_router = detach
        return self

    def capture_residual(self) -> "MoEHookManager":
        """Capture each decoder layer's residual-stream output, keyed by layer index in
        `capture.residual`, with the stream count in `capture.residual_streams`.

        The tensor is stored as-is; architecture adapters can interpret extra axes."""
        self._capture_residual = True
        return self

    def capture_gate_input(self) -> "MoEHookManager":
        """Capture the router's input (T, d_model) in `capture.gate_input`."""
        self._capture_gate_input = True
        return self

    def capture_routing(self, gate_spec: "gate_math.GateSpec") -> "MoEHookManager":
        """Capture faithful per-layer routing in `capture.routing` as
        :class:`~routeaudit.model.gate_math.RouteResult`.

        This is the path for any gate whose semantics aren't `softmax(logits)`: the
        routing is recomputed from the gate's input under `gate_spec`, so it works on
        gates that expose no pre-selection score tensor at all. On a spec with
        `router_output="recompute"` (DeepSeek) the logits are formed from the gate's
        weight matrix; otherwise the gate's own output is used.

        Hash-routed layers are NOT captured — their routing comes from a token-id table,
        not from the gate input, so a recomputed score there would be fiction. Use
        `gate_math.hash_route` for those.
        """
        self._gate_spec = gate_spec
        self._selection_only = False
        return self

    def capture_expert_selection(self, gate_spec: "gate_math.GateSpec") -> "MoEHookManager":
        """Capture only WHICH experts fire, per layer, in `capture.expert_indices`.

        The corpus-sweep counterpart to `capture_routing`: activation-frequency harvesting
        only needs the top-k membership, and storing full `RouteResult`s over a
        16x1024-token batch on a 43-layer, 256-expert model would run to gigabytes.

        Use this instead of `topk(router_logits)` on any gate that isn't
        `GateSpec.is_plain_topk` — that shortcut ignores the balancing bias and the group
        mask, and on DeepSeek there is no logit tensor to top-k in the first place.
        """
        self._gate_spec = gate_spec
        self._selection_only = True
        return self

    # ── mutator wiring (defensive routing + steering) ────────────────────

    def set_router_mutator(self, fn: RouterMutator | None) -> None:
        self._router_mutator = fn

    # ── decoding-step bookkeeping ────────────────────────────────────────

    def reset_step(self) -> None:
        self.step_idx = 0

    def advance_step(self) -> None:
        self.step_idx += 1

    # ── install hooks (internal) ─────────────────────────────────────────

    def _base_module(self):
        """Resolve an ArchSpec base path, including nested multimodal decoders."""
        base = self.model
        for attr in self.spec.base_attr.split("."):
            base = getattr(base, attr)
        return base

    def _iter_moe_blocks(self):
        """Yield (layer_idx, moe_block) using the ArchSpec to locate modules."""
        s = self.spec
        base = self._base_module()
        layers = getattr(base, s.layers_attr)
        for i, layer in enumerate(layers):
            block = None
            for attr in s.moe_block_attrs:   # first existing attr wins (version-robust)
                block = getattr(layer, attr, None)
                if block is not None:
                    break
            if block is None:
                continue
            # MoE blocks expose the router + experts containers named by the spec.
            if hasattr(block, s.experts_attr) and hasattr(block, s.router_attr):
                yield i, block

    def _install(self) -> None:
        # All hooks are installed unconditionally; the capture flags gate only
        # whether data is *stored* inside each hook. This lets callers flip a
        # `capture_*()` switch either before or after entering the context (the
        # documented pattern — see the class docstring). Installing conditionally
        # here is a footgun: a `capture_residual()` called after `__enter__` would
        # otherwise never install its hook and silently capture nothing.
        for layer_idx, block in self._iter_moe_blocks():
            self._install_router_hook(layer_idx, block)
        self._install_residual_hooks()

    def _install_residual_hooks(self) -> None:
        """Hook each decoder layer's forward to capture its residual-stream output.

        HF decoder layers return either a tensor or a tuple whose first element is the
        hidden state. We store it without reshaping and record any explicit stream axis.
        """
        s = self.spec
        base = self._base_module()
        layers = getattr(base, s.layers_attr)
        mgr = self

        for i, layer in enumerate(layers):
            def make_hook(li=i):
                def fwd_hook(_module, _inputs, output):
                    if mgr._capture_residual:
                        tensor = output[0] if isinstance(output, tuple) else output
                        mgr.capture.residual[li] = tensor.detach()
                        mgr.capture.residual_streams[li] = mgr._stream_count(tensor)
                    return output
                return fwd_hook

            self._handles.append(layer.register_forward_hook(make_hook()))

    def _stream_count(self, tensor: torch.Tensor) -> int:
        """Streams in a captured residual. Uses `spec.d_model` when the config supplies
        it; falls back to rank (a 4-D hidden state is multi-stream) when it doesn't."""
        if self.spec.d_model and tensor.shape[-1] == self.spec.d_model:
            return int(tensor.shape[-2]) if tensor.dim() == 4 else 1
        return int(tensor.shape[-2]) if tensor.dim() == 4 else 1

    def _gate_bias(self, module: torch.nn.Module, recompute: bool) -> Optional[torch.Tensor]:
        """The load-balancing bias, if this gate has one.

        Only tried on gates we recompute, or when the GateSpec explicitly asked for a
        bias: on a plain `nn.Linear` gate, `.bias` is the linear layer's own additive
        bias (already inside the logits) and adding it again would corrupt selection.
        On DeepSeek's `Gate`, `.bias` *is* the auxiliary-loss-free balancing bias, which
        is why the fallback exists at all.
        """
        names = [self.spec.router_bias_attr, "e_score_correction_bias"]
        if recompute:
            names.append("bias")
        for attr in names:
            b = getattr(module, attr, None)
            if isinstance(b, torch.Tensor):
                return b
        return None

    def _record_hash_routing(self, layer_idx: int, module: torch.nn.Module,
                             hidden: torch.Tensor, output) -> None:
        """Routing for a hash-gated layer, where selection and weighting come apart.

        A hash router takes `indices = tid2eid[input_ids]` — fixed per token id, so no
        input change and no gradient can move it — but it still computes
        `weights = scores.gather(indices)` from the LEARNED score function. So the layer
        is unsteerable in *which* experts fire and content-dependent in *how much* each
        contributes. Capturing only the selection (or skipping the layer entirely) throws
        away a real signal; capturing it as if it were a learned gate invents one.

        Needs the token ids, which the gate's own inputs don't carry — they arrive via
        `set_token_ids`, populated by a pre-hook on the base model. Without them the layer
        is skipped rather than guessed at.
        """
        gs = self._gate_spec
        ids = self._token_ids
        table = getattr(module, "tid2eid", None)
        if ids is None or not isinstance(table, torch.Tensor):
            return
        logits = self._logits_from_output(output, self.spec.n_experts)
        if logits is None:
            return
        idx = gate_math.hash_route(ids.reshape(-1), table).to(logits.device)
        if idx.shape[0] != logits.shape[0]:
            return                      # ids/hidden mismatch (padding?) — don't guess
        scores = gate_math.affinity(logits.float(), gs.scoring_func)
        w = gate_math.gate_weights(scores, idx, gs)
        dense = torch.zeros_like(scores).scatter_(-1, idx, w)
        if self._selection_only:
            self.capture.expert_indices[layer_idx] = idx
        else:
            # `sel_scores` is the bias-free score: nothing is selected BY score here, so
            # presenting a selection score would imply a contest that does not happen.
            official_weights, official_indices = self._official_topk_from_output(output, gs.top_k)
            self.capture.routing[layer_idx] = gate_math.RouteResult(
                scores=scores,
                sel_scores=scores,
                indices=idx,
                weights=w,
                dense=dense,
                official_weights=official_weights,
                official_indices=official_indices,
            )

    def set_token_ids(self, ids: Optional[torch.Tensor]) -> None:
        """Supply the token ids hash-routed layers need. See `capture_hash_layers`."""
        self._token_ids = ids

    def capture_hash_layers(self) -> "MoEHookManager":
        """Also capture hash-routed layers, by plumbing token ids to the gate hooks.

        Installs a forward pre-hook on the base module that snapshots `input_ids` for the
        current forward. Off by default: for membership-based statistics (which experts
        fire, the usual expert-localization signal) a hash layer contributes only the
        corpus's token distribution and is noise, so including it is usually wrong. Turn
        it on for mass-based analyses, where the learned weighting is real signal.
        """
        base = self._base_module()
        mgr = self

        def pre_hook(_module, args, kwargs):
            ids = kwargs.get("input_ids")
            if ids is None and args and isinstance(args[0], torch.Tensor):
                ids = args[0]
            mgr._token_ids = ids
            return None

        self._handles.append(base.register_forward_pre_hook(pre_hook, with_kwargs=True))
        self._capture_hash = True
        return self

    def _logits_from_output(self, output, n_experts: int) -> Optional[torch.Tensor]:
        """Find the `(T, n_experts)` router logits in whatever the gate returned.

        Gate return shapes differ by family and by transformers version:
          * a bare `(T, E)` tensor                       — OLMoE / Mixtral / Qwen / Phi
          * `(scores, topk_weights, topk_indices)`       — fused HF gates
          * `(logits, weights, indices)`                 — fused learned routers
          * `(weights, indices)`                         — DeepSeek's raw inference impl,
                                                           which exposes no logits at all

        Rather than guess from position, take the first element whose trailing dimension
        is `n_experts`: `weights` and `indices` are `(T, top_k)`, so they can't be
        mistaken for logits unless top_k == n_experts (which would mean no sparsity).
        Returns None when nothing matches, so the caller can fall back.
        """
        cands = output if isinstance(output, tuple) else (output,)
        for t in cands:
            if isinstance(t, torch.Tensor) and t.dim() >= 2 and t.shape[-1] == n_experts:
                return t.detach()
        return None

    def _official_topk_from_output(
        self, output, top_k: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return the router's own ``(weights, indices)`` when present.

        Non-softmax routers may expose either ``(logits, weights, indices)`` or
        ``(weights, indices)``. Detect by dtype and trailing width so the capture also
        survives tuple-order differences without confusing logits for top-k weights.
        """
        tensors = output if isinstance(output, tuple) else (output,)
        weights = next(
            (
                t.detach()
                for t in tensors
                if isinstance(t, torch.Tensor) and t.dim() >= 2
                and t.shape[-1] == top_k and t.is_floating_point()
            ),
            None,
        )
        indices = next(
            (
                t.detach()
                for t in tensors
                if isinstance(t, torch.Tensor) and t.dim() >= 2
                and t.shape[-1] == top_k and not t.is_floating_point()
            ),
            None,
        )
        if weights is None or indices is None or weights.shape != indices.shape:
            return None, None
        return weights, indices

    def _record_lfm2_routing(self, layer_idx: int, output, selection_bias=None) -> None:
        """Record LFM2's exact sigmoid routing and its selection-only expert bias."""
        if not isinstance(output, tuple) or len(output) < 3:
            return
        logits, weights, indices = output[:3]
        if not all(isinstance(x, torch.Tensor) for x in (logits, weights, indices)):
            return
        logits, weights, indices = logits.detach(), weights.detach(), indices.detach()
        scores = logits.float().sigmoid()
        bias = selection_bias.detach().to(scores.dtype) if isinstance(selection_bias, torch.Tensor) else None
        sel_scores = scores if bias is None else scores + bias
        if self._selection_only:
            self.capture.expert_indices[layer_idx] = indices
            return
        dense = torch.zeros_like(scores).scatter_(-1, indices, weights.float())
        self.capture.routing[layer_idx] = gate_math.RouteResult(
            scores=scores,
            sel_scores=sel_scores,
            indices=indices,
            weights=weights,
            dense=dense,
            official_weights=weights,
            official_indices=indices,
        )

    def _record_routing(self, layer_idx: int, module: torch.nn.Module,
                        hidden: torch.Tensor, output, selection_bias=None) -> None:
        """Recompute this layer's routing under the GateSpec and store the RouteResult."""
        if self.spec.router_output == "lfm2":
            self._record_lfm2_routing(layer_idx, output, selection_bias)
            return
        gs = self._gate_spec
        kind = gate_math.routing_kind(layer_idx, gs)
        if kind == gate_math.DENSE:
            return
        if kind == gate_math.HASH:
            if self._capture_hash:
                self._record_hash_routing(layer_idx, module, hidden, output)
            return
        h = hidden.reshape(-1, hidden.shape[-1]) if hidden.dim() == 3 else hidden
        recompute = self.spec.router_output == "recompute"
        n_experts = self.spec.n_experts
        logits = self._logits_from_output(output, n_experts)
        if logits is None and recompute:
            # No usable logit tensor in the output — form them from the gate's weight
            # matrix. This is the fallback for DeepSeek's raw `inference/model.py`, whose
            # gate returns only (weights, indices). Implementations that return logits
            # take the preferred path.
            # Recomputing will NOT work against fp8 weights, which is another reason to
            # prefer returned logits whenever they are available.
            w = getattr(module, "weight", None)
            if not isinstance(w, torch.Tensor):
                return
            logits = torch.nn.functional.linear(h.detach().to(w.dtype), w)
        if logits is None:
            return
        bias = self._gate_bias(module, recompute) if gs.use_bias else None
        if self._selection_only:
            scores = gate_math.affinity(logits.float(), gs.scoring_func)
            sel = gate_math.selection_scores(scores, bias, gs)
            self.capture.expert_indices[layer_idx] = sel.topk(gs.top_k, dim=-1).indices
        else:
            rr = gate_math.route(logits, bias, gs)
            official_weights, official_indices = self._official_topk_from_output(output, gs.top_k)
            rr.official_weights = official_weights
            rr.official_indices = official_indices
            self.capture.routing[layer_idx] = rr

    def _mutate_lfm2_router(self, module: torch.nn.Module, inputs, output,
                            layer_idx: int):
        """Apply a mutation while reproducing LFM2's sigmoid/expert-bias router."""
        if not inputs or not isinstance(output, tuple) or len(output) < 3:
            return output
        routed_logits = output[0]
        if not isinstance(routed_logits, torch.Tensor):
            return output
        routed_logits = self._router_mutator(routed_logits, layer_idx, self.step_idx)
        scores = routed_logits.sigmoid()
        selection_bias = inputs[1] if len(inputs) > 1 and isinstance(inputs[1], torch.Tensor) else None
        selection_scores = scores if selection_bias is None else scores + selection_bias.to(scores.dtype)
        top_k = int(getattr(module, "top_k", output[1].shape[-1]))
        topk_indices = torch.topk(selection_scores, top_k, dim=-1).indices
        topk_weights = scores.gather(-1, topk_indices).to(routed_logits.dtype)
        if bool(getattr(module, "norm_topk_prob", True)):
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
        topk_weights = topk_weights * float(getattr(module, "routed_scaling_factor", 1.0))
        return (routed_logits, topk_weights, topk_indices) + tuple(output[3:])

    def _install_router_hook(self, layer_idx: int, block: torch.nn.Module) -> None:
        """Hook the router (`block.<router_attr>`) forward to capture and optionally
        mutate router logits.

        The gate is a Linear: input (B*T, d_model) -> logits (B*T, n_experts).
        We mutate the *output* before it leaves the gate, so the downstream top-k
        and softmax see our biased logits.
        """
        gate = getattr(block, self.spec.router_attr)
        mgr = self

        def fwd_hook(module, inputs, output):
            # Captures independent of what the gate returns.
            if (mgr._capture_gate_input or mgr._gate_spec is not None) and inputs:
                h = inputs[0]
                if isinstance(h, torch.Tensor):
                    if mgr._capture_gate_input:
                        mgr.capture.gate_input[layer_idx] = h.detach()
                    if mgr._gate_spec is not None:
                        mgr._record_routing(layer_idx, module, h, output, inputs[1] if len(inputs) > 1 else None)

            if mgr.spec.router_output == "lfm2":
                if not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor):
                    return output
                if mgr._capture_router:
                    mgr.capture.router_logits[layer_idx] = (
                        output[0].detach() if mgr._detach_router else output[0])
                if mgr._router_mutator is not None:
                    return mgr._mutate_lfm2_router(module, inputs, output, layer_idx)
                return output

            # OLMoE gate output shape depends on transformers version:
            #   - Old (≤ ~4.46): the gate Linear returns raw logits (B*T, n_experts).
            #   - New: a fused gate returns (routing_scores, top_k_weights, top_k_index)
            #     where routing_scores is (B*T, n_experts) — either raw logits or
            #     softmax probs depending on the version. We treat it as routing
            #     scores and topk it ourselves when a mutator is present.
            if isinstance(output, tuple):
                scores = output[0]
                tail = output[1:]
                if mgr._capture_router:
                    mgr.capture.router_logits[layer_idx] = (
                        scores.detach() if mgr._detach_router else scores)
                if mgr._router_mutator is not None:
                    scores = mgr._router_mutator(scores, layer_idx, mgr.step_idx)
                    # Recompute top-k on the biased scores so the MoE dispatch
                    # actually uses our routing change.
                    if len(tail) >= 2:
                        k = tail[0].shape[-1]   # top_k from old top_k_weights
                        probs = scores.softmax(dim=-1)
                        new_w, new_idx = torch.topk(probs, k=k, dim=-1)
                        new_w = new_w / new_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                        new_w = new_w.to(tail[0].dtype)
                        new_idx = new_idx.to(tail[1].dtype)
                        return (scores, new_w, new_idx) + tuple(tail[2:])
                    return (scores,) + tuple(tail)
                return output
            # Legacy path: gate emits raw logits directly.
            logits = output
            if mgr._capture_router:
                mgr.capture.router_logits[layer_idx] = (
                    logits.detach() if mgr._detach_router else logits)
            if mgr._router_mutator is not None:
                logits = mgr._router_mutator(logits, layer_idx, mgr.step_idx)
            return logits

        self._handles.append(gate.register_forward_hook(fwd_hook))


# ─────────────────────────── Convenience ───────────────────────────


@contextmanager
def captured_forward(model, *, router=False, spec=None):
    """One-shot context for read-only router-logit capture.

        with captured_forward(model, router=True) as cap:
            model(**batch)
        cap.router_logits[2]   # tensor
    """
    hm = MoEHookManager(model, spec)
    if router:
        hm.capture_router_logits()
    with hm:
        yield hm.capture
