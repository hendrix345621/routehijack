"""Local gate-Jacobian spectrum at the generation boundary token.

For each prompt and learned MoE layer, this diagnostic asks a deliberately local
question: which direction in the gate input ``h`` most changes the routing objective at
``t*``?  Stacking those gradients across prompts gives a small ``N x d_model`` matrix.
Its singular spectrum says whether routing control is concentrated in a reusable
low-rank subspace or changes substantially from prompt to prompt.

The gradient is local to the router.  The full model forward is inference-only; after
capturing the real gate input, only the tiny router is replayed with autograd.  This is
important for DeepSeek-V4, whose packed FP4 expert backend need not implement backward.

Two V4-aware objectives are reported:

``mass``
    Bias-free normalized gating mass on the selected target experts.
``margin``
    Positive ``score + balancing_bias`` margin keeping a target expert inside the
    hard top-k.  Hash-routed layers are excluded because they have no selection margin.

Rows are L2-normalized before the SVD.  The result therefore measures the dimension of
the *directions*, not which prompt happened to have the largest raw gradient.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from routeaudit import ui
from routeaudit.model import gate_math
from routeaudit_deepseek_v4.hooks import DeepSeekV4HookManager as MoEHookManager
from routeaudit.model.prompting import encode_prompt


def load_expert_map(path: str | Path) -> dict[int, list[int]]:
    """Load RouteAudit's ``[{layer, expert, ...}]`` artifact or a layer->ids map."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[int, set[int]] = {}
    if isinstance(raw, list):
        for row in raw:
            if isinstance(row, dict) and "layer" in row and "expert" in row:
                out.setdefault(int(row["layer"]), set()).add(int(row["expert"]))
    elif isinstance(raw, dict):
        for layer, experts in raw.items():
            vals = experts if isinstance(experts, list) else [experts]
            out[int(layer)] = {int(v) for v in vals}
    else:
        raise TypeError(f"unsupported expert-map format in {path}")
    return {layer: sorted(experts) for layer, experts in out.items()}


def gradient_spectrum(gradients: list[torch.Tensor], *, eps: float = 1e-12) -> dict:
    """Singular spectrum of row-normalized gradients, using the small prompt Gram matrix."""
    if not gradients:
        return {"n_gradients": 0, "takeaway": "no nonzero gate gradients captured"}

    rows = torch.stack([g.detach().float().cpu().reshape(-1) for g in gradients])
    norms = rows.norm(dim=1)
    keep = torch.isfinite(norms) & (norms > eps)
    rows = rows[keep]
    raw_norms = norms[keep]
    if rows.numel() == 0:
        return {"n_gradients": 0, "takeaway": "all captured gate gradients were zero or non-finite"}

    rows = rows / raw_norms[:, None]
    gram = rows @ rows.T
    eig = torch.linalg.eigvalsh(gram.double()).clamp_min(0).flip(0)
    singular = eig.sqrt()
    energy = eig / eig.sum().clamp_min(eps)
    cumulative = energy.cumsum(0)

    def rank_at(frac: float) -> int:
        return int(torch.searchsorted(cumulative, torch.tensor(frac, dtype=cumulative.dtype)).item() + 1)

    positive = energy[energy > eps]
    effective_rank = float(torch.exp(-(positive * positive.log()).sum())) if positive.numel() else 0.0
    stable_rank = float(eig.sum() / eig[0].clamp_min(eps))
    top = min(10, singular.numel())
    result = {
        "n_gradients": int(rows.shape[0]),
        "d_model": int(rows.shape[1]),
        "singular_values": [round(float(v), 8) for v in singular[:top]],
        "explained_energy": [round(float(v), 8) for v in energy[:top]],
        "top1_energy": float(energy[0]),
        "top5_energy": float(energy[:5].sum()),
        "rank90": rank_at(0.90),
        "rank95": rank_at(0.95),
        "rank99": rank_at(0.99),
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "gradient_norm_mean": float(raw_norms.mean()),
        "gradient_norm_median": float(raw_norms.median()),
    }
    result["takeaway"] = (
        f"top direction explains {result['top1_energy']:.1%}; "
        f"rank90={result['rank90']}, effective_rank={effective_rank:.2f}"
    )
    return result


def _router_modules(dm) -> dict[int, torch.nn.Module]:
    base = dm.model
    for attr in dm.spec.base_attr.split("."):
        base = getattr(base, attr)
    layers = getattr(base, dm.spec.layers_attr)
    out = {}
    for layer_idx, layer in enumerate(layers):
        block = next(
            (getattr(layer, attr) for attr in dm.spec.moe_block_attrs if hasattr(layer, attr)),
            None,
        )
        if block is not None:
            gate = getattr(block, dm.spec.router_attr, None)
            if isinstance(gate, torch.nn.Module):
                out[layer_idx] = gate
    return out


def _logits_from_gate(gate: torch.nn.Module, h: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Replay the official gate when possible, falling back to its linear projection."""
    try:
        output = gate(h)
    except TypeError:
        output = None
    tensors = output if isinstance(output, tuple) else (output,)
    for value in tensors:
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[-1] == n_experts:
            return value

    weight = getattr(gate, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"cannot recover an expert-axis logit tensor from {type(gate).__name__}")
    linear_bias = getattr(gate, "bias", None)
    if linear_bias is not None and not isinstance(linear_bias, torch.Tensor):
        linear_bias = None
    return F.linear(h.to(weight.dtype), weight, linear_bias)


def _selection_bias(dm, gate: torch.nn.Module) -> torch.Tensor | None:
    if not dm.gate_spec.use_bias:
        return None
    for name in (dm.spec.router_bias_attr, "e_score_correction_bias"):
        value = getattr(gate, name, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def local_gate_gradients(
    dm,
    gate: torch.nn.Module,
    hidden: torch.Tensor,
    target_experts: list[int],
    *,
    objectives: tuple[str, ...] = ("mass", "margin"),
) -> dict[str, torch.Tensor]:
    """Gradients of V4-faithful routing objectives with respect to one gate input."""
    if not target_experts:
        return {}
    h = hidden.detach().reshape(1, -1).requires_grad_(True)
    logits = _logits_from_gate(gate, h, dm.spec.n_experts).float()
    rr = gate_math.route(logits, _selection_bias(dm, gate), dm.gate_spec)
    ids = torch.as_tensor(sorted(set(target_experts)), dtype=torch.long, device=logits.device)
    ids = ids[(ids >= 0) & (ids < logits.shape[-1])]
    if ids.numel() == 0:
        return {}

    scalars = {}
    if "mass" in objectives:
        scalars["mass"] = rr.dense[0].index_select(0, ids).sum()
    if "margin" in objectives:
        margins = gate_math.selection_margin(
            rr.sel_scores, dm.gate_spec, ids, eligible=rr.eligible
        )[0]
        # A suppression direction only needs to move target experts that currently
        # survive the hard top-k. Already-excluded experts contribute zero.
        scalars["margin"] = torch.where(
            torch.isfinite(margins) & (margins > 0), margins, torch.zeros_like(margins)
        ).sum()

    out = {}
    names = list(scalars)
    for i, name in enumerate(names):
        scalar = scalars[name]
        grad = torch.autograd.grad(
            scalar, h, retain_graph=i < len(names) - 1, allow_unused=True
        )[0]
        if grad is not None:
            out[name] = grad[0].detach()
    return out


def gate_jacobian_spectrum(
    dm,
    prompts,
    *,
    expert_map: dict[int, list[int]] | None = None,
    objectives: tuple[str, ...] = ("mass", "margin"),
    want_template: bool = True,
) -> dict:
    """Measure per-layer routing-gradient spectra across prompts.

    With no expert map, the strongest currently selected expert is used for each
    prompt/layer. This intrinsic fallback avoids the constant ``sum(top-k weights)``
    objective; for a safety claim, pass the harvested safety-expert artifact instead.
    """
    invalid = sorted(set(objectives) - {"mass", "margin"})
    if invalid:
        raise ValueError(f"unknown Jacobian objectives: {invalid}")

    routers = _router_modules(dm)
    learned = set(dm.learned_layers)
    collected: dict[str, dict[int, list[torch.Tensor]]] = {
        name: {} for name in objectives
    }
    attempted: dict[str, dict[int, int]] = {name: {} for name in objectives}
    model, tok = dm.model, dm.tok
    device = next(model.parameters()).device

    for prompt in ui.iter_with_progress(list(prompts), "gate Jacobian"):
        ids = encode_prompt(tok, prompt, want_template=want_template, device=device).unsqueeze(0)
        with MoEHookManager(model, dm.spec) as hm, torch.no_grad():
            hm.capture_gate_input().capture_routing(dm.gate_spec)
            model(input_ids=ids, use_cache=False)
            gate_inputs = dict(hm.capture.gate_input)
            routing = dict(hm.capture.routing)

        for layer_idx in sorted(learned & set(gate_inputs) & set(routing) & set(routers)):
            rr = routing[layer_idx]
            if expert_map is None:
                best_pos = int(rr.weights[-1].argmax())
                targets = [int(rr.indices[-1, best_pos])]
            else:
                targets = expert_map.get(layer_idx, [])
            if not targets:
                continue
            hidden = gate_inputs[layer_idx].reshape(-1, gate_inputs[layer_idx].shape[-1])[-1]
            with torch.enable_grad():
                gradients = local_gate_gradients(
                    dm, routers[layer_idx], hidden, targets, objectives=objectives
                )
            for name, grad in gradients.items():
                attempted[name][layer_idx] = attempted[name].get(layer_idx, 0) + 1
                if torch.isfinite(grad).all() and float(grad.float().norm()) > 1e-12:
                    collected[name].setdefault(layer_idx, []).append(grad.cpu())

    result = {
        "target_source": "harvested_expert_map" if expert_map is not None else "strongest_selected_expert",
        "gradient_space": "local boundary-token gate input",
        "row_normalized": True,
        "hash_layers_excluded": True,
        "objectives": {},
    }
    summary_ranks = []
    for name in objectives:
        layers = {}
        for layer_idx in sorted(set(attempted[name]) | set(collected[name])):
            spec = gradient_spectrum(collected[name].get(layer_idx, []))
            spec["n_attempted"] = attempted[name].get(layer_idx, 0)
            layers[str(layer_idx)] = spec
            if spec.get("n_gradients", 0):
                summary_ranks.append(spec["effective_rank"])
        result["objectives"][name] = {"layers": layers}

    if summary_ranks:
        ordered = sorted(summary_ranks)
        median = ordered[len(ordered) // 2]
        result["median_effective_rank"] = median
        result["takeaway"] = (
            f"median effective rank across objective/layer spectra is {median:.2f}; "
            + ("routing gradients are strongly concentrated" if median <= 3 else
               "routing control is not uniformly low-rank")
        )
    else:
        result["takeaway"] = "no nonzero learned-layer gate gradients were captured"
    return result
