"""Forward hooks that capture router scores or selected experts from MoE models."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    expert_indices: dict[int, torch.Tensor] = field(default_factory=dict)

    def clear(self) -> None:
        self.router_logits.clear()
        self.expert_indices.clear()


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

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._capture_router = False
        self._detach_router = True
        self._gate_spec: gate_math.GateSpec | None = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def __enter__(self):
        self._install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ── capture switches (set before entering the context, or before forward) ──

    def capture_router_logits(self, detach: bool = True) -> MoEHookManager:
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

    def capture_expert_selection(self, gate_spec: gate_math.GateSpec) -> MoEHookManager:
        """Capture selected expert ids without retaining full router tensors."""
        self._gate_spec = gate_spec
        return self

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
            for attr in s.moe_block_attrs:  # first existing attr wins (version-robust)
                block = getattr(layer, attr, None)
                if block is not None:
                    break
            if block is None:
                continue
            # MoE blocks expose the router + experts containers named by the spec.
            if hasattr(block, s.experts_attr) and hasattr(block, s.router_attr):
                yield i, block

    def _install(self) -> None:
        for layer_idx, block in self._iter_moe_blocks():
            self._install_router_hook(layer_idx, block)

    def _gate_bias(self, module: torch.nn.Module, recompute: bool) -> torch.Tensor | None:
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

    def _logits_from_output(self, output, n_experts: int) -> torch.Tensor | None:
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

    def _record_lfm2_selection(self, layer_idx: int, output) -> None:
        if not isinstance(output, tuple) or len(output) < 3:
            return
        indices = output[2]
        if not isinstance(indices, torch.Tensor):
            return
        self.capture.expert_indices[layer_idx] = indices.detach()

    def _record_routing(self, layer_idx: int, module: torch.nn.Module, hidden: torch.Tensor, output) -> None:
        """Recompute and retain only this layer's selected expert ids."""
        if self.spec.router_output == "lfm2":
            self._record_lfm2_selection(layer_idx, output)
            return
        gs = self._gate_spec
        kind = gate_math.routing_kind(layer_idx, gs)
        if kind != gate_math.LEARNED:
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
        scores = gate_math.affinity(logits.float(), gs.scoring_func)
        sel = gate_math.selection_scores(scores, bias, gs)
        self.capture.expert_indices[layer_idx] = sel.topk(gs.top_k, dim=-1).indices

    def _install_router_hook(self, layer_idx: int, block: torch.nn.Module) -> None:
        """Hook ``block.<router_attr>`` to capture scores or routing decisions."""
        gate = getattr(block, self.spec.router_attr)
        mgr = self

        def fwd_hook(module, inputs, output):
            # Captures independent of what the gate returns.
            if mgr._gate_spec is not None and inputs:
                h = inputs[0]
                if isinstance(h, torch.Tensor):
                    mgr._record_routing(layer_idx, module, h, output)

            if mgr.spec.router_output == "lfm2":
                if not isinstance(output, tuple) or not output or not isinstance(output[0], torch.Tensor):
                    return output
                if mgr._capture_router:
                    mgr.capture.router_logits[layer_idx] = (
                        output[0].detach() if mgr._detach_router else output[0]
                    )
                return output

            # OLMoE gate output shape depends on transformers version:
            #   - Old (≤ ~4.46): the gate Linear returns raw logits (B*T, n_experts).
            #   - New: a fused gate returns (routing_scores, top_k_weights, top_k_index)
            #     where routing_scores is (B*T, n_experts) — either raw logits or
            #     softmax probs depending on the version. We treat it as routing
            #     scores.
            if isinstance(output, tuple):
                scores = output[0]
                if mgr._capture_router:
                    mgr.capture.router_logits[layer_idx] = scores.detach() if mgr._detach_router else scores
                return output
            # Legacy path: gate emits raw logits directly.
            logits = output
            if mgr._capture_router:
                mgr.capture.router_logits[layer_idx] = logits.detach() if mgr._detach_router else logits
            return logits

        self._handles.append(gate.register_forward_hook(fwd_hook))
