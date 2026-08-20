"""Architecture spec — tells :class:`MoEHookManager` how to find the router and
experts on a given MoE family, so the hook layer is not hardcoded to one model.

A spec is resolved from the model config: pick a named preset (``olmoe``,
``mixtral``) and override any field explicitly in YAML. Adding a new family is a
new preset entry plus the dims in that family's config.

Verified layouts:

  - OLMoE    : ``model.model.layers[i].mlp``               (block) → ``.gate`` + ``.experts``
  - Mixtral  : ``model.model.layers[i].block_sparse_moe``  (block) → ``.gate`` + ``.experts``
  - DeepSeek : ``model.model.layers[i].mlp``               (block) → ``.gate`` + ``.experts``
  - GLM-4.5  : ``model.model.layers[i].mlp``               (block) → ``.gate`` + ``.experts``

The first two expose ``gate`` as an ``nn.Linear`` returning ``(T, n_experts)`` logits and
``experts`` as an ``nn.ModuleList`` whose members return ``(n_routed, d_model)`` — the
only contract the kept attacks rely on. DeepSeek's gate breaks the first half of that
(it returns ``(weights, indices)``), which is what ``router_output="recompute"`` and the
companion :mod:`routeaudit.model.gate_math` exist to handle.
"""

from __future__ import annotations

from dataclasses import dataclass

# Per-family module-layout presets. `moe_block_attrs` lists candidate attribute
# names tried in order (first existing wins) so a spec survives cross-version
# renames (e.g. Mixtral moved block_sparse_moe → mlp when experts were fused).
PRESETS: dict[str, dict] = {
    "olmoe": {
        "base_attr": "model",
        "layers_attr": "layers",
        "moe_block_attrs": ("mlp",),
        "router_attr": "gate",
        "experts_attr": "experts",
        "router_output": "auto",
    },
    "mixtral": {
        "base_attr": "model",
        "layers_attr": "layers",
        "moe_block_attrs": ("block_sparse_moe", "mlp"),
        "router_attr": "gate",
        "experts_attr": "experts",
        "router_output": "logits",
    },
    # Qwen2-MoE (Qwen1.5-MoE, Qwen2-57B-A14B) and Qwen3-MoE (Qwen3-30B-A3B, 235B-A22B):
    # per-layer `.mlp` is the SparseMoeBlock with `.gate` (Linear → raw logits) and
    # `.experts` (ModuleList of routed experts). Qwen2-MoE also has an always-on
    # `.shared_expert` — deliberately NOT hooked (it isn't routed, so it carries no
    # routing-level safety signal). One preset covers both; dims live in the YAML.
    "qwen": {
        "base_attr": "model",
        "layers_attr": "layers",
        "moe_block_attrs": ("mlp",),
        "router_attr": "gate",
        "experts_attr": "experts",
        "router_output": "logits",
    },
    # Phi-3.5-MoE (`microsoft/Phi-3.5-MoE-instruct`, model_type `phimoe`):
    # per-layer `.block_sparse_moe` is `PhiMoESparseMoeBlock` with `.gate`
    # (plain nn.Linear → raw `(T, n_experts)` logits) and `.experts` (nn.ModuleList
    # of `PhiMoEBlockSparseTop2MLP`). Structurally identical to Mixtral — the gate
    # is a clean Linear, so router capture and suffix gradients work. VERIFIED layout.
    "phimoe": {
        "base_attr": "model",
        "layers_attr": "layers",
        "moe_block_attrs": ("block_sparse_moe", "mlp"),
        "router_attr": "gate",
        "experts_attr": "experts",
        "router_output": "logits",
    },
    # DeepSeekMoE V2/V3. Layout is standard — `.mlp` holds `.gate` + `.experts`
    # — but the gate is NOT a plain Linear: it returns `(weights, indices)` and never
    # exposes its pre-selection scores, so `router_output="recompute"` tells the hooks to
    # form the logits from the gate's own weight matrix and re-derive routing through a
    # `gate_math.GateSpec` (sqrtsoftplus/sigmoid affinity, selection-only balancing bias,
    # flat or node-limited top-k). `.shared_experts` is always-on and deliberately NOT
    # hooked — it isn't routed, so it carries no routing-level safety signal.
    #
    # Group-limited selection is controlled declaratively through `routing.n_group`.
    "deepseek": {
        "base_attr": "model",
        "layers_attr": "layers",
        "moe_block_attrs": ("mlp",),
        "router_attr": "gate",
        "experts_attr": "experts",
        "router_output": "recompute",
        "router_bias_attr": "e_score_correction_bias",
    },
    "glm4_moe": {
        "base_attr": "model",
        "layers_attr": "layers",
        "moe_block_attrs": ("mlp",),
        "router_attr": "gate",
        "experts_attr": "experts",
        "router_output": "logits",
        "router_bias_attr": "e_score_correction_bias",
    },
    # Liquid LFM2.5 MoE: two initial dense layers, then each decoder layer's
    # `.feed_forward` is a sparse block with `.gate` and fused `.experts`.
    # Its gate uses sigmoid scores and an expert-bias argument for selection only.
    "lfm2": {
        "base_attr": "model",
        "layers_attr": "layers",
        "moe_block_attrs": ("feed_forward",),
        "router_attr": "gate",
        "experts_attr": "experts",
        "router_output": "lfm2",
    },
}

_FIELDS = (
    "base_attr",
    "layers_attr",
    "moe_block_attrs",
    "router_attr",
    "experts_attr",
    "router_output",
    "router_bias_attr",
)


@dataclass(frozen=True)
class ArchSpec:
    """How to reach the MoE router/experts on a model.

    `router_output`:
      - "logits"    — gate forward returns a raw ``(T, n_experts)`` logit tensor.
      - "auto"      — detect tuple-vs-tensor at runtime (some HF versions return a
                      fused ``(scores, topk_w, topk_idx)`` tuple from the gate).
      - "recompute" — the gate's output carries no usable score tensor (DeepSeek returns
                      only ``(weights, indices)``); routing is re-derived from the gate's
                      *input* and weight matrix during expert harvesting.

    `router_bias_attr` names the auxiliary-loss-free load-balancing bias on the gate
    module, used for expert SELECTION only (never for the gating weight). Only read on
    "recompute" specs or when a ``GateSpec`` explicitly enables it.
    """

    name: str = "olmoe"
    base_attr: str = "model"
    layers_attr: str = "layers"
    moe_block_attrs: tuple[str, ...] = ("mlp",)
    router_attr: str = "gate"
    experts_attr: str = "experts"
    router_output: str = "auto"
    router_bias_attr: str = "e_score_correction_bias"
    # Dims (informational / for validation against discovered modules).
    n_layers: int = 0
    n_experts: int = 0
    top_k: int = 0
    d_model: int = 0

    @classmethod
    def from_config(cls, model_ns) -> ArchSpec:
        """Build a spec from a config ``model`` namespace.

        Reads an optional ``model.arch`` block: ``arch.name`` selects a preset
        and any other ``arch.*`` key overrides that preset's field. With no
        ``arch`` block the OLMoE preset is used (backward-compatible default).
        """
        arch = getattr(model_ns, "arch", None)
        name = (getattr(arch, "name", None) if arch is not None else None) or "olmoe"
        merged = dict(PRESETS.get(name, PRESETS["olmoe"]))
        if arch is not None:
            for key in _FIELDS:
                val = getattr(arch, key, None)
                if val is not None:
                    merged[key] = val
        merged["moe_block_attrs"] = tuple(merged["moe_block_attrs"])
        return cls(
            name=name,
            n_layers=int(getattr(model_ns, "n_layers", 0) or 0),
            n_experts=int(getattr(model_ns, "n_experts", 0) or 0),
            top_k=int(getattr(model_ns, "top_k", 0) or 0),
            d_model=int(getattr(model_ns, "d_model", 0) or 0),
            **merged,
        )
