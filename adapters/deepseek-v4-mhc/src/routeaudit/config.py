"""YAML config loader. Returns a SimpleNamespace tree so attribute access works in scripts.

`load` accepts three things, so picking a model is easy and unambiguous:
  1. a config path     — `configs/qwen3_moe.yaml`
  2. a short nickname  — `qwen3`  (see `MODELS`)
  3. a HuggingFace id  — `Qwen/Qwen3-30B-A3B`  → fetches the model's config, detects
                         the MoE family + dims automatically; if the model is not a
                         supported routed-expert MoE it raises `UnsupportedModelError`
                         with an explanation.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

# Repo root = .../<repo>/src/routeaudit/config.py → parents[2]; configs/ live there.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Short model nicknames → config files. Add a line here when you add a config.
MODELS: dict[str, str] = {
    # LFM2.5-8B-A1B is the default target. Keep OLMoE as an explicit,
    # lower-cost regression target rather than making it the implicit target.
    "liquid": "configs/lfm2_5_8b_a1b.yaml",
    "lfm2": "configs/lfm2_5_8b_a1b.yaml",
    "lfm2.5": "configs/lfm2_5_8b_a1b.yaml",
    "lfm2.5-8b-a1b": "configs/lfm2_5_8b_a1b.yaml",
    "lfm2_5_8b_a1b": "configs/lfm2_5_8b_a1b.yaml",
    "base": "configs/lfm2_5_8b_a1b.yaml",
    "olmoe": "configs/olmoe.yaml",
    "mixtral": "configs/mixtral.yaml",
    "qwen2": "configs/qwen2_moe.yaml",
    "qwen2_moe": "configs/qwen2_moe.yaml",
    "qwen3": "configs/qwen3_moe.yaml",
    "qwen3_moe": "configs/qwen3_moe.yaml",
    "qwen3-think-smoke": "configs/qwen3_30b_a3b_fp8_think_smoke.yaml",
    "qwen3_think_smoke": "configs/qwen3_30b_a3b_fp8_think_smoke.yaml",
    "qwen3-235b": "configs/qwen3_235b_a22b.yaml",
    "qwen3_235b": "configs/qwen3_235b_a22b.yaml",
    "qwen3-235b-a22b": "configs/qwen3_235b_a22b.yaml",
    "qwen3.5": "configs/qwen3_5_moe.yaml",  # best-effort / unverified — see the config header
    "qwen3_5": "configs/qwen3_5_moe.yaml",
    "qwen3.6": "configs/qwen3_6_35b_a3b.yaml",  # dims verified from config.json (hybrid attention)
    "qwen3_6": "configs/qwen3_6_35b_a3b.yaml",
    "qwen3.6-think": "configs/qwen3_6_35b_a3b_think.yaml",  # same model, thinking mode ON (A2 attack)
    "qwen3_6_think": "configs/qwen3_6_35b_a3b_think.yaml",
    # DeepSeekMoE V2-Lite exercises the grouped-gate path. Architecture-specific
    # integrations live in optional adapters rather than the universal evaluator.
    "deepseek-v2-lite": "configs/deepseek_v2_lite.yaml",
    "deepseek_v2_lite": "configs/deepseek_v2_lite.yaml",
    "smoke": "configs/smoke.yaml",
}


def list_models() -> list[str]:
    return sorted(set(MODELS))


def resolve_config_path(spec: str | Path) -> Path:
    """Resolve a path *or* a model nickname to a concrete config file.

    Tries, in order: an existing path as given; `MODELS[nickname]` under the repo
    root; `configs/<nickname>.yaml` under the repo root. Raises with the known
    nicknames if nothing matches.
    """
    p = Path(spec)
    if p.exists():
        return p
    key = str(spec).lower()
    key = key[:-5] if key.endswith(".yaml") else key
    if key in MODELS:
        cand = _REPO_ROOT / MODELS[key]
        if cand.exists():
            return cand
    cand2 = _REPO_ROOT / "configs" / f"{key}.yaml"
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"config '{spec}' not found. Use a path, or a model nickname: {list_models()}.")


# ─────────────────────── HuggingFace model-id auto-detection ───────────────────────

# HF `model_type` → our ArchSpec preset. Only routed-expert MoE families whose layout
# our hooks support. Add a line here (and an ArchSpec preset) to support a new family.
_HF_TYPE_TO_PRESET: dict[str, str] = {
    "olmoe": "olmoe",
    "mixtral": "mixtral",
    "qwen2_moe": "qwen",
    "qwen3_moe": "qwen",  # covers Qwen3-30B-A3B, Qwen3-235B-A22B
    "qwen3_next": "qwen",  # best-effort: Qwen3-Next hybrid MoE (standard Linear gate)
    "qwen3_5_moe": "qwen",  # Qwen3.5 / Qwen3.6 hybrid-attention MoE (every layer has a standard
    # MoE mlp; linear/full attention sublayers don't affect router capture)
    "lfm2_moe": "lfm2",
    "phimoe": "phimoe",
    # DeepSeekMoE. The gate is not a softmax — its semantics come from the `routing:`
    # block via `gate_math.GateSpec`, filled in by `_routing_ns_from_hf` below.
    "deepseek_v2": "deepseek",
    "deepseek_v3": "deepseek",
}

# Per-family gate defaults, applied when the HF config doesn't state them.
_ROUTING_DEFAULTS: dict[str, dict] = {
    "deepseek_v2": dict(scoring_func="sigmoid", use_bias=True, norm_topk_prob=True),
    "deepseek_v3": dict(scoring_func="sigmoid", use_bias=True, norm_topk_prob=True),
}


class UnsupportedModelError(ValueError):
    """Raised when a HuggingFace model id is not a supported routed-expert MoE."""


def _hf_get(cfg, *names, default=None):
    for n in names:
        v = getattr(cfg, n, None)
        if v is not None:
            return v
    return default


def _routing_ns_from_hf(hf_cfg, mt: str, top_k: int) -> SimpleNamespace | None:
    """Build the `model.routing` block (read by `gate_math.GateSpec.from_config`).

    Returns None for families whose gate is a plain softmax over logits — those need no
    routing block, and omitting it keeps their configs byte-identical to before.
    """
    defaults = _ROUTING_DEFAULTS.get(mt or "")
    if defaults is None:
        return None
    ns = dict(
        defaults,
        top_k=top_k,
        n_group=int(_hf_get(hf_cfg, "n_group", default=defaults.get("n_group", 1)) or 1),
        topk_group=int(_hf_get(hf_cfg, "topk_group", default=defaults.get("topk_group", 0)) or 0),
        first_k_dense_replace=int(_hf_get(hf_cfg, "first_k_dense_replace", default=0) or 0),
        num_hash_layers=int(
            _hf_get(hf_cfg, "num_hash_layers", default=defaults.get("num_hash_layers", 0)) or 0
        ),
    )
    for key, names in (
        ("scoring_func", ("scoring_func",)),
        ("norm_topk_prob", ("norm_topk_prob",)),
        ("routed_scaling_factor", ("routed_scaling_factor", "route_scale")),
    ):
        v = _hf_get(hf_cfg, *names)
        if v is not None:
            ns[key] = v
    return SimpleNamespace(**ns)


def _model_ns_from_hf(hf_cfg, model_id: str, *, dtype: str, device_map: str) -> SimpleNamespace:
    """Build the `model` config block from a fetched HF config, or raise
    UnsupportedModelError with an explanation. Pure (no network) — testable."""
    mt = getattr(hf_cfg, "model_type", None)
    preset = _HF_TYPE_TO_PRESET.get(mt or "")
    n_experts = _hf_get(hf_cfg, "num_experts", "num_local_experts", "n_routed_experts")
    top_k = _hf_get(hf_cfg, "num_experts_per_tok", "num_experts_per_token", "moe_topk")
    n_layers = _hf_get(hf_cfg, "num_hidden_layers")
    d_model = _hf_get(hf_cfg, "hidden_size", "d_model")
    if preset is None or not n_experts:
        raise UnsupportedModelError(
            f"'{model_id}' (model_type={mt!r}) is not a supported MoE. This tool needs a "
            f"routed-expert Mixture-of-Experts; supported families: OLMoE, Mixtral, "
            f"Qwen2/3-MoE, Liquid LFM2.5-MoE, Phi-MoE, and DeepSeekMoE V2/V3 (model_type "
            f"{sorted(_HF_TYPE_TO_PRESET)}). Other MoE variants (e.g. DBRX, GPT-OSS, "
            f"Granite-MoE) need a hand-written config in configs/ plus a matching "
            f"ArchSpec preset in model/archspec.py."
        )
    ns = SimpleNamespace(
        hf_id=model_id,
        dtype=dtype,
        device_map=device_map,
        n_layers=int(n_layers),
        n_experts=int(n_experts),
        top_k=int(top_k or 0),
        d_model=int(d_model),
        d_expert=int(_hf_get(hf_cfg, "moe_intermediate_size", "intermediate_size", default=0) or 0),
        arch=SimpleNamespace(name=preset),
    )
    routing = _routing_ns_from_hf(hf_cfg, mt or "", int(top_k or 0))
    if routing is not None:
        ns.routing = routing
    if mt == "lfm2_moe":
        ns.routing = SimpleNamespace(
            scoring_func="sigmoid",
            use_bias=bool(_hf_get(hf_cfg, "use_expert_bias", default=False)),
            norm_topk_prob=bool(_hf_get(hf_cfg, "norm_topk_prob", default=True)),
            routed_scaling_factor=float(_hf_get(hf_cfg, "routed_scaling_factor", default=1.0)),
            top_k=int(top_k or 0),
        )
    # QAT-native checkpoints can declare their shipped precision so the loader can
    # refuse a second quantization pass that would add measurement error.
    for key in ("expert_dtype", "weights_dtype", "quantization_dtype"):
        v = _hf_get(hf_cfg, key)
        if isinstance(v, str):
            setattr(ns, "expert_dtype" if key == "expert_dtype" else "weights_dtype", v)
    return ns


def from_hf(
    model_id: str, *, template: str = "base", dtype: str = "bfloat16", device_map: str = "auto"
) -> SimpleNamespace:
    """Build a full config for a HuggingFace MoE model id, auto-detecting arch + dims.

    Non-model blocks (identify/attacks/eval) are inherited from the `template`
    config (defaults).
    """
    try:
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except UnsupportedModelError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"could not fetch HuggingFace config for '{model_id}': {type(e).__name__}: {e}. "
            f"Check the model id is correct, you have network access, and (for gated models) "
            f"that you ran `huggingface-cli login`."
        ) from e
    cfg = load(template)
    cfg.model = _model_ns_from_hf(hf_cfg, model_id, dtype=dtype, device_map=device_map)
    return cfg


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        # Coerce keys to str: YAML parses mapping keys like `0:` (e.g. a max_memory
        # device map) as ints, which `SimpleNamespace(**...)` rejects. `_coerce_max_memory`
        # in model/loader.py turns "0" back into a device int where needed.
        return SimpleNamespace(**{str(k): _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load(path: str | Path) -> SimpleNamespace:
    """Resolve a path / nickname / HuggingFace model id to a config namespace."""
    spec = str(path)
    try:
        resolved = resolve_config_path(spec)
    except FileNotFoundError:
        resolved = None
    if resolved is not None:
        with open(resolved, "r", encoding="utf-8") as fh:
            return _to_ns(yaml.safe_load(fh))
    if spec.lower().endswith((".yaml", ".yml")):
        raise FileNotFoundError(f"config file '{spec}' not found.")
    # Not a file or nickname → treat as a HuggingFace model id.
    return from_hf(spec)


def to_dict(ns: SimpleNamespace) -> dict:
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [to_dict(v) for v in ns]
    return ns
