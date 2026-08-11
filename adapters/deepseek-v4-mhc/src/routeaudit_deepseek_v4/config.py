"""Adapter registration and Hugging Face config augmentation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from routeaudit import config as core_config

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "deepseek_v4_flash.yaml"


def register() -> None:
    """Register adapter-owned model aliases and routing defaults with RouteAudit."""

    for alias in ("deepseek-v4-flash", "deepseek_v4_flash", "deepseek-v4"):
        core_config.MODELS[alias] = str(CONFIG_PATH)
    core_config._HF_TYPE_TO_PRESET["deepseek_v4"] = "deepseek"
    core_config._ROUTING_DEFAULTS["deepseek_v4"] = {
        "scoring_func": "sqrtsoftplus",
        "use_bias": True,
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 0,
        "routed_scaling_factor": 1.5,
        "num_hash_layers": 3,
    }


def model_ns_from_hf(hf_cfg, model_id: str, *, dtype: str, device_map: str) -> SimpleNamespace:
    """Build a core model namespace and add the adapter's residual metadata."""

    register()
    ns = core_config._model_ns_from_hf(hf_cfg, model_id, dtype=dtype, device_map=device_map)
    streams = getattr(hf_cfg, "hc_mult", None) or getattr(hf_cfg, "n_hc", None)
    if streams and int(streams) > 1:
        ns.mhc = SimpleNamespace(
            hc_mult=int(streams),
            hc_sinkhorn_iters=int(getattr(hf_cfg, "hc_sinkhorn_iters", 20) or 20),
        )
    return ns
