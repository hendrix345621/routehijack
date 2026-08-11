"""Precision policy — when quantization is a measurement error and when it isn't.

The pipeline's standing rule is "no quantization, it perturbs the router logits"
(see `model/loader.py`), and for a bf16 checkpoint that is right. DeepSeek-V4 inverts
the framing, and getting this backwards costs accuracy in both directions:

1. **The shipped low precision IS the model.** V4's MoE experts (and the indexer QK
   path) were quantized to FP4 during post-training via Quantization-Aware Training.
   The routing behavior of the fp8/fp4 release is the real deployed behavior — there is
   no cleaner full-precision reference that it approximates.

2. **FP4→FP8 dequantization is lossless.** FP8 (E4M3) has two more exponent bits than
   FP4 (E2M1), and the released weights satisfy the scale-ratio condition under which
   the per-1x32-tile FP4 scales are absorbed exactly. So the experts can be materialized
   in fp8 with zero error — and a bitsandbytes-style re-quantization pass on top would
   introduce genuine perturbation that the deployed model does not have.

3. **Kernels are deterministic and batch-invariant.** DeepGEMM, the dual-kernel
   attention decoder, and the deterministic mHC reduction make outputs bitwise identical
   across runs and batch compositions, so "exact scores and margins" are well-defined
   observables rather than run-to-run noise.

There is still a published error bar to respect: the indexer runs QK in FP4 with 99.7%
top-k recall, so ~0.3% of selected KV blocks can differ from an FP32 indexer. Selection
margins finer than that are below the architecture's own noise floor — see
`below_noise_floor`.
"""
from __future__ import annotations

from enum import Enum
from types import SimpleNamespace
from typing import Optional

# Published FP4 indexer top-k recall (DeepSeek-V4 §5.2.1). 1 - this is the fraction of
# selected blocks that may differ from an FP32 indexer.
INDEXER_RECALL = 0.997
SELECTION_NOISE_RATE = 1.0 - INDEXER_RECALL

# Dtype strings that mean "this checkpoint already ships below bf16".
_NATIVE_LOW_PRECISION = ("fp8", "float8", "e4m3", "fp4", "float4", "e2m1", "nvfp4", "mxfp4")


class Claim(Enum):
    """What a measurement is going to be used to claim. Determines how much precision
    discipline it needs — structural claims tolerate quantization, quantitative ones
    don't."""

    STRUCTURAL = "structural"      # directions, subspaces, coarse routing statistics
    QUANTITATIVE = "quantitative"  # exact scores, margins, flip thresholds


PRECISION_POLICY: dict[Claim, dict] = {
    Claim.STRUCTURAL: dict(
        weights="as-shipped (experts fp4→fp8 lossless dequant); NF4/int8 acceptable on a "
                "bf16 checkpoint",
        kernels="any",
        note="never re-quantize below the shipped precision",
    ),
    Claim.QUANTITATIVE: dict(
        weights="as-shipped fp8/fp4 — this IS native precision (QAT), not an approximation",
        kernels="deterministic stack (DeepGEMM + batch-invariant attention + deterministic "
                "mHC reduction)",
        note=f"report margins against the {INDEXER_RECALL:.1%} indexer-recall noise floor",
    ),
}


def native_precision(model_ns) -> Optional[str]:
    """The checkpoint's shipped precision, if the config declares one below bf16.

    Reads `model.dtype`, `model.weights_dtype` and `model.expert_dtype` — the last is how
    a V4 config records that routed experts ship in fp4 while the rest is fp8.
    """
    for key in ("expert_dtype", "weights_dtype", "dtype"):
        v = getattr(model_ns, key, None)
        if isinstance(v, str) and any(t in v.lower() for t in _NATIVE_LOW_PRECISION):
            return v
    return None


def check_quant_policy(model_ns, quant: Optional[str],
                       claim: Claim = Claim.STRUCTURAL) -> tuple[bool, str]:
    """Should this (model, quantization, claim) combination be allowed?

    Returns `(ok, message)`. `ok=False` means the load would produce numbers that are
    wrong in a way no downstream analysis can correct for; callers should refuse rather
    than warn. A non-empty message with `ok=True` is a caveat worth printing.
    """
    q = (quant or "none").lower()
    native = native_precision(model_ns)
    mid = getattr(model_ns, "hf_id", "<model>")

    if native and q not in ("none", "", "native"):
        return False, (
            f"{mid} ships in {native} — those weights are QAT-native, so they ARE the "
            f"ground truth, not an approximation of a bf16 model. Applying quant={q!r} on "
            f"top adds error the deployed model does not have (and fp4→fp8 dequant is "
            f"lossless, so there is nothing to gain). Load as-shipped: quant='none'."
        )

    if claim is Claim.QUANTITATIVE and q not in ("none", "", "native"):
        return True, (
            f"quant={q!r} with a QUANTITATIVE claim: exact scores, margins and flip "
            f"thresholds are not trustworthy under re-quantization. Use this run to "
            f"PROBE, then confirm the numbers as-shipped."
        )

    if q not in ("none", "", "native"):
        return True, (
            f"DIAGNOSTIC quant={q!r} — routing is perturbed. Fine for structural claims "
            f"(which layers/experts, coarse shifts); confirm any exact number as-shipped."
        )

    return True, ""


def below_noise_floor(margin: float, noise_rate: float = SELECTION_NOISE_RATE) -> bool:
    """Is a selection margin too small to distinguish from the architecture's own noise?

    The indexer's FP4 top-k recall means ~`noise_rate` of selections can differ run to
    run against an FP32 reference. A "routing flip" whose margin is within that band is
    not evidence of anything — discard it rather than reporting it.
    """
    return abs(margin) <= noise_rate


def policy_banner(model_ns, quant: Optional[str], claim: Claim = Claim.STRUCTURAL) -> dict:
    """Compact, printable record of the precision decisions a run made — meant to be
    written into the artifacts alongside the numbers, so a result carries its own
    reproducibility context."""
    ok, msg = check_quant_policy(model_ns, quant, claim)
    return {
        "claim": claim.value,
        "quant": quant or "none",
        "native_precision": native_precision(model_ns) or "bf16/fp16 (no QAT)",
        "policy": PRECISION_POLICY[claim],
        "allowed": ok,
        "note": msg,
        "indexer_recall_noise_floor": SELECTION_NOISE_RATE,
    }


def _demo() -> SimpleNamespace:  # pragma: no cover - documentation helper
    """The two cases that matter, side by side."""
    flash = SimpleNamespace(hf_id="deepseek-ai/DeepSeek-V4-Flash", dtype="fp8", expert_dtype="fp4")
    lite = SimpleNamespace(hf_id="deepseek-ai/DeepSeek-V2-Lite-Chat", dtype="bfloat16")
    return SimpleNamespace(
        flash_nf4=check_quant_policy(flash, "nf4"),   # (False, "...ARE the ground truth...")
        lite_nf4=check_quant_policy(lite, "nf4"),     # (True,  "DIAGNOSTIC quant...")
    )
