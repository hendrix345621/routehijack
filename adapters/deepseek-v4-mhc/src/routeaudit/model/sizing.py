"""Model-size-aware defaults for the white-box attack.

The attack's per-step memory is dominated by (a) the batched candidate forward and
(b) the grad pass's backward activations. The hardcoded `128 / 8 / 16`
(candidate_batch_size / grad_batch_size / n_prompts) is fine at ~7B but OOMs well
before 235B. `autoscale_attack_batches` picks conservative starting points by total
parameter count so a large run doesn't crash on its first step; everything here is
**quality-neutral** (∇ of a sum = sum of ∇s; candidates are scored identically) — it
only changes how work is chunked. Always overridable.
"""
from __future__ import annotations

B = 1_000_000_000

# (upper_bound_total_params, defaults). First tier whose bound the model is under wins.
_TIERS = [
    (20 * B, dict(candidate_batch_size=128, grad_batch_size=8, n_prompts=16)),   # ≤7–14B
    (60 * B, dict(candidate_batch_size=48, grad_batch_size=4, n_prompts=16)),    # 30–47B
    (120 * B, dict(candidate_batch_size=24, grad_batch_size=2, n_prompts=12)),   # ~70–100B
    (300 * B, dict(candidate_batch_size=12, grad_batch_size=1, n_prompts=8)),    # ~235B
]
_HUGE = dict(candidate_batch_size=8, grad_batch_size=1, n_prompts=8)            # >300B


def param_count(model) -> int:
    """Total parameters (works for sharded / offloaded models)."""
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:  # noqa: BLE001
        return 0


def autoscale_attack_batches(total_params: int) -> dict:
    """Return conservative {candidate_batch_size, grad_batch_size, n_prompts} for a
    model of `total_params`. Also suggests enabling prefix-cache + grad checkpointing
    for anything past the smallest tier."""
    chosen = _HUGE
    for bound, defaults in _TIERS:
        if total_params < bound:
            chosen = defaults
            break
    out = dict(chosen)
    out["use_prefix_cache"] = total_params >= 20 * B
    out["grad_checkpointing"] = total_params >= 20 * B
    return out
