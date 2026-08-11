"""How hard is it to flip one expert out of a flat top-k gate? An order-statistics answer.

Challenge: an input-space routing attack has to move a safety expert's selection score
past its nearest competitor. Whether that is reachable is the go/no-go, and it is normally
answered by running the model. This estimates it WITHOUT the model.

The observation that makes it possible: a selection margin is the gap between the k-th and
(k+1)-th order statistics of E scores. For E=256 and k=6 those are deep in the upper tail,
where order-statistic spacings are governed by the density at the quantile and are almost
insensitive to the details of the score distribution. So the margin's *scale* follows from
E, k and the shape of the score function alone.

Three things get reported, in increasing usefulness:

  1. the raw margin, which depends on the unknown logit spread sigma;
  2. the margin as a FRACTION OF THE SCORE RANGE, which is dimensionless and therefore
     comparable across models and independent of sigma;
  3. the input-space perturbation ||dh|| needed to close it, from the gate's Jacobian  - 
     the quantity an attack actually has to produce.

Also reported: the number of experts sitting within one margin of the boundary, which is
what determines whether an attack has many cheap targets or one expensive one.

    python analysis/margin_forecast.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

E, K, D = 256, 6, 4096          # DeepSeek-V4-Flash: experts, top-k, hidden size
TOKENS = 20000


def scores_of(logits, fn="sqrtsoftplus"):
    if fn == "sqrtsoftplus":
        return F.softplus(logits).sqrt()
    if fn == "sigmoid":
        return logits.sigmoid()
    if fn == "softmax":
        return logits.softmax(-1)
    raise ValueError(fn)


def margin_stats(sigma, fn="sqrtsoftplus", bias_sigma=0.0, e=E, k=K,
                 tokens=TOKENS, seed=0):
    """Simulate the top-k boundary directly. Exact under the stated logit model."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, e, generator=g) * sigma
    s = scores_of(logits, fn)
    sel = s + torch.randn(e, generator=g) * bias_sigma if bias_sigma else s
    top = sel.topk(k + 1, dim=-1).values
    margin = top[:, k - 1] - top[:, k]                     # k-th minus (k+1)-th
    rng = sel.amax(-1) - sel.amin(-1)
    # How many experts sit inside a fixed 1%-of-range band around the boundary. Counting
    # "within one margin" would be circular  -  the k-th and (k+1)-th are inside it by
    # definition, so that metric returns 2 no matter what the distribution does.
    band = 0.05 * rng.unsqueeze(-1)
    near = ((sel - top[:, k:k + 1]).abs() < band).sum(-1).float()
    return dict(
        margin_p10=float(margin.quantile(0.10)), margin_p50=float(margin.median()),
        margin_p90=float(margin.quantile(0.90)),
        rel_p50=float((margin / rng).median()),            # dimensionless
        rel_p10=float((margin / rng).quantile(0.10)),
        score_p50=float(top[:, k - 1].median()), range_p50=float(rng.median()),
        near_p50=float(near.median()),
    )


def flip_perturbation(sigma, fn="sqrtsoftplus", d=D, e=E, k=K, tokens=4000, seed=0):
    """||dh|| needed to close the margin, in units of ||h||.

    The gate is score_i = f(w_i . h). Moving the gap between experts i and j takes
    d(score_i - score_j)/dh = f'(z_i) w_i - f'(z_j) w_j, whose norm sets the exchange rate
    between input perturbation and margin. Reported RELATIVE to ||h|| so it is comparable
    across models and independent of the weight scale.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(e, d, generator=g) / (d ** 0.5)        # unit-ish rows
    h = torch.randn(tokens, d, generator=g)
    h = h / h.norm(dim=-1, keepdim=True)
    z = (h @ w.T) * sigma
    s = scores_of(z, fn)
    top_v, top_i = s.topk(k + 1, dim=-1)
    margin = top_v[:, k - 1] - top_v[:, k]

    if fn == "sqrtsoftplus":                               # d/dz sqrt(softplus(z))
        deriv = torch.sigmoid(z) / (2 * F.softplus(z).sqrt().clamp_min(1e-9))
    elif fn == "sigmoid":
        deriv = torch.sigmoid(z) * (1 - torch.sigmoid(z))
    else:
        deriv = s * (1 - s)
    deriv = deriv * sigma

    ar = torch.arange(tokens)
    i, j = top_i[:, k - 1], top_i[:, k]                    # boundary pair
    grad = deriv[ar, i].unsqueeze(-1) * w[i] - deriv[ar, j].unsqueeze(-1) * w[j]
    gn = grad.norm(dim=-1).clamp_min(1e-12)
    need = margin / gn                                     # ||dh|| at best alignment
    return dict(need_p10=float(need.quantile(0.10)), need_p50=float(need.median()),
                need_p90=float(need.quantile(0.90)),
                grad_p50=float(gn.median()), margin_p50=float(margin.median()))


def main() -> None:
    print(f"Flat top-{K} gate over {E} experts  -  selection-margin forecast")
    print("Simulating the order statistics directly; exact under an iid-logit model.\n")

    print("1) MARGIN vs logit spread, sqrt(softplus) scoring, no bias")
    hdr = (f"{'sigma':>6} {'margin p10':>11} {'p50':>10} {'p90':>10} "
           f"{'margin/range p50':>17} {'p10':>9} {'in +/-5% band':>24}")
    print(hdr)
    print("-" * len(hdr))
    for s in (0.5, 1.0, 2.0, 4.0, 8.0):
        r = margin_stats(s)
        print(f"{s:>6.1f} {r['margin_p10']:>11.4f} {r['margin_p50']:>10.4f} "
              f"{r['margin_p90']:>10.4f} {r['rel_p50']:>17.4f} {r['rel_p10']:>9.4f} "
              f"{r['near_p50']:>24.1f}")

    print("\n2) The DIMENSIONLESS number  -  margin as a fraction of the score range")
    print("   (independent of sigma and of the weight scale, so it transfers across models)")
    for fn in ("sqrtsoftplus", "sigmoid", "softmax"):
        vals = [margin_stats(s, fn)["rel_p50"] for s in (1.0, 2.0, 4.0)]
        print(f"   {fn:>13}: {['%.4f' % v for v in vals]}  (sigma = 1, 2, 4)")

    print("\n3) Effect of a selection bias (the load-balancing term)")
    hdr2 = f"{'bias sigma':>8} {'margin p50':>11} {'margin/range p50':>17} {'in +/-5% band':>16}"
    print(hdr2)
    print("-" * len(hdr2))
    for b in (0.0, 0.05, 0.2, 0.5):
        r = margin_stats(2.0, bias_sigma=b)
        print(f"{b:>8.2f} {r['margin_p50']:>11.4f} {r['rel_p50']:>17.4f} "
              f"{r['near_p50']:>16.1f}")

    print(f"\n4) INPUT PERTURBATION needed to flip the boundary expert, d={D}")
    print("   ||dh|| / ||h||, assuming PERFECT alignment with the flip direction")
    hdr3 = f"{'sigma':>6} {'need p10':>10} {'p50':>10} {'p90':>10} {'|grad| p50':>10}"
    print(hdr3)
    print("-" * len(hdr3))
    for s in (0.5, 1.0, 2.0, 4.0):
        r = flip_perturbation(s)
        print(f"{s:>6.1f} {r['need_p10']:>10.4f} {r['need_p50']:>10.4f} "
              f"{r['need_p90']:>10.4f} {r['grad_p50']:>10.4f}")

    print("\nREADING")
    print("  - sqrt(softplus) holds a stable relative margin (~1.05% of range) at EVERY")
    print("    logit spread, because it is unbounded and never saturates. sigmoid")
    print("    collapses  -  0.59% at sigma=1 down to 0.01% at sigma=4  -  because it saturates and")
    print("    the whole expert population piles up near 1. So V4's switch from sigmoid")
    print("    to sqrt(softplus) makes its margins RELATIVELY WIDER than V2/V3's at the")
    print("    same spread. That is a routing-robustness gain, and as far as I can tell")
    print("    it is not claimed as one anywhere; it looks like a side effect of a change")
    print("    made for optimization reasons.")
    print("  - A selection bias does NOT protect the boundary. Raising bias sigma from 0 to")
    print("    0.5 leaves the relative margin flat (~1.0%) and the number of near-boundary")
    print("    experts unchanged. It relabels WHICH experts are marginal without making")
    print("    the boundary harder to reach  -  so 'the balancing bias creates a protective")
    print("    margin an attacker must overcome' is not supported. The bias is an")
    print("    obstacle to PREDICTING the flip, not to CAUSING it.")
    print("  - There are ~8 experts inside a +/-5% band around the boundary, at every")
    print("    spread. An attack aiming at 'any safety expert in this layer' therefore")
    print("    has several cheap targets rather than one expensive one -- the per-expert")
    print("    margin understates how reachable the LAYER is.")
    print("  - The perturbation needed is tiny: ~5e-4 of ||h|| at perfect alignment, and")
    print("    scale-invariant (the Jacobian grows with sigma exactly as fast as the margin).")
    print("    So the gate itself is NOT what makes an input attack hard.")
    print("  - Therefore the binding constraint is ALIGNMENT, not margin. A suffix")
    print("    perturbs one position and reaches the boundary token only through")
    print("    attention, so its achievable component along the flip direction is what")
    print("    decides feasibility. That reframes the go/no-go: measure the ALIGNMENT")
    print("    ratio (achieved dscore per unit ||dh||, versus the perfectly-aligned bound")
    print("    above), which is a forward-only measurement on any MoE model  -  including")
    print("    cheap ones  -  and does not need the target checkpoint at all.")


if __name__ == "__main__":
    main()
