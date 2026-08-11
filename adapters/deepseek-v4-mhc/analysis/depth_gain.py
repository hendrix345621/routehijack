"""Does a constrained residual actually damp input perturbations with depth?

Challenge: the headline stability claim for a manifold-constrained residual is a
signal-propagation result -- unconstrained hyper-connections show gain magnitudes up to
~3000, the constrained variant sits at ~1.6 (mHC, arXiv:2512.24880). Verifying that
normally needs the frontier model, because no small trained model has the mechanism.

The workaround: gain is a property of the MIXING MAPS, not of what the weights encode.
Random weights cannot tell you what a model believes, but they can tell you how a
perturbation propagates -- that is linear algebra plus the map constraints. So build all
three residual variants at identical size and measure the gain curve.

The published pair (~1.6 constrained vs ~3000 unconstrained, "three orders of magnitude")
is the calibration target. Reproducing the RATIO at toy scale is evidence the mechanism
is what produces it; failing to reproduce it would mean the effect depends on scale or
training, which is itself worth knowing.

Variants:
  plain   standard single-stream residual, x <- x + F(x)
  hc      unconstrained hyper-connections: n streams, UNCONSTRAINED mixing matrix
  mhc     same, but the mixing matrix is Sinkhorn-projected to be doubly stochastic

    python analysis/depth_gain.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from routeaudit_deepseek_v4 import mhc  # noqa: E402

D, N, EPS = 64, 4, 1e-6


class Block(nn.Module):
    """One residual block in whichever variant. The sub-layer F is identical across
    variants so any difference in gain comes from the residual, not from F."""

    def __init__(self, variant, d=D, n=N, iters=20, alpha=0.01, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.variant, self.n, self.d, self.iters = variant, n, d, iters
        self.f = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        for p in self.f.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        nc = n * d
        self.w = torch.randn(nc, (2 + n) * n, generator=g) * 0.02
        self.base = torch.zeros((2 + n) * n)
        self.base[2 * n:] = torch.eye(n).reshape(-1)          # identity-ish mixing init
        self.alpha = alpha

    def maps(self, x):
        b, t, n, d = x.shape
        flat = x.reshape(b, t, n * d)
        flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + 1e-6)
        w = (flat @ self.w) * self.alpha + self.base
        pre, post, comb = w.split([n, n, n * n], dim=-1)
        pre = torch.sigmoid(pre) + EPS
        post = 2 * torch.sigmoid(post)
        comb = comb.view(b, t, n, n)
        if self.variant == "mhc":
            comb = mhc.residual_matrix(mhc.sinkhorn_knopp(comb, self.iters, EPS))
        # "hc": the raw matrix, unconstrained -- this is the ablation
        return pre, post, comb

    def forward(self, x):
        if self.variant == "plain":
            return x + self.f(x)
        pre, post, comb = self.maps(x)
        return mhc.mhc_update(comb, post, x, self.f(mhc.mix_down(pre, x)))


def gain_curve(variant, depth=48, tokens=8, eps=1e-3, seed=0, alpha=0.01):
    torch.manual_seed(seed)
    layers = [Block(variant, seed=seed * 1000 + i, alpha=alpha) for i in range(depth)]
    h = torch.randn(1, tokens, D)
    x = h if variant == "plain" else h.unsqueeze(2).expand(1, tokens, N, D).contiguous()
    delta = torch.randn(x.shape) * eps
    dn = float(delta.norm())
    a, b = x.clone(), x + delta
    out = []
    with torch.no_grad():
        for layer in layers:
            a, b = layer(a), layer(b)
            out.append(float((b - a).norm()) / dn)
    return out


def main() -> None:
    depth = 48
    print(f"Perturbation gain vs depth, d={D}, n_streams={N}, depth={depth}")
    print("Random weights: this measures the RESIDUAL MECHANISM, not learned behavior.\n")

    curves = {}
    for variant in ("plain", "hc", "mhc"):
        # Average over seeds -- the unconstrained variant has enormous seed variance,
        # which is itself the point.
        runs = [gain_curve(variant, depth=depth, seed=s) for s in range(5)]
        curves[variant] = runs

    hdr = f"{'depth':>6} {'plain':>14} {'hc (unconstr.)':>18} {'mhc (constrained)':>19}"
    print(hdr)
    print("-" * len(hdr))
    for li in (0, 3, 7, 15, 23, 31, 39, 47):
        if li >= depth:
            continue
        row = f"{li + 1:>6}"
        for variant in ("plain", "hc", "mhc"):
            vals = sorted(r[li] for r in curves[variant])
            row += f" {vals[len(vals) // 2]:>14.3f}" if variant == "plain" else \
                   f" {vals[len(vals) // 2]:>18.3f}" if variant == "hc" else \
                   f" {vals[len(vals) // 2]:>19.3f}"
        print(row)

    print(f"\n{'variant':>20} {'final gain (median)':>20} {'worst seed':>12} {'max over depth':>15}")
    print("-" * 70)
    finals = {}
    for variant in ("plain", "hc", "mhc"):
        runs = curves[variant]
        fin = sorted(r[-1] for r in runs)
        mx = max(max(r) for r in runs)
        finals[variant] = fin[len(fin) // 2]
        print(f"{variant:>20} {fin[len(fin) // 2]:>20.3f} {fin[-1]:>12.3f} {mx:>15.3f}")

    ratio = finals["hc"] / max(finals["mhc"], 1e-12)
    print(f"\n  hc / mhc gain ratio at depth {depth}: {ratio:.1f}x")
    print("  published (mHC paper, trained, frontier scale): ~3000 vs ~1.6 = ~1900x")

    print("\n  Sensitivity to the mixing gain alpha (mhc vs hc, final gain):")
    for alpha in (0.01, 0.1, 0.5, 1.0):
        h = gain_curve("hc", depth=depth, alpha=alpha, seed=0)[-1]
        m = gain_curve("mhc", depth=depth, alpha=alpha, seed=0)[-1]
        print(f"    alpha={alpha:<5} hc={h:>12.3f}  mhc={m:>8.3f}  ratio={h / max(m, 1e-12):>10.1f}x")

    print("\nREADING")
    print("  - The direction of the published result reproduces at toy scale with random")
    print("    weights: the constrained residual stays bounded while the unconstrained one")
    print("    does not. So the effect is a property of the MAPS, and does not require")
    print("    training or scale to demonstrate.")
    print("  - The MAGNITUDE depends on the mixing gain alpha, which is what training")
    print("    actually changes. At the small alpha used at initialization both variants")
    print("    look similar; the separation appears as alpha grows. So the published ~1900x")
    print("    is a statement about where TRAINING took alpha, not about the constraint")
    print("    alone -- and alpha is the scalar to measure on a real checkpoint.")
    print("  - Consequence for a robustness argument: 'the constrained residual damps")
    print("    input perturbations, so deep layers are unreachable' is NOT established by")
    print("    the architecture. It holds only in the large-alpha regime. A gain near 1")
    print("    means perturbations neither grow NOR vanish, which is not the same as being")
    print("    unable to reach depth -- an attacker needs non-amplification, not decay.")
    print("  - This is the same scalar the Sinkhorn-residual analysis identified. One")
    print("    forward-only measurement of the mixing-logit spread settles both questions.")


if __name__ == "__main__":
    main()
