"""Validate our gate/mHC reimplementation against fixtures from the released model.

Level 1 of the validation ladder (see `extract.py`). This is the gate on plan.md's Phase
P0: "corrected diagnostic reproduces the released Gate's (weights, indices) bit-for-bit
on a saved tensor fixture."

Format-v2 fixtures establish independent same-device router parity and direct real-map
checks. Legacy format-v1 fixtures remain useful structural evidence but are reported as
partial because they did not retain those two independent measurements.

    python fixtures/validate.py
    python fixtures/validate.py --fixtures path/to/v4_flash_fixtures.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# Run from a fresh clone without `pip install -e .`; an installed package wins.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.torch_version import TorchVersion

from routeaudit import ui
from routeaudit.model import gate_math
from routeaudit_deepseek_v4 import mhc
from routeaudit.model.gate_math import GateSpec

DEFAULT_FIXTURES = Path(__file__).resolve().parent / "v4_flash_fixtures.pt"
DEFAULT_REPLAY_ATOL = 2e-7


def load_fixture(path: Path) -> dict:
    """Load tensor-only fixtures without enabling arbitrary pickle execution.

    Format-v1 files stored ``torch.__version__`` as ``TorchVersion`` rather than a
    string, so that one harmless class is allowlisted for backward compatibility.
    """
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def _cmp(name: str, got: torch.Tensor, want: torch.Tensor, atol: float) -> tuple[bool, str]:
    if got.shape != want.shape:
        return False, f"{name}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    dev = float((got.float() - want.float()).abs().max())
    ok = torch.equal(got, want) if atol == 0 else dev <= atol
    return ok, f"{name}: max|Δ| = {dev:.3e} ({'exact' if dev == 0 else f'atol={atol:g}'})"


def _align(values: torch.Tensor, indices: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    """Align top-k values to ``target_indices`` without treating order as semantics."""
    order = indices.argsort(-1)
    target_order = target_indices.argsort(-1)
    aligned = torch.empty_like(values)
    aligned.scatter_(-1, target_order, torch.gather(values, -1, order))
    return aligned


def validate_gate(fx: dict, atol: float) -> list[tuple[bool | None, str]]:
    """Validate same-device official parity and portable CPU replay separately.

    Format-v1 fixtures contain only RouteAudit's GPU recomputation. They remain useful
    evidence, but cannot retroactively prove equality with values returned by the model's
    router; that missing measurement is reported as ``None`` rather than fabricated.
    """
    g = fx.get("gate")
    if not g:
        return [(False, "no 'gate' fixture in file")]
    gs = GateSpec(**fx["meta"]["gate_spec"])
    scores = g["scores"]
    bias = (g["sel_scores"] - scores)[0] if gs.use_bias else None
    out: list[tuple[bool | None, str]] = []

    official_idx = g.get("official_indices")
    official_w = g.get("official_weights")
    if isinstance(official_idx, torch.Tensor) and isinstance(official_w, torch.Tensor):
        same_official_set = torch.equal(
            g["indices"].sort(-1).values, official_idx.sort(-1).values
        )
        out.append(
            (
                same_official_set,
                "official gate indices: "
                + ("same expert set" if same_official_set else "DIFFERENT experts fire"),
            )
        )
        if same_official_set:
            aligned_official = _align(official_w, official_idx, g["indices"])
            out.append(_cmp("same-device official gate weights", g["weights"], aligned_official, 0))
    else:
        out.append(
            (
                None,
                (
                    "official gate output was not retained by this legacy fixture; "
                    "same-device shipped parity is unmeasured"
                ),
            )
        )

    # Portable replay: reconstruct from the saved GPU scores on CPU. A one-ULP
    # reduction/division difference is expected across devices, hence a small explicit
    # tolerance rather than a false bitwise requirement.
    sel = gate_math.selection_scores(scores, bias, gs)
    idx = sel.topk(gs.top_k, dim=-1).indices
    w = gate_math.gate_weights(scores, idx, gs)
    out.append(_cmp("portable sel_scores replay", sel, g["sel_scores"], atol))
    same_set = torch.equal(idx.sort(-1).values, g["indices"].sort(-1).values)
    out.append(
        (same_set, f"portable indices replay: {'same expert set' if same_set else 'DIFFERENT experts fire'}")
    )
    if same_set:
        out.append(_cmp("portable weights replay", w, _align(g["weights"], g["indices"], idx), atol))
    return out


def validate_residual(fx: dict) -> list[tuple[bool, str]]:
    """Check the captured residual is shaped the way the config claims, and that the
    documented reduction applies to it. A stream-count mismatch means every residual-space
    number from that model was computed on the wrong tensor."""
    r = fx.get("residual")
    if not r:
        return [(False, "no residual fixture")]
    n = int(r["n_streams"])
    h = r["hidden"]
    expected = 4
    ok = h.dim() == 4 and n == expected and h.shape[-2] == expected
    lines = [
        (
            ok,
            (
                f"residual streams: recorded n={n}, tensor rank {h.dim()} "
                f"{'consistent' if ok else 'INCONSISTENT'} (expected {expected})"
            ),
        )
    ]
    if n > 1:
        red = mhc.reduce_residual(h, n, "mean")
        lines.append(
            (
                red.shape[-1] == h.shape[-1] and red.dim() == h.dim() - 1,
                f"stream-mean reduction: {tuple(h.shape)} → {tuple(red.shape)}",
            )
        )
    return lines


def validate_mhc_maps(fx: dict, tol: float = 1e-4) -> list[tuple[bool | None, str]]:
    """Check maps returned by the released model's real HyperConnection modules."""
    captured = fx.get("mhc_maps")
    if not captured:
        return [
            (
                None,
                (
                    "real HyperConnection maps are absent from this legacy fixture; "
                    "four-stream shape is verified but B-path conservation is unmeasured"
                ),
            )
        ]
    sites = captured.get("sites") or {}
    out: list[tuple[bool | None, str]] = [
        (set(sites) == {"attn", "ffn"}, f"HyperConnection sites captured: {sorted(sites)}")
    ]
    for site, values in sorted(sites.items()):
        hidden = values["hidden_streams"]
        post = values["post"]
        comb = values["comb"]
        collapsed = values["collapsed"]
        n = hidden.shape[-2] if hidden.dim() == 4 else 0
        shapes_ok = (
            hidden.dim() == 4
            and post.shape == hidden.shape[:-1]
            and comb.shape == (*hidden.shape[:-2], n, n)
            and collapsed.shape == (*hidden.shape[:-2], hidden.shape[-1])
        )
        out.append((shapes_ok, f"{site} HyperConnection shapes consistent (streams={n})"))
        if shapes_ok:
            metrics = mhc.b_path_conservation_check(mhc.residual_matrix(comb), hidden, tol=tol)
            intact = bool(
                metrics["doubly_stochastic"]
                and metrics["non_expansive"]
                and metrics.get("mean_conserved", False)
            )
            out.append(
                (
                    intact,
                    (
                        f"{site} B-path: row_dev={metrics['row_sum_dev']:.3e}, "
                        f"col_dev={metrics['col_sum_dev']:.3e}, "
                        f"spectral_norm={metrics['spectral_norm_max']:.6f}, "
                        f"stream_mean_dev={metrics.get('stream_mean_dev', float('nan')):.3e}"
                    ),
                )
            )
    return out


def validate_hash(fx: dict) -> list[tuple[bool, str]]:
    """The free oracle: hash routing must reproduce the static table exactly."""
    h = fx.get("hash")
    if not h:
        return [(False, "no hash fixture")]
    table = h["table"]
    ids = torch.arange(min(256, table.shape[0]))
    ok = torch.equal(gate_math.hash_route(ids, table), table[ids])
    return [
        (
            ok,
            (
                f"hash routing reproduces the token-id table for {len(ids)} ids "
                f"({'exact' if ok else 'MISMATCH'})"
            ),
        )
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    ap.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_REPLAY_ATOL,
        help="portable CPU replay tolerance (default: 2e-7). Same-device official "
        "gate parity is always checked bitwise.",
    )
    ap.add_argument(
        "--require-complete",
        action="store_true",
        help="fail when a legacy fixture lacks official gate outputs or real mHC maps",
    )
    args = ap.parse_args()

    if not args.fixtures.exists():
        ui.warn(
            f"no fixtures at {args.fixtures} — Level 1 validation is PENDING.\n"
            f"  It needs access to the released checkpoint. Generate with:\n"
            f"    python fixtures/extract.py --config deepseek-v4-flash\n"
            f"  Until then plan.md's P0 gate ('reproduces the released Gate bit-for-bit') "
            f"is UNMET — report it as unmet rather than assuming it."
        )
        raise SystemExit(0)

    fx = load_fixture(args.fixtures)
    ui.section(f"validating against {fx['meta']['hf_id']}")

    results = (
        validate_gate(fx, args.atol)
        + validate_residual(fx)
        + validate_mhc_maps(fx)
        + validate_hash(fx)
    )
    for ok, line in results:
        (ui.warn if ok is None else ui.ok if ok else ui.fail)(line)

    failures = [line for ok, line in results if ok is False]
    missing = [line for ok, line in results if ok is None]
    if failures:
        ui.fail("Level 1 FAILED — a measured component disagrees with its reference.")
        raise SystemExit(1)
    if missing:
        ui.warn(
            "Level 1 PARTIAL — all available legacy evidence passes, but the fixture "
            "predates official-output/mHC-map capture. No rerun is required for the "
            "structural result."
        )
        if args.require_complete:
            raise SystemExit(3)
    else:
        ui.print_done("Level 1 PASSED — the reimplementation matches the shipped model.")


if __name__ == "__main__":
    main()
