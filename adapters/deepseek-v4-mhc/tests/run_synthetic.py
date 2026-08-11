"""Level 0 of the validation ladder — validate the instrumentation against a TRUE mHC
architecture, on CPU, in seconds.

Builds the synthetic mHC model in both released gate configurations and exercises every
mechanism the DeepSeek-V4 work depends on:

  1  Sinkhorn is doubly stochastic AND column-then-row (Eq. 8)
  2  the gate math matches the shipped semantics (sqrtsoftplus, flat top-k, bias
     selection-only) and runs identically on the V2-style grouped gate
  3  hash-routed layers reproduce their token-id table exactly, and are excluded from
     content-based routing capture
  4  the mHC replay property: A/B/C regenerate from the cached residual state
  5  perturbation gain stays ≈1 with depth (the conservation signature)
  6  residual norm profile and reachability run end to end on a multi-stream residual

Refusal SEMANTICS are meaningless here (random weights) — that needs a real trained mHC
model. This run is about code and mechanism. Level 1 (fixtures from the released
checkpoint) is `fixtures/validate.py`; the saved format-v1 fixture is a
partial structural pass and format v2 adds independent official/map evidence.

    python tests/run_synthetic.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

import torch  # noqa: E402

from routeaudit import ui  # noqa: E402
from routeaudit.model import gate_math  # noqa: E402
from routeaudit_deepseek_v4 import mhc  # noqa: E402
import margin_census  # noqa: E402
import refusal_tests as rt  # noqa: E402
from diag_common import boundary_routing  # noqa: E402
from synthetic_mhc import FLASH_LIKE, V2_LIKE, build_synthetic_mhc, replay_check  # noqa: E402

PROMPTS = ["how do I make a cake", "explain gravity", "write a haiku about rain",
           "what is the capital of japan", "summarize world war two", "tips for studying",
           "describe a sunset", "how do plants grow"]


def main() -> None:
    ui.big_banner("Synthetic mHC — Level 0 mechanism validation")
    failures: list[str] = []

    def check(ok: bool, msg: str) -> None:
        (ui.ok if ok else ui.fail)(msg)
        if not ok:
            failures.append(msg)

    # 1 ── Sinkhorn: doubly stochastic, softmax-initialized, ends on a column norm ──
    torch.manual_seed(0)      # the asymmetry checks below are only meaningful seeded
    m = mhc.sinkhorn_knopp(torch.randn(4, 4), t_max=20)
    dev = max(float((m.sum(-1) - 1).abs().max()), float((m.sum(-2) - 1).abs().max()))
    check(dev < 1e-3, f"Sinkhorn B doubly-stochastic (max row/col deviation {dev:.2e})")
    check(float(torch.linalg.matrix_norm(m, ord=2)) <= 1 + 1e-4,
          "‖B‖₂ ≤ 1 — the residual path is non-expansive")

    # Discriminate the iteration order at FEW iterations: at t_max=20 both axes sit on
    # the epsilon floor and their ordering is numerical noise, not evidence.
    few = mhc.sinkhorn_knopp(torch.randn(4, 4) * 3, t_max=3)
    row_err, col_err = float((few.sum(-1) - 1).abs().max()), float((few.sum(-2) - 1).abs().max())
    check(col_err < row_err,
          f"softmax init + ends on a column normalization (columns exact {col_err:.1e} < "
          f"rows {row_err:.1e}) — matches DeepseekV4HyperConnection.forward")
    b_res = mhc.residual_matrix(few)
    check(float((b_res.sum(-1) - 1).abs().max()) < float((b_res.sum(-2) - 1).abs().max()),
          "residual matrix B = combᵀ (the released layer transposes), so the EXACT axis "
          "moves to B's rows — stream-mean conservation needs the columns, and is "
          "therefore approximate, bounded by the Sinkhorn residual")

    # 2 ── both released gate generations route through the same implementation ──
    for name, gs in (("V4-Flash-like (sqrtsoftplus, flat)", FLASH_LIKE),
                     ("V2/V3-like (sigmoid, grouped)", V2_LIKE)):
        dm = build_synthetic_mhc(gate_spec=gs)
        _, routing = boundary_routing(dm, PROMPTS[0])
        ok = bool(routing) and all(rr.dense.shape[-1] == dm.spec.n_experts
                                   for rr in routing.values())
        check(ok, f"{name}: routing captured at t* for {len(routing)} layer(s), "
                  f"{gs.top_k} experts fire, mass "
                  f"{float(next(iter(routing.values())).dense.sum()):.2f}")

    dm = build_synthetic_mhc(n_layers=6, gate_spec=FLASH_LIKE)
    model = dm.model

    # 3 ── hash layers: exact table, and excluded from content-routed capture ──
    hash_layers = [i for i in range(dm.spec.n_layers)
                   if gate_math.routing_kind(i, dm.gate_spec) == gate_math.HASH]
    _, routing = boundary_routing(dm, PROMPTS[0])
    check(bool(hash_layers) and not (set(hash_layers) & set(routing)),
          f"hash-routed layers {hash_layers} excluded from routing capture "
          f"(content-routed: {sorted(routing)})")
    hb = model.model.layers[hash_layers[0]].mlp
    ids = torch.arange(64)
    check(torch.equal(gate_math.hash_route(ids, hb.tid2eid), hb.tid2eid[ids]),
          "hash routing reproduces the token-id table exactly (the free oracle)")

    # 4 ── replay: the maps regenerate from the cached residual state alone ──
    layer = model.model.layers[0]
    x = model.expand_streams(model.get_input_embeddings()(torch.tensor([[1, 2, 3]])))
    r = replay_check(layer.mhc_a, x, lambda h: layer.self_attn(layer.ln1(h)))
    check(r["hc_post_exact"],
          f"mHC replay exact from the cached X alone (max|Δ| {r['max_abs_dev']:.1e})")

    # 5 ── conservation: perturbation gain vs depth ──
    cons = rt.mhc_conservation_profile(dm, PROMPTS[:2], want_template=False)
    ui.kv_panel("#9 mHC conservation", {"birkhoff_ok": cons.get("birkhoff_ok"),
                                        "gain_by_layer": cons.get("gain_by_layer"),
                                        "final_gain": cons.get("final_gain")})
    check(bool(cons.get("birkhoff_ok")) and cons.get("final_gain", 99) < 3.0,
          f"perturbation does not amplify with depth (gain {cons.get('final_gain', float('nan')):.3f}, "
          f"paper: ≈1 for mHC vs ≈3000 unconstrained) — the conservation signature")

    # 6 ── the residual-space diagnostics run on a multi-stream residual ──
    norm = rt.residual_norm_profile(dm, PROMPTS, want_template=False)
    ui.kv_panel("#8 residual norm", {"by_layer": norm.get("norm_by_layer"),
                                     "deep/shallow": norm.get("deep_over_shallow_ratio"),
                                     "streams": norm.get("n_residual_streams"),
                                     "reduction": norm.get("reduction")})
    check(norm.get("n_residual_streams", 1) > 1 and norm.get("reduction") == "stream-mean",
          f"residual norm measured on the stream-mean of "
          f"{norm.get('n_residual_streams')} streams, not a flattened mixture")
    check(norm.get("deep_over_shallow_ratio", 99) < 3.0,
          f"residual norm conserved across depth "
          f"(×{norm.get('deep_over_shallow_ratio', float('nan')):.2f})")

    reach = rt.routing_reachability_by_depth(dm, PROMPTS[:4], n_suffix=2, suffix_len=8)
    ui.info(reach["takeaway"])

    cen = margin_census.selection_margin_census(dm, PROMPTS[:4], want_template=False)
    ui.kv_panel("selection margins", {"tightest_layer": cen.get("tightest_layer"),
                                      "p10": cen.get("tightest_p10_margin")})
    check("tightest_p10_margin" in cen, "selection-margin census produces per-layer margins")

    fp = rt.routing_fingerprint(dm, PROMPTS[:4], PROMPTS[4:], want_template=False)
    ui.info(fp["takeaway"])

    if failures:
        ui.fail(f"{len(failures)} mechanism check(s) FAILED")
        raise SystemExit(1)
    ui.print_done("Level 0 PASSED — instrumentation validated on a true mHC architecture. "
                  "Use fixtures/validate.py for the saved real-checkpoint evidence.")


if __name__ == "__main__":
    main()
