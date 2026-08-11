"""DeepSeekMoE / mHC routing diagnostic  --  SEPARATE EXPERIMENT (not part of `make all`).

Reports the safety / harmful **routing mass at the boundary token t***, computed through
the model's real gate rather than the pipeline's default `softmax(logits)`.

Gate semantics come from the config's `routing:` block via `gate_math.GateSpec`, so this
one script covers every DeepSeekMoE generation:

    DeepSeek-V4-Flash   sqrt(softplus(Wh)) -> flat top-6 over 256 experts,
                        selection-only bias, weights renormalized x1.5
    DeepSeek-V2/V3      sigmoid(Wh) -> node-limited (grouped) top-k

Layers whose routing is hash-based (`routing.num_hash_layers`) or dense
(`routing.first_k_dense_replace`) are excluded: their routing is a token-id lookup or
absent entirely, so including them would report mass that no input can move.

This is a DIAGNOSTIC. The suffix search is NOT ported to this gate  --  its losses assume
`softmax(logits)` and would need bias-free gating weights plus a selection-margin hinge
(phase P2 in plan.md, gated on the P1 feasibility verdict). `--suffix` only *evaluates* an
already-derived suffix, e.g. one transferred from a sibling model.

    python scripts/route_mhc.py --config deepseek-v4-flash
    python scripts/route_mhc.py --config deepseek-v2-lite \
        --suffix artifacts/routeaudit_universal.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_ADAPTER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ADAPTER_ROOT / "src"))

import torch

from routeaudit import config as cfg_mod
from routeaudit import ui
from routeaudit.data import read_jsonl
from routeaudit.identify.select import load_experts
from routeaudit.model import load_model
from routeaudit_deepseek_v4 import precision
from routeaudit.model.gate_math import GateSpec, learned_router_layers
from routeaudit_deepseek_v4.hooks import DeepSeekV4HookManager as MoEHookManager
from routeaudit_deepseek_v4.config import CONFIG_PATH
from routeaudit.model.prompting import encode_prompt

_DEFAULT_CONFIG = str(CONFIG_PATH)


def _layer_map(experts):
    m: dict[int, list[int]] = {}
    for e in experts:
        m.setdefault(e.layer, []).append(e.expert)
    return m


@torch.no_grad()
def boundary_mass(model, tok, spec, gs: GateSpec, prompt: str, use_tmpl: bool) -> dict:
    """Per-layer dense gating weights at t*, for the content-routed layers only."""
    device = next(model.parameters()).device
    ids = encode_prompt(tok, prompt, want_template=use_tmpl, device=device).unsqueeze(0)
    with MoEHookManager(model, spec) as hm:
        hm.capture_routing(gs)
        model(input_ids=ids, use_cache=False)
        return {l: rr.dense[-1].float().cpu() for l, rr in hm.capture.routing.items()}


def _masses(model, tok, spec, gs, prompts, use_tmpl, safety_map, harmful_map):
    safe_rows, harm_rows = [], []
    for p in ui.iter_with_progress(prompts, "route t*"):
        b = boundary_mass(model, tok, spec, gs, p, use_tmpl)
        ps = [float(b[l][safety_map[l]].sum()) for l in safety_map if l in b]
        ph = [float(b[l][harmful_map[l]].sum()) for l in harmful_map if l in b]
        if ps:
            safe_rows.append(sum(ps) / len(ps))
        if ph:
            harm_rows.append(sum(ph) / len(ph))
    return (sum(safe_rows) / max(1, len(safe_rows)),
            sum(harm_rows) / max(1, len(harm_rows)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=_DEFAULT_CONFIG)
    p.add_argument("--safety", default="artifacts/safety_experts.json")
    p.add_argument("--harmful", default="artifacts/harmful_experts.json")
    p.add_argument("--advbench", default="data/advbench.jsonl")
    p.add_argument("--n-prompts", type=int, default=32)
    p.add_argument("--suffix", default=None,
                   help="optional path to a routeaudit_universal.json — EVALUATE its routing "
                        "effect (e.g. a suffix transferred from a sibling model). Not optimized.")
    p.add_argument("--out", default="artifacts/route_mhc_diagnostics.json")
    args = p.parse_args()

    cfg = cfg_mod.load(args.config)
    arch_name = getattr(getattr(cfg.model, "arch", None), "name", None)
    if arch_name != "deepseek":
        ui.fail(f"this experiment expects a DeepSeek config (model.arch.name: deepseek); "
                f"got {arch_name!r} from {args.config}.")
        raise SystemExit(2)

    ui.step_header(25, "mHC / DeepSeek routing diagnostic (separate experiment)", total=4)
    for f in (args.safety, args.harmful):
        if not Path(f).exists():
            ui.fail(f"{f} not found — run stage 01 harvest first.")
            raise SystemExit(2)

    loaded = load_model(cfg)
    model, tok, spec = loaded.model, loaded.tokenizer, loaded.spec
    gs = GateSpec.from_config(cfg.model)
    ui.kv_panel("gate", {
        "scoring_func": gs.scoring_func, "top_k": gs.top_k,
        "selection": "grouped" if gs.grouped else "flat",
        "selection_bias": gs.use_bias, "routed_scaling_factor": gs.routed_scaling_factor,
        "hash_layers": gs.num_hash_layers, "dense_layers": gs.first_k_dense_replace,
    })
    ui.kv_panel("precision", precision.policy_banner(cfg.model, "none",
                                                     precision.Claim.QUANTITATIVE))

    safety_map = _layer_map(load_experts(args.safety))
    harmful_map = _layer_map(load_experts(args.harmful))
    # Hash/dense layers never appear in the capture, so drop them from the expert maps
    # too rather than silently averaging over layers that contribute nothing.
    routable = set(learned_router_layers(int(getattr(cfg.model, "n_layers", 0) or 0), gs))
    dropped = [l for l in (set(safety_map) | set(harmful_map)) if routable and l not in routable]
    if dropped:
        ui.warn(f"excluding {len(dropped)} non-content-routed layer(s) from the mass: "
                f"{sorted(dropped)} (hash-routed or dense — no steerable routing).")
        safety_map = {l: v for l, v in safety_map.items() if l in routable}
        harmful_map = {l: v for l, v in harmful_map.items() if l in routable}

    use_tmpl = getattr(cfg.model, "use_chat_template", True)
    prompts = [r["prompt"] for r in list(read_jsonl(args.advbench))[: args.n_prompts]]
    if not prompts:
        ui.fail(f"no prompts in {args.advbench} — run stage 00 data first.")
        raise SystemExit(2)

    ui.section("Clean routing mass at boundary token t*")
    clean_safe, clean_harm = _masses(model, tok, spec, gs, prompts, use_tmpl,
                                     safety_map, harmful_map)
    out = {
        "model": getattr(cfg.model, "hf_id", args.config),
        "gate": {"scoring_func": gs.scoring_func, "top_k": gs.top_k,
                 "grouped": gs.grouped, "use_bias": gs.use_bias,
                 "routed_scaling_factor": gs.routed_scaling_factor,
                 "num_hash_layers": gs.num_hash_layers},
        "content_routed_layers": sorted(routable),
        "n_prompts": len(prompts),
        "clean_safety_mass": clean_safe, "clean_harmful_mass": clean_harm,
        "note": "diagnostic only; the RouteAudit suffix search is not ported to this gate.",
    }

    if args.suffix:
        suffix = json.load(open(args.suffix, encoding="utf-8")).get("suffix", "")
        if suffix:
            ui.section(f"Evaluating provided suffix ({len(suffix)} chars) — NOT optimized here")
            attacked = [f"{p} {suffix}" for p in prompts]
            atk_safe, atk_harm = _masses(model, tok, spec, gs, attacked, use_tmpl,
                                         safety_map, harmful_map)
            out.update(attacked_safety_mass=atk_safe, attacked_harmful_mass=atk_harm,
                       TESR=atk_safe - clean_safe, THPR=atk_harm - clean_harm)
        else:
            ui.warn(f"no 'suffix' field in {args.suffix}; skipping suffix evaluation.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    ui.kv_panel("Routing diagnostic", out)
    ui.ok(f"diagnostics → {args.out}")
    ui.print_done("mHC routing diagnostic complete (diagnostic only).")


if __name__ == "__main__":
    main()
