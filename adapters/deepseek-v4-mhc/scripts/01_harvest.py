"""Stage 01 — harvest: expert identification (model loaded once).

  • Identify: F_l(e|safe), F_l(e|harm), F_l(e|gen) over the contrast pairs (response
    tokens) → Score_safe / Score_harm → top-pct safety + harmful experts (RouteHijack paper, §5).

Outputs:
  artifacts/safety_experts.json, artifacts/harmful_experts.json
  artifacts/identify_diagnostics.pt   (score_safe / score_harm tensors)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from routeaudit import config as cfg_mod
from routeaudit import ui
from routeaudit.model import load_model
from routeaudit.pipeline import harvest_run
from routeaudit_deepseek_v4.config import CONFIG_PATH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--out-safety", dest="out_safety", default="artifacts/safety_experts.json")
    p.add_argument("--out-harmful", dest="out_harmful", default="artifacts/harmful_experts.json")
    p.add_argument("--out-diag", dest="out_diag", default="artifacts/identify_diagnostics.pt")
    p.add_argument("--freq-batch-size", dest="freq_batch_size", type=int, default=16,
                   help="sequences per forward in the expert-frequency sweeps (lower if VRAM-tight)")
    p.add_argument("--resume", action="store_true",
                   help="reuse activation-frequency sweeps already cached on disk (spot-friendly)")
    args = p.parse_args()

    cfg = cfg_mod.load(args.config)
    ui.step_header(2, "Harvest — identify experts", total=4)
    loaded = load_model(cfg)
    harvest_run(loaded, cfg, args)
    ui.print_done("Harvest complete")


if __name__ == "__main__":
    main()
