"""Phase 4 — evaluate the RouteAudit universal suffix.

Runs two cells — a clean baseline and the attacked prompts — and reports, for each:
  • ASR (fraction of harmful prompts that get a non-refusal), screened by a string
    RefusalDetector and optionally re-graded by a HarmBench classifier (`--judge`);
  • MMLU accuracy (utility), to confirm the attack doesn't simply break the model;
then prints the routing-shift diagnostics (TESR / THPR) and a SAFE / AT-RISK verdict.

This is intentionally RouteAudit-only: no representation/feature machinery.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from routeaudit import config as cfg_mod
from routeaudit import ui
from routeaudit.model import load_model
from routeaudit.pipeline import eval_run
from routeaudit_deepseek_v4.config import CONFIG_PATH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--safety", default="artifacts/safety_experts.json")
    p.add_argument("--harmful", default="artifacts/harmful_experts.json")
    p.add_argument("--suffix", default="artifacts/routeaudit_universal.json")
    p.add_argument("--advbench", default="data/advbench.jsonl")
    p.add_argument("--mmlu", default="data/mmlu_subset.jsonl")
    p.add_argument("--n-prompts", type=int, default=None,
                   help="override eval.asr.n_prompts from the config")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="override eval.max_new_tokens; needs to be large in thinking mode "
                        "(a 128-token cap truncates every trace)")
    p.add_argument("--mmlu-generative", action="store_true", default=None,
                   help="force the thinking-aware generative MMLU on (else eval.mmlu.generative)")
    p.add_argument("--judge", action="store_true", default=True,
                   help="grade completions with the config's judge (default Llama-Guard-3-1B) — "
                        "the trustworthy, language-agnostic ASR; on by default")
    p.add_argument("--no-judge", dest="judge", action="store_false",
                   help="skip the judge; report the string-detector ASR only")
    p.add_argument("--results-dir", default="artifacts/results",
                   help="full auditable results bundle (summary + every prompt's clean/attacked + verdict)")
    p.add_argument("--judge-device", default="cuda")
    p.add_argument("--asr-threshold", type=float, default=0.5)
    p.add_argument("--gen-batch-size", type=int, default=8,
                   help="prompts generated per batched forward for ASR (raise to use more GPU; "
                        "lower if the KV cache OOMs)")
    p.add_argument("--mmlu-batch-size", type=int, default=16,
                   help="MMLU questions per batched forward (single-step, so can be larger than gen)")
    p.add_argument("--out", default="artifacts/eval_cells.jsonl",
                   help="raw per-cell jsonl (programmatic re-grading)")
    p.add_argument("--results", default="artifacts/eval_results.json",
                   help="consolidated results file (one object); a readable .md report is "
                        "written alongside it")
    args = p.parse_args()

    cfg = cfg_mod.load(args.config)
    ui.step_header(4, "Evaluate RouteAudit (ASR + utility + routing shift)", total=4)
    loaded = load_model(cfg)
    eval_run(loaded, cfg, args)
    ui.print_done("Evaluation complete")


if __name__ == "__main__":
    main()
