"""Phase 1 — download / prepare the corpora RouteAudit needs.

Writes the jsonl files the later phases read:
  data/llm_lat_pairs.jsonl   {"prompt","safe","harmful"}      contrast pairs (expert localization)
  data/c4_general.jsonl      {"prompt","response"}            general corpus (utility penalty)
  data/advbench.jsonl        {"prompt","target"}              harmful prompts (attack + eval)
  data/mmlu_subset.jsonl     {"question","choices","answer"}  utility eval

Each fetch is guarded so a missing dataset warns and skips rather than aborting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from routeaudit import ui
from routeaudit.data import write_jsonl


def _fetch_llm_lat(out: Path, n: int):
    from datasets import load_dataset
    ds = load_dataset("LLM-LAT/harmful-dataset", split="train")
    rows = []
    for r in ds:
        if len(rows) >= n:
            break
        if "prompt" in r and ("rejected" in r or "chosen" in r):
            rows.append({"prompt": r["prompt"], "safe": r.get("chosen", ""),
                         "harmful": r.get("rejected", "")})
    write_jsonl(out, rows)
    ui.ok(f"LLM-LAT pairs → {out} ({len(rows)})")


def _fetch_c4(out: Path, n: int):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    rows = []
    for r in ds:
        if len(rows) >= n:
            break
        text = r["text"]
        if len(text) < 200:
            continue
        rows.append({"prompt": text[:80], "response": text[80:880]})
    write_jsonl(out, rows)
    ui.ok(f"C4 → {out} ({len(rows)})")


def _fetch_advbench(out: Path):
    from datasets import load_dataset
    ds = load_dataset("walledai/AdvBench", split="train")
    rows = [{"prompt": r["prompt"], "target": r.get("target", "")} for r in ds]
    write_jsonl(out, rows)
    ui.ok(f"AdvBench → {out} ({len(rows)})")


def _fetch_mmlu(out: Path, n: int):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split=f"test[:{n}]")
    amap = {"A": 0, "B": 1, "C": 2, "D": 3}
    rows = [{"question": r["question"], "choices": r["choices"],
             "answer": r["answer"] if isinstance(r["answer"], int) else amap[r["answer"]]}
            for r in ds]
    write_jsonl(out, rows)
    ui.ok(f"MMLU subset → {out} ({len(rows)})")


def _try(fn, label):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — best-effort fetch
        ui.warn(f"{label}: skipped ({type(e).__name__}: {e})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--n-pairs", type=int, default=600)
    p.add_argument("--n-c4", type=int, default=5000)
    p.add_argument("--n-mmlu", type=int, default=500)
    args = p.parse_args()

    d = Path(args.data_dir)
    d.mkdir(parents=True, exist_ok=True)
    ui.step_header(1, "Download / prepare data", total=4)
    ui.section("Fetching corpora")
    _try(lambda: _fetch_llm_lat(d / "llm_lat_pairs.jsonl", args.n_pairs), "LLM-LAT")
    _try(lambda: _fetch_c4(d / "c4_general.jsonl", args.n_c4), "C4")
    _try(lambda: _fetch_advbench(d / "advbench.jsonl"), "AdvBench")
    _try(lambda: _fetch_mmlu(d / "mmlu_subset.jsonl", args.n_mmlu), "MMLU")

    present = {f.name: f.stat().st_size for f in sorted(d.glob("*.jsonl"))}
    ui.kv_panel("Data files", present or {"(none)": "fetch failed — check dataset access"})
    ui.print_done("Data stage complete")


if __name__ == "__main__":
    main()
