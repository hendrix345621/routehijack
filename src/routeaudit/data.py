"""Prepare and read the normalized corpora used by the pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from . import ui


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)


def iter_safe_pairs(pairs_path: str | Path) -> Iterator[dict]:
    """LLM-LAT-style records → safe-side sequences for F_l(e | a_safe).

    Expected jsonl schema:
      {"prompt": "...harmful query...", "safe": "...refusal...", "harmful": "..."}
    """
    for r in read_jsonl(pairs_path):
        yield {"prompt": r["prompt"], "response": r["safe"]}


def iter_harm_pairs(pairs_path: str | Path) -> Iterator[dict]:
    for r in read_jsonl(pairs_path):
        yield {"prompt": r["prompt"], "response": r["harmful"]}


def iter_general(corpus_path: str | Path) -> Iterator[dict]:
    """General-purpose generations for P_l(e | D_gen) utility penalty.

    Schema (one record per generated continuation):
      {"prompt": "<any prompt>", "response": "<generation>"}
    """
    yield from read_jsonl(corpus_path)


def prepare_datasets(data_dir: Path, *, n_pairs: int, n_general: int, n_mmlu: int) -> None:
    """Download and normalize every dataset required by the pipeline.

    Dataset errors are fatal: continuing with a partial directory only moves the
    failure into an expensive model phase.
    """
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ValueError('dataset support is not installed; run `pip install -e ".[data]"`') from exc

    data_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_dataset("LLM-LAT/harmful-dataset", split="train")
    pair_rows = []
    for row in pairs:
        if len(pair_rows) >= n_pairs:
            break
        if "prompt" in row and ("rejected" in row or "chosen" in row):
            pair_rows.append(
                {"prompt": row["prompt"], "safe": row.get("chosen", ""), "harmful": row.get("rejected", "")}
            )
    if not pair_rows:
        raise ValueError("the safety-pair dataset did not contain the expected prompt/chosen/rejected schema")
    write_jsonl(data_dir / "llm_lat_pairs.jsonl", pair_rows)

    general = load_dataset("allenai/c4", "en", split="train", streaming=True)
    general_rows = []
    for row in general:
        if len(general_rows) >= n_general:
            break
        text = row["text"]
        if len(text) >= 200:
            general_rows.append({"prompt": text[:80], "response": text[80:880]})
    if not general_rows:
        raise ValueError("the general corpus did not yield any usable rows")
    write_jsonl(data_dir / "c4_general.jsonl", general_rows)

    advbench = load_dataset("walledai/AdvBench", split="train")
    write_jsonl(
        data_dir / "advbench.jsonl",
        ({"prompt": row["prompt"], "target": row.get("target", "")} for row in advbench),
    )

    mmlu = load_dataset("cais/mmlu", "all", split=f"test[:{n_mmlu}]")
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    write_jsonl(
        data_dir / "mmlu_subset.jsonl",
        (
            {
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"] if isinstance(row["answer"], int) else answer_map[row["answer"]],
            }
            for row in mmlu
        ),
    )
    ui.ok(
        f"datasets ready in {data_dir} ({len(pair_rows)} pairs, {len(general_rows)} general, {n_mmlu} MMLU)"
    )
