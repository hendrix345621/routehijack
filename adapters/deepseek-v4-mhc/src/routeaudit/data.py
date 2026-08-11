"""Minimal jsonl loaders. Heavy dataset wrangling stays out of the library —
scripts/00_data.py is the single place that knows about specific corpora."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


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
