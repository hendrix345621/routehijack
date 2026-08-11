"""Top-fraction selection over Score_safe.

The RouteHijack paper (§5, Table 10, p. 11) uses top-20% of (layer, expert) pairs by score.
We replicate that as the default but keep `top_pct` configurable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class SafetyExpert:
    layer: int
    expert: int
    score: float


def select_safety_experts(
    score: torch.Tensor,
    *,
    top_pct: float = 0.20,
) -> list[SafetyExpert]:
    """Pick the top `top_pct` fraction of layer-expert pairs globally.

    Returns a list sorted by descending score.
    """
    L, E = score.shape
    flat = score.reshape(-1)
    k = max(1, int(L * E * top_pct))
    vals, idx = flat.topk(k)
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        layer = i // E
        expert = i % E
        out.append(SafetyExpert(layer=layer, expert=expert, score=float(v)))
    return out


# Harmful-side identification reuses the same selection mechanic — the only
# difference is which score tensor you pass in. RouteHijack paper, p. 4: Score_harm
# omits the utility penalty so chosen harmful experts stay fluent.
select_harmful_experts = select_safety_experts


def save_experts(experts: list[SafetyExpert], path: str | Path) -> None:
    payload = [{"layer": e.layer, "expert": e.expert, "score": e.score} for e in experts]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_experts(path: str | Path) -> list[SafetyExpert]:
    with open(path, "r", encoding="utf-8") as fh:
        return [SafetyExpert(**row) for row in json.load(fh)]
