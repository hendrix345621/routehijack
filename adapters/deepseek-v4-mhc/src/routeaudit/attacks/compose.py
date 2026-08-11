"""Prompt-side RouteAudit suffix helper."""
from __future__ import annotations


def apply_routeaudit_suffix(prompts: list[str], suffix: str) -> list[str]:
    """Append a previously-derived universal RouteAudit suffix to every prompt."""
    return [f"{p} {suffix}" for p in prompts]
