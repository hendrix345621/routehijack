"""Safety/harmful differentials per the RouteHijack paper's Eqs. 4-5 (pp. 4-5).

    Δ_S(l, e)         = F_l(e | a_safe) − F_l(e | a_harm)
    Score_safe(l, e)  = Δ_S(l, e) − P_l(e | D_gen)²     # utility penalty (Eq. 5)
    Score_harm(l, e)  = −Δ_S(l, e)                       # no penalty — preserves fluency
"""
from __future__ import annotations

import torch

from .activation_freq import ExpertFreq


def delta_s(safe: ExpertFreq, harm: ExpertFreq) -> torch.Tensor:
    """Δ_S(l,e) = F(safe) - F(harm). Returns (L, E) tensor."""
    return safe.freq - harm.freq


def score_safe(safe: ExpertFreq, harm: ExpertFreq, general: ExpertFreq) -> torch.Tensor:
    """Per Eq. 5 — quadratic utility penalty suppresses general-purpose experts."""
    return delta_s(safe, harm) - general.freq.pow(2)


def score_harm(safe: ExpertFreq, harm: ExpertFreq) -> torch.Tensor:
    """Per paper p. 4 — no utility penalty for harmful experts (preserves attack fluency)."""
    return -delta_s(safe, harm)
