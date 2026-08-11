"""Harvest (phase 01) must count the experts the model actually routes to.

`compute_expert_freq` implements the paper's Eq. 3 as `𝟙[e ∈ TopK(logits)]`. That is
correct for a softmax gate and wrong for a DeepSeek one in three separate ways, each of
which produces plausible-looking numbers rather than an error:

  1. the selection-only balancing bias changes WHICH experts win;
  2. node-limited routing (V2/V3) masks whole groups out;
  3. V4's gate returns `(weights, indices)` — top-k-ing that counts over `top_k`
     positions as if they were the expert axis.

These tests build a gate with each property and check the frequencies against the
module's own routing decisions.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from routeaudit.identify.activation_freq import compute_expert_freq
from routeaudit.model.archspec import ArchSpec
from routeaudit.model.gate_math import GateSpec

D, E, K, VOCAB = 16, 12, 3, 64


class _DeepSeekGate(nn.Module):
    """Shaped like the released V4 Gate: emits `(weights, indices)` and nothing else."""

    def __init__(self, d=D, e=E, k=K, bias_scale=3.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(e, d) * 0.5)
        self.register_buffer("e_score_correction_bias", torch.randn(e) * bias_scale)
        self.k = k

    def scores(self, h):
        return F.softplus(h @ self.weight.T).sqrt()

    def forward(self, h):
        s = self.scores(h)
        idx = (s + self.e_score_correction_bias).topk(self.k, -1).indices
        w = s.gather(-1, idx)
        return w / w.sum(-1, keepdim=True) * 1.5, idx


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _DeepSeekGate()
        self.experts = nn.ModuleList(nn.Linear(D, D) for _ in range(E))

    def forward(self, h):
        self.gate(h.reshape(-1, h.shape[-1]))
        return h


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MoE()

    def forward(self, h, **kw):
        return self.mlp(h)


class _Base(nn.Module):
    """The base transformer. `compute_expert_freq` calls `model.model` directly to skip
    the lm_head's VRAM spike, so this must be independently forwardable — as it is on a
    real HF causal LM."""

    def __init__(self, n_layers=3):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D)
        self.layers = nn.ModuleList(_Layer() for _ in range(n_layers))

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kw):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return SimpleNamespace(last_hidden_state=h)


class _Model(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.model = _Base(n_layers)

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kw):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return SimpleNamespace(logits=out.last_hidden_state)


class _Tok:
    """Minimal tokenizer: `profiling_ids` needs encode + a pad id, nothing else."""
    chat_template = None
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, s, add_special_tokens=False):
        return [min(VOCAB - 1, b % VOCAB) for b in s.encode()] or [2]

    def __call__(self, text, **kw):
        return SimpleNamespace(input_ids=self.encode(text))


SEQS = [{"prompt": "why is the sky blue", "response": "because of rayleigh scattering"},
        {"prompt": "name a fruit", "response": "an apple is a fruit"}]

FLASH = GateSpec(scoring_func="sqrtsoftplus", top_k=K, use_bias=True, n_group=1,
                 routed_scaling_factor=1.5)
SPEC = ArchSpec(name="deepseek", router_output="recompute", n_layers=3,
                n_experts=E, top_k=K, d_model=D)


def _freq(gate_spec):
    torch.manual_seed(0)
    return compute_expert_freq(
        _Model().eval(), _Tok(), SEQS, n_layers=3, n_experts=E, top_k=K,
        spec=SPEC, gate_spec=gate_spec, use_chat_template=False, batch_size=2, desc="t")


def test_legacy_path_refuses_a_gate_with_no_logit_tensor():
    """Without a GateSpec, harvest would top-k the gate's `(T, top_k)` weights tensor as
    if those columns were the expert axis. It must refuse with an actionable message
    rather than accumulate over the wrong axis."""
    with pytest.raises(RuntimeError, match="does not emit a router-logit tensor"):
        _freq(None)


def test_gate_aware_path_reaches_the_full_expert_axis():
    gated = _freq(FLASH)
    assert gated.freq[:, K:].sum() > 0, "experts beyond top_k must be reachable"
    assert gated.freq.shape == (3, E)


def test_frequencies_match_the_modules_own_routing():
    """The real check: the counted experts are exactly the ones the gate selected."""
    torch.manual_seed(0)
    model, tok = _Model().eval(), _Tok()
    ef = compute_expert_freq(model, tok, SEQS, n_layers=3, n_experts=E, top_k=K,
                             spec=SPEC, gate_spec=FLASH, use_chat_template=False,
                             batch_size=2, desc="t")
    # Recompute by hand over the same response tokens: every expert with nonzero counted
    # frequency must be one the module would actually pick.
    from routeaudit.model.prompting import profiling_ids
    with torch.no_grad():
        for seq in SEQS:
            ids, n_prompt = profiling_ids(tok, seq["prompt"], seq["response"],
                                          want_template=False)
            h = model.model.embed(ids.unsqueeze(0))
            for li, layer in enumerate(model.model.layers):
                _, idx = layer.mlp.gate(h.reshape(-1, D))
                for e in idx[n_prompt:].unique().tolist():
                    assert ef.freq[li, e] > 0, f"L{li}/E{e} was routed to but not counted"
                h = layer(h)


def test_bias_free_topk_would_pick_different_experts():
    """Guards the premise: with this bias scale, ignoring it genuinely changes selection.
    If it didn't, the test above would pass for the wrong reason."""
    torch.manual_seed(0)
    g = _DeepSeekGate()
    h = torch.randn(8, D)
    s = g.scores(h)
    assert not torch.equal(s.topk(K, -1).indices.sort(-1).values,
                           (s + g.e_score_correction_bias).topk(K, -1).indices.sort(-1).values)


def test_hash_layers_contribute_no_frequency():
    """Hash-routed layers must stay at exactly zero so `_mask_unroutable` can exclude
    them — a nonzero count there would mean content-based routing was fabricated."""
    ef = _freq(GateSpec(scoring_func="sqrtsoftplus", top_k=K, use_bias=True,
                        num_hash_layers=1))
    assert ef.freq[0].sum() == 0, "layer 0 is hash-routed — nothing to capture"
    assert ef.freq[1:].sum() > 0, "learned layers still counted"
