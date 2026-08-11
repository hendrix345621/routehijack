"""Gate semantics — the corrections that make DeepSeek-V4 routing come out right.

Each test here pins one thing the pre-correction code got wrong (sigmoid instead of
sqrt(softplus), grouped selection that Flash doesn't have, bias leaking into the gating
weight, hash layers treated as content-routed). A regression on any of them silently
produces plausible but wrong routing numbers, which is worse than a crash.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from routeaudit.model import gate_math
from routeaudit.model.gate_math import GateSpec

FLASH = GateSpec(scoring_func="sqrtsoftplus", top_k=6, use_bias=True, n_group=1,
                 norm_topk_prob=True, routed_scaling_factor=1.5, num_hash_layers=3)
V3 = GateSpec(scoring_func="sigmoid", top_k=6, use_bias=True, n_group=8, topk_group=4,
              norm_topk_prob=True, routed_scaling_factor=2.5)
SOFTMAX = GateSpec(scoring_func="softmax", top_k=2)


# ── scoring ──────────────────────────────────────────────────────────────────

def test_sqrtsoftplus_is_not_sigmoid():
    """The single most consequential correction: V4-Flash scores with sqrt(softplus),
    not sigmoid. They differ in range and in ordering-under-scaling, so swapping them
    changes which experts fire, not just by how much."""
    x = torch.linspace(-4, 4, 64)
    got = gate_math.affinity(x, "sqrtsoftplus")
    assert torch.allclose(got, F.softplus(x).sqrt())
    assert not torch.allclose(got, x.sigmoid())
    assert (got >= 0).all(), "sqrt(softplus) is a non-negativity floor — never negative"
    assert got.max() > 1.0, "unlike sigmoid, it is unbounded above"


def test_elementwise_scores_are_not_a_distribution():
    """sigmoid/sqrtsoftplus scores do not sum to 1. Anything that treats them as a
    probability distribution (entropy, KL against a softmax gate) is comparing units."""
    logits = torch.randn(8, 32)
    for fn in ("sigmoid", "sqrtsoftplus"):
        assert not torch.allclose(gate_math.affinity(logits, fn).sum(-1), torch.ones(8))
    assert torch.allclose(gate_math.affinity(logits, "softmax").sum(-1), torch.ones(8))


def test_unknown_scoring_func_rejected():
    with pytest.raises(ValueError, match="sqrtsoftplus"):
        GateSpec(scoring_func="swish")


# ── selection ────────────────────────────────────────────────────────────────

def test_flash_selection_is_flat():
    """Flash has no node-limited selection. With n_group=1 every expert stays eligible,
    so the choice is a plain top-k — the phantom grouped bottleneck the old code modeled
    would have masked most of the experts out."""
    scores = torch.rand(4, 256)
    mask = gate_math.group_mask(scores, FLASH)
    assert mask.all(), "flat gate must leave every expert eligible"
    idx = gate_math.select(scores, None, FLASH)
    assert torch.equal(idx.sort(-1).values, scores.topk(6, -1).indices.sort(-1).values)


def test_grouped_selection_excludes_whole_groups():
    """V2/V3 node-limited routing: an expert's score is irrelevant if its group isn't
    chosen. This is why a grouped margin is combinatorial and a flat one is 1-D."""
    scores = torch.zeros(1, 64)    # 8 groups of 8; topk_group=4
    scores[0, :32] = 1.0           # groups 0-3 win the group race outright
    scores[0, 60] = 0.9            # a strong expert stranded in group 7
    mask = gate_math.group_mask(scores, V3)
    assert mask[0, 0] and not mask[0, 60]
    assert 60 not in gate_math.select(scores, None, V3)[0].tolist()


def test_bias_moves_selection_only():
    """The load-bearing distinction. The balancing bias decides WHICH experts fire; the
    gating weight is computed from the bias-free score. Conflating them attributes a
    load-balancing artifact to content."""
    torch.manual_seed(0)
    logits = torch.randn(1, 32)
    bias = torch.zeros(32)
    gs = GateSpec(scoring_func="sqrtsoftplus", top_k=4, use_bias=True)

    base = gate_math.route(logits, bias, gs)
    excluded = int(base.scores.argsort(-1, descending=True)[0, -1])
    bias[excluded] = 100.0                     # force an otherwise-unselected expert in
    biased = gate_math.route(logits, bias, gs)

    assert excluded in biased.indices[0].tolist(), "bias must change selection"
    assert torch.equal(base.scores, biased.scores), "bias must NOT change the scores"
    # The forced expert's gating weight comes from its (tiny) bias-free score, not from
    # the +100 that got it selected.
    slot = biased.indices[0].tolist().index(excluded)
    assert biased.weights[0, slot] < biased.weights[0].max()


def test_weights_are_bias_free_normalized_and_scaled():
    logits = torch.randn(3, 32)
    bias = torch.randn(32)
    rr = gate_math.route(logits, bias, FLASH)
    assert torch.allclose(rr.weights.sum(-1), torch.full((3,), 1.5), atol=1e-5), \
        "norm_topk_prob then x routed_scaling_factor → weights sum to the scale factor"
    assert torch.allclose(rr.dense.sum(-1), rr.weights.sum(-1), atol=1e-5)
    assert (rr.dense > 0).sum(-1).tolist() == [6, 6, 6], "exactly top_k experts fire"


# ── margins ──────────────────────────────────────────────────────────────────

def test_selection_margin_sign_and_magnitude():
    """Hand-built case: scores 10,9,8,7,... with top_k=3.
    Expert 2 (score 8) is the weakest selected; it can drop by 8-7=1 before expert 3
    takes its place. Expert 3 (score 7) must gain 7-8=-1, i.e. +1, to get in."""
    scores = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0]])
    gs = GateSpec(scoring_func="softmax", top_k=3)
    m = gate_math.selection_margin(scores, gs)[0]
    assert m[0] == pytest.approx(3.0)      # 10 - 7, most secure
    assert m[2] == pytest.approx(1.0)      # weakest selected
    assert m[3] == pytest.approx(-1.0)     # best excluded
    assert (m[:3] > 0).all() and (m[3:] < 0).all()


def test_selection_margin_uses_selection_scores_not_weights():
    """A margin measured on gating weights would be a different, wrong number: weights
    are bias-free and renormalized, so they can't tell you how far from the top-k an
    expert sits once the bias is applied."""
    logits = torch.randn(1, 32)
    bias = torch.zeros(32)
    bias[7] = 5.0
    gs = GateSpec(scoring_func="sqrtsoftplus", top_k=4, use_bias=True)
    rr = gate_math.route(logits, bias, gs)
    m = gate_math.selection_margin(rr.sel_scores, gs)[0]
    assert m[7] > 0 and 7 in rr.indices[0].tolist()
    assert m[7] > gate_math.selection_margin(rr.scores, gs)[0][7], \
        "the bias makes expert 7 harder to deselect — visible only in sel_scores"


def test_selection_margin_subset_matches_full():
    scores = torch.randn(4, 64)
    gs = GateSpec(scoring_func="sigmoid", top_k=6)
    ids = [3, 17, 40]
    full = gate_math.selection_margin(scores, gs)
    subset = gate_math.selection_margin(scores, gs, ids)
    assert torch.equal(subset, full[:, ids])


# ── layer classification ─────────────────────────────────────────────────────

def test_routing_kind_boundaries():
    gs = GateSpec(num_hash_layers=3, first_k_dense_replace=2)
    kinds = [gate_math.routing_kind(i, gs) for i in range(8)]
    assert kinds == ["dense", "dense", "hash", "hash", "hash",
                     "learned", "learned", "learned"]


def test_flash_hash_layers_excluded_from_learned():
    """Flash's first 3 MoE layers route by token id. Content-based analysis over them is
    structurally meaningless, so they must not appear in the learned set."""
    learned = gate_math.learned_router_layers(43, FLASH)
    assert learned[0] == 3 and len(learned) == 40
    assert not any(gate_math.routing_kind(i, FLASH) == gate_math.LEARNED for i in (0, 1, 2))


def test_no_hash_layers_by_default():
    """Every currently-supported family is fully content-routed; the default GateSpec
    must not accidentally exclude layer 0 from anything."""
    assert gate_math.learned_router_layers(12, SOFTMAX) == list(range(12))


def test_hash_route_is_the_table():
    table = torch.randint(0, 16, (128, 2))
    ids = torch.tensor([0, 5, 127])
    assert torch.equal(gate_math.hash_route(ids, table), table[ids])


def test_hash_layer_selection_is_fixed_but_weighting_is_not():
    """The half-observability that makes hash layers awkward: `indices` depend only on
    the token id, while `weights` come from the learned scores. So the same token routes
    to the same experts in every context but contributes a different amount."""
    torch.manual_seed(0)
    table = torch.randint(0, 32, (64, 2))
    ids = torch.tensor([7, 7, 7])
    gs = GateSpec(scoring_func="sqrtsoftplus", top_k=2, norm_topk_prob=True)

    idx = gate_math.hash_route(ids, table)
    assert (idx == idx[0]).all(), "same token id must give the same experts, always"

    # Same ids, different hidden states -> same selection, different weights.
    w1 = gate_math.gate_weights(gate_math.affinity(torch.randn(3, 32), "sqrtsoftplus"), idx, gs)
    w2 = gate_math.gate_weights(gate_math.affinity(torch.randn(3, 32), "sqrtsoftplus"), idx, gs)
    assert not torch.allclose(w1, w2), "gating weights on a hash layer ARE content-dependent"


# ── config plumbing ──────────────────────────────────────────────────────────

def test_from_config_defaults_to_flat_softmax():
    """A config with no `routing:` block must behave exactly as before this module
    existed — that is what keeps OLMoE/Mixtral/Qwen untouched."""
    from types import SimpleNamespace
    gs = GateSpec.from_config(SimpleNamespace(top_k=8))
    assert gs == GateSpec(scoring_func="softmax", top_k=8)
    assert not gs.grouped and not gs.use_bias and gs.num_hash_layers == 0


def test_from_config_reads_flash_routing_block():
    from types import SimpleNamespace
    ns = SimpleNamespace(top_k=6, routing=SimpleNamespace(
        scoring_func="sqrtsoftplus", use_bias=True, n_group=1, topk_group=0,
        norm_topk_prob=True, routed_scaling_factor=1.5, num_hash_layers=3,
        first_k_dense_replace=0))
    assert GateSpec.from_config(ns) == FLASH
