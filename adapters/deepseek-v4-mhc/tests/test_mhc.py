"""mHC residual mechanism — the constraint, the invariant, and the conservation claim.

These pin the properties every mHC-specific conclusion rests on. If Sinkhorn isn't
producing a doubly-stochastic matrix, "mHC conserves signal" is unfounded; if
`reduce_streams` isn't the conserved reduction, every single-vector residual measurement
is on the wrong quantity.
"""
from __future__ import annotations

import pytest
import torch

from routeaudit_deepseek_v4 import mhc


# ── the Sinkhorn projection ──────────────────────────────────────────────────

def test_sinkhorn_is_doubly_stochastic():
    torch.manual_seed(0)
    b = mhc.sinkhorn_knopp(torch.randn(8, 4, 4), t_max=20)
    assert torch.allclose(b.sum(-1), torch.ones(8, 4), atol=1e-4), "rows"
    assert torch.allclose(b.sum(-2), torch.ones(8, 4), atol=1e-4), "columns"
    assert (b >= 0).all()


def test_sinkhorn_ends_on_a_column_normalization():
    """Transcribed from `DeepseekV4HyperConnection.forward`: init with `softmax(dim=-1)`
    (a row normalization), then the loop's LAST operation is the column normalization.

    So columns come out exact and rows carry the finite-iteration residual. An
    implementation ending on a row normalization converges to the same limit but is a
    different matrix after 20 steps — this test pins which one we reproduce.
    """
    torch.manual_seed(0)
    b = mhc.sinkhorn_knopp(torch.randn(4, 4) * 3, t_max=3)   # few iters → error visible
    row_err = float((b.sum(-1) - 1).abs().max())
    col_err = float((b.sum(-2) - 1).abs().max())
    assert col_err < 1e-5, "columns must be exact — column normalization runs last"
    assert row_err > col_err, (
        "rows should carry the residual error; if they don't, the iteration order has "
        "been flipped to end on a row normalization")


def test_sinkhorn_initializes_with_softmax_not_bare_exp():
    """The init is `softmax(B̃)`, i.e. exp followed by a row normalization. Starting from
    bare `exp` and running the same loop gives a measurably different matrix at t_max=20,
    because Sinkhorn is only asymptotically insensitive to its initialization."""
    torch.manual_seed(7)
    logits = torch.randn(4, 4) * 2

    def from_bare_exp(x, t_max=20, eps=1e-6):
        m = torch.exp(x) + eps
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
        for _ in range(t_max - 1):
            m = m / (m.sum(dim=-1, keepdim=True) + eps)
            m = m / (m.sum(dim=-2, keepdim=True) + eps)
        return m

    ours = mhc.sinkhorn_knopp(logits, t_max=20)
    assert not torch.allclose(ours, from_bare_exp(logits), atol=1e-7)


def test_residual_matrix_is_the_transpose_of_the_sinkhorn_output():
    """The released decoder layer applies `comb.transpose(-1,-2)` to the residual, so the
    exact axis moves: `comb` has exact columns, B has exact rows."""
    torch.manual_seed(0)
    comb = mhc.sinkhorn_knopp(torch.randn(2, 3, 4, 4), t_max=20)
    b = mhc.residual_matrix(comb)
    assert torch.equal(b, comb.transpose(-1, -2))
    assert float((b.sum(-1) - 1).abs().max()) < float((b.sum(-2) - 1).abs().max())


def test_stream_mean_conservation_is_approximate_on_the_released_form():
    """Conservation needs B's COLUMNS to sum to 1. With B = combᵀ those are `comb`'s
    rows — the axis carrying the Sinkhorn residual. So the released model conserves the
    stream mean only to that residual, not exactly. Measure it; don't assume it."""
    torch.manual_seed(11)
    x = torch.randn(1, 4, 4, 8)
    b = mhc.residual_matrix(mhc.sinkhorn_knopp(torch.randn(1, 4, 4, 4) * 2, t_max=20))
    r = mhc.b_path_conservation_check(b, x, tol=1e-4)
    assert r["non_expansive"]
    assert r["stream_mean_dev"] < 1e-2, "drift should be small, even if not exactly zero"


def test_sinkhorn_is_non_expansive():
    """‖B‖₂ ≤ 1 — the residual path cannot amplify. This is what makes the depth-wise
    gain measurement a test of the C·F(·) branch rather than of the residual."""
    torch.manual_seed(1)
    b = mhc.sinkhorn_knopp(torch.randn(16, 4, 4), t_max=20)
    assert (torch.linalg.matrix_norm(b, ord=2) <= 1 + 1e-4).all()


def test_sinkhorn_is_differentiable():
    x = torch.randn(4, 4, requires_grad=True)
    mhc.sinkhorn_knopp(x, t_max=5).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_birkhoff_closed_under_multiplication():
    """Products across depth stay doubly stochastic, so conservation composes rather than
    decaying — that is why the guarantee survives 43 layers."""
    torch.manual_seed(2)
    prod = torch.eye(4)
    for _ in range(10):
        prod = prod @ mhc.sinkhorn_knopp(torch.randn(4, 4), t_max=20)
    assert torch.allclose(prod.sum(-1), torch.ones(4), atol=1e-3)
    assert torch.allclose(prod.sum(-2), torch.ones(4), atol=1e-3)


# ── the conserved reduction ──────────────────────────────────────────────────

def test_stream_mean_is_conserved_by_the_b_path():
    """The reason "mean" is the default reduction and not an arbitrary convention:
    mean_streams(B X) = mean_streams(X) whenever B's columns sum to 1. Here B is taken
    directly from Sinkhorn (exact columns), which is the ideal case; see
    `test_stream_mean_conservation_is_approximate_on_the_released_form` for what the
    released transpose actually delivers."""
    torch.manual_seed(3)
    x = torch.randn(2, 5, 4, 16)
    b = mhc.sinkhorn_knopp(torch.randn(2, 5, 4, 4), t_max=20)
    mixed = mhc.mix_residual(b, x)
    assert torch.allclose(mixed.mean(-2), x.mean(-2), atol=1e-4)
    # "last" is NOT conserved — the documented reason not to use it by default.
    assert not torch.allclose(mixed[..., -1, :], x[..., -1, :], atol=1e-4)


def test_reduce_streams_modes():
    x = torch.randn(2, 3, 4, 8)
    assert mhc.reduce_streams(x, "mean").shape == (2, 3, 8)
    assert mhc.reduce_streams(x, "last").shape == (2, 3, 8)
    assert mhc.reduce_streams(x, "per_stream").shape == x.shape
    with pytest.raises(ValueError):
        mhc.reduce_streams(x, "sum")


def test_reduce_residual_leaves_single_stream_alone():
    """The trap this API exists to close: a (B, T, d) residual has the same rank as a
    3-D multi-stream one, and mean(-2) on it would average over TOKENS — a wrong answer
    that looks entirely reasonable."""
    plain = torch.randn(2, 7, 16)
    assert torch.equal(mhc.reduce_residual(plain, n_streams=1), plain)
    multi = torch.randn(2, 7, 4, 16)
    assert mhc.reduce_residual(multi, n_streams=4).shape == (2, 7, 16)


def test_stream_count_detection():
    assert mhc.stream_count(torch.randn(2, 7, 16), d_model=16) == 1
    assert mhc.stream_count(torch.randn(2, 7, 4, 16), d_model=16) == 4
    assert mhc.stream_count(torch.randn(2, 7, 4, 16), d_model=99) == 1   # unknown → safe


# ── conservation oracles ─────────────────────────────────────────────────────

def test_b_path_conservation_check_passes_on_a_real_projection():
    torch.manual_seed(4)
    x = torch.randn(1, 3, 4, 8)
    b = mhc.sinkhorn_knopp(torch.randn(1, 3, 4, 4), t_max=20)
    r = mhc.b_path_conservation_check(b, x)
    assert r["doubly_stochastic"] and r["non_expansive"] and r["mean_conserved"]


def test_b_path_conservation_check_catches_an_off_polytope_matrix():
    """A check that can't fail isn't a check. An unprojected exp(B̃) must be rejected."""
    torch.manual_seed(5)
    bad = torch.randn(1, 1, 4, 4).exp()
    r = mhc.b_path_conservation_check(bad, torch.randn(1, 1, 4, 8))
    assert not r["doubly_stochastic"]
    assert not r["mean_conserved"]


# ── end-to-end on the synthetic model ────────────────────────────────────────

def test_perturbation_gain_stays_near_one_on_a_true_mhc_model():
    """The H1 signature. mHC's constrained residual should damp an input perturbation
    rather than amplify it with depth (the paper: gain ≈ 1 vs ≈ 3000 for unconstrained
    Hyper-Connections). Random weights, so this tests the MECHANISM, not semantics."""
    from synthetic_mhc import build_synthetic_mhc
    dm = build_synthetic_mhc(n_layers=6, seed=0)
    model = dm.model
    ids = torch.tensor([[5, 9, 12, 30, 44]])
    x = model.expand_streams(model.get_input_embeddings()(ids))
    prof = mhc.perturbation_profile(list(model.model.layers), x, eps=1e-3, token_ids=ids)
    assert prof["max_gain"] < 10.0, f"perturbation amplified: {prof['gain_by_layer']}"
    assert prof["final_gain"] < 3.0


def test_mhc_replay_is_exact():
    """Blocker 1's property: A/B/C are pure functions of X, so a cached residual state is
    enough to reproduce the layer output exactly. If this drifts, the layer is carrying
    hidden state the capture didn't record and offline analysis is unsound."""
    from synthetic_mhc import build_synthetic_mhc, replay_check
    dm = build_synthetic_mhc(n_layers=2, seed=1)
    layer = dm.model.model.layers[0]
    x = dm.model.expand_streams(dm.model.get_input_embeddings()(torch.tensor([[1, 2, 3]])))
    r = replay_check(layer.mhc_a, x, lambda h: layer.self_attn(layer.ln1(h)))
    assert r["hc_post_exact"], f"replay diverged by {r['max_abs_dev']:.3e}"
