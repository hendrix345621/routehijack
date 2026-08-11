"""mHC — Manifold-Constrained Hyper-Connections (DeepSeek-V4 §2.2, arXiv:2512.24880).

mHC replaces the single residual stream with `n` parallel streams and three per-layer
maps generated dynamically from the residual state itself:

    X_{l+1} = B_l · X_l  +  C_l · F_l(A_l · X_l),        X_l ∈ R^{n × d}

    A_l = σ(Ã_l)                    mixes n streams DOWN to the one d-vector F sees
    C_l = 2σ(C̃_l)                   broadcasts F's output back INTO the n streams
    B_l = Sinkhorn(exp(B̃_l))        doubly-stochastic mixing between streams

Two consequences this module exists to handle:

1. **Tooling that assumes a (T, d) residual is silently wrong.** Under mHC the layer
   output is (T, n, d). Flattening it norms all n streams together. Every single-vector
   analysis must go through `reduce_streams`, and "mean" is not an arbitrary convention
   — because B is doubly stochastic, `mean_streams(B X) = mean_streams(X)`, so the
   stream-mean is exactly the quantity the manifold constraint conserves.

2. **The maps are recomputable from X_l alone**, so caching the residual state is enough
   to regenerate A/B/C exactly and replay a layer offline. That is what makes the
   bit-for-bit replay test possible.

What mHC does NOT break: the MoE gate. `A_l · X_l` is d-dimensional, so gate-input
capture works unchanged — see `hooks.capture_routing`.
"""

from __future__ import annotations

import torch

_EPS = 1e-9


# ─────────────────────────── the constrained mixing map ───────────────────────────


def sinkhorn_knopp(b_tilde: torch.Tensor, t_max: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Project raw mixing parameters onto the Birkhoff polytope, as the released model does.

    Transcribed from `DeepseekV4HyperConnection.forward` in transformers'
    `modeling_deepseek_v4.py`. The exact sequence is:

        M = softmax(B̃, dim=-1) + eps          # row normalization, via softmax
        M = M / (M.sum(dim=-2) + eps)          # T_c — the first EXPLICIT Sinkhorn step
        repeat t_max-1 times:
            M = M / (M.sum(dim=-1) + eps)      # T_r
            M = M / (M.sum(dim=-2) + eps)      # T_c

    Three details that are easy to get wrong and that change the result at a finite
    `t_max = 20`:

    * **Initialization is `softmax`, not bare `exp`.** softmax over the last dim IS a row
      normalization, which is why the first explicit step is the column one — the
      implementation is often described as "column-normalization first" for that reason,
      even though a row normalization has already happened.
    * **The loop runs `t_max - 1` times**, not `t_max`, because the initial column step is
      counted as the first iteration.
    * **The last operation is a COLUMN normalization**, so columns are exact and rows carry
      the residual error. An implementation ending on a row normalization has the error on
      the other axis — the same limit, a different matrix after 20 steps.

    `eps` is added to every denominator (config `hc_eps`, 1e-6), not clamped — a small but
    real difference from `clamp_min` when a sum is already large.

    b_tilde: (..., n, n) raw mixing logits. Returns (..., n, n), doubly stochastic to
    within the finite-iteration residual. Differentiable throughout.

    Two properties this buys, usable as measurement oracles on real activations:
      * ‖B‖₂ ≤ 1 — the residual path is non-expansive, so nothing amplifies through it;
      * the Birkhoff polytope is closed under multiplication — products across depth stay
        doubly stochastic, so conservation composes rather than decaying.
    """
    m = torch.softmax(b_tilde, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)  # T_c (iteration 1)
    for _ in range(max(0, t_max - 1)):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)  # T_r
        m = m / (m.sum(dim=-2, keepdim=True) + eps)  # T_c — ends here
    return m


# ─────────────────────────── the residual update ───────────────────────────
#
# Shared by the model's forward and by offline replay, so a replay check tests map
# regeneration rather than a second copy of the same einsum.


def mix_down(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A_l · X_l — n streams → the single (B, T, d) input the sub-layer sees."""
    return torch.einsum("btn,btnd->btd", a, x)


def residual_matrix(comb: torch.Tensor) -> torch.Tensor:
    """The mixing matrix B actually applied to the residual, from the Sinkhorn output.

    The released decoder layer computes `matmul(comb.transpose(-1, -2), hidden_states)`,
    so **B is the TRANSPOSE of what Sinkhorn returned**. This is not cosmetic. Sinkhorn
    ends on a column normalization, so `comb`'s columns are exact and its rows carry the
    residual error; transposing moves the exact axis to the rows of B.

    Stream-mean conservation needs `1ᵀB = 1ᵀ`, i.e. B's COLUMNS to sum to 1 — which are
    `comb`'s rows, the approximate axis. So the released model conserves the stream mean
    only to the finite-iteration Sinkhorn residual, not exactly. Measure it with
    `b_path_conservation_check` rather than assuming it.
    """
    return comb.transpose(-1, -2)


def mix_residual(b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """B_l · X_l — mixing between streams. `b` must already be the residual matrix; pass
    Sinkhorn output through `residual_matrix` first."""
    return torch.einsum("btnm,btmd->btnd", b, x)


def write_back(c: torch.Tensor, h_out: torch.Tensor) -> torch.Tensor:
    """C_l · F_l(·) — broadcast the sub-layer output back into the n streams."""
    return c.unsqueeze(-1) * h_out.unsqueeze(-2)


def mhc_update(b: torch.Tensor, c: torch.Tensor, x: torch.Tensor, h_out: torch.Tensor) -> torch.Tensor:
    """Eq. 1: X_{l+1} = B_l X_l + C_l F_l(A_l X_l)."""
    return mix_residual(b, x) + write_back(c, h_out)


# ─────────────────────────── stream reduction ───────────────────────────


def stream_count(tensor: torch.Tensor, d_model: int, n_batch_dims: int = 2) -> int:
    """How many residual streams a captured tensor carries: 1 for a plain (B, T, d)
    residual, n for an mHC (B, T, n, d) one.

    `n_batch_dims` is how many leading axes precede the stream/feature axes — 2 for the
    usual (batch, tokens, ...) capture, 1 if the batch axis was already squeezed.
    Returns 1 (with no error) when the trailing dim isn't d_model, so callers on an
    unknown model degrade to the single-stream reading rather than crashing.
    """
    if tensor.shape[-1] != d_model:
        return 1
    return int(tensor.shape[-2]) if tensor.dim() == n_batch_dims + 2 else 1


def is_multi_stream(tensor: torch.Tensor, d_model: int, n_batch_dims: int = 2) -> bool:
    return stream_count(tensor, d_model, n_batch_dims) > 1


def reduce_streams(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Collapse an mHC (..., n, d) residual to (..., d) under a documented convention.

    "mean"       the invariant of the doubly-stochastic B-path — the default, and the
                 only reduction the residual mixing conserves.
    "per_stream" no reduction; for stream-resolved analysis.
    "last"       stream n-1 only. Available because some tooling implicitly does this,
                 but it is NOT conserved by B and will drift across depth — pick it
                 deliberately or not at all.

    `x` MUST actually be multi-stream. This function cannot tell a (B, T, n, d) residual
    from a (B, T, d) one by rank alone, and applied to the latter it would average over
    *tokens* — a silent, plausible-looking wrong answer. When the stream count is only
    known at runtime, use `reduce_residual`, which takes it explicitly.
    """
    if mode == "per_stream":
        return x
    if mode == "mean":
        return x.mean(dim=-2)
    if mode == "last":
        return x[..., -1, :]
    raise ValueError(f"unknown reduction mode {mode!r} (expected mean|per_stream|last)")


def reduce_residual(x: torch.Tensor, n_streams: int, mode: str = "mean") -> torch.Tensor:
    """Architecture-agnostic residual reduction: reduce when there are streams to reduce,
    pass through when there aren't.

    Pair with `HookCapture.residual_streams[layer]`, which the residual hook fills in, so
    the same analysis code runs over a standard model and an mHC one without either
    branching or silently mangling one of them.
    """
    return x if n_streams <= 1 else reduce_streams(x, mode)


# ─────────────────────────── measurement oracles ───────────────────────────


@torch.inference_mode()
def b_path_conservation_check(b: torch.Tensor, x: torch.Tensor | None = None, tol: float = 1e-4) -> dict:
    """Verify the paper's three guarantees on real activations rather than asserting them.

    Returns measurements plus booleans, so a diagnostic can report *how far off* a real
    fp8 model is instead of only pass/fail:
      doubly_stochastic  rows and columns of B sum to 1
      non_expansive      ‖B‖₂ ≤ 1 — nothing amplifies through the residual path
      mean_conserved     mean_streams(B X) == mean_streams(X)

    A failure here means B is off the Birkhoff polytope, which invalidates every
    conservation claim downstream — check it before interpreting a norm profile.
    """
    b = b.float()
    metrics = [
        (b.sum(dim=-1) - 1).abs().max(),
        (b.sum(dim=-2) - 1).abs().max(),
        torch.linalg.matrix_norm(b, ord=2).max(),
    ]
    if x is not None:
        mixed = mix_residual(b, x.float())
        metrics.append((mixed.mean(dim=-2) - x.float().mean(dim=-2)).abs().max())
    # Sync once for the complete diagnostic instead of once per scalar metric.
    values = torch.stack(metrics).cpu().tolist()
    row_dev, col_dev, spec = values[:3]
    out = {
        "row_sum_dev": row_dev,
        "col_sum_dev": col_dev,
        "spectral_norm_max": spec,
        "doubly_stochastic": row_dev <= tol and col_dev <= tol,
        "non_expansive": spec <= 1 + tol,
    }
    if x is not None:
        dev = values[3]
        out["stream_mean_dev"] = dev
        out["mean_conserved"] = dev <= tol
    return out


@torch.inference_mode()
def perturbation_profile(
    layers, x0: torch.Tensor, *, eps: float = 1e-3, inject_at: int = 0, seed: int = 0, **layer_kw
) -> dict:
    """Depth-resolved perturbation gain: inject a perturbation into the residual state
    at `inject_at`, then track how it grows or decays through the remaining layers.

    `layers` is any sequence of callables mapping residual state → residual state
    (single- or multi-stream; both work).

    Returns:
      gain[l]        ‖X'_l − X_l‖_F / ‖δ‖_F at each depth
      norm[l]        ‖X_l‖_F, the unperturbed residual-norm profile
      final_gain     gain at the last layer — the headline number
      max_gain       the largest gain at any depth

    The paper's claim is gain ≈ 1 for mHC versus ≈ 3000 for unconstrained
    Hyper-Connections. Because the B-path is non-expansive, any growth must come through
    the C·F(·) branch, which is itself bounded (C ∈ (0,2) elementwise). So this is a null
    hypothesis testable against the model itself — no external baseline needed.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    delta = (eps * torch.randn(x0.shape, generator=g)).to(x0.device, x0.dtype)
    dn = delta.norm().clamp_min(_EPS)

    xa, xb = x0.clone(), x0.clone()
    gain_t, norm_t = [], []
    for i, layer in enumerate(layers):
        if i == inject_at:
            xb = xb + delta
        xa = layer(xa, **layer_kw)
        xb = layer(xb, **layer_kw)
        gain_t.append((xb - xa).norm() / dn)
        norm_t.append(xa.norm())
    # Defer host reads so CUDA does not synchronize twice per layer.
    gain = torch.stack(gain_t).cpu().tolist() if gain_t else []
    norm = torch.stack(norm_t).cpu().tolist() if norm_t else []
    return {
        "gain_by_layer": gain,
        "norm_by_layer": norm,
        "final_gain": gain[-1] if gain else float("nan"),
        "max_gain": max(gain) if gain else float("nan"),
    }
