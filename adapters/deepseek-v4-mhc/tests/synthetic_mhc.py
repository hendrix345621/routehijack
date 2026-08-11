"""Synthetic mHC model — a TINY, CPU-runnable transformer that is genuinely mHC
(DeepSeek-V4's residual mechanism) with a configurable DeepSeekMoE gate.

Why this exists: there is NO small public mHC model (the mHC paper's 3B/9B/27B aren't
released; Qwen3.6 is NOT mHC — verified). So to validate the instrumentation and the
paper-grounded conservation tests we build the architecture from its equations:

  • B = Sinkhorn-Knopp(exp(B̃)), column-then-row, t_max=20 (Eq. 8) → doubly stochastic
  • A = σ(·), C = 2σ(·) (Eqs. 6-7); n-stream residual (Eq. 1); RMSNorm over vec(X)
  • hash-routed leading layers (Roller et al. 2021) then learned routing
  • the real gate math, imported from `routeaudit.model.gate_math` — the same code the
    diagnostics run on a real model, so a bug here is a bug there

Weights are RANDOM → token-level OUTPUTS are meaningless. Refusal semantics need a real
trained mHC model. This is for MECHANISM and CODE validation, and it exposes the module
structure the diagnostics expect (`model.layers[i].mlp.gate` with `.weight` +
`e_score_correction_bias`, `.experts`) so capture runs end to end on a true mHC residual.

Two configurations are supported through `GateSpec`, which is the point — one builder
covers both released gate generations:
    V4-Flash-like   sqrtsoftplus + FLAT top-k + hash layers + scaling 1.5
    V2/V3-like      sigmoid + node-limited (grouped) top-k
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from routeaudit.model import gate_math
from routeaudit_deepseek_v4 import mhc
from routeaudit.model.gate_math import GateSpec

# Re-exported so existing callers keep working; the implementation (and the corrected
# column-then-row iteration order) now lives in the package.
sinkhorn_knopp = mhc.sinkhorn_knopp


class _RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class MHCResidual(nn.Module):
    """The mHC residual around a sub-layer F (Eqs. 1, 3-8) on an n-stream x ∈ (B,T,n,d).

    A, B and C are pure functions of x, which is the property the replay test relies on:
    caching x alone is enough to regenerate all three maps and reproduce the update
    exactly, with no hidden state.
    """

    def __init__(self, n: int, d: int, iters: int = 20, alpha: float = 0.01,
                 eps: float = 1e-6):
        super().__init__()
        self.n, self.d, self.iters, self.eps = n, d, iters, eps
        nc = n * d
        self.phi_pre = nn.Linear(nc, n, bias=False)
        self.phi_post = nn.Linear(nc, n, bias=False)
        self.phi_res = nn.Linear(nc, n * n, bias=False)
        for lin in (self.phi_pre, self.phi_post, self.phi_res):
            nn.init.normal_(lin.weight, std=0.02)
        self.a_pre = nn.Parameter(torch.tensor(alpha))
        self.a_post = nn.Parameter(torch.tensor(alpha))
        self.a_res = nn.Parameter(torch.tensor(alpha))
        self.b_pre = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.eye(n))          # identity-ish init (S_res)
        self.norm = _RMSNorm(nc)

    def generate_maps(self, x):
        """(A, B, C) from x alone. Matches `DeepseekV4HyperConnection.forward`: `pre`
        carries a `+hc_eps` and `post` does not, and B is the TRANSPOSE of the Sinkhorn
        output — the released layer applies `comb.transpose(-1, -2)` to the residual."""
        b, t, n, d = x.shape
        xf = self.norm(x.reshape(b, t, n * d))
        a = torch.sigmoid(self.a_pre * self.phi_pre(xf) + self.b_pre) + self.eps  # (B,T,n)
        c = 2 * torch.sigmoid(self.a_post * self.phi_post(xf) + self.b_post)      # (B,T,n)
        b_tilde = self.a_res * self.phi_res(xf).view(b, t, n, n) + self.b_res
        comb = mhc.sinkhorn_knopp(b_tilde, self.iters, self.eps)
        return a, mhc.residual_matrix(comb), c

    def forward(self, x, sublayer):                       # x: (B,T,n,d)
        a, b, c = self.generate_maps(x)
        h_in = mhc.mix_down(a, x)                         # A·X → (B,T,d), what F sees
        return mhc.mhc_update(b, c, x, sublayer(h_in))    # B·X + C·F(A·X)


class SwiGLUClamped(nn.Module):
    """Expert FFN with DeepSeek-V4's activation clamps (§4.2.3). A documented
    reimplementation divergence point — omitting the clamps is one of the things that
    quietly sinks from-scratch reproductions, so bake them in from the start."""

    def __init__(self, d, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)
        nn.init.normal_(self.down_proj.weight, std=0.01)   # small F → clean conservation

    def forward(self, x):
        g = self.gate_proj(x).clamp(max=10)
        u = self.up_proj(x).clamp(-10, 10)
        return self.down_proj(F.silu(g) * u)


class RoutedMoE(nn.Module):
    """DeepSeekMoE block. The gate is an `nn.Linear` carrying an
    `e_score_correction_bias` buffer — exactly what the diagnostics hook and recompute —
    and selection/weighting go through `gate_math.route`, so this model and a real one
    share one implementation of the gate."""

    def __init__(self, d, n_exp, gate_spec: GateSpec, hidden=None):
        super().__init__()
        self.gate = nn.Linear(d, n_exp, bias=False)
        self.gate.register_buffer("e_score_correction_bias", torch.zeros(n_exp))
        self.experts = nn.ModuleList(
            SwiGLUClamped(d, hidden or 2 * d) for _ in range(n_exp))
        self.gate_spec = gate_spec

    def forward(self, h):                                  # (B,T,d)
        b, t, d = h.shape
        x = h.reshape(b * t, d)
        logits = self.gate(x)                              # ← hooks read the gate INPUT
        rr = gate_math.route(logits, self.gate.e_score_correction_bias, self.gate_spec,
                             dtype=x.dtype)
        allout = torch.stack([e(x) for e in self.experts], 0)      # (E, BT, d)
        return torch.einsum("te,etd->td", rr.dense, allout).reshape(b, t, d)


class HashMoE(nn.Module):
    """Hash-routed MoE (DeepSeek-V4's leading layers, Roller et al. 2021).

    Experts come from a static token-id table, so routing here is content-independent and
    non-differentiable — no gradient reaches the choice, and no input suffix can change
    it. Deliberately exposes NO `.gate`, so the hook layer skips it the way it skips a
    dense layer, and `tid2eid` stays reachable as a correctness oracle.
    """

    def __init__(self, d, n_exp, top_k, vocab, hidden=None, seed=0):
        super().__init__()
        self.experts = nn.ModuleList(
            SwiGLUClamped(d, hidden or 2 * d) for _ in range(n_exp))
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("tid2eid", torch.randint(0, n_exp, (vocab, top_k), generator=g))
        self.top_k = top_k

    def forward(self, h, token_ids=None):                  # (B,T,d)
        b, t, d = h.shape
        x = h.reshape(b * t, d)
        if token_ids is None:                              # embeds-only path: expert 0
            idx = torch.zeros(b * t, self.top_k, dtype=torch.long, device=x.device)
        else:
            idx = gate_math.hash_route(token_ids.reshape(-1), self.tid2eid)
        w = torch.full(idx.shape, 1.0 / self.top_k, device=x.device, dtype=x.dtype)
        dense = torch.zeros(b * t, len(self.experts), device=x.device, dtype=x.dtype)
        dense.scatter_(1, idx, w)
        allout = torch.stack([e(x) for e in self.experts], 0)
        return torch.einsum("te,etd->td", dense, allout).reshape(b, t, d)


class _Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.o.weight, std=0.01)

    def forward(self, x):                                  # (B,T,d)
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, -1)
        q, k, v = (z.view(b, t, self.h, self.dk).transpose(1, 2) for z in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(o.transpose(1, 2).reshape(b, t, d))


class _Layer(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.kind = gate_math.routing_kind(layer_idx, cfg.gate_spec)
        self.self_attn = _Attn(cfg.d, cfg.heads)
        if self.kind == gate_math.HASH:
            self.mlp = HashMoE(cfg.d, cfg.n_exp, cfg.gate_spec.top_k, cfg.vocab)
        elif self.kind == gate_math.DENSE:
            self.mlp = SwiGLUClamped(cfg.d, 2 * cfg.d)
        else:
            self.mlp = RoutedMoE(cfg.d, cfg.n_exp, cfg.gate_spec)
        self.ln1, self.ln2 = _RMSNorm(cfg.d), _RMSNorm(cfg.d)
        self.mhc_a = MHCResidual(cfg.n_streams, cfg.d, cfg.sinkhorn_iters)
        self.mhc_m = MHCResidual(cfg.n_streams, cfg.d, cfg.sinkhorn_iters)

    def _mlp(self, h, token_ids):
        return self.mlp(h, token_ids=token_ids) if self.kind == gate_math.HASH else self.mlp(h)

    def forward(self, x, token_ids=None):                  # x: (B,T,n,d)
        x = self.mhc_a(x, lambda h: self.self_attn(self.ln1(h)))
        return self.mhc_m(x, lambda h: self._mlp(self.ln2(h), token_ids))


class _Inner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        self.layers = nn.ModuleList(_Layer(cfg, i) for i in range(cfg.n_layers))
        self.norm = _RMSNorm(cfg.d)


class TinyMHCModel(nn.Module):
    """HF-causal-LM-ish: forward(input_ids|inputs_embeds) -> ns(logits)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = _Inner(cfg)
        self.lm_head = nn.Linear(cfg.d, cfg.vocab, bias=False)

    def get_input_embeddings(self):
        return self.model.embed

    def expand_streams(self, h):
        """(B,T,d) → the n-stream residual state X (B,T,n,d)."""
        b, t, d = h.shape
        return h.unsqueeze(2).expand(b, t, self.cfg.n_streams, d).contiguous()

    def forward(self, input_ids=None, inputs_embeds=None, use_cache=False, **kw):
        h = inputs_embeds if inputs_embeds is not None else self.model.embed(input_ids)
        if h.dim() == 2:
            h = h.unsqueeze(0)
        x = self.expand_streams(h)
        for layer in self.model.layers:
            x = layer(x, token_ids=input_ids)
        # `reduce_streams("mean")` is the invariant of the doubly-stochastic B-path — the
        # same reduction every analysis of this model must use.
        return SimpleNamespace(logits=self.lm_head(self.model.norm(mhc.reduce_streams(x))))


class _CharTokenizer:
    """Minimal byte tokenizer (no chat template → raw-text path)."""
    chat_template = None
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, vocab):
        self.vocab = vocab

    def _enc(self, s):
        return [min(self.vocab - 1, b + 2) for b in s.encode("utf-8")] or [2]

    def __call__(self, text, add_special_tokens=True, return_tensors=None, padding=None,
                 truncation=None, max_length=None):
        return SimpleNamespace(input_ids=self._enc(text))

    def encode(self, s, add_special_tokens=False):
        return self._enc(s)

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return bytes(max(0, min(255, i - 2)) for i in ids if i >= 2).decode("utf-8", "replace")


# ─────────────────────────── builders ───────────────────────────

#: V4-Flash's gate, scaled down. The mechanism is identical to the released model:
#: sqrt(softplus) affinity, FLAT top-k, selection-only bias, hash-routed leading layers.
FLASH_LIKE = GateSpec(scoring_func="sqrtsoftplus", top_k=2, use_bias=True, n_group=1,
                      norm_topk_prob=True, routed_scaling_factor=1.5, num_hash_layers=1)

#: V2/V3's gate: sigmoid affinity with node-limited (grouped) selection.
V2_LIKE = GateSpec(scoring_func="sigmoid", top_k=2, use_bias=True, n_group=4,
                   topk_group=2, norm_topk_prob=True, routed_scaling_factor=1.0)


def build_synthetic_mhc(*, d=64, n_streams=4, n_layers=4, n_exp=16, heads=4, vocab=256,
                        sinkhorn_iters=20, seed=0, gate_spec: GateSpec = FLASH_LIKE):
    """Return a DiagModel wrapping a true mHC model — drop-in for refusal_tests /
    margin_census / run_diagnostics."""
    from diag_common import DiagModel
    from routeaudit.model.archspec import ArchSpec
    torch.manual_seed(seed)
    cfg = SimpleNamespace(d=d, n_streams=n_streams, n_layers=n_layers, n_exp=n_exp,
                          heads=heads, vocab=vocab, sinkhorn_iters=sinkhorn_iters,
                          gate_spec=gate_spec)
    model = TinyMHCModel(cfg).eval()
    spec = ArchSpec(name="deepseek", base_attr="model", layers_attr="layers",
                    moe_block_attrs=("mlp",), router_attr="gate", experts_attr="experts",
                    router_output="recompute", n_layers=n_layers, n_experts=n_exp,
                    top_k=gate_spec.top_k, d_model=d)
    model_cfg = SimpleNamespace(hf_id="synthetic-mhc", n_layers=n_layers, n_experts=n_exp,
                                top_k=gate_spec.top_k, d_model=d, use_chat_template=False)
    return DiagModel(model=model, tok=_CharTokenizer(vocab), spec=spec,
                     gate_spec=gate_spec, cfg=SimpleNamespace(model=model_cfg))


@torch.no_grad()
def replay_check(residual: MHCResidual, x, sublayer) -> dict:
    """Blocker 1's bit-for-bit property: A/B/C are pure functions of x, so replaying a
    cached residual state must reproduce the layer output exactly.

    A failure means the layer carries hidden state the capture didn't record — which
    would invalidate every offline analysis built on cached residuals.
    """
    out = residual(x, sublayer)                       # the forward pass being replayed
    a, b, c = residual.generate_maps(x)               # regenerated from the cached x alone
    h_in = mhc.mix_down(a, x)
    recon = mhc.mhc_update(b, c, x, sublayer(h_in))
    return {
        "hc_post_exact": bool(torch.equal(recon, out)),
        "max_abs_dev": float((recon - out).abs().max()),
    }
