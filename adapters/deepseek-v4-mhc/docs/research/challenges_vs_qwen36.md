# Key challenges: DeepSeek-V4 (mHC) vs. architectures like Qwen 3.6

Why RouteAudit ports cleanly to Qwen3.6-35B-A3B but hits a wall on DeepSeek-V4-Flash.
Every claim on the DeepSeek side is verified against the **released** weights
(`deepseek-ai/DeepSeek-V4-Flash` `config.json` + `inference/model.py`); the Qwen side is
verified against `Qwen/Qwen3.6-35B-A3B` `config.json`. See `plan.md` for the porting plan
and `scoping.md` for the threat model.

RouteAudit's premise: append a text suffix that, at the boundary token `t*`, steers the
MoE **router** off the safety experts and onto compliant ones, so refusal never fires. It
needs (a) a differentiable-ish handle on router scores, (b) an input that can actually move
those scores, and (c) a residual/attention path that carries the perturbation to the layers
that gate refusal. Qwen3.6 grants all three; DeepSeek-V4 contests each one.

---

## Side-by-side

| Axis | Qwen3.6-35B-A3B | DeepSeek-V4-Flash | Method impact |
|---|---|---|---|
| MoE gate | plain `Linear` → `(T,256)` logits, **softmax** top-8 | `Linear` → **`sqrt(softplus(·))`** score, **flat** top-6 | different score fn + a non-negativity floor (below) |
| Selection bias | none | learned per-expert `bias`, **selection-only** | selection ≠ weighting; the loss must split them |
| Grouping | none (flat) | **none** (flat) — released Flash is *not* grouped | one fewer obstacle than `scoping.md` assumed |
| First layers | all 40 are content-routed MoE | **first 3 layers hash-routed** (`tid2eid[input_ids]`) | those layers are unsteerable — dead attack surface |
| Residual | standard single-stream | **mHC**: 4 doubly-stochastic streams (`hc_mult=4`) | breaks residual probes; may damp input leverage |
| Attention | hybrid **linear/full** (full every 4) | **MLA + CSA/HCA** compression + sliding-window 128 | dilutes a few appended suffix tokens at `t*` |
| Router capture | `MoEHookManager.capture_router_logits` works as-is | needs custom gate-input recompute (done in `route_mhc.py`) | more plumbing, but reachable |
| Runnable size | ~35B / ~3B active, ~19 GB @ 4-bit, 1×24 GB | ~284B / ~13B active, ~160 GB **fp8**, 2×80 GB | white-box iteration is expensive |
| Net portability | **method works, `qwen` preset unchanged** | **method does not port unmodified** | this doc = why |

---

## The challenges, ranked by how much they actually bite

### C1 — Router scores aren't a clean softmax logit tensor
Qwen's gate emits `(T, n_experts)` logits; RouteAudit differentiates a softmax over them,
and `capture_router_logits` reads them with one hook. DeepSeek's gate never exposes a
softmax: it computes `score = sqrt(softplus(Wh))`, adds a **selection-only** bias, takes a
flat top-6, and returns `(weights, indices)` — the pre-top-k tensor never leaves the module.
- **Consequence:** "routing mass on safety experts" (the original softmax loss) isn't the
  same object. The quantity that gates an expert is the bias-free normalized weight
  `g_e`; the quantity that decides *which* experts fire is `score+bias`. The loss must
  target **two different tensors**, and capture has to recompute the selection from the
  gate input (which `experiments/mhc/route_mhc.py` does — but currently with the *wrong* sigmoid; see
  P0 in `plan.md`).
- **Severity:** medium. Mechanical, not fundamental — it's plumbing plus a corrected loss.

### C2 — `sqrt(softplus)` non-negativity floor
Softplus is strictly positive, so every expert score is `≥ 0` and the sqrt compresses the
top range. Compared to raw softmax logits (unbounded, sign-carrying), there's less dynamic
range for a suffix to exploit, and no way to drive a competitor score *negative* to knock
it out — you can only add. Selection flips must come from **out-scoring** the 6th expert,
not from suppressing below zero.
- **Severity:** low–medium. Narrows the lever; doesn't remove it.

### C3 — Selection-only bias creates a fixed margin
`indices = topk(score + bias)` but `weights = score.gather(idx)` (bias-free). The learned
bias sets a per-expert *entry toll*: to make a harmful expert fire, the input must lift its
`score` past `(6th-place score + bias gap)`; to silence a safety expert, push it below.
Qwen has no such toll — any score change directly reshuffles the softmax top-8.
- **Consequence:** the load-bearing loss term becomes a **selection-margin hinge** on
  `score_e + bias_e` vs the 6th competitor (this replaces the phantom `λ_grp` group term in
  the old scoping). Whether the margins are crossable is the P1 go/no-go.
- **Severity:** medium — this is the genuine gate-level obstacle now that grouping is gone.

### C4 — Hash-routed first layers are unsteerable
`num_hash_layers = 3`: the first three layers pick experts by `tid2eid[input_ids]`, a fixed
token-id→expert table. No dependence on activations, non-differentiable, immune to any
suffix. Qwen content-routes all 40 layers.
- **Consequence:** if refusal signal concentrates early, part of it sits behind a wall no
  input attack can touch. Skip these layers in harvest and loss (−7% of layers, 0 surface).
- **Severity:** low if safety lives deep; potentially high if it lives shallow (P1 measures
  this by crossing the refusal fingerprint with layer depth).

### C5 — mHC damps input leverage with depth (the paper-grounded one)
mHC's residual mixing `H^res` is projected onto **doubly-stochastic** matrices
(Sinkhorn-Knopp, 20 iters) — norm-preserving, mean-conserving, spectral norm ≤ 1, so the
residual transform is **non-expansive**. The mHC paper reports gain ≈ 1 vs standard
hyper-connections ≈ 3000. Qwen's ordinary residual lets perturbations grow with depth.
- **Consequence (hypothesis H1):** a suffix perturbation injected at the input may decay
  before it reaches the deep layers that gate refusal — starving an input-only attack of
  leverage exactly where it's needed. This is the *architecture-level* robustness story and
  the most interesting result if it holds. Measured by `routing_reachability_by_depth` +
  `residual_norm_profile` on matched vanilla/HC/mHC siblings.
- **Severity:** unknown → **the crux.** Could be the reason the attack is infeasible, or a
  non-effect. It's a research question, not a settled blocker.

### C6 — mHC breaks residual probing (tooling, not defense)
Under mHC the per-layer residual is `(T, 4, C)`, not `(T, C)`. Any hidden-state / SAE /
`capture_residual` probe needs a "which stream" decision the main project doesn't make.
The **gate-input** path is unaffected (`hc_pre` collapses 4→1 before the gate), so routing
capture survives; only residual-space analysis needs new handling.
- **Severity:** low — it's a measurement inconvenience, and it cuts both ways (also our
  norm-conservation probe in C5).

### C7 — Compressed attention dilutes the suffix at `t*`
DeepSeek's MLA + CSA/HCA compress the KV cache (CSA every-4-tokens, HCA every-128) and add a
sliding-window-128 branch. A handful of appended suffix tokens contribute a smaller,
compressed share of the boundary-token context than they would under Qwen's linear/full
attention. Less of the suffix "reaches" `t*` per token spent.
- **Severity:** low–medium; compounds C5 (both reduce achievable Δ at `t*`).

### C8 — Scale, precision, access
Qwen3.6 runs white-box on a single 24 GB card at 4-bit — you can optimize a suffix directly
on the target. DeepSeek-V4-Flash is ~160 GB **fp8** (2×80 GB to load natively; already
quantized, so no bitsandbytes). White-box gradients on the real target are out of reach for
most, forcing **surrogate + transfer** — and transfer across a *learned-bias + hash-routed*
gate is more fragile than across Qwen's flat softmax, because the binding constraints
(bias margins, hash tables) are model-specific.
- **Severity:** high for practical attack; it's why P1–P3 run on a cheap sibling and only
  P4 touches the real model, forward-only.

---

## What is *not* a blocker (myth-busting)
- **"mHC blocks the routing attack."** No — mHC is a *residual* mechanism; the gate still
  gets a normal C-dim input via `hc_pre`. mHC's plausible effect is C5 (damping leverage),
  not gating reachability.
- **"The grouped gate is the hard part."** Not for released Flash — it's **flat top-k**.
  The old `scoping.md`/`route_mhc.py` grouped-selection modeling describes V2/V3, not the
  shipped V4-Flash. (This is corrected in `plan.md` P0.)
- **"Non-differentiable top-k kills GCG."** No — GCG never differentiates the hard select;
  it differentiates a soft surrogate to *propose* swaps, then scores candidates with the
  real gate. True for Qwen and DeepSeek alike.

## One-line summary
Qwen3.6's "hybrid" is only in its **attention**; its MoE gate and residual are textbook, so
RouteAudit runs unchanged. DeepSeek-V4 changes the **gate score function (C2), adds a
selection-only bias margin (C3), hash-routes its first layers (C4), and wraps everything in
a norm-conserving mHC residual (C5)** — a stack of small, mutually-reinforcing frictions
whose combined effect on input-only routing leverage is an open, measurable question, most
likely resolving toward *robustness*.
