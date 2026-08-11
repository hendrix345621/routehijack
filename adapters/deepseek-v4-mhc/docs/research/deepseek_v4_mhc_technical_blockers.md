# DeepSeek-V4 / Manifold-Constrained Hyper-Connections: technical blockers and requirements

A neutral survey of the concrete engineering and research obstacles to **instrumenting,
reproducing, and experimenting with** the DeepSeek-V4 architecture — in particular its
Manifold-Constrained Hyper-Connections (mHC) residual stream and its Mixture-of-Experts
router. Each blocker lists what is technically required to resolve it. Facts are taken from
the released `deepseek-ai/DeepSeek-V4-Flash` weights (`config.json`, `inference/model.py`)
and the DeepSeek-V4 and mHC papers (arXiv:2512.24880).

Reference configuration (DeepSeek-V4-Flash): 43 layers, hidden size 4096, 256 routed
experts + 1 shared, 6 active/token, mHC expansion `hc_mult = 4` with 20 Sinkhorn–Knopp
iterations, MLA + compressed (CSA/HCA) attention, sliding window 128, weights in fp8 /
routed experts in fp4.

---

## 1. The residual stream is multi-stream, so standard per-layer hidden-state tooling is invalid

**Blocker.** Conventional transformer instrumentation assumes a single residual vector of
shape `(tokens, hidden)` between layers. Under mHC the residual is `(tokens, n, hidden)`
with `n = 4` parallel streams; a sub-layer reads a **mixed-down** `(tokens, hidden)` view
(`hc_pre`) and writes back into all `n` streams (`hc_post`). Any probe, cache, or
activation-patch that hooks "the residual" captures a tensor of the wrong rank and silently
mixes streams.

**What is needed.**
- A stream-aware capture layer that exposes each of the `n` streams separately and records
  the `hc_pre` (down-mix) and `hc_post` (write-back) weights per layer.
- A defined convention for reducing `n` streams to one when a single-vector analysis is
  required (mean, learned mix, or per-stream analysis), documented and held constant.
- Unit tests that confirm the reconstructed `(n, hidden)` residual matches the model's
  internal state bit-for-bit on a fixed input.

## 2. The residual mixing is a doubly-stochastic (Sinkhorn) operator, which changes signal-propagation behavior

**Blocker.** mHC constrains the inter-stream mixing matrix to the Birkhoff polytope
(doubly-stochastic) via 20 Sinkhorn–Knopp iterations, giving a spectral norm ≤ 1 and a
non-expansive, norm/mean-preserving residual update. Empirically the mHC paper reports a
signal "gain" ≈ 1 versus ≈ 3000 for unconstrained hyper-connections. Consequently
perturbations injected at the input do **not** amplify with depth the way they do in a
standard residual, and depth-wise propagation studies calibrated on ordinary transformers
do not transfer.

**What is needed.**
- A faithful, differentiable re-implementation of the Sinkhorn projection (exact iteration
  count and normalization order) to reason about, or backprop through, the mixing.
- Depth-resolved measurement tooling: per-layer input-perturbation → per-layer response,
  and a residual-norm-vs-depth profile, to quantify the conservation empirically.
- Matched baselines (standard residual and unconstrained hyper-connections at the same
  scale) to isolate what the doubly-stochastic constraint specifically contributes — the
  released model gives only the constrained variant.

## 3. The router is not a softmax over logits

**Blocker.** Much MoE tooling assumes the gate emits `(tokens, experts)` logits and applies
softmax top-k. The DeepSeek-V4 gate instead scores experts with `sqrt(softplus(Wh))`, adds
a learned per-expert bias **for selection only**, takes a flat top-6, then computes gating
weights from the **bias-free** score (renormalized, scaled by 1.5). The pre-selection score
tensor is never returned — the module emits only `(weights, indices)`.

**What is needed.**
- A gate re-implementation that recomputes the score from the gate input with the correct
  `sqrt(softplus)` activation (not sigmoid/softmax), applies the selection-only bias, and
  reproduces the released module's `(weights, indices)` exactly, validated against a saved
  input/output fixture.
- A clear separation, in any downstream analysis, between the quantity that decides *which*
  experts fire (`score + bias`) and the quantity that *weights* them (bias-free normalized
  score) — they are different tensors.

## 4. The first layers use non-differentiable hash routing

**Blocker.** The first `num_hash_layers = 3` layers do not route on content: experts are
selected by a fixed token-id → expert lookup table (`tid2eid[input_ids]`). This path is
non-differentiable and independent of activations, so any method relying on
gradient/activation signals through those layers has no purchase there.

**What is needed.**
- Detection of hash-routed layers from config and their explicit exclusion from any
  content/gradient-based routing analysis.
- If those layers matter for a study, a separate treatment of the static token→expert map
  (e.g. reading the table directly) rather than treating them as learned routers.

## 5. Compressed attention (MLA + CSA/HCA) complicates position- and token-level analysis

**Blocker.** DeepSeek-V4 uses Multi-head Latent Attention with two compression schemes —
Compressed Sparse Attention (compress every ~4 tokens, then top-k select) and Heavily
Compressed Attention (every ~128 tokens) — plus a sliding-window-128 branch, with per-layer
compression ratios. The KV cache is a heterogeneous, compressed object, so the contribution
of an individual input token to a given position is diluted and layer-dependent, and
standard dense-attention attribution does not apply.

**What is needed.**
- An attention implementation (or careful use of the shipped one) that exposes the
  per-layer compression ratio and the CSA top-k selection, so token→position influence can
  be reasoned about rather than assumed uniform.
- Analyses that account for the mixed dense/CSA/HCA schedule per layer instead of treating
  all layers as full attention.

## 6. Low-precision weights (fp8 / fp4) perturb numerically sensitive measurements

**Blocker.** The released weights are fp8 with routed experts in fp4. Quantization is fine
for coarse structural probing but shifts routing decisions and fine numerical quantities,
so any measurement that depends on exact scores, margins, or reproducibility must account
for it. Naive 4-bit re-quantization for cheaper experimentation perturbs the router further.

**What is needed.**
- A precision policy: use quantized loads for structural/direction-finding work, and
  confirm any quantitative claim in the model's native precision.
- Awareness that the model ships already-quantized (fp8) — it does **not** require an
  additional bitsandbytes pass to fit, unlike an unquantized model of the same size.

## 7. Scale and access limit white-box experimentation

**Blocker.** DeepSeek-V4-Flash is ~284B parameters (~160 GB on disk in fp8); DeepSeek-V4-Pro
is ~1.6T. Loading Flash natively needs on the order of a 2×80 GB node; gradient-based work
on the full model is out of reach on commodity hardware. This forces either forward-only
work on the target or development on a smaller sibling with transfer.

**What is needed.**
- Multi-GPU (≈2×80 GB) for native fp8 inference on Flash; MoE expert **CPU-offload**
  (keeping router + attention resident) to run forward-only probes on a smaller box, slowly.
- A smaller sibling (e.g. an earlier DeepSeek-MoE model) for gradient-based development,
  plus an explicit transfer study — noting that the sibling gates differ (some use a
  grouped/node-limited router and sigmoid scoring, unlike Flash's flat `sqrt(softplus)`),
  so mechanism transfer must be validated, not assumed.

## 8. Reproducing the architecture from scratch is non-trivial

**Blocker.** A faithful mHC + DeepSeek-V4 reimplementation must combine several
interacting, individually subtle components: the `n`-stream residual with dynamically
generated pre/post/mixing maps, the Sinkhorn projection, the two-quantity router, hash
layers, MLA with CSA/HCA, MTP heads, and the fp8/fp4 numerics. No small pre-trained mHC
model is publicly available (the mHC paper's 3B–27B validation models are unreleased), so
mechanism-level code cannot be validated end-to-end on a cheap real checkpoint.

**What is needed.**
- A small **synthetic** mHC model built directly from the paper's equations (doubly-
  stochastic `H^res`, `n`-stream residual, the grouped/flat gate variants) with random
  weights, sufficient to validate *code and mechanism* (not semantics) on CPU.
- Component-level fixtures extracted from the released model (one gate's I/O, one layer's
  residual update) to check each reimplemented piece against ground truth.
- Native-precision confirmation of any semantic result on the real checkpoint once the
  mechanism-level code is validated.

---

## Summary of the critical path
The two hardest, most novel blockers are the **multi-stream Sinkhorn residual** (items 1–2
— it invalidates standard hidden-state tooling and changes depth-wise signal propagation)
and **scale/access** (item 7). The gate and hash-routing differences (items 3–4) are
mechanical once the exact score function and selection rule are reproduced. Compressed
attention and low precision (items 5–6) are secondary corrections. A synthetic mHC model
plus component-level fixtures (item 8) is the practical prerequisite that makes the rest
verifiable without a frontier-scale training run.
