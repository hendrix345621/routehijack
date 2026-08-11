# Proposed resolutions to the open challenges

Companion to `technical_challenges.md`. For each challenge: the best attack I can
construct, what it actually buys, and what remains genuinely open. Every number below is
either from a cited paper or produced by a script in `analysis/` that runs on CPU in
under a minute — no GPU, no checkpoint.

The organizing idea across all seven: **most of these are not blocked on compute, they are
blocked on having framed the question as needing compute.** Four of them collapse once you
notice that the quantity in question is a property of a small matrix, an order statistic,
or open-source code rather than of trained weights.

| # | Challenge | Status after this pass |
|---|---|---|
| 1 | Steering a biased non-differentiable top-k | **Reframed** — margin was the wrong obstacle; alignment is the real one |
| 2 | Verifying a reimplementation | **Largely solved** — needs the released *code*, not the *weights* |
| 3 | No small model with the mechanism | **Solved for mechanism** — and it falsifies the robustness story |
| 4 | Half-observable fixed-selection layers | **Solved** — implemented |
| 5 | No full-precision reference for QAT weights | **Workaround** — self-consistency + published floors |
| 6 | Conservation exact in math, approximate in code | **Solved** — reduces to one measurable scalar |
| 7 | Attribution through compressed attention | **Partial** — a computable bound, not the exact chain |

---

## 1. Steering a biased, non-differentiable top-k

### What the literature does

The closest published work optimizes the **soft routing probabilities** directly rather
than differentiating through top-k:

> `L_route(δ; x) = Σ ΔU_i · p_i(x̃)` where `p_i = softmax(z_i)`
> — *Misrouter: Exploiting Routing Mechanisms for Input-Only Attacks on MoE LLMs*
> ([arXiv:2605.04446](https://arxiv.org/html/2605.04446v1))

It reports ASR of 6–96% depending on target, but **no soft-to-hard fidelity analysis**, no
selection-margin measurement, no random-suffix control, and it assumes softmax gating
throughout — so it says nothing about a `sqrt(softplus)` score with a selection-only bias.
[RouteHijack](https://arxiv.org/abs/2605.02946) reports 69.3% average ASR across seven MoE
models with the same softmax assumption. So the specific obstacle is unaddressed in the
literature, not solved elsewhere.

### The reframe

Run `analysis/margin_forecast.py`. A selection margin is the gap between the k-th and
(k+1)-th order statistics of E scores, which for E=256, k=6 is deep enough in the tail that
its *scale* follows from E, k and the score function alone — no weights needed.

```
                              margin / score-range (median)
  scoring fn        sigma=1     sigma=2     sigma=4
  sqrt(softplus)     0.0103      0.0105      0.0105     <- flat: unbounded, never saturates
  sigmoid            0.0059      0.0018      0.0001     <- collapses: saturates toward 1
  softmax            0.0211      0.0182      0.0071

  experts within a +/-5% band of the boundary:  ~8, at every spread
  ||dh||/||h|| to flip the boundary expert:     ~5e-4 at perfect alignment, scale-invariant
```

Four things fall out, three of which contradict the standing assumptions:

1. **The bias does not protect the boundary.** Raising the bias spread from 0 to 0.5 leaves
   the relative margin flat (~1.0%) and the near-boundary expert count unchanged. It
   changes *which* experts are marginal, not how hard the boundary is to reach. "The
   balancing bias creates a protective margin an attacker must overcome" — the premise the
   whole difficulty framing rested on — is not supported. The bias obstructs *predicting* a
   flip, not *causing* one.
2. **The margin is tiny in input terms**: ~5e-4 of `||h||` at perfect alignment, and
   scale-invariant, because the gate's Jacobian grows with the logit spread exactly as fast
   as the margin does. The gate is not what makes an input attack hard.
3. **~8 experts sit within ±5% of the boundary**, so an attack targeting "any safety expert
   in this layer" has several cheap targets. Per-expert margins understate layer
   reachability.
4. **`sqrt(softplus)` gives systematically wider relative margins than `sigmoid`** and
   holds them at any spread, because it is unbounded and never saturates while sigmoid
   piles the expert population up near 1. So V4's scoring change makes its routing *harder*
   to perturb than V2/V3's at equal spread. I have not seen this claimed anywhere; it
   looks like a robustness side effect of a change made for optimization reasons.

**So the binding constraint is alignment, not margin.** A suffix perturbs one position and
reaches the boundary token only through attention; what decides feasibility is the
achievable component of `Δh` along the flip direction. That reframing is useful because
the alignment ratio — achieved `Δscore` per unit `||Δh||`, versus the perfectly-aligned
bound above — is **forward-only, needs no gradients, and can be measured on any MoE model,
including cheap ones.** The go/no-go no longer needs the target checkpoint.

### What remains open

Soft-to-hard fidelity is still unmeasured, in this work and in the literature. The
cheapest decisive experiment: run the relaxed objective on any small MoE, log both the
relaxed loss and the hard top-k set every step, and report their correlation. Near-zero
correlation falsifies the entire method class in a few GPU-hours on a 1B model, and would
be a stronger contribution than another ASR number.

---

## 2. Verifying a reimplementation against weights you cannot run

**Largely solved, by noticing the ladder was mis-specified.** Level 1 was described as
needing the released *weights*. It needs the released *code*. `transformers` ships
`DeepseekV3TopkRouter`; instantiating it at toy size with random weights
exercises the identical arithmetic the 671B checkpoint runs.

Random weights are **stronger** than a real checkpoint here, because they sweep regions of
score space a trained model may never enter — which is exactly where ties, group masking
and bias inversions live.

`tests/test_reference_parity.py` implements this: 17 tests, ~11 s on CPU, no download. It
caught a real compatibility defect after the Transformers 5.9 API change:

> Routing now lives in `DeepseekV3TopkRouter.forward`, which returns
> `(router_logits, topk_weights, topk_indices)`, and excluded groups are masked with
> `-inf`. A finite zero sentinel is incorrect when balancing bias makes every eligible
> score negative because an excluded expert can then re-enter the final top-k.

That is precisely the class of silent, plausible-looking error the challenge described,
and it was found without a single GPU.

**What this does and does not cover.** It covers selection semantics, weighting semantics,
module discovery and the hook layer. It does not cover DeepSeek-V4's two deltas
(`sqrt(softplus)`, flat selection) until `transformers` ships `deepseek_v4` — until then
those are tested against a transcription of the released router, which is weaker because
both sides could share a transcription error. And it says nothing about semantics: random
weights have no behavior.

**Generalization:** for any released model, check whether the modeling code is public
*before* concluding that verification requires the checkpoint. Code-parity at toy scale
covers implementation correctness; only semantic claims need real weights.

---

## 3. No small model exists with the mechanism

**Solved for mechanism — and the answer undercuts the robustness hypothesis.**

Gain is a property of the mixing maps, not of what the weights encode. Random weights
cannot tell you what a model believes, but perturbation propagation is linear algebra plus
the map constraints. `analysis/depth_gain.py` builds all three residual variants at
identical size (d=64, n=4, 48 layers) and measures the gain curve.

```
  depth      plain     hc (unconstrained)   mhc (constrained)
      1      1.006              1.012              0.594
     16      1.145              1.279              0.935
     32      1.289              2.103              1.883
     48      1.464              4.025              3.843

  sensitivity to the mixing gain alpha (final gain at depth 48):
    alpha=0.01   hc=      3.66   mhc=3.59   ratio      1.0x
    alpha=0.1    hc=      3.58   mhc=3.59   ratio      1.0x
    alpha=0.5    hc=     20.37   mhc=3.65   ratio      5.6x
    alpha=1.0    hc=  73060.38   mhc=3.84   ratio  19050x
```

The published pair is ~3000 (unconstrained) versus ~1.6 (constrained), a ~1900× ratio
([mHC, arXiv:2512.24880](https://arxiv.org/pdf/2512.24880)). The toy model **brackets
that ratio** at α=1.0 — so the effect is a property of the maps and reproduces without
training or scale.

Two consequences that matter more than the reproduction:

- **The separation is entirely a function of α**, the mixing gain. At the small α used at
  initialization the two variants are *identical* (ratio 1.0×). The published ~1900× is a
  statement about where training took α, not about the constraint alone.
- **The constrained residual does not damp perturbations.** Its gain at depth 48 is ~3.8 —
  mild *growth*, not decay. The hypothesis that a constrained residual makes deep layers
  unreachable from the input is **not supported by the mechanism**. Bounded ≠ decaying, and
  an attacker needs only non-vanishing, not amplification.

This is the most useful negative result in the set: it removes the main a-priori reason to
expect the architecture to be robust, and it cost nothing.

**Still open:** semantics. No claim about refusal behavior follows from random weights.

---

## 4. Layers with fixed selection but learned weighting

**Solved and implemented.** `MoEHookManager.capture_hash_layers()` installs a forward
pre-hook on the base module that snapshots `input_ids`, so the gate hook can compute
`indices = tid2eid[input_ids]` and pair them with the learned weights.

The design decision worth stating: it is **off by default**, because the two statistic
families want opposite treatment.

- *Membership* statistics (which experts fire — the standard expert-localization signal)
  should exclude these layers. On a hash layer, membership is a deterministic function of
  the token id, so the statistic measures the corpus's token distribution and nothing about
  the model.
- *Mass* statistics (how much each expert contributes) should include them, because the
  weights come from the learned score function and do vary with context.

Also fixed: layers that register no activations produce a wall of identically-zero cells,
and since difference-based scores go negative, those zeros **outrank** informative cells.
On a 43-layer, 256-expert model that is 768 phantom candidates competing for a top-20%
selection. They are now masked to −inf before selection rather than left at zero.

**Still open:** whether the content-dependent weight magnitude on these layers carries
usable signal. Now measurable in one forward pass; previously not capturable at all.

---

## 5. No full-precision reference for QAT weights

**Workaround, not a solution.** Three substitutes for the missing reference, in descending
order of how much they buy:

1. **Well-definedness before accuracy.** Verify determinism (two runs, same batch, bitwise
   equal) and batch-invariance (same token in different batch compositions, bitwise equal).
   These are achievable properties of the shipped inference stack and they make an exact
   score a well-defined observable even without a more accurate reference. A quantity you
   cannot reproduce twice cannot be attributed to anything.
2. **Published component floors.** The FP4 index path ships at 99.7% top-k recall, so ~0.3%
   of selected blocks may differ from an FP32 indexer. Any margin or flip claim finer than
   that is below the architecture's own noise. This is implemented as
   `precision.below_noise_floor`.
3. **Refuse the inverted operation.** Applying a generic 4-bit loader on top of QAT fp8/fp4
   weights introduces error the deployed model does not have. `precision.check_quant_policy`
   refuses that combination rather than warning, keyed off the config's `expert_dtype`.

The FP4-training literature ([arXiv:2501.17116](https://openreview.net/forum?id=uK7JArZEJM),
[NVFP4 QAD](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf)) reports
model-level accuracy recovery but not per-tensor error bounds, so it cannot substitute for
(2). **The honest position is that a model-level "accuracy preserved" claim does not bound
the error on a per-token routing margin, and no published number currently does.**

---

## 6. Conservation exact in math, approximate in code

**Solved: it reduces to one cheap, forward-only scalar.** The mixing matrix is 4×4, so its
residual is a property of the matrix, not of the model. `analysis/sinkhorn_residual.py`
sweeps it densely and cross-checks against Knight's asymptotic contraction factor
σ₂(P)² per iteration ([Knight 2008](https://www.cerfacs.fr/algor/reports/2006/TR_PA_06_42.pdf)).

```
  logit sigma   median mean-drift    worst    |B|_2 max    sigma2^2 (contraction/iter)
          0.1          1.01e-06   1.13e-06       1.0000        0.0041
          1.0          1.01e-06   1.29e-04       1.0000        0.2653
          2.0          1.67e-06   3.07e-02       1.0020        0.5381
          4.0          2.89e-03   6.25e-02       1.0131        0.7889
          8.0          2.00e-02   7.62e-02       1.0193        0.9080
         16.0          2.46e-02   8.07e-02       1.0203        0.9479

  median drift crosses bf16 resolution (~8e-3) at mixing-logit sigma ~ 5.2
```

The answer is **scale-dependent, and the intuitive "20 iterations is plenty" is only half
right**:

- σ ≤ 1: residual pinned at the ~1e-6 epsilon floor, converged in 3–30 steps. Conservation
  holds far below bf16 resolution; treat it as exact.
- σ ≥ 2: **convergence stalls** — σ₂² → 1, so 20 iterations is *not* enough. Median drift
  reaches 2e-2 at σ=8, worst case 8e-2, above bf16 and comparable to fp8 e4m3.
- And `‖B‖₂` exceeds 1 in the same regime (1.0203 at σ=16). Compounded over 43 layers that
  is ~2.4×, so **the non-expansiveness guarantee degrades exactly where conservation does,
  for the same reason.**

So the open question collapses to: *what is the mixing-logit spread in the trained model?*
One forward pass, no labels, and it settles both this and challenge 3 — they turn out to be
the same scalar. Initialization favors the safe regime (small mixing gain, near-identity
bias), so the burden of proof is on showing training moved it.

---

## 7. Attribution through compressed, sparsely-selected attention

**Partial: a computable bound, not the exact chain.** The exact attribution is a product of
three tensors — per-token compression weight × block-selection indicator × sink-inclusive
attention weight — and all three live inside a fused kernel.

What is obtainable without touching the kernel:

- **An upper bound per token.** Compression weights within a block are a softmax over `2m`
  entries (`m=4` for the finer tier), so no single token can carry more than the softmax
  max, and a uniform-attribution assumption is wrong by at most `1 − 1/(2m)`. That bounds
  the error of the standard shortcut instead of leaving it unquantified.
- **The selection mask, from outside.** Which blocks a query attends to is recoverable by
  ablation: mask a candidate block from the input and check whether the output changes at
  all. Selection is hard top-k, so the response is binary and unambiguous — expensive
  (`O(blocks)` forwards) but exact, and it needs no kernel access.
- **The sink correction.** Attention mass per head does not sum to 1. Renormalizing to 1 —
  the standard move — inflates every weight by `1/(1 − sink_share)`. Reporting the
  un-renormalized weights alongside the sink share preserves the information; the
  normalization is the discardable step.

**Still open:** the compression weights themselves. Deriving them from outside the kernel
would require solving for `2m` unknowns per block from ablation responses, which is
possible in principle and probably not worth it. For routing-level work — where the object
of study is expert selection, not token attribution — the bound above is sufficient and
this challenge can be parked.

---

## What actually changed

Four of seven challenges dissolved once the question was posed correctly:

- #2 needed the released **code**, not the released weights.
- #3 and #6 are properties of **small matrices and their maps**, not of trained weights.
- #1's obstacle was **misidentified** — the margin is easy to cross; alignment is the
  hard part, and alignment is measurable on cheap models.

Three findings run against the prevailing story and are worth stating plainly:

1. The balancing bias does **not** create a protective margin.
2. The constrained residual does **not** damp perturbations with depth — gain ~3.8 over 48
   layers, and no separation from unconstrained at initialization-scale mixing gain.
3. `sqrt(softplus)` scoring **does** widen relative margins versus `sigmoid`, so the one
   real robustness gain in the newer gate appears to be an unclaimed side effect.

The remaining hard core is small: soft-to-hard fidelity (a few GPU-hours on any small MoE),
semantic validation (needs the real checkpoint), and per-tensor error bounds for QAT
numerics (needs publication that does not exist yet).

## Reproducing

```bash
python experiments/mhc/analysis/margin_forecast.py     # challenge 1
python experiments/mhc/analysis/depth_gain.py          # challenge 3
python experiments/mhc/analysis/sinkhorn_residual.py   # challenge 6
pytest experiments/mhc/tests/test_reference_parity.py  # challenge 2
```

All CPU, all under a minute, no checkpoint.
