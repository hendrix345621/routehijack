# Open technical challenges in analyzing sparse-MoE models with constrained residual streams

> **Update:** proposed resolutions for all seven are in
> [`technical_solutions.md`](technical_solutions.md), with CPU-only reproductions in
> `analysis/`. Four dissolved once re-posed; #1's obstacle turned out to be misidentified.
> This document is kept as the statement of the problems.

A standalone statement of problems that currently have no clean engineering solution. Each
entry describes the problem, why the obvious workarounds fail, and what specifically would
have to be measured or discovered to close it. Nothing here is a matter of writing more
code against known behavior; these are the points where the required information does not
exist yet, or exists only behind a resource wall.

Architecture assumed throughout: a decoder-only mixture-of-experts transformer with

- a router whose affinity is an **unnormalized elementwise function** of the projection
  (e.g. `sqrt(softplus(Wh))`), not a softmax over experts;
- a **selection-only bias** added before top-k but excluded from the gating weight;
- a **multi-stream residual** where each block keeps `n` parallel copies of the hidden
  state and mixes them with a doubly-stochastic matrix produced by a finite-iteration
  Sinkhorn projection;
- **hash-routed leading layers** whose expert *indices* come from a static token-id table;
- **quantization-aware-trained** weights shipped below bf16;
- compressed / sparsely-selected attention over long contexts.

---

## 1. Steering a selection-biased, non-differentiable top-k gate

**The problem.** Discrete input-optimization methods (GCG and its descendants) need a
differentiable scalar that tracks the quantity being attacked. When the attack target is
*routing*, the natural objective is the mass assigned to a chosen set of experts. On a
softmax gate that objective is differentiable everywhere and its gradient is informative.
On the architecture above it is neither:

- expert **selection** is `topk(score + bias)` — piecewise constant, zero gradient almost
  everywhere, undefined at the flip points that are the only places anything happens;
- expert **weighting** is computed from the *bias-free* score and renormalized over the
  selected set, so the weight of an expert is a function of *which other experts were
  selected*. The two quantities move under different conditions, and an objective written
  in one does not control the other;
- the balancing bias is a trained constant per expert. It sets a fixed margin an input must
  overcome, and that margin encodes training-time load pressure rather than anything about
  the input, so a flip achieved by crossing it may not correspond to any semantic change.

**Why the obvious workarounds fail.** Replacing the hard top-k with a temperature-annealed
softmax mask, or a straight-through estimator, or a perturbed-optimizer relaxation
(Berthet et al. 2020) all produce *a* gradient. Whether that gradient points anywhere
useful is the open question: the relaxed objective can be driven down substantially while
the hard top-k never changes, because the relaxation redistributes mass among experts that
are all comfortably inside or outside the selected set. This is a soft↔hard **fidelity**
problem, and it is not detectable by looking at the loss curve — only by measuring flips.

**What must be discovered.**

1. The empirical distribution of **selection margins** at the decision position: for each
   expert of interest, the gap in `score + bias` to the k-th competitor. This is the
   quantity an input has to move, and it is measurable with forward passes only.
2. The **achievable input leverage**: how far a continuous (soft-embedding) perturbation
   can move those selection scores. A soft suffix is strictly stronger than any tokenized
   one, so it upper-bounds the attack.
3. **Soft↔hard fidelity**, quantified: over many optimization steps, the correlation
   between a decrease in the relaxed objective and an actual change in the hard top-k set.
   Near-zero correlation falsifies the whole method class, and that is a publishable
   result rather than a failure.

If (2) is smaller than (1) across the layers that matter, no input-only routing attack
exists on this gate and the correct output is a robustness characterization. **This
ordering matters**: building the optimizer before measuring (1) and (2) risks weeks of
compute on a method that was ruled out by a forward-only experiment.

---

## 2. Verifying a reimplementation against weights you cannot run

**The problem.** Any independent implementation of the gate, the residual mixing, or the
routing statistics is a *transcription* of a reference implementation until it has been
executed side by side with the real one. Transcription errors in this architecture are
systematically silent — they produce plausible numbers with the wrong meaning rather than
exceptions. Three real examples, all of which look correct in isolation:

- Sinkhorn iteration order. Initializing with `softmax` (already a row normalization)
  versus bare `exp`, and ending on a column versus a row normalization, converge to the
  same limit but give different matrices at the finite iteration count actually used. The
  axis that comes out *exact* changes, and with it which conservation property holds.
- Whether the residual mixing matrix is the Sinkhorn output or its transpose. Both are
  doubly stochastic; only one conserves the mean across streams.
- Normalization epsilon placement: `x / (sum + ε)` versus `x / max(sum, ε)` differ
  whenever the sum is not tiny, which is always.

**Why the obvious workarounds fail.** Comparing end-to-end outputs (logits, perplexity)
detects that *something* is wrong but localizes nothing, and small transcription errors
often fall below the noise of any end-to-end metric. Reading the reference source more
carefully has diminishing returns — the errors above all survived source review and were
only caught by executing against ground truth.

**What must be discovered.** Nothing conceptual; this is purely a **resource wall**. The
resolution is a component-fixture ladder, which is cheap to build and impossible to run
without the weights:

```
Level 0  synthetic model, random weights, full precision, CPU  →  mechanism correctness
Level 1  one captured input/output pair per component          →  component correctness
Level 2  full forward parity on a fixed prompt                 →  integration correctness
Level 3  semantic experiments on the real checkpoint           →  the actual results
```

Level 0 is achievable by anyone. Levels 1–3 require loading a checkpoint that, for a
frontier-scale model, is O(100 GB) even at native precision. **Until Level 1 passes, every
downstream number is spec-faithful but not model-verified, and should be reported that
way.** There is no substitute — a second independent reimplementation agreeing with the
first only proves the two share an assumption.

---

## 3. No small model exists with the architecture under study

**The problem.** Standard practice for validating analysis tooling is to develop against a
small model of the same family and scale up once the code is right. For a novel residual
mechanism this fails: the mechanism is introduced by one frontier model, small models in
the same family predate it, and the ablation-scale models from the originating paper are
typically not released. So there is no cheap model that is *both* small and has the
property being studied.

Two things get conflated when a "similar" small model is substituted:

- **Mechanism validation** — does the instrumentation compute what it claims? This can be
  answered by a synthetic model built from the published equations with random weights.
- **Semantic validation** — does the measured quantity mean anything about behavior?
  This *cannot*: random weights have no behavior, and a differently-architected small model
  has behavior driven by a different mechanism.

**Why the obvious workarounds fail.** A same-family sibling from an earlier generation
shares the *gate* lineage but has a plain single-stream residual, so every conservation,
propagation, or depth-reachability result from it is uninformative about the mechanism in
question. Using it and quietly generalizing is the failure mode to avoid; it produces
confident claims about a mechanism the measured model does not have.

**What must be discovered.** Either (a) access to the frontier checkpoint, or (b) a
*trained* small model with the same residual mechanism — which means training one, at
which point the question becomes whether a small trained model exhibits the property at
all, itself an open empirical question. Note the property under study here is a
**signal-propagation** claim (perturbation gain ≈ 1 versus ≈ 10³ for the unconstrained
variant), and propagation behavior is depth-dependent, so a 4-layer synthetic model
cannot settle it even if it is architecturally faithful.

---

## 4. Layers with fixed selection but learned weighting are only half-observable

**The problem.** In a hash-routed layer, expert *indices* come from a static token-id
lookup while the gating *weights* are still computed from the learned score function. The
layer is therefore:

- **unsteerable in selection** — no input change moves which experts fire, and no gradient
  flows to the choice, so it is structurally immune to activation- or input-space attacks;
- **content-dependent in weighting** — the magnitude each selected expert contributes does
  vary with the input.

Analyses that count top-k *membership* (the standard definition of expert activation
frequency) see nothing on these layers: membership is a deterministic function of the token
id, so the statistic reflects the token distribution of the corpus and nothing about the
model's behavior. Analyses that measure routing *mass* would see a real signal there, but
capturing it requires the token ids at the router, which is a different hook contract than
"read the gate's activations."

There is also a scoring hazard. Layers that register no activations produce a wall of
identically-zero cells. If the downstream score can go negative — and difference-based
scores routinely do — those zeros **outrank** genuinely informative cells and dominate any
top-fraction selection. The failure is silent and can consume a large share of the selected
set.

**What must be discovered.** Whether the content-dependent weight magnitude on
fixed-selection layers carries usable signal at all. This is answerable with forward passes
once the ids are plumbed to the capture point, and the answer determines whether these
layers should be excluded from analysis entirely or measured with a mass-based statistic
instead of a membership-based one. Excluding them is the conservative default and is
correct for membership statistics; it is an assumption, not a result.

---

## 5. Quantization-aware-trained weights have no full-precision reference

**The problem.** The usual mental model — "quantization is noise on top of the true
model" — inverts when the weights were quantized *during* training. The low-precision
weights are the deployed model; there is no higher-precision original that they
approximate. Consequences that trip up standard practice:

- Applying a further quantization pass (a generic 4-bit loader, say) to make the model fit
  introduces error the deployed model does not have. The result is not a cheaper view of
  the real model, it is a different model.
- Conversely, "dequantizing to bf16 for accuracy" can be exactly lossless when the
  narrower format's per-block scales fit inside the wider format's dynamic range — in
  which case the wider representation buys nothing but memory.
- Standard advice to "confirm final numbers in full precision" has no referent.

**Why this is hard rather than merely unfamiliar.** Establishing a noise floor normally
means comparing against a more accurate computation. Here the most accurate available
computation *is* the shipped one, so the floor has to come from published component
characterizations — e.g. a stated top-k recall for a low-precision retrieval or indexing
path — rather than from measurement. Any claimed effect finer than that floor is
indistinguishable from arithmetic.

**What must be discovered.** Per-component error characterizations for the shipped
numerics, at the granularity of the quantities being claimed. A model-level "accuracy is
preserved" statement does not bound the error on a per-token routing margin. Absent that,
the honest fallback is to (i) verify run-to-run and batch-composition determinism, so the
observable is at least well-defined, and (ii) report every margin alongside the coarsest
published error bar that applies to it.

---

## 6. Exact conservation claims survive the math but not the implementation

**The problem.** Constrained-residual designs are motivated by an exact algebraic property
— a doubly-stochastic mixing matrix is non-expansive, and the mean across streams is
invariant under it. Both properties hold for a *converged* Sinkhorn projection. Shipped
implementations run a fixed, small iteration count and add an epsilon to every
denominator, so the matrix is doubly stochastic only to the finite-iteration residual, and
only on the axis normalized last. If the residual update then transposes the matrix, the
exact axis is the one that does *not* underwrite the conservation claim.

The practical consequence: "the stream mean is conserved" becomes a quantitative statement
with a model-, layer-, and input-dependent error, not an identity. Any analysis that
reduces the multi-stream residual to a single vector "because that reduction is invariant"
inherits that error, and it compounds across depth.

**What must be discovered.** The actual magnitude of the deviation on real activations,
per layer and across depth, and whether it stays negligible relative to the effects being
measured. This is cheap to measure *given the model* — it is a property of the mixing
matrices alone, needing no labels or generation — and it is not safely assumed. The
related question, whether the deviation accumulates or cancels over dozens of layers,
follows from the same capture.

---

## 7. Token-level attribution through compressed, sparsely-selected attention

**The problem.** Long-context attention variants replace dense attention with a chain of
transformations, each of which breaks the usual "attention weight = token influence"
reading:

1. blocks of tokens are compressed into fewer entries by a learned weighted sum, so an
   original token's contribution is spread across compressed entries — and with
   overlapping compression windows, across more than one;
2. a scoring module emits per-(query, block) scores and takes a hard top-k, so most blocks
   are simply invisible to a given query;
3. the surviving attention scores are normalized against a denominator that includes a
   learned sink term, so attention mass per head does **not** sum to 1.

Reading step 3's weights as token attribution — the standard move — is wrong three times
over: it ignores the compression weights, treats unselected blocks as merely low-weight
rather than absent, and renormalizes away the sink.

**Why the obvious workaround fails.** Assuming uniform attribution within a compressed
block is the common shortcut and is wrong by exactly the amount the compression weights are
non-uniform, which is the interesting part.

**What must be discovered.** Nothing conceptually unknown — the exact chain is
recoverable, and it is the product of three recorded tensors (per-token compression
weight × block-selection indicator × sink-inclusive attention weight), summed over the
compressed entries a token contributes to. The obstacle is that all three live inside a
fused kernel path optimized for throughput, so obtaining them means either instrumenting
that path or running an unfused reference implementation whose numerical agreement with
the fused one is itself unverified. The open question is how much attribution fidelity is
lost to whichever compromise is chosen.

---

## Summary

| # | Challenge | Blocked by |
|---|---|---|
| 1 | Steering a biased non-differentiable top-k | Open research — soft↔hard fidelity is unmeasured |
| 2 | Verifying a reimplementation | Resource wall — needs the checkpoint |
| 3 | No small model with the mechanism | Nothing released; mechanism ≠ semantics |
| 4 | Half-observable fixed-selection layers | Unmeasured whether weight magnitude carries signal |
| 5 | No full-precision reference for QAT weights | Missing per-component error characterizations |
| 6 | Conservation exact in math, approximate in code | Needs measurement on real activations |
| 7 | Attribution through compressed attention | Fused kernels hide the intermediate tensors |

Challenges 1, 4 and 6 are answerable with forward passes alone, given access. Challenge 2
is purely resources. Challenges 3, 5 and 7 need something that does not currently
exist — a released small model, published per-component error bars, or an instrumented
reference kernel.
