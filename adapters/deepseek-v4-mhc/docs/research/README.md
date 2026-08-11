# mHC / DeepSeek-V4-Flash — separate experiment

Target: **DeepSeek-V4-Flash** — ~284B total / ~13B active, 43 layers, hidden 4096, 256 routed
experts + 1 shared, 6 active/token, MLA + hybrid compressed attention (CSA/HCA), and the **mHC**
(Manifold-Constrained Hyper-Connections) residual stream.

This folder is isolated from the main pipeline on purpose: `make all` does not run any of it.
What *is* now shared is the mechanism — gate semantics, mHC residual handling and the precision
policy live in the `routeaudit` package so they are version-controlled and reusable, and this
folder holds the research on top.

```
experiments/mhc/
  README.md                      ← you are here
  changes_and_runbook.md         ← what changed, and how to run everything. START HERE.
  plan.md                        ← the phase plan (P0–P4). The source of truth.
  technical_challenges.md        ← the open problems, stated project-independently
  technical_solutions.md         ← proposed resolutions + measured numbers
  analysis/                      ← CPU-only studies backing them (margins, gain, Sinkhorn)
  scoping.md                     ← older research scoping; superseded where it conflicts
  challenges_vs_qwen36.md        ← DeepSeek-V4 vs Qwen3.6, challenges C1–C8
  deepseek_v4_mhc_technical_blockers.md   ← the neutral 8-blocker survey
  reasoning_model_blockers.md    ← thinking-mode / boundary-token issues
  route_mhc.py                   ← routing-mass diagnostic at t*
  fixtures/{extract,validate}.py ← ladder Level 1 — saved legacy fixture + strict v2 format
  tests/                         ← the diagnostic battery + the pytest suite
```

The configs are **not** here — they moved into the version-controlled tree, because
`experiments/` is gitignored and a config nobody can recover is worse than useless:
`configs/deepseek_v4_flash.yaml` (nickname `deepseek-v4-flash`) and
`configs/deepseek_v2_lite.yaml` (`deepseek-v2-lite`).

---

## What changed, and why the old README was wrong

This folder was written before the weights shipped and argued the method fails because the gate
is *"sigmoid, grouped, and never exposes a score tensor."* Verified against the released
`config.json` + `inference/model.py`, two thirds of that is false for V4-Flash:

| Old claim | Released V4-Flash | Consequence |
|---|---|---|
| `affinity = σ(Wh)` | **`sqrt(softplus(Wh))`** | the old code computed the wrong affinity → wrong routing mass at every layer |
| **Grouped / node-limited** top-k (`n_group=8`) | **Flat top-6 over 256 experts.** No `n_group` anywhere | the "grouped bottleneck" was the scoping's central obstacle (O2). It does not exist. A flat top-k is a clean 1-D margin — the gate is *more* portable than feared |
| (not modeled) | first **3 MoE layers are HASH-routed** (`tid2eid[input_ids]`) | content-independent and non-differentiable → zero attack surface, and they must be excluded from every per-layer statistic |

Two things the old README got right, and that still hold:

- **The bias is selection-only.** `indices = topk(score + bias)` but `weights = score.gather(idx)`
  from the *bias-free* score. So "which experts fire" and "how much each contributes" are
  different tensors, and a flip caused by the bias is a load-balancing artifact, not content.
  `gate_math.RouteResult` keeps both so they cannot be conflated.
- **Backend outputs differ.** DeepSeek's raw reference gate returns `(weights, indices)`,
  while Transformers V4 returns `(logits, weights, indices)`. RouteAudit consumes the
  official logits when present and otherwise recomputes them from the gate input
  (`arch.router_output: recompute`). Fixture v2 also retains the official returned
  weights and indices, so same-device parity is checked independently of CPU replay.

The corrected reference gate, per token:

```
raw   = W · h                          # W: (256, 4096); h is the gate input, d-dim
score = sqrt(softplus(raw))            # NOT sigmoid; elementwise, >= 0, unbounded above
idx   = topk(score + bias, k=6)        # FLAT over all 256; bias is SELECTION-ONLY
w     = score.gather(idx) / sum * 1.5  # gating weight uses the BIAS-FREE score
# layers 0..2: idx = tid2eid[token_id]  (hash routing — ignore for any attack)
```

## Where mHC actually bites

```
X_{l+1} = B_l · X_l  +  C_l · F_l(A_l · X_l),     X_l ∈ R^{4 × 4096}
```

- **It does not break routing capture.** `A_l` (`hc_pre`) mixes the 4 streams down to one
  4096-dim vector before the sub-layer, so the gate input is `(T, d)` as usual.
- **It does break residual probing.** The per-layer hidden state is `(T, 4, d)`. Flattening it —
  which the previous `residual_norm_profile` did — norms a mixture of 4 streams and turns the
  conservation signature into noise. `HookCapture.residual_streams` now records the count and
  `mhc.reduce_residual` reduces on the **stream-mean**, which is the quantity the doubly-
  stochastic `B` conserves (`mean(B X) = mean(X)`) — not an arbitrary convention.
- **It may harden routing — the open question (H1).** `‖B‖₂ ≤ 1` makes the residual path
  non-expansive, so an input perturbation should not amplify with depth (the paper reports gain
  ≈1 vs ≈3000 for unconstrained Hyper-Connections). If the layers that gate refusal are
  unreachable from the input, that is an incidental routing-robustness mechanism and a result in
  its own right. `refusal_tests.mhc_conservation_profile` measures it directly — the Birkhoff
  constraint and the depth-wise gain, not just the norm-profile symptom.

## What this folder gives you

**A correct routing diagnostic.** `route_mhc.py` reports safety/harmful routing mass at the
boundary token `t*` through the real gate, over content-routed layers only.

**The two halves of the P1 go/no-go.** `tests/margin_census.py` measures how far a safety expert
sits from falling out of the top-6, in selection-score units; `refusal_tests.suffix_leverage_probe`
measures how far a *continuous* suffix can move the decision. A soft suffix is strictly stronger
than any text suffix, so:

> achievable Δ < margins in the safety-bearing layers ⇒ no input-only attack works there.
> Report robustness. That is a first-class result, not a failure.

**A validation ladder.** Level 0 passes on CPU. A real B200 run produced a legacy Level 1
fixture: all retained structural evidence passes, while official returned gate values and
real HyperConnection maps were not retained by that old format. Fixture v2 captures both.

```
Level 0  synthetic mHC model, random weights, fp32, CPU     ← run_synthetic.py     PASSING
Level 1  component fixtures from the released checkpoint    ← fixtures/validate.py PARTIAL (legacy)
Level 2  full forward parity on a fixed prompt              ← fixtures/validate.py PENDING
Level 3  semantic experiments on the real checkpoint        ← the diagnostics      PENDING
```

See [V4_ADAPTER_RESULTS.md](V4_ADAPTER_RESULTS.md) for the evidence, exact limitation, and
the patched v2 format. Do not describe the successful real forward as checkpoint access
still being pending.

Everything that stays open after this, stated without reference to this project, is in
[technical_challenges.md](technical_challenges.md). Short version: steering the biased
top-k (open research), verifying against the real weights (resources), the absence of any
small model with an mHC residual, the half-observability of hash layers, the missing
full-precision reference for QAT weights, and attribution through fused compressed
attention.

## What is NOT here

- **The suffix attack.** `src/routeaudit/attacks/suffix_search.py` losses assume
  `softmax(logits)`. Porting them to bias-free gating weights plus a selection-margin hinge is
  phase P2, gated on P1's verdict. `route_mhc.py --suffix` only *evaluates* a suffix derived
  elsewhere.
- **CSA/HCA token attribution.** Deliberately out of scope — nothing in the pipeline consumes
  token-level attention attribution. The compression schedule sits in the config as
  informational only.
- **A small, cheap real mHC checkpoint.** DeepSeek-V4-Flash is public but its native
  FP4 experts require a large Blackwell deployment; the paper's smaller checkpoints are
  not available. V2-Lite is a *gate* proxy with a plain residual, so no conservation
  result from it says anything about mHC.

## Running it

```bash
pip install -e .                                              # once

pytest experiments/mhc/tests/ -q                              # CPU tests, seconds
python experiments/mhc/tests/run_synthetic.py                 # Level 0, CPU, seconds

# real sibling (~9 GB NF4) — exercises the grouped gate on real weights
python experiments/mhc/tests/run_diagnostics.py --config deepseek-v2-lite --quant nf4 \
    --tests margin,affirm,leverage,selection,routing,reachability,norm

python experiments/mhc/fixtures/validate.py --fixtures PATH   # offline saved-fixture check
```

`route_mhc.py` additionally needs `data/` plus `artifacts/{safety,harmful}_experts.json` from a
harvest run.

**Precision.** V4-Flash's fp8 weights and fp4 experts are QAT-native — that *is* the deployed
model, and FP4→FP8 dequantization is lossless, so a bitsandbytes NF4 pass adds error the real
model does not have. `model/precision.py` refuses that combination; NF4 remains correct for the
bf16 sibling. Quote selection margins against the published 99.7% indexer top-k recall — anything
finer is below the architecture's own noise floor.

## References

- DeepSeek-V4 report: mHC §2.2, `sqrt(softplus)` affinity §2.1, hash routing §2.1,
  Sinkhorn `t_max=20` column-then-row (Eq. 8), QAT / precision §5.2.1.
- Released weights: `deepseek-ai/DeepSeek-V4-Flash` `config.json` + `inference/model.py`
  (`Gate`, `hc_pre`/`hc_post`) — the source of the corrections above.
- mHC: arXiv:2512.24880 · RouteHijack: arXiv:2605.02946 · hash routing: Roller et al. 2021 ·
  auxiliary-loss-free balancing: Wang et al. 2024a (arXiv:2408.15664).
