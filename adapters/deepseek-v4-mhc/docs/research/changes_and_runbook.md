# mHC / DeepSeek work — changes and how to run things

Covers 2026-08-02 → 2026-08-03. Two parts: **what changed** (and why), then **how to run
the experiments**, starting with the one that gates everything else.

Git state: the first tranche is committed as `5bda3d0 "Mhc attempt"` (15 files, +1430).
Three files are uncommitted on top of it — `attacks/suffix_search.py`,
`model/gate_math.py`, `model/hooks.py` — carrying the gradient fix and the group-mask fix
below. **Everything under `experiments/` is gitignored and therefore untracked**: the
analysis scripts, the tests, and these docs exist only on this machine.

---

# Part 1 — What changed

## 1.1 Bugs found and fixed

Ranked by how much they affected results. The first three were silent — they produced
plausible numbers rather than errors.

### A. Routing losses contributed zero gradient (main pipeline, every model)

`MoEHookManager` stored `logits.detach()`. `_loss_suppress_bi` / `_loss_promote_bi` are
built from that tensor, so they were **constants** in the proposal gradient:

```
detach=True   router_logits.requires_grad=False  routing-loss grad -> ERROR: does not require grad
detach=False  router_logits.requires_grad=True   routing-loss grad -> 1.2873e+00
```

The backward pass still succeeded because `L_refusal` carries gradient, the loss still
went down, and nothing errored. The routing objective only ever *scored* candidates; it
never *steered* the search. This is not DeepSeek-specific — it affected OLMoE, Mixtral,
Qwen, Phi-MoE equally.

Fixed: `capture_router_logits(detach=False)` on the gradient path
([hooks.py](../../src/routeaudit/model/hooks.py),
[suffix_search.py](../../src/routeaudit/attacks/suffix_search.py)). Candidate scoring is
under `@torch.no_grad()`, so nothing is retained there.

> **Consequence you must handle:** `lambda_suppress: 3.0` vs `lambda_refusal: 1.0` were
> set while λ_suppress contributed *nothing* to the search direction. It is now the
> largest-weighted gradient term. Those weights are stale and need re-tuning.

### B. Group mask fill value follows the supported installed reference

The supported Transformers `DeepseekV3MoE.route_tokens_to_experts` fills excluded groups
with **zero**. RouteAudit now matches that behavior exactly, including the unusual edge
case where a sufficiently negative balancing bias lets the sentinel enter top-k.
`RouteResult.eligible` remains separate so analytical margin code can identify which
groups were selected by the group contest. DeepSeek-V4 uses a flat gate, so this shared
V2/V3 adapter correction does not affect its result.

### C. Sinkhorn projection did not match the released implementation

Three deltas from `DeepseekV4HyperConnection.forward`, all of which change the matrix at
a finite 20 iterations:

- init is `softmax(dim=-1)` (already a row normalization), not bare `exp`;
- the loop runs `t_max - 1` times, because the initial column step counts as the first;
- it **ends on a column normalization**, so columns are exact and rows carry the residual.

Also: the decoder applies `comb.transpose(-1,-2)`, so **B is the transpose**. Combined
with the above, stream-mean conservation depends on the *inexact* axis — it is
approximate, not exact. Added `mhc.residual_matrix()` to make that explicit.

### D. Harvest counted the wrong experts on non-softmax gates

`compute_expert_freq` implemented Eq. 3 literally as `topk(router_logits)`. On a DeepSeek
gate that ignores the selection bias and the group mask, and on V4 the gate returns
`(logits, weights, indices)` — top-k-ing the wrong element counts over `top_k` positions
as if they were the expert axis. Fixed via `capture_expert_selection(gate_spec)`, which
recomputes selection through the real gate and stores only top-k membership (a full
`RouteResult` per layer over a 16×1024 batch would run to gigabytes).

### E. Hash layers flooded expert selection

Hash-routed layers register no activations, so every cell scores exactly `0.0`. Since
`Score_safe = ΔS − P_gen²` goes negative for any expert firing more on harmful text, a
wall of zeros **outranks real experts**. On a 43-layer/256-expert model that is 768
phantom candidates competing for a top-20% selection. Now masked to −inf before selection
(`pipeline._mask_unroutable`).

### F. Smaller ones

- `reduce_streams` guarded on tensor *rank*, so a standard `(B,T,d)` residual would have
  been averaged over **tokens**. Split into `reduce_residual(x, n_streams)`.
- Residual capture flattened `(T, n, d)` into one vector, mixing 4 streams together in
  every norm profile. Now records the stream count alongside.
- `load_model` raised `KeyError('fp8')` on a QAT-native config. Added `_resolve_dtype`.
- Two unseeded Sinkhorn checks in `run_synthetic.py` compared values both sitting on the
  epsilon floor — a flaky test. Seeded and moved to a low iteration count.

## 1.2 New package modules

| Module | Purpose |
|---|---|
| [`model/gate_math.py`](../../src/routeaudit/model/gate_math.py) | `GateSpec` / `RouteResult`. All gate semantics in one place: scoring function, flat vs node-limited selection, selection-only bias, `selection_margin`, hash/dense layer classification. |
| [`model/mhc.py`](../../src/routeaudit/model/mhc.py) | Sinkhorn (release-faithful), `residual_matrix`, `reduce_residual`, `b_path_conservation_check`, `perturbation_profile`. |
| [`model/precision.py`](../../src/routeaudit/model/precision.py) | Refuses bitsandbytes on QAT-native fp8/fp4 weights; exposes the 99.7% indexer noise floor. |

Plus: `deepseek` ArchSpec preset, `deepseek_v2/v3/v4` HF autodetect, `configs/deepseek_v4_flash.yaml`
and `configs/deepseek_v2_lite.yaml` (moved out of gitignored `experiments/` so they survive).

## 1.3 New experiment tooling

```
experiments/mhc/
  analysis/
    soft_hard_fidelity.py     THE experiment — see Part 2
    margin_forecast.py        selection margins from order statistics, no model needed
    depth_gain.py             perturbation gain vs depth, plain/hc/mhc ablation
    sinkhorn_residual.py      how far from doubly-stochastic 20 iterations actually gets
  fixtures/
    extract.py, validate.py   ladder Level 1 — pending checkpoint access
  tests/                      76 pytest tests (the repo previously had zero)
    test_reference_parity.py  differential parity vs official DeepseekV3, CPU
    test_gate_math.py, test_mhc.py, test_harvest_gate.py, test_no_regression.py
  technical_challenges.md     the open problems, stated project-independently
  technical_solutions.md      proposed resolutions + measured numbers
```

## 1.4 Findings worth carrying forward

From the CPU analyses, all reproducible in under a minute:

1. **The balancing bias does not protect the boundary.** Raising bias spread 0 → 0.5
   leaves the relative margin flat at ~1.0%. It changes *which* experts are marginal, not
   how hard the boundary is to reach.
2. **The constrained residual does not damp perturbations with depth** — gain ~3.8 over 48
   layers (growth, not decay), and *zero* separation from unconstrained at
   initialization-scale mixing gain (ratio 1.0× at α=0.01, 19,050× at α=1.0). The
   published ~1900× is a statement about where training took α.
3. **`sqrt(softplus)` widens relative margins vs `sigmoid`** (1.05% flat vs 0.59% → 0.01%
   as sigmoid saturates). The one real routing-robustness gain in the newer gate looks
   like an unclaimed side effect.
4. **Conservation reduces to one measurable scalar** — the mixing-logit spread. Below
   σ≈2 it holds at ~1e-6; median drift crosses bf16 resolution at σ≈5.2, and `‖B‖₂`
   exceeds 1 in the same regime.

---

# Part 2 — Running things

## 2.0 Setup

```bash
pip install -e .                 # the routeaudit package
pip install bitsandbytes         # only if using --quant nf4/int8
```

The tests fall back to the in-repo `src/` if the package isn't installed; the analysis
scripts add it to `sys.path` themselves. So the CPU-only work needs nothing but torch.

## 2.1 The fidelity experiment (run this first)

`analysis/soft_hard_fidelity.py`. **This is the gate on whether porting the suffix attack
to a DeepSeek gate is worth doing at all**, and it is now also the only way to tell
whether bug (A) helped, was neutral, or made things worse.

### What it measures

Optimizes a soft-embedding suffix and logs, every step: the **soft** surrogate the attack
differentiates, the **hard** count of target experts actually in top-k under the real
gate, and their **summed selection margin**. Then reports:

- **gradient vs a matched-step random control** — the load-bearing number. If gradient
  flips no more experts than an equal-magnitude random walk, the gradient carries no
  routing information regardless of what the loss curve does. Neither Misrouter nor
  RouteHijack reports this.
- **dead-zone fraction** — steps where soft improved but hard did not move at all.
- **two relaxations**: `prob` (`Σ softmax(logits)_e`, what the literature uses) vs
  `boundary` (`Σ sigmoid((score − kth)/T)`, a soft version of the actual indicator).
- **margin traversal** — diagnoses *why* on failure: did the optimizer move targets
  toward their boundary, or shuffle mass among experts that were never close?

### Running it

```bash
# cheapest useful run: OLMoE-1B-7B, ~14GB bf16, fits one 24GB card, minutes not hours
python experiments/mhc/analysis/soft_hard_fidelity.py --config olmoe

# a real DeepSeek gate (sigmoid + node-limited), ~9GB under NF4
python experiments/mhc/analysis/soft_hard_fidelity.py --config deepseek-v2-lite --quant nf4

# scoped to harvested safety experts instead of the default targets
python experiments/mhc/analysis/soft_hard_fidelity.py --config olmoe \
    --safety artifacts/safety_experts.json
```

| flag | default | notes |
|---|---|---|
| `--config` | `olmoe` | any supported MoE, or a raw HF id |
| `--quant` | `none` | `nf4`/`int8` to fit a bigger model; refused on QAT-native weights |
| `--steps` | 120 | optimization steps per arm (4 arms total) |
| `--lr` | 0.05 | Adam lr; also sets the random control's step size |
| `--n-soft` | 20 | soft suffix length in tokens |
| `--temp` | 0.05 | temperature for the `boundary` relaxation |
| `--layers` | 8 | max target layers — the main cost knob |
| `--safety` | `artifacts/safety_experts.json` | falls back to top-2 firing experts per layer |

Without a harvest it targets the experts that actually fire at the boundary token on the
clean prompt. That is the right default: those are exactly what a suppression attack aims
at, and starting them *inside* the top-k means every flip is a real removal.

Writes `artifacts/soft_hard_fidelity.json`.

### Reading the output

| Verdict | Meaning | Do next |
|---|---|---|
| **DEAD** — no flips | Routing objective steers nothing; reported ASR comes from the refusal/target terms | Write it up, stop |
| **NO SIGNAL** — gradient ≤ random | Loss drop is real, routing effect not attributable to it | Write it up, stop |
| **WEAK** — beats random, ρ < 0.2 | Flips found incidentally, not tracked | Search will be very inefficient; consider the `boundary` surrogate |
| **TRACKS** | Surrogate tracks the hard selection | Port stage 02 |

**The good outcome is `boundary` tracking while `prob` does not** — that makes the method
salvageable by changing the surrogate rather than abandoning it.

**Caveat on ρ:** the hard series is integer-valued and mostly zero, so Spearman is
tie-inflated. Read it as secondary. The gradient-vs-random flip count has no such problem.

### Prediction (recorded before running, so it is falsifiable)

Expect **WEAK** overall. `prob` should underperform for a mechanical reason: minimizing
`Σ softmax(logits)_e` is dominated by whichever target has the *largest* probability —
the one sitting deepest inside the top-k, i.e. hardest to dislodge — so gradient goes to
the expert least likely to flip. `boundary` puts it where the decision is. Guesses:
`prob` dead-zone > 60%, `boundary` meaningfully lower, random achieving 20–40% of
gradient's flips.

One asymmetry to watch: softmax normalizes, so suppressing safety experts automatically
promotes others. `sigmoid`/`sqrtsoftplus` are elementwise — suppression promotes nothing.
If that shows up, the **promote** half of the objective needs separate work on
DeepSeek-family gates.

### Verification status

Internals were driven end to end on a CPU stand-in (targets, both relaxations, both arms,
analysis, verdict). **It has not been run against a downloaded model** — no GPU here. Treat
the first real run as also a shakedown of argument plumbing.

## 2.2 CPU analyses (no model, no GPU, under a minute each)

```bash
python experiments/mhc/analysis/margin_forecast.py     # selection margins, order statistics
python experiments/mhc/analysis/depth_gain.py          # gain vs depth: plain / hc / mhc
python experiments/mhc/analysis/sinkhorn_residual.py   # 20-iteration residual + contraction
```

All three back the findings in §1.4 and print their own reading. No arguments needed.

## 2.3 Tests

```bash
pytest experiments/mhc/tests/ -q          # 76 tests, ~35s, no downloads
python experiments/mhc/tests/run_synthetic.py   # Level 0 mechanism validation, exits non-zero on failure
```

`test_reference_parity.py` is the important one: it diffs our gate against the **official**
`DeepseekV3MoE` from `transformers`, instantiated at toy size with random weights. That is
how bug (B) was found. It skips cleanly on a transformers build without `deepseek_v3`.

## 2.4 Pipeline stages on a DeepSeek gate

```bash
python scripts/00_data.py
make harvest MODEL=deepseek-v2-lite        # works — gate-aware selection
make eval    MODEL=deepseek-v2-lite        # works — set judge_kind: llamaguard for a 1B judge
python experiments/mhc/route_mhc.py --config deepseek-v2-lite    # routing mass at t*
make routeaudit MODEL=deepseek-v2-lite     # RAISES UnsupportedGateError — not ported
```

Stage 02 is deliberately blocked with an actionable error rather than failing deep in a
GPU run with an `IndexError`. Porting it is phase P2 in [plan.md](plan.md), gated on the
fidelity result above.

## 2.5 Saved-checkpoint fixture

```bash
python experiments/mhc/fixtures/validate.py --fixtures PATH_TO_v4_flash_fixtures.pt
```

The completed B200 run is execution-verified and all evidence retained in its legacy
fixture passes. It remains partial for strict independent parity because that old fixture
did not retain the router's returned values or the real HyperConnection maps. See
[V4_ADAPTER_RESULTS.md](V4_ADAPTER_RESULTS.md); no repeat rental is recommended for
development.

---

## Suggested next steps, in order

1. **Run the fidelity experiment on OLMoE** (~$5 of rented GPU, minutes). It decides
   everything downstream.
2. **Re-tune `lambda_suppress` / `lambda_promote`** — they were set while contributing
   zero gradient and are now the dominant terms.
3. Optionally add a third arm ablating the gradient fix itself (routing gradient on vs
   off, same seeds), turning "we fixed a bug" into a number.
4. Only if fidelity says TRACKS: port stage 02.
5. **Back up `experiments/`** — it is gitignored, so all of the above lives on one machine.
