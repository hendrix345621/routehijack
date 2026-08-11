# Plan: make RouteAudit actually run on DeepSeek-V4 (mHC) architectures

Status: implementation plan, grounded in the **released** `deepseek-ai/DeepSeek-V4-Flash`
weights (verified against `config.json` + `inference/model.py`), the V4 paper, and the
existing `experiments/mhc/` scoping. It supersedes the "best-effort / UNVERIFIED" assumptions in
`experiments/mhc/config/deepseek_v4_flash.yaml` and `route_mhc.py` where they now conflict with the
real model.

The honest prior is unchanged from `scoping.md`: the most probable publishable outcome
is a **routing-robustness characterization**, not a working suffix search. This plan is
designed to reach a real result either way, and — critically — to **fix the parts of the
existing `experiments/mhc/` code that model a gate the released model does not have.**

---

## 0. What changed once the weights were inspected (read this first)

The `experiments/mhc/` folder was written before the weights shipped and encodes three assumptions
that are now **falsified** for DeepSeek-V4-Flash. Any plan that "gets it working" has to
correct them, because the current diagnostic silently computes the wrong routing.

| `experiments/mhc/` code assumes | Released Flash actually does | Consequence |
|---|---|---|
| `scoring_func: sigmoid` — `affinity = σ(Wh)` | **`sqrtsoftplus`**: `sqrt(softplus(Wh))` | `route_mhc.py`/`diag_common.py` compute the wrong affinity → wrong safety/harmful mass. Must fix before any number is trusted. |
| **Grouped / node-limited** top-k (`n_group=8, topk_group=4`) | **Flat top-k over 256 experts.** No `n_group`/`topk_group` anywhere in the released Gate or config. | Scoping's "O2 grouped bottleneck — the genuinely new structure" **does not exist in Flash.** `_grouped_topk` models a phantom. The gate is a *flat, selection-biased* top-k — much closer to a standard MoE gate, so the attack is *more* portable than feared, not less. |
| Bias affects routing | Bias is **selection-only**: `indices=topk(scores+bias)`, but `weights=original_scores.gather(idx)` (bias-free), `/=sum`, `*=1.5` | The suppress/promote loss must target *selection margins* (bias-shifted) for which-experts-fire, but *original scores* for the gating weight. Two different quantities. |
| (not modeled) first 3 layers route by content | **First `num_hash_layers=3` layers use HASH routing**: experts = `tid2eid[input_ids]`, a fixed token-id→expert table. Input-content-independent, non-differentiable. | Those layers are **unsteerable by any activation/suffix attack** — skip them in harvest and in the loss. A real, paper-grounded constraint the scoping missed. |

Corrected reference gate (released Flash), per token:
```
raw   = W · h                              # W: (256, 4096), h = hc_pre(residual) is C-dim
score = sqrt(softplus(raw))                # sqrtsoftplus, elementwise, ≥ 0
idx   = topk(score + bias, k=6)            # bias selection-only; FLAT (no groups)
w     = score.gather(idx); w /= w.sum(); w *= 1.5   # gating weights use bias-free score
# layers 0..2: idx = tid2eid[token_id] instead (hash) — ignore for the attack
```
mHC (`hc_mult=4`, sinkhorn 20 iters): `hc_pre` mixes the 4 residual streams down to one
C-dim vector that the gate sees (so the gate is reachable and capture is dimensionally
fine); `hc_post` writes the sub-layer output back into the 4 streams. Confirmed: **mHC
breaks per-layer residual hooks, not gate-input capture.**

---

## Phase P0 — Correct the harness so it models the real gate (1–2 days, no GPU)

Goal: the diagnostic must compute DeepSeek-V4-Flash routing *exactly*, or every
downstream number is noise. All CPU / synthetic-testable.

1. **Fix affinity + selection in the three places that hardcode sigmoid+grouped:**
   `experiments/mhc/route_mhc.py` (`_grouped_topk`, hook), `experiments/mhc/tests/diag_common.py`
   (`_grouped_topk`, `_grouped_boundary`), `experiments/mhc/tests/synthetic_mhc.py` (GroupedMoE).
   - Replace `logits.sigmoid()` → `F.softplus(logits).sqrt()` (make `scoring_func`
     config-driven: `sigmoid|softmax|sqrtsoftplus`).
   - Replace grouped top-k with **flat** `topk(score+bias, k)`; keep grouped path only
     behind an explicit `n_group>1` flag for V2/V3 siblings.
   - Compute weights from **bias-free** `score.gather(idx)`, normalize, `*1.5`.
2. **Add hash-layer awareness.** Read `num_hash_layers`; mark layers `< num_hash_layers`
   as hash-routed and exclude them from routing capture, harvest, and the loss (they
   carry no steerable safety signal).
3. **Rewrite the config** `experiments/mhc/config/deepseek_v4_flash.yaml`: `scoring_func:
   sqrtsoftplus`, drop `n_group`/`topk_group` (or set `n_group: 1`),
   `routed_scaling_factor: 1.5`, `num_hash_layers: 3`, `n_layers: 43`, `d_model: 4096`,
   `top_k: 6`, `sliding_window: 128`, `hc_mult: 4`, `hc_sinkhorn_iters: 20`. Delete the
   stale "grouped/sigmoid" prose.
4. **Loader reality.** The repo ships raw `inference/model.py` (ModelArgs), but
   `config.json` declares `DeepseekV4ForCausalLM` on transformers 4.57.1 — check whether
   `AutoModelForCausalLM` resolves it natively; if not, wire `trust_remote_code` to the
   repo's module or the DeepSeek raw impl. Record which path loads.
5. **Re-validate on the synthetic mHC model** (`run_synthetic.py`) with the corrected
   flat-sqrtsoftplus gate, so the code path is exercised before spending GPU.

Gate: corrected diagnostic reproduces the released Gate's `(weights, indices)` bit-for-bit
on a saved tensor fixture (extract one real gate's I/O to compare).

## Phase P1 — Feasibility on a cheap sibling (the go/no-go) (1 wk, ~1×24 GB)

Nothing here needs the 160 GB model. Use **DeepSeek-V2-Lite** as the gate-mechanism proxy
(sigmoid+grouped — *different* gate, so treat as a lower-fidelity control) **and** run the
paper-grounded probes that only need forward passes, which do port.

- **Margin census** (§5 scoping, corrected): at `t*`, per safety expert, the Δ in
  `score+bias` needed to fall out of the flat top-6. Report margin distribution per
  (non-hash) layer. Flat top-k margins are simpler than grouped — this is now a clean 1-D
  gap-to-6th-competitor.
- **Soft-embedding leverage upper bound** (`suffix_leverage_probe`, already implemented):
  optimize a continuous suffix to move refusal/affirm at `t*`. **If soft can't cross the
  margins, no text suffix can → stop, write robustness.**
- **Reachability + norm-conservation** (`routing_reachability_by_depth`,
  `residual_norm_profile`): does mHC's doubly-stochastic residual damp input perturbation
  with depth (paper's "gain ≈ 1")? Cross with the refusal fingerprint: can the input even
  reach the layers that gate refusal? This is the mHC-specific robustness test (H1).

Kill criterion: soft upper-bound < margins across safety-bearing layers ⇒ report
"V4 gate robust to input-only routing steering," attribute to bias / sqrtsoftplus
floor / mHC conservation. **This is a first-class result, not a failure.**

## Phase P2 — Port the optimizer to the flat-biased gate (1–2 wk, sibling)

Only if P1 shows leverage ≥ margins. The good news from P0: with grouped selection gone,
the reframed objective collapses to something close to the existing RouteAudit loss.

- **Reuse the existing GCG loop** in `src/routeaudit/attacks/suffix_search.py` almost
  as-is. The only gate-specific change is the **routing loss inner term**: swap the
  `softmax`-mass `_loss_suppress`/`_loss_promote` for the corrected quantity —
  `suppress = Σ_{E_safe} g_e`, `promote = hinge(m − Σ_{E_harm} g_e)` where `g_e` is the
  real bias-free normalized weight, and selection is credited via `score+bias` top-k.
  Add a differentiable surrogate for the top-k (softmax(score/τ) mask) for the *proposal*
  gradient; **score candidates with the corrected hard gate** (P0) — the propose-soft /
  score-hard split RouteAudit already uses.
- Because bias only moves *selection*, add a small **selection-margin hinge** on
  `score_e + bias_e` vs the 6th competitor for `E_safe`/`E_harm` — this is the load-bearing
  term now (replaces scoping's `λ_grp`).
- Measure **soft↔hard fidelity** (does lowering the surrogate flip the real top-6?). Low
  fidelity kills the method — report it.

Gate: selection-flip rate at `t*` responds to the gradient on the sibling.

## Phase P3 — End-to-end on a sibling: ASR + utility (1 wk, sibling)

- Full discrete universal-suffix optimization. Metrics via existing eval harness:
  HarmBench-judged **ASR** (clean vs attacked) + MMLU (confirm the model isn't just
  broken) + routing shift (TESR/THPR redefined on `g_e`, via the corrected `route_mhc.py`).
- **Routing shift without ASR lift is not an attack** — say so plainly if that's the result.

## Phase P4 — The real target + mHC ablation (the frontier cost) (2×80 GB node, or offload)

- **Run the corrected forward-only diagnostics on DeepSeek-V4-Flash itself.** At ~160 GB
  fp8 this fits a 2×80 GB node natively (it's already fp8 — no bitsandbytes needed);
  routing/refusal probes are forward-only so MoE-expert CPU-offload makes even a smaller
  box viable, slowly. This is where mHC is *actually* present.
- **Transfer study**: does a sibling-optimized suffix move routing / lift ASR on Flash?
  Prior: transfer is weak because the binding constraint (selection bias + hash layers)
  is model-specific. Quantify vs a Mixtral/Qwen flat-softmax control.
- **mHC ablation (H1)**: compare input-leverage on matched vanilla-residual vs HC vs mHC.
  If mHC demonstrably lowers achievable Δ at `t*`, that's the clean novel finding:
  *a stability-motivated architecture that also hardens routing.*
- **Pro** (`DeepSeek-V4-Pro`, 1.6T) only if Flash shows a real effect worth confirming at
  scale — not before.

---

## Efficiency & access notes
- Flash ships **fp8** already (~160 GB): 2×80 GB loads it natively; no 4-bit needed for
  the target, though 4-bit siblings stay the cheap-iteration path (P1–P3).
- Expert **CPU-offload** (`max_memory` spill) keeps router+attention resident for
  forward-only probes → single-box routing diagnostics on the real model, slowly.
- Skip the 3 hash layers everywhere: −7% of layers with zero attack surface.

## Deliverables (any branch)
1. A **corrected** `experiments/mhc/` diagnostic that reproduces the released V4-Flash gate exactly.
2. A feasibility verdict (P1) with the soft-embedding upper bound vs margins per layer.
3. Either (a) a robustness characterization attributing V4's resistance to
   bias / sqrtsoftplus / hash-routing / mHC-conservation, or (b) a sibling suffix
   search + transfer result — whichever the gates produce. No distributable suffix
   is built before P1 says input-only steering is even reachable.

## References
- DeepSeek-V4 paper (uploaded): CSA/HCA, mHC §2.2, sqrt(softplus) affinity §2.1, hash
  routing for first layers §2.1, Sinkhorn t_max=20.
- Released weights: `deepseek-ai/DeepSeek-V4-Flash` `config.json` + `inference/model.py`
  (`Gate`, `hc_pre`/`hc_post`) — the source of the P0 corrections.
- mHC: arXiv:2512.24880. RouteHijack: arXiv:2605.02946. Berthet et al. 2020 (perturbed
  optimizers, still available if a differentiable-selection fallback is needed).
