# Scoping: input-only routing suffix search on DeepSeek-V4-Flash (grouped-biased gate + mHC)

Status: research scoping, not an implementation plan. The honest prior is that the
**most probable publishable outcome is a robustness characterization** (this gate
class resists input-only routing steering), not a working suffix search. The plan below
is designed to find that out cheaply *before* building a full optimizer, and to be a
real result either way. This is red-team / alignment-robustness work on a model you
are authorized to test.

---

## 1. Problem statement & threat model

RouteAudit's premise: optimize a **text suffix** that, appended to a harmful prompt,
shifts the model's routing at the **boundary token t\*** away from the safety experts
`E_safe` and toward harmful/compliant experts `E_harm`, so refusal never engages —
*input-only at deployment, white-box surrogate at attack time.*

We keep that threat model. The question is narrow and concrete:

> Can a gradient-guided discrete suffix search move the **selection and gating
> weights of a DeepSeekMoE gate** (sigmoid affinity + auxiliary-loss-free bias +
> node-limited *grouped* top-k), under an **mHC** residual stream, enough to suppress
> `E_safe` / promote `E_harm` at t\*?

Reference gate (DeepSeek-V3, inherited by V4): `s_e = σ(w_eᵀh)`; select on
`s_e + b_e` (loss-free bias `b_e`, selection-only); experts in `n_group` groups,
group score = sum of top-2 affinities, keep `topk_group` groups, then top-`k`;
gating weight `g_e = routed_scaling · normalize(s over selected)`.
V3 dims: 256 routed experts, k=8, n_group=8, topk_group=4, scaling=2.5,
58 MoE layers. V4-Flash: 256 routed / 6 active / 1 shared (verify the rest).

---

## 2. What is actually hard (and what is NOT)

Rank the obstacles, because the naive "swap softmax→sigmoid and re-run GCG" fails for
specific, fixable reasons — and some "scary" properties are not real blockers.

**Not real blockers (GCG/RouteAudit already handle these):**
- *Non-differentiable top-k.* The existing attack never differentiates the hard
  selection; it differentiates a soft router objective to *propose* token swaps, then
  scores candidates with the hard gate. Discreteness is the search's job, not a wall.
- *The loss-free bias `b_e`.* It is a fixed additive constant at inference. It shifts
  the selection *margin* but is fully differentiable to pass through; it does not
  destroy gradient signal. It does change *what* you must overcome (see O3).
- *mHC.* The gate still receives a C-dim input (via `H^pre`); the n-stream lives only
  between layers. The token→gate-input map is more nonlinear (doubly-stochastic stream
  mixing) but differentiable. mHC changes the *landscape*, not the *reachability of
  gradients*. (It may still help the model resist — that's hypothesis H1, §7.)

**Real obstacles (the research content):**
- **O1 — Objective mismatch.** The original loss is "softmax routing mass on
  `E_safe`." DeepSeek has no softmax over experts and the quantity that actually
  gates an expert's output is `g_e` (zero unless the expert's *group* is selected).
  A flat softmax-mass loss produces gradients that push a harmful expert's score up
  even when its whole group is unselected → wasted/ misleading signal. **The loss must
  mirror the gate's two-level structure.**
- **O2 — Grouped (node-limited) selection bottleneck.** `g_e = 0` unless `e`'s group
  is among the `topk_group`. So promoting a harmful expert is gated by a *group*
  decision (a coarse, combinatorial variable), and suppressing a safety expert may
  require deselecting its group (which also drops benign experts that share it,
  perturbing utility). This two-level coupling is the genuinely new structure vs flat
  top-k and is where a faithful surrogate is essential.
- **O3 — Selection margins vs input leverage (the feasibility crux).** The bias +
  grouping create *margins*: how much must `s_e` (or a group score) move to flip a
  selection at t\*? And how much can an input suffix actually move boundary-token
  affinities, given MLA/CSA compression dilutes a few appended tokens? If
  margins ≫ achievable Δ, input-only is infeasible — independent of optimizer
  cleverness. **This must be measured first.**
- **O4 — Soft↔hard fidelity.** Any differentiable surrogate for grouped top-k is only
  useful if lowering it actually flips the *hard* selection. Two-level relaxations can
  have gradients that point "uphill" in soft space but never cross a hard boundary.
  Fidelity must be measured, not assumed.
- **O5 — Scale & access.** White-box gradients on 284B (MLA, long ctx) are out of
  reach for most. Requires a surrogate (smaller DeepSeek sibling / the mHC paper's
  3B–27B models) + transfer — and transfer across *grouped, learned-bias* gates is
  more fragile than across flat softmax gates (group structure + biases are
  scale/seed-specific).

---

## 3. Reframed objective (addresses O1, O2)

Target the **effective gating weight** `g_e` at t\*, with a loss that matches the
gate hierarchy. For groups `G`, let `S(G)` be the group score and `1[G sel]` the
group-selection indicator:

```
L = λ1 · Σ_{e∈E_safe} g_e            # suppress: drives g→0 (group deselect OR within-group loss)
  + λ2 · max(0, m − Σ_{e∈E_harm} g_e) # promote (bounded), but only credited via real g_e
  + λ_grp · [ group-selection terms ] # push E_harm groups in, E_safe groups out
  + λ_ref · L_refusal                 # unchanged: refusal-token unlikelihood at step 1
```

The `λ_grp` term is new and load-bearing: it operates on group scores `S(G)` so the
optimizer spends gradient budget on the *binding* constraint (getting the right groups
selected) instead of futile within-group nudges on unselected groups.

---

## 4. Candidate methods for the gradient (addresses O2, O4)

Three options, primary first. All reuse the existing GCG outer loop and — crucially —
**score candidates with the faithful hard gate already implemented in
`experiments/mhc/route_mhc.py`** (sigmoid+bias+grouped top-k). So the design is: *soft/
perturbed gradient proposes token swaps → real grouped gate scores them → keep best.*

### 4a. Perturbed-optimizer gradients (PRIMARY — differentiates the *real* selection)
Treat grouped top-k as a linear-objective combinatorial solver `y*(θ)=argmax⟨y,θ⟩`
over the structured (group-constrained) selection polytope. Berthet et al. (2020)
*Differentiable Perturbed Optimizers*: `y_ε(θ)=E_Z[y*(θ+εZ)]` is differentiable, with
a Monte-Carlo Jacobian — **you perturb the scores, run the actual DeepSeek grouped
top-k, and average.** No surrogate drift (O4 largely sidestepped): gradients are of
the true selection rule, bias and grouping included. Cost is benign: the expensive
full-model forward to produce `h` at each gate runs once; only the *gate selection*
(cheap) is resampled `m`≈8–32 times. Chain back `θ = σ(Wh+…) → h → suffix embeddings`.
Why it likely works: it is the only method that gets unbiased-ish gradients through
the genuine two-level combinatorial gate. Failure mode: high-variance gradients if
margins are large (ties O3) — variance ∝ how flat the selection is around t\*.

### 4b. Temperature-annealed two-level relaxation (surrogate baseline)
Relax *both* levels: group-selection weight = differentiable top-k over group scores
(SOFT / entropic-OT / `softmax(S/τ)` masked), expert weight = group_weight ×
within-group soft top-k; anneal τ→0 toward the hard gate. Cheaper than 4a (no
sampling), fully differentiable. Risk = exactly O4: the soft optimum may not cross a
hard boundary; the anneal schedule is delicate and can stall. Good as a fast baseline
and for ablations, not trusted as the primary.

### 4c. Straight-through (cheap control)
Forward = real hard grouped gate; backward = Jacobian of the 4b relaxation. Simplest,
most biased, noisiest for two-level selection. Use only as a lower-bound control to
show whether the fancier gradients buy anything.

---

## 5. Feasibility study FIRST (O3) — the go/no-go gate

Before any optimizer, answer "is there enough input leverage to flip selections at
t\*?" Cheap, uses only forward passes + `experiments/mhc/route_mhc.py`.

- **Margin census.** At t\* over AdvBench prompts, per safety expert / per group,
  compute the Δ in `s_e` / `S(G)` needed to deselect (the gap to the k-th / topk_group-th
  competitor, including bias). Report the margin distribution per layer.
- **Leverage probe.** Append (i) random, (ii) hand-crafted, (iii) short
  *unconstrained soft-embedding* suffixes (optimize continuous embeddings — an upper
  bound on what any discrete suffix can do) and measure achievable Δ in `s_e`/`S(G)` at
  t\*. The soft-embedding upper bound is the key number: **if even unconstrained soft
  suffixes can't cross the margins, no text suffix can** → report robustness, stop.
- **Layer concentration.** Cross the harvest `Score_safe` with the margins: are safety
  experts concentrated in a few layers with *small* margins? If yes, the effective
  problem is low-dimensional (target those layers/groups only — §8).

**Kill criterion:** soft-embedding upper bound < margins across the safety-bearing
layers ⇒ input-only routing steering is infeasible on this gate; write it up as a
positive robustness result (and test whether bias/grouping/mHC are *why*).

---

## 6. Surrogate & transfer (O5)

- Optimize on the smallest faithful DeepSeekMoE sibling (e.g. V2-Lite / a small V3 /
  the mHC paper's 3B–27B models), which share the gate *form* but not the learned
  biases / group assignment.
- **Transfer study:** does a suffix optimized on the sibling move routing on the
  target? Hypothesis: transfer is *weaker* than for flat-softmax families because the
  binding constraint (which group is selected) depends on model-specific learned
  biases. Quantify TESR/THPR transfer vs a Mixtral/Qwen flat-gate control.
- If transfer fails and full-model white-box is out of reach ⇒ the practical attack
  surface is "labs with the weights only" — itself a meaningful exposure statement.

---

## 7. mHC robustness hypothesis (H1)

mHC projects the residual mixing `H^res` onto doubly-stochastic matrices (norm-
preserving, mean-conserving convex combinations of streams). **H1: this incidentally
regularizes the boundary-token hidden state, shrinking the input leverage on gate
scores (O3) relative to a vanilla residual.** Test by running the §5 leverage probe on
matched siblings: vanilla-residual vs HC vs mHC (the paper trains all three). If mHC
demonstrably lowers achievable Δ at t\*, that is a clean, novel alignment finding:
*a stability-motivated architecture choice that also hardens routing.*

---

## 8. Efficiency levers (make a hard attack tractable IF §5 passes)

- **Layer/group targeting.** Don't steer all 58×256; from harvest, attack only the
  few layers where `Score_safe` concentrates, and only the *groups* containing
  `E_safe`/`E_harm`. Collapses dimensionality and focuses the `λ_grp` term.
- **Group-first curriculum.** Optimize group-selection terms to convergence before
  within-group terms (respect the bottleneck order).
- **Reuse experiments/mhc/route_mhc.py** as the candidate scorer and as the eval metric source.

---

## 9. Metrics & validation

- **Soft↔hard fidelity** (gates the whole approach): correlation between Δsoft-loss and
  Δhard-selection (selection-flip rate) per step. Low fidelity ⇒ method 4a/4b is dead.
- **Selection-flip rate** at t\*: fraction of `E_safe` deselected / `E_harm` selected.
- **Routing metrics** = TESR/THPR redefined on `g_e` (not softmax mass), via experiments/mhc/route_mhc.py.
- **The only metric that counts:** end-to-end **ASR** (HarmBench-judged), clean vs
  attacked, + MMLU to confirm the model isn't merely broken. Routing shift without ASR
  is not an attack.

---

## 10. Phased plan with explicit gates

| Phase | Work | Cost | Go/no-go |
|---|---|---|---|
| P0 | Verify gate/module layout + dims vs real `modeling_deepseek*.py` (the §VERIFY checklist) | low | layout matches |
| P1 | **Feasibility (§5):** margin census + soft-embedding leverage upper bound on a sibling | low–med | upper bound ≥ margins in safety layers — else STOP (robustness result) |
| P2 | Reframed objective (§3) + perturbed-optimizer gradient (4a), measure soft↔hard fidelity (§9) | med | flip-rate responds to gradient |
| P3 | Full discrete suffix optimization on sibling; ASR + routing | med–high | ASR lift over clean & over random-suffix control |
| P4 | Transfer to target (§6) + mHC ablation (§7) | high | report transfer + H1 either way |

Kill criteria are first-class: P1 failing is a *result*, not a failure.

---

## 11. Honest expected outcomes

1. *Most likely:* feasibility (P1) shows grouped-biased selection margins exceed input
   leverage in the safety-bearing layers ⇒ **DeepSeek-V4's gate is robust to input-only
   routing steering**; quantify how much bias / grouping / mHC each contribute. This is
   the valuable alignment result and directly serves "better-aligned models."
2. *Possible:* P2–P3 get routing flips on a sibling but ASR barely moves (boundary-token
   steering doesn't derail autoregressive refusal under MLA) ⇒ a negative result about
   the *attack premise* on this architecture.
3. *Less likely:* a working white-box attack on a sibling that does NOT transfer ⇒
   exposure limited to weight-holders; document and defend.

In all three, the deliverable is a characterization that helps harden routing — not a
distributable suffix. No working suffix search on DeepSeek is built until P1 says it's
even reachable, and any artifact stays within the existing local/audit guardrails.

---

### References (mechanism)
- DeepSeek-V3 Technical Report, arXiv:2412.19437 (MoEGate: sigmoid, grouped top-k, scaling 2.5, 58 MoE layers).
- Auxiliary-loss-free load balancing, arXiv:2408.15664 (the `e_score_correction_bias`).
- mHC: Manifold-Constrained Hyper-Connections, arXiv:2512.24880 (doubly-stochastic residual mixing).
- Berthet et al., *Learning with Differentiable Perturbed Optimizers*, NeurIPS 2020 (4a).
- DeepSeek-V2 (MLA), arXiv:2405.04434; RouteHijack, arXiv:2605.02946 (boundary-token premise).
