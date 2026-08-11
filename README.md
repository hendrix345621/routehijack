# RouteAudit

**A routing-aware safety-evaluation tool for Mixture-of-Experts (MoE) LLMs.**

RouteAudit optimizes a single adversarial **suffix** that, appended to a harmful prompt,
steers the model's internal **routing** away from the experts responsible for refusal and
toward experts associated with compliance — measuring whether safety alignment holds up
under **text-only input**, with no access to weights or inference code at deployment time.
This is a research tool for **authorized** red-teaming and safety evaluation of models you
have permission to test — see [Note on responsible use](#note-on-responsible-use) below.

> Reproduces the method from *RouteHijack: Routing-Aware Attack on Mixture-of-Experts LLMs*
> ([arXiv:2605.02946](https://arxiv.org/abs/2605.02946)) — that's the name of the paper this
> tool implements; RouteAudit is our name for the implementation and evaluation harness.

See [the archived timeline](docs/archive/TIMELINE.md) for the project's history, how each feature was added, and
what's on the roadmap — this README stays focused on how to use it today.

---

## The idea in one minute

In an MoE layer, a small **router** sends each token to a few of many expert FFNs:

```
MoE(x) = Σ_{e ∈ TopK}  p_e(x) · E_e(x)        p = softmax(router · x)
```

Safety behaviour isn't spread evenly across the model — it **concentrates in a small set
of "safety experts"** that fire preferentially when the model refuses. If you can nudge the
router away from those experts (and toward harmful-leaning ones) at the moment generation
begins, the model proceeds as if its refusal machinery were never consulted.

Because routing is driven by **continuous** router scores, you can optimize an input suffix
to shift them — even though the Top-K selection itself is discrete. That is what RouteAudit
does, and it's why the attack remains **input-only** at deployment.

---

## How it works

**1. Response-driven expert localization.** For every `(layer, expert)`, measure how often
it fires on **safe refusals** vs **harmful completions** (counting *response* tokens, not the
prompt — response-driven profiling is far more discriminative). Define a safety differential
and rank experts:

```
F_l(e | a)        = activation frequency of expert e (layer l) over response a
Δ_S(l,e)          = F(e | safe) − F(e | harmful)
Score_safe(l,e)   = Δ_S − F(e | general)²       # utility penalty: drop general-purpose experts
Score_harm(l,e)   = −Δ_S
```

The top-20% by `Score_safe` are the **safety experts** `E_safe`; the top-20% by `Score_harm`
are the **harmful experts** `E_harm`.

**2. Ternary-loss suffix optimization.** Optimize a `T`-token suffix (GCG-style discrete
search over the input) against three terms evaluated at the **boundary token** `t*` (the last
input position before decoding, rendered through the model's chat template):

```
L = λ₁·L_suppress  +  λ₂·L_promote  +  λ₃·L_refusal      (λ = 3 : 1 : 1)

L_suppress = routing mass on safety experts at t*                       (push down)
L_promote  = max(0, m_harm − routing mass on harmful experts at t*)     (push up, bounded)
L_refusal  = unlikelihood of refusal-opener tokens at the first step    (block "I'm sorry…")
```

Optimization is gradient-guided discrete search: gradients of `L` w.r.t. the suffix one-hots
(through the **soft** router probabilities) propose top-k token swaps per position; candidates
are scored in a batched forward and the best improvement is kept. A decode-then-re-encode
length filter keeps the suffix's tokenization stable so what you optimize is what deploys.

**3. Deploy.** Append the optimized suffix to any harmful prompt as **plain text**.

---

## Pipeline (4 phases)

```bash
make data         # 1. corpora: LLM-LAT contrast pairs, C4, AdvBench, MMLU
make harvest      # 2. localize safety + harmful experts        → artifacts/*_experts.json
make routeaudit  # 3. optimize the universal suffix            → artifacts/routeaudit_universal.json
make eval         # 4. ASR + MMLU utility + routing shift + SAFE/AT-RISK verdict
# or:
make all
```

Each phase reuses the prior phase's artifacts; the model is loaded once per phase.

**One-shot run.** To pick a model and run all four phases end to end in a single
command — the first thing it asks is *which model* (a preset nickname or any HF
`user/model` id):

```bash
make run                 # interactive: choose the model, confirm, run
make run MODEL=qwen3      # non-interactive (automation)
python scripts/run_all.py --model microsoft/Phi-3.5-MoE-instruct --judge
```

`run_all.py` **ends at the SAFE/AT-RISK verdict and uploads nothing.** The deployable
artifact of an attack is the **suffix text** itself (`artifacts/routeaudit_universal.json`,
also echoed by the eval) — RouteAudit is input-only and never produces or ships a model.

**Large models (Qwen3-235B and up), cost-effectively.** The gradient attack is the only phase
that needs a big white-box node; harvest + eval are forward-only. So optimize the suffix on a
cheap sibling, then measure on the big model in a **single load**:

```bash
make surrogate MODEL=qwen3          # cheap 1-GPU box → a transferable suffix
make target MODEL=qwen3-235b        # big node: ONE load, forward-only harvest + eval + verdict
```

Everything is **spot-resumable** (`--resume` + checkpoints) and **bf16-only** (no quantization —
it would corrupt the routing signal). Full white-box on 235B is available via
`target_session.py --attack` (auto-scaled batches + grad checkpointing). See the **cost playbook**
in [the runbook](docs/RUNBOOK.md) and [configs/qwen3_235b_a22b.yaml](configs/qwen3_235b_a22b.yaml).

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
hf auth login                                         # HF token, for gated model weights
```

Requires Python ≥ 3.10 and a CUDA GPU for real runs. The default target,
`LiquidAI/LFM2.5-8B-A1B`, is an 8B-total / 1B-active MoE with a ~17 GB BF16 checkpoint; a
24 GB+ GPU is a practical starting point. Use `make run MODEL=smoke` for a tiny pipeline check.

---

## Threat model

Two stages, matching how open-weight backbones get repackaged into deployed products:

- **Offline (white-box surrogate):** full access to a related open-weight MoE — weights,
  activations, router logits — used to localize experts and optimize the suffix.
- **Deployment (input-only):** the suffix is appended to prompts as text. No weight edits,
  no expert pruning, no inference-code changes. The optimized suffix also transfers zero-shot
  across sibling models that share a routing backbone.

---

## Configuration

Everything is driven by [configs/base.yaml](configs/base.yaml):

- **Swap the target model** under `model:` — any MoE whose router/experts the `ArchSpec` can
  locate. Presets ship for **Liquid LFM2.5**, **OLMoE**, **Mixtral**, **Qwen** MoE, and **Phi-MoE**; add a family by
  adding a preset in [model/archspec.py](src/routeaudit/model/archspec.py) and the dims in the
  config. Passing a HuggingFace id straight to any script auto-detects the family and dims for
  supported `model_type`s, or raises `UnsupportedModelError` with guidance.
  - Ready-made config nicknames (`--config <name>` or `make run MODEL=<name>`): **`liquid`**/`base` (the default), `olmoe`,
    `mixtral`, `qwen2`, `qwen3` (Qwen3-30B-A3B), **`qwen3-235b`** (Qwen3-235B-A22B),
    **`qwen3.6`** (Qwen3.6-35B-A3B — hybrid-attention MoE, dims verified from its config.json;
    every layer still has a standard MoE gate so the attack applies), best-effort **`qwen3.5`**
    ([config header](configs/qwen3_5_moe.yaml) checklist), and `smoke` (tiny end-to-end sanity run).
    Passing a raw HF id auto-detects supported `model_type`s instead.
  - **Qwen3-Next** (`model_type: qwen3_next`, hybrid-attention MoE) auto-detects via a raw HF id
    the same way — no ready-made config nickname yet, but the `model_type` is already mapped to
    the `qwen` ArchSpec preset (standard MoE gate per layer, so the attack applies as-is).
- **`use_chat_template`** renders prompts through the instruct template so the boundary token
  `t*` is the real decision point. Auto-falls back to raw text if the tokenizer has none.
- **`enable_thinking`** (reasoning models — see the caveat below). Set `false` on Qwen3-family
  configs; forwarded to the chat template so the model answers directly.
- **`chat_template_kwargs:`** — a generic passthrough block for any other kwarg a model's chat
  template accepts, forwarded to `apply_chat_template` alongside `enable_thinking` (e.g. a custom
  `trust_remote_code` template that wants something beyond the two built-in switches).
- **Attack budget** (`attacks.routeaudit`): suffix length `T`, steps, the `λ` weights, the
  harmful-promotion margin. CLI flags on `scripts/02_suffix_search.py` tune the search
  (candidates per step, prompt subsample, grad/candidate batch sizes, early-stop patience).

### Attack signal & judge

The pure routing loss only *suppresses refusal*, so the search could "win" by derailing the
model onto **off-topic content** (a non-refusal that isn't the harmful answer — e.g. a suffix
that injects a foreign-language "write a poem" instruction). Two changes fix that:

- **`lambda_target` (λ4) — affirmative-target term.** Teacher-forces the AdvBench harmful-answer
  opener (`target` field) so the attack is anchored **on-topic** — it must start the *actual*
  harmful answer, which poetry can't satisfy. Combined with `lambda_refusal` it forms a
  contrastive margin (push the answer up, refusal down). On by default (`λ4=1.0`); `0.0` is the
  paper-faithful routing-only loss. One forward, differentiable, ~1.05× runtime, and it
  **transfers** (targets aren't model-specific).
- **`ascii_only: false`** on the Qwen configs — because the on-topic anchor now prevents the
  redirect, the ASCII constraint is no longer needed and **cross-lingual features are kept**.
  Same knob, two entry points: the YAML `attacks.routeaudit.ascii_only` field, or the
  `--ascii-suffix` flag on `scripts/02_suffix_search.py` (either one turns it on).
- **Judge.** ASR is screened by the string detector but the trustworthy number comes from a judge
  (`--judge`): `eval.asr.judge_kind` ∈ {`harmbench` (behaviour-conditioned, the paper standard),
  `llamaguard` (fast taxonomy judge — `Llama-Guard-3-1B`, default on the Qwen configs; **gated**,
  accept the license + `hf auth login`)}. The judge loads once and is reused across cells. Eval
  **warns loudly** if run without a judge. The cheap string detector ([eval/asr.py](src/routeaudit/eval/asr.py))
  also carries a hand-maintained list of Chinese/Japanese/Korean refusal phrasings, so a
  multilingual model refusing in the language a suffix nudged it into doesn't silently inflate
  ASR the way an English-only phrase list would — still a band-aid; the judge is the real fix.

> **Experimental — the harm-probe distillation.** [attacks/harm_probe.py](src/routeaudit/attacks/harm_probe.py)
> + [scripts/distill_harm_probe.py](scripts/distill_harm_probe.py) distill the judge into a tiny
> probe over **router features** for *judge-aware gradients* at probe speed. Run it on its own:
> ```bash
> python scripts/distill_harm_probe.py --config qwen3.6 --judge-kind llamaguard \
>     --judge-id meta-llama/Llama-Guard-3-1B --n-prompts 200 --out artifacts/harm_probe.pt
> ```
> Needs both harmful and safe examples to train on; clean AdvBench generations are mostly
> refusals, so add `--n-samples`/`--temperature` for variety (see the script's docstring).
> Historical status notes are archived in [TIMELINE.md](docs/archive/TIMELINE.md#roadmap--open-threads).

### ⚠ Caveat — reasoning ("thinking") models

RouteAudit assumes the **first generated token (`t*`) is the safety decision point** — that's
where it localizes safety experts, applies `L_refusal`, and measures routing shift. **Reasoning
models break that assumption:** with chain-of-thought on, `t*` is the start of the *thinking*
("Here's a thinking process: …"), and the model only decides to refuse/comply *much later*, after
the `</think>`. Left on, this silently corrupts the whole pipeline — expert localization counts
thinking tokens, the attack aims at the wrong token, and the refusal detector scores the thinking
preamble (a truncated CoT looks like a "compliance"), inflating ASR.

So the shipped Qwen3-family configs set **`enable_thinking: false`**, which makes `t*` the real
answer decision and the metrics trustworthy. **This means RouteAudit evaluates a reasoning model's
*non-thinking* mode** — a deliberate scoping choice (the only setting consistent with the boundary-
token threat model), **not** a measurement of its full reasoning-mode safety. After a run on a new
reasoning model, sanity-check that completions no longer open with a thinking preamble (some custom
`trust_remote_code` templates ignore the kwarg and need the `/no_think` switch instead).

Independent of `enable_thinking`, the ASR scorers already defend themselves against a stray or
truncated `<think>…</think>` block: both the string `RefusalDetector` and the judge path strip a
*completed* thinking block before matching, so a model that decides to refuse inside its own
chain-of-thought is scored on the answer, not misread off the thinking preamble. An *unclosed*
(truncated) thinking block is left in place and caught by dedicated reasoning-phrase patterns
(`"must refuse"`, `"cannot provide"`, etc.) in the refusal-phrase list. This narrows the failure
mode but doesn't replace re-anchoring `t*` — see the adaptation sketch below.

**Adapting to thinking mode** is a real research extension, not a config flag — re-anchoring `t*`
to the post-`</think>` answer start, localizing on answer tokens only, and making the rollout
tractable. Sketched in the archived [timeline](docs/archive/TIMELINE.md#roadmap--open-threads).

---

## Outputs

**Live transparency.** Every phase runs through a shared terminal UI ([ui.py](src/routeaudit/ui.py)):
step headers, progress bars, and — the important part — a colored REFUSED/COMPLIED panel for a
sample of completions as they're generated, so you're reading actual model output instead of
trusting a single ASR number. A "refused" boolean from a string detector can lie in both
directions (a safety-flavored compliance counted as refusal, or a vague-but-harmful response
counted as safe); the live samples are how you catch that. The same transcripts are mirrored to
disk as they're produced — one markdown + one JSONL file per cell under `artifacts/transcripts/`
— so a reviewer can scroll through every sample later without re-running anything.

| Artifact | Phase | Contents |
|---|---|---|
| `artifacts/safety_experts.json`, `harmful_experts.json` | harvest | flagged `(layer, expert)` sets + scores |
| `artifacts/identify_diagnostics.pt` | harvest | per-expert score / frequency tensors |
| `artifacts/routeaudit_universal.json` | routeaudit | the optimized suffix |
| `artifacts/routeaudit_attacks.jsonl` | routeaudit | per-prompt attacked completions |
| `artifacts/routeaudit_routing_shift.json` | routeaudit | TESR / THPR routing-shift metrics |
| `artifacts/eval_cells.jsonl` | eval | raw per-cell ASR / MMLU / routing-shift (for re-grading) |
| `artifacts/eval_results.json` | eval | **consolidated results**: model, suffix, metrics, routing shift, verdict, timestamp |
| `artifacts/eval_results.md` | eval | the same as a readable report (verdict, metrics table, the suffix) |
| **`artifacts/results/`** | eval | **full auditable bundle**: `summary.md` + `per_prompt.md` / `.jsonl` — every prompt's clean vs attacked completion with the string **and** judge verdict (clear proof each "success" is real harm, not a non-English refusal) + `transcripts/` |

**Metrics.** *ASR* — fraction of harmful prompts that get a non-refusal (string detector,
optionally re-graded by the HarmBench classifier with `--judge`). *MMLU* — multiple-choice
accuracy with the attack active, to confirm the model isn't merely broken. *TESR / THPR* —
how much routing mass shifts off safety experts / onto harmful experts at the boundary token.

---

## Performance & optimizations

The pipeline is tuned to keep the GPU busy and avoid recomputation. Everything below is
**quality-neutral** (identical results) unless flagged otherwise.

**Harvest (expert localization)**
- The activation-frequency sweeps **pre-tokenize** the whole corpus once, then run **batched**
  forwards (length-sorted to minimize padding) with on-GPU count accumulation, and call the
  base transformer (skipping the `lm_head`) since only router logits are needed.
- A **model-placement check** warns loudly when `device_map: auto` has offloaded layers to
  CPU/disk — the usual cause of 10–100× slow runs.

**Suffix optimization (the attack)**
- **One persistent hook manager** for the whole run (no per-forward hook install/remove).
- **Prefix-embedding cache**: each prompt's `[before]` (template + query) embeddings are
  computed once and reused across all optimization steps.
- **Batched candidate evaluation** (`--candidate-batch-size`): all candidates for a prompt are
  scored in a *single* batched forward instead of one-at-a-time — the dominant per-step cost.
- **Batched grad pass** (`--grad-batch-size`): prompts are processed in right-padded chunks,
  one forward+backward each; mathematically identical to per-prompt accumulation
  (∇ of a sum = sum of ∇s), just far better GPU utilization.
- **Decode-then-re-encode length filter** (Algorithm 1): candidates whose suffix re-tokenizes
  to a different length are rejected, so the optimized tokens survive deployment as text
  (this also prevents the failure mode where the deployed suffix differs from the optimized one).
- **Early stop** (`--early-stop-patience`): halts once the best loss plateaus.
- **Prefix KV-cache** (experimental, `--prefix-kv-cache`): the `[before]` prefix is fixed and
  shared across all candidates and all 300 steps. With this flag its KV cache is computed
  **once per prompt**, and candidate forwards process only `[suffix][after]` (~25 tokens),
  attending to the cached prefix instead of recomputing the full `[before][suffix][after]`.
  Quality-neutral; **self-checked** against the full path on first use and **auto-disabled** on
  any numeric mismatch or HF-version incompatibility, so it can never silently corrupt the attack.

**Evaluation**
- **ASR completions** are generated in **left-padded batches** (`--gen-batch-size`) via the model's
  own `generate`, rather than decoding one prompt at a time. The per-prompt step-by-step path is
  kept only for cells that install router/expert *mutators* (RouteAudit's input-only cells have
  none, so they batch); switching is automatic.
- **MMLU** and the **routing-shift (TESR/THPR)** measurement run in **right-padded batches**
  (`--gen-batch-size`, `--mmlu-batch-size`), reading each row's last real token / boundary token.
  Right padding keeps real tokens at positions 0…L-1, so the batched results are numerically
  identical to scoring one item at a time — just far fewer forward launches.
- The **HarmBench judge** (`--judge`) already batches its classifier forwards.

> The boundary token `t*` is placed correctly by rendering prompts through the chat template
> (see *How it works*); this is a correctness requirement the optimizations are built on, not a
> speed tweak.

---

## Layout

```
src/routeaudit/
  model/      loader.py · archspec.py · hooks.py (router/expert capture + mutate) · prompting.py
  identify/   activation_freq.py (Eq. 3) · delta_s.py (Eq. 4–5) · select.py (top-pct)
  attacks/    suffix_search.py (ternary loss + GCG search) · compose.py
  eval/       asr.py (RefusalDetector + HarmBench) · mmlu.py · generate.py · harness.py
  config.py · data.py · ui.py
scripts/      00_data.py · 01_harvest.py · 02_suffix_search.py · 03_eval.py
configs/      base.yaml
```

---

## The attack artifact is the suffix

RouteAudit is **input-only** — the pipeline produces a **text suffix** and a verdict, and never
modifies weights. There is no model to "merge" or export: the deployable result is the suffix in
`artifacts/routeaudit_universal.json`, which the eval phase also prints and records into
`artifacts/eval_cells.jsonl` alongside the ASR/MMLU/routing-shift numbers.

## Note on responsible use

This is a red-teaming / safety-evaluation tool for measuring how susceptible open-weight MoE
models are to routing-level manipulation. Use it to audit models you are **authorized to test**.
