# RouteAudit runbook

## Quick start (cloud pod)

```bash
cd /workspace
git clone https://github.com/hendrix345621/routeaudit && cd routeaudit

source ./setup_ram.sh        # run everything in RAM (disk too small); see notes below
pip install -e .
hf auth login                 # HF token (gated weights)

make run                      # asks which model first, then runs all 4 phases → verdict
```

`make run` is the one-shot: it prompts for the model (a preset nickname or any HF `user/model`
id), runs `data → harvest → routeaudit → eval`, and **stops at the SAFE/AT-RISK verdict.**
It uploads nothing. For non-interactive use: `make run MODEL=qwen3`.

> `setup_ram.sh` grows `/dev/shm` to 26 GB and points HF cache + `data/ cache/ artifacts/` at
> RAM. It is intended for small checkpoints; size GPU memory and storage explicitly for larger models.

## What the phases produce

- `artifacts/eval_results.json` + `eval_results.md` — **consolidated results**: model, suffix,
  ASR/MMLU/routing-shift, SAFE/AT-RISK verdict, timestamp (the `.md` is the readable report). In
  thinking mode it also carries a **Thinking mode** table (scored/truncated counts, the `[lo, hi]`
  ASR interval, mean think tokens, format-audit pass) and a **Generative reasoning utility** table.
- `artifacts/eval_cells.jsonl` — raw per-cell numbers for programmatic re-grading.
- `artifacts/results/` — **full auditable bundle**: `summary.md` + `per_prompt.md`/`.jsonl`
  (every prompt's clean vs attacked completion + string **and** judge verdict) + `transcripts/`.
  The judge (Llama-Guard-3-1B by default) is language-agnostic, so non-English refusals are
  scored correctly — `make run` runs it by default (`--no-judge` to skip).
- `artifacts/routeaudit_universal.json` — the optimized suffix (the deployable artifact).
- `artifacts/safety_experts.json`, `harmful_experts.json` — localized experts.

## Supported MoE families

| Family | nickname / how to select | attack | notes |
|---|---|---|---|
| Liquid LFM2.5-8B-A1B | `liquid` / `base` | ✓ | default target; 24L · 32 experts · top-4 · ~17 GB BF16 |
| OLMoE | `olmoe` (· `smoke` = tiny sanity run) | ✓ | cheap regression target |
| Mixtral | `mixtral` | ✓ | Mixtral-8x7B; fused experts on newer HF → router capture still fine |
| Qwen2-MoE | `qwen2` | ✓ | Qwen1.5-MoE-A2.7B; shared_expert intentionally not hooked |
| Qwen3-MoE | `qwen3` | ✓ | Qwen3-30B-A3B; no shared expert |
| Qwen3-235B-A22B | `qwen3-235b` | ✓ | 94L · 128 experts · top-8 · no shared expert; ~470 GB → multi-GPU |
| Qwen3.6-35B-A3B | `qwen3.6` | ✓ | hybrid attention (linear+full); 40L · 256 experts · top-8 · shared expert (unhooked); dims verified |
| Qwen3.6-35B-A3B (thinking) | `qwen3.6-think` | ✓ | same model, chain-of-thought ON + A2 attack; see "Running a reasoning model in THINKING mode" below |
| Qwen3.5 MoE | `qwen3.5` | ~ best-effort | hybrid-attention MoE; dims unconfirmed — verify (config header) |
| Phi-3.5-MoE | HF id `microsoft/Phi-3.5-MoE-instruct` | ✓ | clean Linear gate, Mixtral-like |

Passing a raw HF id to any script (or `make run MODEL=<id>`) auto-detects family + dims for
supported `model_type`s, else raises `UnsupportedModelError` with guidance.

DBRX / GPT-OSS / Granite-MoE are **not** wired in: their gates return tuples / sit at non-standard
paths and need an ArchSpec router-path generalization first.

### Non-softmax routers — routing analysis supported, attack not ported

The core supports declarative non-softmax gate semantics for routing analysis. The suffix
optimizer is limited to differentiable softmax routers and fails fast on unsupported gates.

## Running a reasoning model in THINKING mode (A2 attack)

By default reasoning models run with `enable_thinking: false` so the boundary token `t*` is the
answer decision. To attack and evaluate them *with chain-of-thought ON*, use a thinking config —
`configs/qwen3_6_35b_a3b_think.yaml` (nickname `qwen3.6-think`) is the reference:

```bash
make run MODEL=qwen3.6-think        # harvest (answer-span) → A2 attack → eval → verdict
```

What differs from a normal run, and why:

- **Answer-span metrics.** ASR (string + judge) and expert localization read the answer *after*
  `</think>`, not the thinking preamble. Generations that never close their trace have no answer;
  they are **excluded** from ASR and the report shows the `[lo, hi]` interval the exclusion could
  move the rate by. A wide interval means `max_new_tokens` was too small — raise it.
- **`max_new_tokens` must be large.** Traces are long; the 128-token default truncates every one.
  The think config sets `eval.max_new_tokens: 2048`. Calibrate on your model with a pilot batch —
  do not reuse token budgets from another model; calibrate the value with a pilot batch.
- **A judge is mandatory.** With thinking on, the string detector scores the trace, not the answer,
  so `--no-judge` is refused. Llama-Guard-3-1B is gated: accept its license and `hf auth login`, or
  the run fails fast (by design — it no longer silently downgrades to string-only).
- **A2 target mode.** `attacks.routeaudit.target_mode: thought` teacher-forces a compliant
  *reasoning* opener so the suffix steers what the model deliberates. `t*` stays at the boundary;
  only the target string changes. Use `target_len: ~32` (a framing sentence is longer than
  "Sure, here is") and `ascii_only: true` (an English thought target and a multilingual suffix
  fight each other).
- **Harvest span.** `identify.span: answer` counts only answer-span tokens when localizing safety
  experts. Harvest prints a **general-expert overlap** number first: if think/answer safety experts
  overlap the top general-purpose experts by >30%, suppressing them will cost reasoning utility —
  raise the Eq. 5 penalty or expect an MMLU/reasoning drop. **Your harvest corpus responses must
  still contain their `<think>…</think>` markup**, or there is no trace to segment.
- **Utility that sees thinking.** Set `eval.mmlu.generative: true` (the think config does) for a
  generative reasoning score read past `</think>`. The old log-prob MMLU column is kept but applies
  no suffix and cannot observe thinking — the report says so; don't read a retained log-prob MMLU as
  "reasoning intact".

**Phase 0c (10 s, do it once per model):** confirm where the template puts `<think>`. If the
generation prompt already emits it, the A2 target must not add its own — `build_thought_target`
handles this from `prompting.generation_prompt_tail`, but verify the **format audit passes** on a
pilot batch (the eval warns loudly if the requested mode wasn't actually applied). Read the
`## Thinking mode` table in `artifacts/results/summary.md`: `trace_rate` near 1.0 and
`format_audit_passed: true` mean the mode took.

To study the deliberation itself rather than attack it, set `identify.span: think` (or `delimiter`
for the refusal-cliff region) and compare the localized experts against the answer-span set.

## Large models on spot/rented GPUs — cost playbook (Qwen3-235B and up)

For ~235B (~470 GB bf16) the gradient attack is the only phase needing a big white-box node;
**harvest and eval are forward-only**. So the cheapest correct flow never runs the attack on the
big model — it transfers a suffix from the closest sibling and uses the big node only for
forward-only work, loaded once.

```bash
# 1) CHEAP 1-GPU box — optimize the suffix on the closest sibling (shares the routing backbone)
make surrogate MODEL=qwen3            # = run_all --stop-after attack (+ checkpoint/resume)
#   → artifacts/routeaudit_universal.json   (copy this to the big node's persistent volume)

# 2) BIG NODE — ONE model load does forward-only harvest + eval with that suffix
make target MODEL=qwen3-235b          # = target_session --suffix … --judge --resume
```

Full white-box on 235B instead (faithful, expensive): `python scripts/target_session.py
--model qwen3-235b --attack --checkpoint artifacts/attack.ckpt.json --resume` — one load runs
harvest → attack → eval with auto-scaled batches, gradient checkpointing, and prefix cache.

**Why it's cheap & smooth**
- **Surrogate split** keeps the expensive grad attack off the 235B node entirely (per the threat
  model, suffixes transfer across siblings sharing a routing backbone). Transfer is still *measured*
  on the target, so the verdict is honest.
- **Single load** (`target_session.py`) loads the 470 GB once for harvest+eval — no 2-3× reloads.
- **Auto-batch** (`--auto-batch`, on by default in `run_all`) sizes candidate/grad batches +
  n_prompts to the model so the attack doesn't OOM on step 1 (quality-neutral). bf16 only.
- **`model.load:`** in [configs/qwen3_235b_a22b.yaml](configs/qwen3_235b_a22b.yaml) sets
  `attn_implementation`, per-GPU `max_memory`, and an optional `offload_folder` (forward-only).

**Spot-resilience (critical):** everything is resumable — harvest caches each frequency sweep,
the attack checkpoints the best suffix and warm-resumes (`--checkpoint` + `--resume`), and eval
re-runs are cheap. **Put `data/`, `artifacts/`, and `HF_HOME` on a PERSISTENT volume — NOT
`/dev/shm`** ([setup_ram.sh](setup_ram.sh)), or a preemption wipes the checkpoints and the 470 GB
download and resume buys nothing.

**bf16 only:** quantization is deliberately unsupported — it shifts the router logits harvest
localizes and the attack/verdict depend on. Fit via more GPUs / the surrogate / forward-only
offload instead.

## The attack artifact is the suffix (no model export)

RouteAudit is input-only and modifies no weights — there is no checkpoint to "merge" or export.
The deployable result is the **suffix text** in `artifacts/routeaudit_universal.json`. The eval
phase (`scripts/03_eval.py`) prints it and records it into `artifacts/eval_cells.jsonl` next to the
ASR / MMLU / routing-shift numbers, so the verdict and the exact suffix that produced it travel
together.
