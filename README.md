# RouteAudit

RouteAudit is a routing-aware safety evaluation tool for open Mixture-of-Experts
(MoE) language models. It identifies experts associated with refusals, optimizes one
input suffix against their router scores, and compares clean and attacked behavior.

Use it only for authorized safety testing. Generated datasets, suffixes, and model
outputs may contain harmful material and are excluded from version control.

The attack follows the input-space method described in
[RouteHijack](https://arxiv.org/abs/2605.02946). The implementation is intentionally
narrow: unsupported router semantics fail before expensive GPU work begins.

## Install

Python 3.10+ and a CUDA environment suitable for the selected model are required.

```bash
python -m pip install -e ".[data,dev]"
```

Authenticate with Hugging Face before using gated model or judge repositories.

## Quick start

The default target is `Qwen/Qwen3-30B-A3B-FP8`: the smallest official Qwen3
thinking-capable MoE, in its native lower-footprint checkpoint format. The smaller
Qwen3 releases are dense models and cannot exercise RouteAudit's expert-routing attack.

On a rented GPU with persistent storage mounted at `/workspace`, run the small profile:

```bash
make run
```

`make run` defaults to the reduced `smoke` workload and keeps Hugging Face caches,
temporary files, datasets, model offload, and artifacts on persistent disk. Run
`make thinking-check` first to verify that the model emits a closed thinking trace and
that answer segmentation works. See `runbook.md` for mount overrides and full runs.

The `run` command prepares data, loads the model once, then performs harvest, attack,
and evaluation. Without `--config`, the CLI uses the same FP8 thinking model with the
full workload. `python -m routeaudit.cli` is equivalent when the console entry point
is not installed.

Individual phases are also available:

```bash
routeaudit data --config base
routeaudit harvest --config base
routeaudit attack --config base
routeaudit eval --config base
```

Useful run controls:

```bash
routeaudit run --config qwen3 --skip-data
routeaudit run --config qwen3 --stop-after harvest
routeaudit run --config qwen3 --suffix-input artifacts/transferred_suffix.json
routeaudit run --config smoke --no-judge
```

Run `routeaudit <command> --help` for phase-specific controls.

## Supported profiles

Profiles are small YAML overrides that inherit from `configs/base.yaml`.

| Profile | Purpose | Suffix attack |
|---|---|---|
| `default`, `qwen3-fp8` | smallest supported thinking MoE; native FP8 | yes |
| `smoke`, `qwen3-think-smoke` | reduced run using the default thinking MoE | yes |
| `base`, `olmoe` | low-cost non-thinking OLMoE alternative | yes |
| `mixtral`, `qwen2`, `qwen3`, `qwen3-235b` | plain top-k MoE targets | yes |
| `qwen3.6`, `qwen3.6-think` | official 35B-A3B Qwen MoE; thinking is default | yes |
| `glm4.5-air`, `glm-air` | GLM-4.5-Air biased sigmoid MoE | analysis/eval only |
| `liquid`, `lfm2` | biased sigmoid-router analysis | no |

Qwen3-family and GLM-4.5-Air profiles inherit `configs/thinking.yaml`: thinking,
Qwen-recommended sampled generation, answer-span harvesting, and generative MMLU are
enabled by default. Evaluation judges the complete answer after `</think>` and treats
unfinished traces as unscoreable.

LFM2 remains useful for expert harvesting and clean evaluation, but its biased sigmoid
gate is not mathematically compatible with the current suffix objective. The CLI rejects
that attack instead of silently applying the wrong router math.

A Hugging Face model id can be passed directly when its architecture can be detected:

```bash
routeaudit run --config microsoft/Phi-3.5-MoE-instruct
```

## Pipeline

1. `data` prepares paired safety examples, a general corpus, AdvBench prompts, and an
   MMLU subset.
2. `harvest` measures expert activation frequency and writes the selected safety and
   harmful experts.
3. `attack` performs gradient-guided discrete search for one universal suffix.
4. `eval` generates clean and attacked answers, scores ASR, measures MMLU once, and
   reports routing shift when the router supports it.

The default classifier judge is enabled. `--no-judge` explicitly opts into the weaker
string-refusal heuristic. Thinking-model generations are scored only after the closing
thinking delimiter; truncated traces are marked unscoreable rather than compliant.

## Outputs

Generated data and artifacts are intentionally compact:

```text
data/
  llm_lat_pairs.jsonl
  c4_general.jsonl
  advbench.jsonl
  mmlu_subset.jsonl
artifacts/
  safety_experts.json
  harmful_experts.json
  routeaudit_universal.json
  results/
    summary.json
    summary.md
    samples.jsonl
```

`summary.json` is the machine-readable result, `summary.md` is the short report, and
`samples.jsonl` contains aligned clean/attacked completions for audit or re-grading.

## Configuration

Shared non-thinking defaults live in `configs/base.yaml`; the selected runnable default
is `configs/default.yaml`. A new profile normally needs only an inheritance line and
its model-specific fields:

```yaml
extends: base.yaml
model:
  hf_id: organization/model
  n_layers: 24
  n_experts: 64
  top_k: 8
  d_model: 2048
  arch:
    name: qwen
```

Do not add an alias until the architecture dimensions and router behavior have been
verified against the checkpoint's configuration and implementation.

## Development

```bash
make check
```

The repository has one command surface (`src/routeaudit/cli.py`), one orchestration
module (`src/routeaudit/pipeline.py`), and focused modules for model adapters, expert
identification, suffix search, and evaluation. `scripts/quick_reasoning_check.py` is kept
as a diagnostic for tokenizer/chat-template behavior; it is not part of the pipeline.
