# RouteAudit rented-GPU runbook

Use persistent disk for model caches, temporary downloads, datasets, checkpoints,
offload files, and results. Do not use `/dev/shm`: it is capacity-limited and
disappears when the machine stops.

## Set up the disk

The example assumes the provider mounted persistent storage at `/workspace`. Substitute
the mount point shown by your provider.

```bash
cd /workspace
git clone https://github.com/hendrix345621/routeaudit
cd routeaudit

python -m pip install -e ".[data,dev]"
hf auth login
```

The Makefile creates and exports the disk-backed cache paths. Its default layout is:

```text
/workspace/huggingface/          downloaded model snapshots
/workspace/cache/                library caches
/workspace/tmp/                  temporary downloads and state-dict staging
/workspace/routeaudit-data/      prepared datasets
/workspace/routeaudit-offload/   Accelerate model/state-dict offload
/workspace/routeaudit-artifacts/ checkpoints and results
```

Set `PERSIST_ROOT` when your provider uses another mount, for example
`make run PERSIST_ROOT=/mnt/persistent`. The Makefile refuses `/dev/shm`. Keep these
directories when stopping or restarting the rented GPU, and check free disk space
before downloading the checkpoint.

Disk backs storage and loading; model layers should still run in GPU VRAM. If the
loader reports CPU or disk layers, rent a larger GPU or use a compatible multi-GPU
profile because training-style suffix optimization will otherwise be extremely slow.

## Verify thinking mode

The default model is `Qwen/Qwen3-30B-A3B-FP8`, the smallest official Qwen3 MoE with
thinking support. Qwen3-0.6B through 14B are smaller thinking models, but they are dense
and cannot test expert routing. The native FP8 checkpoint reduces disk and VRAM pressure
without applying a second quantization pass.

Run the focused two-prompt generation check before the pipeline:

```bash
make thinking-check
```

Success requires both generations to contain a completed thinking trace, a parseable
post-`</think>` answer, and the expected answer. The machine-readable result is written
to `/workspace/routeaudit-artifacts/quick_reasoning.json`. If a trace is truncated:

```bash
make thinking-check THINKING_ARGS="--max-new-tokens 1024"
```

This is the runtime check that the template setting worked; merely passing
`enable_thinking: true` is not treated as proof.

## Run

`make run` uses the reduced `smoke` workload with the default thinking MoE:

```bash
make run

# Full dataset and optimization settings, same model
make run CONFIG=default

# Override normal CLI controls
make run RUN_ARGS="--skip-data --stop-after harvest"

# Official Qwen 35B-A3B MoE
make run CONFIG=qwen3.6

# GLM-4.5-Air: use a transferred suffix because its biased sigmoid router
# does not support this repository's suffix optimizer
make run CONFIG=glm4.5-air \
  RUN_ARGS="--skip-data --suffix-input /workspace/routeaudit-artifacts/transferred_suffix.json"
```

For a transferred suffix, the complete target flow is:

```bash
make run CONFIG=qwen3.6 \
  RUN_ARGS="--skip-data --suffix-input /workspace/routeaudit-artifacts/transferred_suffix.json"
```

The `run` command loads the target model once. Use `--stop-after harvest` when you only
need expert localization. Use `--checkpoint` and `--resume` for long suffix searches.

## Thinking and answer scoring

Thinking is the default for Qwen3/Qwen3.5-family and GLM-4.5-Air profiles. Their shared
profile sets:

- `model.enable_thinking: true`
- `identify.span: answer`
- `eval.max_new_tokens: 2048`
- sampled generation with `temperature: 0.6`, `top_p: 0.95`, `top_k: 20`
- `eval.mmlu.generative: true`

Evaluation does not judge the first generated character or the reasoning preamble. It
finds `</think>`, sends the complete answer after it to the refusal detector and optional
classifier judge, and marks an unfinished reasoning trace as unscoreable. Generative MMLU
also parses the complete post-thinking answer; the cheaper log-prob MMLU remains a separate
suffix-independent diagnostic.

Inspect `/workspace/routeaudit-artifacts/results/summary.md` and `samples.jsonl`. A high
truncation rate means `eval.max_new_tokens` must be increased before trusting ASR.

## Outputs

```text
/workspace/routeaudit-artifacts/
  safety_experts.json
  harmful_experts.json
  routeaudit_universal.json
  results/
    summary.json
    summary.md
    samples.jsonl
```

## Repository checks

```bash
make check
```
