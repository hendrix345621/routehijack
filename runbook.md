# RouteAudit rented-GPU runbook

Use persistent disk for model caches, datasets, checkpoints, and results. Do not use
`/dev/shm`: it is capacity-limited and disappears when the machine stops.

## Set up the disk

The example assumes the provider mounted persistent storage at `/workspace`. Substitute
the mount point shown by your provider.

```bash
cd /workspace
git clone https://github.com/hendrix345621/routeaudit
cd routeaudit

export HF_HOME=/workspace/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HOME" data artifacts cache
python -m pip install -e ".[data,dev]"
hf auth login
```

Keep `/workspace/huggingface`, `data`, `cache`, and `artifacts` when stopping or
restarting the rented GPU. Check free disk space before downloading a checkpoint.

## Run

The CLI owns execution; the Makefile only runs repository checks.

```bash
# Small pipeline check
routeaudit run --config smoke

# Official Qwen 35B-A3B MoE, thinking enabled by default
routeaudit run --config qwen3.6

# GLM-4.5-Air: use a transferred suffix because its biased sigmoid router
# does not support this repository's suffix optimizer
routeaudit run --config glm4.5-air \
  --skip-data \
  --suffix-input artifacts/transferred_suffix.json
```

For a transferred suffix, the complete target flow is:

```bash
routeaudit run --config qwen3.6 \
  --skip-data \
  --suffix-input artifacts/transferred_suffix.json
```

The `run` command loads the target model once. Use `--stop-after harvest` when you only
need expert localization. Use `--checkpoint` and `--resume` for long suffix searches.

## Thinking and answer scoring

Thinking is the default for Qwen3/Qwen3.5-family and GLM-4.5-Air profiles. Their shared
profile sets:

- `model.enable_thinking: true`
- `identify.span: answer`
- `eval.max_new_tokens: 2048`
- `eval.mmlu.generative: true`

Evaluation does not judge the first generated character or the reasoning preamble. It
finds `</think>`, sends the complete answer after it to the refusal detector and optional
classifier judge, and marks an unfinished reasoning trace as unscoreable. Generative MMLU
also parses the complete post-thinking answer; the cheaper log-prob MMLU remains a separate
suffix-independent diagnostic.

Inspect `artifacts/results/summary.md` and `samples.jsonl`. A high truncation rate means
`eval.max_new_tokens` must be increased before trusting ASR.

## Outputs

```text
artifacts/
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
