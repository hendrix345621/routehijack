# Fast, low-cost GPU runbook

This runbook answers two narrow questions:

1. Does RouteAudit correctly request, detect, and segment reasoning output?
2. Does the real DeepSeek-V4-Flash checkpoint agree with the project's mHC/gate assumptions?

It is a smoke-test ladder, not the full safety experiment. Stop at the first failure so
you do not pay for a larger GPU while debugging ordinary code or environment problems.

## Recommended order

| Order | Check | Hardware | What a pass means | Stop condition |
|---|---|---|---|---|
| 0 | Unit tests + synthetic mHC | Local CPU, no rental | Project mechanisms and device paths work | Any test fails |
| 1 | Qwen3-4B reasoning smoke | 1 x 16–24 GB GPU | Thinking is emitted, closes, segments, and yields answers | Format, truncation, or answer check fails |
| 2 | Real DeepSeek-V4 fixture | Blackwell SM100+, checkpoint + 20 GiB free VRAM | Shipped gate, hash route, and four-stream residual are compatible | Any preflight or fixture mismatch |

Do reasoning before real mHC. It uses a much smaller checkpoint and catches template,
generation, and answer-span bugs before the expensive rental. The synthetic mHC check is
real code coverage but **not** evidence that DeepSeek's released weights are compatible.

## 0. Free preflight (run before renting)

From the repository root:

```bash
python -m pip install -e '.[dev]'
pytest tests experiments/mhc/tests -q
python experiments/mhc/tests/run_synthetic.py
```

Pass gate: all tests pass and the synthetic script reports all Level 0 checks as passed.
If not, fix locally and do not rent a GPU yet.

## Vast.ai selection

Use an Ubuntu/PyTorch image with direct SSH. Select a **verified** host, reliability of at
least 0.99, a recent CUDA stack, fast download bandwidth, and adequate fixed disk space.
Sort by total dollars/hour, then compare download speed: model download and load time can
cost more than the tiny forwards in this runbook.

For the cheapest reasoning-only rental:

- 1 x 16–24 GB GPU; an RTX 3090/4090-class offer is plenty;
- 25 GB disk minimum; 40 GB is comfortable;
- use `Qwen/Qwen3-4B` in its normal BF16/automatic checkpoint dtype.

For the genuine mHC rental:

- Blackwell only (compute capability >=10.0): one B200/B300 or two 96 GB Blackwell
  cards on the same host are the practical shapes;
- the strict preflight requires the 159.63 GB checkpoint plus 20 GiB of free aggregate
  VRAM, 64 GiB CPU RAM, and the checkpoint plus 40 GiB of free disk;
- allocate 250 GB disk and prefer at least 128 GB CPU RAM;
- use CUDA toolkit 12.9+ with `nvcc`, PyTorch 2.9+, Transformers 5.14+, and `kernels`;
- do not add NF4/int8 quantization: DeepSeek-V4-Flash already ships as mixed FP4/FP8.

Do not rent H100/H200/A100 for this as-shipped fixture. Current Transformers requires
Blackwell SM100+ for DeepSeek-V4-style FP4-packed expert weights. Dequantizing to BF16
would require far more VRAM and would no longer validate the deployed checkpoint.

## Common setup on a rented instance

Connect using the SSH command shown by Vast, then copy or clone the project. In the
project root:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
nvidia-smi
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Keep the shell alive with `tmux` if desired. Authenticate to Hugging Face directly on the
instance; do not send the token through chat or put it in a committed file.

## 1. Cheap reasoning smoke

Run:

```bash
python scripts/quick_reasoning_check.py \
  --max-new-tokens 512 \
  --batch-size 1 \
  --out artifacts/quick_reasoning.json
```

The script uses Qwen's recommended thinking sampling (`temperature=0.6`, `top_p=0.95`,
`top_k=20`) rather than greedy decoding. It passes only when both outputs contain a
completed reasoning trace, the post-`</think>` answer is scoreable, and the two trivial
answers are correct.

Interpret the JSON outcome as follows:

- `passed`: both samples completed; the strict zero-truncation gate passed.
- `partial`: thinking mode and post-trace answer extraction worked on at least one sample,
  but another trace was truncated or wrong. This verifies the feature path, but the token
  budget is not ready for real evaluation.
- `failed`: no correctly segmented thinking answer was observed; diagnose before continuing.

If it fails only because a trace was truncated, rerun once:

```bash
python scripts/quick_reasoning_check.py --max-new-tokens 1024
```

Do not keep raising the budget repeatedly. A missing trace or malformed output after that
is a feature/configuration failure to diagnose, not a reason to start the mHC rental.

For a real reasoning safety evaluation, do not reuse the smoke-test budget blindly. Start
at 2,048 generated tokens, retain the reported truncation rate, and raise the budget if it
is above 5%. A strict score must never count an unfinished reasoning trace as an answer.

This cheapest run proves the **model-independent thinking path**: chat-template switching,
token-level delimiter handling, truncation accounting, and answer extraction. It does not
exercise a MoE router. If it passes and you specifically want a Qwen MoE confirmation, run:

```bash
python scripts/quick_reasoning_check.py --config qwen3-think-smoke
```

That optional target is the official ~32.5 GB Qwen3-30B-A3B FP8 checkpoint and requires a
compute-capability 9+ GPU in the Transformers fine-grained FP8 path (normally an H100).
It is still a structural smoke result, not an exact router-score claim. Use `--config qwen3`
on an 80 GB GPU when you later need the 61.1 GB BF16 checkpoint for quantitative routing.

Download `artifacts/quick_reasoning.json`, then destroy the reasoning instance if using
the lowest-cost two-rental plan.

## 2. Genuine mHC compatibility check

Run the strict hardware/software preflight before downloading weights:

```bash
python -m pip install -e '.[mhc]'
python scripts/run_mhc_smoke.py --preflight-only
```

If and only if that passes, run the complete job:

```bash
python scripts/run_mhc_smoke.py
```

The wrapper runs the synthetic checks, one real forward, exact CPU validation, strict
capture checks, and records the environment and logs under `artifacts/mhc_real/`.

Pass gate:

- fixture extraction captures `gate` and `residual` entries;
- validation reports the same selected expert set and matching gate weights;
- residual validation records four streams and a valid stream-mean reduction;
- hash routing is captured and reproduced exactly;
- `manifest.json` reports `"status": "passed"`;
- the command exits zero.

A missing hash fixture is now a failure, not a skipped pass. Likewise, this fixture check
validates compatibility, not the full semantic safety claim.

Copy the entire `artifacts/mhc_real/` directory out immediately. The fixture is small
compared with the model cache and supports repeat CPU-side validation.

## Cheapest versus fastest rental strategy

**Lowest expected spend:** use two rentals. Run Qwen3-4B on one cheap 16–24 GB GPU,
destroy it after the JSON is copied, and rent the compatible Blackwell node only for the
one-forward DeepSeek fixture. Choose on-demand for the expensive first attempt; an
interruptible instance is sensible only after the procedure has already succeeded and
you know the run can restart cleanly.

**Fastest/simple administration:** use one compatible Blackwell node with 250 GB disk. Run
the Qwen smoke first, unload/exit that process, then run the DeepSeek fixture. This avoids
a second setup but pays for an idle second GPU during the Qwen step, so it is usually not
the cheapest.

Do not run harvesting, AdvBench, a judge, MMLU, or suffix optimization in this smoke
session. Those are full experiments and multiply GPU time without answering the two
compatibility questions above.

## Cost shutdown checklist

1. Confirm both artifact files exist and are non-empty.
2. Copy them off the instance.
3. Record GPU model/count, CUDA, PyTorch, Transformers, checkpoint revision, and command.
4. **Destroy** the Vast instance when finished. Stopping ends compute billing but storage
   charges continue; destroying ends both and deletes the instance data.

Useful primary references:

- [Qwen3-30B-A3B-FP8 model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8)
- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B)
- [Transformers fine-grained FP8 hardware requirements](https://huggingface.co/docs/transformers/quantization/finegrained_fp8)
- [Transformers expert backends and packed-FP4 requirement](https://huggingface.co/docs/transformers/main/experts_interface)
- [DeepSeek-V4-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [Vast.ai offer filters](https://docs.vast.ai/api-reference/search/search-offers)
- [Vast.ai instance lifecycle](https://docs.vast.ai/guides/instances/manage-instances)
