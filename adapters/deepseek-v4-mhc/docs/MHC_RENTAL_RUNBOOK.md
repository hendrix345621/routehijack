# DeepSeek-V4 mHC: one-rental runbook

Use this guide for the **real checkpoint compatibility fixture**, not the larger semantic
safety evaluation. The objective is one short forward that proves the released model's
gate, four-stream mHC residual, and hash routing agree with this project.

## Rent this

Required:

- NVIDIA **Blackwell**, compute capability at least 10.0 (SM100+).
- Free aggregate VRAM of at least the 159.63 GB checkpoint plus 20 GiB. The included
  preflight calculates the exact threshold from the pinned Hugging Face revision.
- 250 GB container disk.
- 64 GiB CPU RAM minimum; 128 GiB or more recommended.
- Linux, CUDA toolkit 12.9+ including `nvcc`, and a Blackwell-compatible PyTorch image.
- Verified Vast host, reliability >=0.99, direct SSH, and fast/low-cost download bandwidth.

Practical offer shapes:

1. One B300/GB300 with 288 GB: safest capacity, usually most expensive.
2. One B200 with 180 GB: likely lowest simple single-GPU shape; rent only if the free-VRAM
   preflight passes with its 20 GiB reserve.
3. Two 96 GB Blackwell cards on the same host: adequate aggregate capacity; prefer a fast
   interconnect, although the one-forward test can tolerate PCIe at extra load time.

Do **not** rent A100, H100, or H200 for the pinned as-shipped checkpoint. Its experts are
FP4-packed, and the current native Transformers expert backend requires Blackwell. Do not
apply NF4/int8 or substitute a community quantization: that would test different routing
and different weights.

Suggested Vast CLI filter, if you use the CLI:

```bash
vastai search offers 'verification=verified reliability>0.99 compute_cap>=1000 gpu_total_ram>=180000 cpu_ram>=65536 disk_space>=250' --order=dph_total
```

In the web UI, also compare `inet_down`, download cost, disk bandwidth, and total hourly
price. Use **on-demand**, not interruptible, for the first and ideally only 160 GB download.

## Before starting the paid download

Choose a PyTorch **development** image with CUDA 12.9 or newer; a runtime-only image has no
`nvcc` and fails the native kernel preflight. Connect by SSH, copy the repository, then:

```bash
cd routehijack
mkdir -p /workspace/hf-cache
export HF_HOME=/workspace/hf-cache
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Replace `/workspace` with the large mounted volume shown by `df -hT`. Keep `HF_HOME`
set for both preflight and the real run. The preflight reports the exact path it checks;
do not continue unless that path has at least 190 GiB free.

`--system-site-packages` deliberately keeps the CUDA-matched PyTorch supplied by the
Vast image. The preflight rejects it if it is too old or lacks the required CUDA runtime.

The checkpoint is public. A Hugging Face token is optional but can reduce anonymous rate
limits. If used, enter it directly on the instance and never commit it or send it in chat.

Run the no-weight preflight:

```bash
python scripts/run_mhc_smoke.py --preflight-only
```

It checks and records:

- exact GPU names, compute capabilities, free and total VRAM;
- CUDA runtime, full toolkit/`nvcc`, PyTorch, Transformers, and `kernels` versions;
- CPU RAM, free disk, GPU topology, platform, and `nvidia-smi` output;
- the pinned model revision and repository byte size;
- the configured native `deepgemm` expert backend and an actual load of stable API v2 of
  the compatible `kernels-community/deep-gemm` extension before model weights download.

If it prints `PREFLIGHT FAILED`, do not download the model. Copy
`artifacts/mhc_real/preflight.json`, then destroy the instance and use the failed check to
select the corrected offer. Do not override a required failure to save time.

## Run once

Use `tmux` so an SSH disconnect does not end the process:

```bash
tmux new -s mhc
python scripts/run_mhc_smoke.py
```

The model is pinned to:

```text
deepseek-ai/DeepSeek-V4-Flash
revision 60d8d70770c6776ff598c94bb586a859a38244f1
repository size 159,630,041,626 bytes
```

The wrapper performs, in order:

1. The strict preflight again.
2. Free synthetic Level 0 validation.
3. One short real-checkpoint forward using native FP4/FP8 weights.
4. Exact same-device parity against the router's returned expert ids and weights.
5. Portable CPU replay with an explicit cross-device tolerance.
6. Four-stream residual shape plus direct B-path checks at `attn_hc` and `ffn_hc`.
7. Exact static hash-routing-table validation.

CPU/disk model offload, a missing gate/residual/hash capture, a non-four-stream residual,
or any parity mismatch is a hard failure.

## Required success evidence

The directory `artifacts/mhc_real/` will contain:

- `preflight.json`: complete hardware/software inventory and every pass/fail check;
- `run.log`: combined output from the synthetic, extraction, and validation stages;
- `v4_flash_fixtures.pt`: portable real-checkpoint fixture;
- `manifest.json`: commands, timings, git commit/status, artifact hash, fixture shapes,
  checkpoint metadata, and the final status.

Success requires all of the following:

```text
command exit code: 0
manifest.json status: passed
fixture required captures: all true
residual streams: 4
official gate selected expert set: same
official same-device gate weights: exact
portable CPU replay: within 2e-7
real HyperConnection sites: attn and ffn
hash routing: exact
```

This is a compatibility result. It does not establish a semantic safety effect or run the
full RouteAudit attack.

## Copy and shut down

Before destroying the instance:

```bash
tar -czf mhc_real_artifacts.tar.gz artifacts/mhc_real
ls -lh mhc_real_artifacts.tar.gz artifacts/mhc_real/*
```

Copy `mhc_real_artifacts.tar.gz` to your computer. If the job failed, copy it anyway—the
manifest and log should identify the exact stage without another blind rental.

Finally **destroy** the Vast instance. Stopping ends compute billing but continues storage
billing; destruction ends both and deletes the remote data.

## Primary references

- [DeepSeek-V4-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [Transformers DeepSeek-V4 implementation](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)
- [Transformers expert backends and native FP4 hardware](https://huggingface.co/docs/transformers/main/experts_interface)
- [Transformers fine-grained FP8/UE8M0 requirements](https://huggingface.co/docs/transformers/quantization/finegrained_fp8)
- [Hugging Face Kernels installation](https://huggingface.co/docs/kernels/main/installation)
- [Vast.ai offer filters](https://docs.vast.ai/api-reference/search/search-offers)
- [Vast.ai instance lifecycle and billing](https://docs.vast.ai/guides/instances/manage-instances)
