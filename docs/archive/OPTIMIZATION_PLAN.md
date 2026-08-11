# RouteAudit optimisation plan

Updated: 2026-08-03

## Completed in this pass

| Priority | Change | Expected effect | Verification |
|---|---|---|---|
| P0 | Accumulate custom-decoder tokens in a list and concatenate once | Removes quadratic token-buffer copies on long thinking traces | Cached decoder regression test |
| P0 | Use `torch.inference_mode()` for generation and harvest | Removes autograd bookkeeping from read-only work | Full test suite |
| P0 | Count top-k expert membership with `bincount` | Removes the dense `(B,T,E)` boolean allocation in every captured layer | Exact old/new equivalence test |
| P0 | Accumulate harvest counts as integers | Avoids slow GPU FP64 accumulation; converts the small result on CPU once | Exact frequency arithmetic |
| P0 | Defer MHC diagnostic host reads | One CPU/GPU synchronization per report instead of two per layer | MHC conservation tests |
| P0 | Match grouped routing to the Transformers 5.9 reference | Restores DeepSeek V3 compatibility after its routing API moved into the gate | Official-router parity test |
| P1 | Allow thinking masks on the source tensor's device | Makes device placement explicit for GPU consumers | CPU/CUDA-parameterized test |

Representative CPU microbenchmark (`B=16`, `T=1024`, `E=256`, top-8): expert
counting fell from 13.58 ms to 0.38 ms and temporary membership storage from 4.0 MiB
to about 0.5 MiB per layer. Accumulating a 2,048-token custom decode fell from
7.06 ms to 0.42 ms. These are microbenchmarks, not end-to-end GPU claims.

## Next work, in order

1. **Resolve the default-target mismatch.** `configs/base.yaml` selects Liquid
   LFM2.5, but the suffix optimiser correctly rejects its biased sigmoid gate. Either
   make OLMoE the runnable end-to-end default or implement a gate-aware LFM loss before
   advertising `make all` for Liquid.
2. **Run the device suite on the deployment GPU.** Execute `pytest tests
   experiments/mhc/tests -q`; CUDA cases activate automatically. Record GPU model,
   PyTorch/CUDA versions, peak VRAM and tokens/s.
3. **Add automatic batch calibration to harvest and evaluation.** Probe a small batch,
   grow until a target VRAM headroom (10-15%), and cache the selected batch size by
   model/GPU. Keep manual overrides for reproducibility.
4. **Prefetch harvest batches.** Move padding into a small DataLoader with pinned-memory
   prefetch and non-blocking transfers so CPU preparation overlaps the current forward.
5. **Bucket deterministic evaluation prompts by token length.** Restore original order
   after generation. This reduces left-padding compute; retain original order when
   sampling so seeded samples remain associated with the same prompts.
6. **Create an end-to-end benchmark command.** Measure wall time, peak allocated/reserved
   VRAM, tokens/s and prompts/s for harvest, attack and evaluation separately. A profile
   without these stage metrics makes regressions hard to detect.
7. **Pin the tested dependency set.** The ignored MHC suite had drifted across a
   Transformers API change. Commit a lock file or an upper-bounded compatibility matrix,
   and run the official-router parity test in CI.
8. **Validate real MHC weights.** Synthetic MHC tests validate mechanics only. The final
   compatibility claim still needs the DeepSeek-V4 fixture/replay test on a sufficiently
   large GPU host.

## Operational settings

- Keep the model fully on accelerators where possible; CPU/disk offload usually dominates
  all Python-level optimisations.
- Increase `--freq-batch-size` and `--gen-batch-size` until VRAM is 85-90% utilised without
  OOM; reduce `--candidate-batch-size` first when suffix search is memory-bound.
- Use `--auto-batch`, prefix KV caching and gradient checkpointing for large attack runs,
  but record the chosen values with results.
- In thinking mode, raise `max_new_tokens` until truncation is below 5%; a faster truncated
  run is not a valid safety measurement.
