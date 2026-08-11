# DeepSeek-V4 mHC adapter: implementation and results

Updated 2026-08-03. This is the current hand-off for the saved B200 run and supersedes
older notes that describe real-checkpoint validation as pending.

## Bottom line

The adapters were operational before this patch: the released checkpoint loaded fully on
one B200, completed a forward pass, and produced a fixture containing a learned V4 gate,
a four-stream residual, a hash-routing table, and output logits. The saved learned-gate
expert set replays exactly and its weights replay within `5.960e-08` (one float32 ULP).

That run did **not** retain two independent reference values: the gate tensors returned by
the shipped router and the `comb` maps returned by the real HyperConnection modules.
Consequently it proves execution and structural compatibility, but cannot retroactively
prove bitwise official gate parity or direct B-path conservation. The old manifest's
`fixture_validate_failed` status was caused only by requiring cross-device bitwise equality
for a one-ULP CPU/GPU reduction difference; it was not a model-load, routing-selection, or
forward failure.

No further paid run is recommended for development. Only repeat the B200 fixture if an
external report requires the two missing measurements to be conclusive rather than clearly
reported as limitations.

## Evidence from the completed rental

- Checkpoint: `deepseek-ai/DeepSeek-V4-Flash` at pinned revision
  `60d8d70770c6776ff598c94bb586a859a38244f1` (148.7 GiB repository).
- Hardware/software: one NVIDIA B200 (178.4 GiB visible), PyTorch 2.12.0+cu130,
  Transformers 5.14.1, `kernels` 0.15.2, CUDA/nvcc 13.0, native `deepgemm` backend.
- Preflight: every required check passed; the full model was placed on `cuda:0` with no
  CPU or disk offload.
- Synthetic Level 0: passed in 9.025 seconds.
- Real extraction: passed in 115.677 seconds, including download/load and one short forward.
- Saved learned gate: layer 23, exact selected expert set; portable weight replay maximum
  absolute difference `5.960e-08`.
- Saved residual: shape `(1, 5, 4, 4096)`, confirming four mHC streams and a valid
  stream-mean reduction to `(1, 5, 4096)`.
- Saved hash routing: the first 256 entries reproduce the retained `(4096, 6)` token-to-
  expert table exactly.
- Saved fixture SHA-256:
  `3F110A8D5B337AE50E7E5FA48FC5D3AA485DE4D5E769FD7F6895552BCE4FD5F1`.

Offline validation, with no model download or GPU:

```powershell
py -3 experiments/mhc/fixtures/validate.py `
  --fixtures C:\Users\kenna\Downloads\v4_flash_fixtures.pt
```

This intentionally reports `Level 1 PARTIAL`: every measurement available in the legacy
fixture passes, while the two values the old format never recorded remain unmeasured.

## What changed

### 1. Official router parity is now independent

`MoEHookManager` recognizes either DeepSeek router output form:

- Transformers V4: `(logits, weights, indices)`;
- the raw/reference form: `(weights, indices)`.

`RouteResult` retains `official_weights` and `official_indices` separately from
RouteAudit's recomputation. Fixture format v2 compares those tensors on the same GPU and
requires exact expert-set and weight parity. This avoids using RouteAudit output as its own
oracle.

Portable replay is a separate test. It reconstructs top-k on CPU from saved scores using a
documented `2e-7` tolerance, enough for the observed one-ULP device difference. It cannot
weaken same-device official parity, which remains exact.

### 2. Real mHC maps are captured at both residual sites

Each V4 decoder layer has two HyperConnection calls, `attn_hc` and `ffn_hc`. Their official
forward result is `(post, comb, collapsed)`. The new hook retains those outputs plus the
four-stream input and reconstructs the matrix actually applied by the decoder:

```text
B = comb.transpose(-1, -2)
X_next = B X + C F(A X)
```

The fixture validator now checks both sites for:

- correct four-stream shapes;
- row and column sums near one;
- spectral norm no greater than one within tolerance;
- conservation of the mean across streams on the B path.

The runtime `mhc_conservation_profile` uses the same real hooks instead of looking only for
the synthetic module's `generate_maps` method. It does not claim an end-to-end perturbation
gain from those map captures; that remains explicitly unmeasured.

### 3. The runner now preserves useful evidence on failure

- Fixture format and strict-capture status are written into the manifest immediately after
  extraction, even if validation later fails.
- Required captures now include official gate output and both real HyperConnection sites.
- Tensor fixtures load with `weights_only=True`; the legacy `TorchVersion` metadata type is
  narrowly allowlisted.
- Missing legacy measurements are warnings in ordinary offline validation and a distinct
  exit code under `--require-complete`. Actual measured mismatches still fail.

### 4. Loading and compatibility were tightened

- V4 explicitly uses its supported eager attention path; the runner no longer attempts
  SDPA and reloads the 149 GiB checkpoint after rejection.
- Transformers' current `dtype=` argument replaces deprecated `torch_dtype=`.
- Native FP4/FP8 and `deepgemm` remain unchanged; no surrogate quantization is introduced.
- Grouped V2/V3 routing follows the supported official zero-mask sentinel. V4 is flat
  top-6, so this shared-adapter correction does not change the V4 result.
- Local Hugging Face cache output is ignored by git.

## How the adapter maps DeepSeek-V4 into RouteAudit

1. `ArchSpec` locates decoder layers, MoE blocks, router, expert containers, and the
   selection-bias tensor across supported Transformers layouts.
2. Learned V4 layers consume official logits when exposed, then reproduce
   `sqrt(softplus(logit))`, flat top-6 selection over 256 experts, selection-only bias,
   bias-free gathered weights, normalization, and scaling.
3. Hash-routed leading MoE layers use `tid2eid[input_id]` for expert identity. They are
   treated as non-steerable selection rather than being misclassified as learned gates.
4. Residual hooks preserve `(batch, token, stream, hidden)` instead of flattening four
   streams. All conservation summaries use the stream mean.
5. HyperConnection hooks observe both attention and feed-forward maps without modifying
   the released model or its kernels.

## Limitations that remain

- The saved format-v1 fixture cannot be upgraded after the instance is gone. Official
  returned gate tensors and real `comb` maps simply are not present.
- Structural compatibility is not a semantic safety result. It does not show whether an
  adversarial suffix changes refusal, reasoning, or harmful-completion behavior.
- Direct B-path constraints are now measurable, but end-to-end perturbation gain through
  attention, experts, and all layers is not implemented for the real hash-routed model.
- Full DeepSeek-V4 suffix optimization is still gated separately: the existing generic
  suffix loss assumes softmax routing and must not be presented as a faithful V4 attack.
- The implementation is pinned to the tested checkpoint revision and Transformers 5.14
  line. A dependency or checkpoint update should rerun the free CPU differential tests.

All of these are solvable. The first requires the model to be resident once more; the
second and fourth require actual experiment design and evaluation rather than adapter
plumbing; the third needs a controlled full-model perturbation injection that preserves
token IDs for hash routing. None blocks using the current saved result as evidence that the
checkpoint loads and the structural adapters work.

## Next steps without another expensive run

1. Keep the downloaded fixture, manifest, preflight, and log together and record the
   fixture SHA-256 above. They are the audit trail.
2. Use the offline validator command above whenever gate/mHC code changes.
3. Run the free local suite before committing:

   ```powershell
   py -3 -m pytest experiments/mhc/tests tests/test_device_paths.py -q
   ```

4. Treat the current conclusion as **structural compatibility: pass; strict independent
   parity: unmeasured in the legacy artifact**.
5. Do not rent another B200 merely to turn the word `PARTIAL` into `PASSED`. If strict
   independent evidence becomes necessary for publication, use the patched runner once;
   fixture v2 now captures every required value and fails early if anything is absent.

