# Testing

Run the full CPU suite from the repository root:

```bash
python -m pytest tests -q
```

Run the end-to-end synthetic mHC validation:

```bash
python tests/run_synthetic.py
```

The suite covers gate semantics, routing capture, mHC conservation, reference parity,
device placement, fixture adapters, and regressions in the vendored runtime.

Real-checkpoint validation is intentionally separate because it requires checkpoint
access and suitable GPUs:

```bash
python scripts/run_mhc_smoke.py --preflight-only
python fixtures/extract.py
python fixtures/validate.py --require-complete
```

Do not treat CPU-only success as validation of trained-model behavior. It validates the
implementation and instrumentation layers only.
