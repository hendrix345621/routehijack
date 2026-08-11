# Test suite

From the repository root:

```bash
python -m pytest tests -q
python tests/run_synthetic.py
```

The pytest suite covers unit, parity, fixture-adapter, device, and regression behavior.
The synthetic runner exercises the complete mHC instrumentation path on CPU.

See [../docs/TESTING.md](../docs/TESTING.md) for the validation ladder and real-model
commands.
