# Architecture

The repository contains two Python packages in one distribution:

- `routeaudit` provides the small, architecture-neutral evaluation runtime used for
  configuration, loading, routing capture, expert harvesting, and ASR/MMLU evaluation.
- `routeaudit_deepseek_v4` adds the V4 model aliases, checkpoint defaults, mHC tensor
  operations, precision rules, and HyperConnection map capture.

The dependency direction is one-way:

```text
scripts / fixtures / tests
          │
          ├── routeaudit_deepseek_v4
          │          │
          └──────────┴── routeaudit
```

`routeaudit` does not import the V4 extension. Importing `routeaudit_deepseek_v4`
registers the V4 configuration with the local runtime. This keeps generic evaluation
code testable while isolating checkpoint-specific behavior.

## Supported workflow

The included runtime supports data preparation, expert harvesting, routing diagnostics,
and evaluation. The V4 integration does not implement a suffix optimizer because the
existing softmax-routing objective does not represent the checkpoint's router.

The synthetic model and fixture ladder deliberately separate mechanism validation from
claims about trained-model behavior.
