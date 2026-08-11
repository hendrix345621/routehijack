# RouteAudit for DeepSeek V4

Standalone research tooling for routing-safety evaluation on DeepSeek V4 and its
Manifold-Constrained Hyper-Connections (mHC) residual stream.

This directory is self-contained and can be moved into its own Git repository. It
includes the minimal RouteAudit runtime, the V4 integration, checkpoint configuration,
CPU validation, real-checkpoint fixture tooling, and evaluation commands.

> Use only on models and infrastructure you are authorized to evaluate. The real
> checkpoint workflow requires specialized hardware and can be expensive.

## What is included

| Capability | Status | Command |
|---|---:|---|
| CPU-only mHC mechanism validation | Ready | `make synthetic` |
| Unit and reference-parity tests | Ready | `make test` |
| Hardware/software preflight | Ready | `make preflight` |
| Expert harvesting | Ready | `make harvest` |
| Boundary routing diagnostics | Ready | `make diagnose` |
| Real-checkpoint fixture extraction | Requires checkpoint access | `python fixtures/extract.py` |
| ASR/MMLU evaluation of a supplied suffix | Ready after artifacts exist | `make eval` |
| V4-specific suffix optimization | Not implemented | — |

The suffix optimizer in the original RouteAudit workflow assumes a differentiable
softmax router. This project does not claim that optimizer works on V4. Evaluation can
use a supplied or transferred suffix, while the V4 tooling measures the actual routing
behavior independently.

## Quick start

Python 3.10+ is required. The pinned V4 runtime currently expects Transformers 5.14.

```bash
git clone <your-new-repository-url>
cd <your-new-repository>

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

make test
make synthetic
```

On PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest tests -q
python tests/run_synthetic.py
```

## Real-checkpoint workflow

Run the preflight before downloading weights:

```bash
python scripts/run_mhc_smoke.py --preflight-only
```

Prepare the evaluation data and identify routed experts:

```bash
python scripts/00_data.py
python scripts/01_harvest.py
python scripts/route_mhc.py
```

Capture and validate evidence from the real checkpoint:

```bash
python fixtures/extract.py
python fixtures/validate.py --require-complete
```

For the complete rental procedure and hardware requirements, see
[docs/MHC_RENTAL_RUNBOOK.md](docs/MHC_RENTAL_RUNBOOK.md).

## Evaluation

`scripts/03_eval.py` compares clean prompts with prompts containing a supplied suffix.
It requires:

- `data/advbench.jsonl` and `data/mmlu_subset.jsonl`;
- `artifacts/safety_experts.json` and `artifacts/harmful_experts.json`;
- `artifacts/routeaudit_universal.json`, containing `{"suffix": "..."}`.

Run:

```bash
python scripts/03_eval.py --no-judge
```

Remove `--no-judge` for the configured classifier-based safety grade.

## Repository layout

```text
.
├── src/routeaudit/                 # vendored minimal evaluation runtime
├── src/routeaudit_deepseek_v4/     # V4 registration, hooks, mHC, precision policy
├── configs/deepseek_v4_flash.yaml  # checkpoint and routing configuration
├── scripts/                        # data, harvest, diagnostics, preflight, evaluation
├── fixtures/                       # real-checkpoint capture and offline validation
├── tests/                          # CPU, parity, regression, and synthetic tests
├── analysis/                       # research utilities
└── docs/                           # runbooks and archived research notes
```

The public interfaces are the root README, `Makefile`, `scripts/`, `fixtures/`, and the
two packages under `src/`. Files under `docs/research/` are historical working notes.

## Verification levels

1. **Unit/parity:** gate math, device paths, hooks, and regression behavior.
2. **Synthetic:** a small genuine mHC architecture validates the mechanism on CPU.
3. **Fixture:** saved outputs from the released checkpoint validate gate and mHC maps.
4. **Real model:** semantic routing and refusal experiments on the full checkpoint.

Passing levels 1–2 validates the implementation mechanics; it is not evidence about the
behavior of the trained checkpoint. Claims about the released model require levels 3–4.

## Development

```bash
python -m pytest tests -q
python -m ruff check --select E9,F63,F7,F82 src tests scripts fixtures analysis
python -m compileall -q src tests scripts fixtures analysis
```

Generated data, model caches, fixtures, and evaluation artifacts are intentionally not
committed. Review `.gitignore` before publishing large or sensitive outputs.
