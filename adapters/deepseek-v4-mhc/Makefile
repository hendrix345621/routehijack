PYTHON ?= python
CONFIG ?= configs/deepseek_v4_flash.yaml

.PHONY: help install data harvest diagnose synthetic preflight eval test

help:
	@echo "DeepSeek V4 RouteAudit"
	@echo "  make install    Install the standalone project"
	@echo "  make data       Prepare evaluation datasets"
	@echo "  make harvest    Identify safety/harmful experts"
	@echo "  make diagnose   Measure routing mass at the decision boundary"
	@echo "  make synthetic  Run CPU-only mHC validation"
	@echo "  make preflight  Validate a real-checkpoint GPU host"
	@echo "  make eval       Run ASR/MMLU evaluation (requires artifacts)"
	@echo "  make test       Run the complete CPU test suite"

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/00_data.py

harvest:
	$(PYTHON) scripts/01_harvest.py --config $(CONFIG)

diagnose:
	$(PYTHON) scripts/route_mhc.py --config $(CONFIG)

synthetic:
	$(PYTHON) tests/run_synthetic.py

preflight:
	$(PYTHON) scripts/run_mhc_smoke.py --config $(CONFIG) --preflight-only

eval:
	$(PYTHON) scripts/03_eval.py --config $(CONFIG)

test:
	$(PYTHON) -m pytest tests -q
