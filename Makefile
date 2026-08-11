# RouteAudit — routing-aware safety evaluation for MoE LLMs. Four phases:
#   1 data        → corpora (LLM-LAT pairs, C4, AdvBench, MMLU)
#   2 harvest     → localize safety + harmful experts (one model load)
#   3 routeaudit → optimize the universal adversarial suffix
#   4 eval        → ASR + MMLU utility + routing-shift (TESR/THPR) + SAFE/AT-RISK verdict

PY := python
DATA := data
ART := artifacts
CONFIG := configs/base.yaml

install:
	pip install -e .

data:
	$(PY) scripts/00_data.py --data-dir $(DATA)

harvest:
	$(PY) scripts/01_harvest.py --config $(CONFIG)

routeaudit:
	$(PY) -u scripts/02_suffix_search.py --config $(CONFIG) \
		--n-prompts 16 --n-steps 300 --candidates-per-step 128 \
		--candidate-prompt-subsample 0 --grad-batch-size 8 --candidate-batch-size 128 \
		--early-stop-patience 30

eval:
	$(PY) scripts/03_eval.py --config $(CONFIG)

all: data harvest routeaudit eval

# Interactive one-shot: pick a model, run all four phases, stop at the verdict.
#   make run                      # prompts for the model
#   make run MODEL=qwen3          # non-interactive
run:
	$(PY) scripts/run_all.py $(if $(MODEL),--model $(MODEL) --yes,)

# Cost-effective large-model flow (e.g. Qwen3-235B). See the COST PLAYBOOK in
# configs/qwen3_235b_a22b.yaml + docs/RUNBOOK.md.
#   make surrogate MODEL=qwen3            # cheap box: produce a transferable suffix
#   make target MODEL=qwen3-235b          # big node: ONE load, forward-only harvest+eval
surrogate:
	$(PY) scripts/run_all.py --model $(MODEL) --yes --stop-after attack \
		--checkpoint $(ART)/attack.ckpt.json --resume

target:
	$(PY) scripts/target_session.py --model $(MODEL) \
		--suffix $(ART)/routeaudit_universal.json --judge --resume

.PHONY: install data harvest routeaudit eval all run surrogate target clean

clean:
	rm -rf $(ART)
