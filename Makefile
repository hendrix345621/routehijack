PYTHON ?= python
PERSIST_ROOT ?= /workspace
HF_HOME = $(PERSIST_ROOT)/huggingface
HF_HUB_CACHE = $(HF_HOME)/hub
XDG_CACHE_HOME = $(PERSIST_ROOT)/cache
TMPDIR = $(PERSIST_ROOT)/tmp
DATA_DIR = $(PERSIST_ROOT)/routeaudit-data
ARTIFACTS_DIR = $(PERSIST_ROOT)/routeaudit-artifacts
OFFLOAD_DIR = $(PERSIST_ROOT)/routeaudit-offload
CONFIG ?= smoke
RUN_ARGS ?=
THINKING_ARGS ?=

GPU_ENV = HF_HOME="$(HF_HOME)" HF_HUB_CACHE="$(HF_HUB_CACHE)" \
	XDG_CACHE_HOME="$(XDG_CACHE_HOME)" TMPDIR="$(TMPDIR)" \
	ROUTEAUDIT_OFFLOAD_DIR="$(OFFLOAD_DIR)" ROUTEAUDIT_OFFLOAD_STATE_DICT=1 \
	TOKENIZERS_PARALLELISM=false \
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run: disk-dirs
	$(GPU_ENV) $(PYTHON) -m routeaudit.cli run --config "$(CONFIG)" \
		--data-dir "$(DATA_DIR)" --artifacts-dir "$(ARTIFACTS_DIR)" $(RUN_ARGS)

thinking-check: disk-dirs
	$(GPU_ENV) $(PYTHON) scripts/quick_reasoning_check.py --config "$(CONFIG)" \
		--out "$(ARTIFACTS_DIR)/quick_reasoning.json" $(THINKING_ARGS)

disk-dirs:
	@for path in "$(PERSIST_ROOT)" "$(HF_HOME)" "$(HF_HUB_CACHE)" \
		"$(XDG_CACHE_HOME)" "$(TMPDIR)" "$(DATA_DIR)" "$(ARTIFACTS_DIR)" "$(OFFLOAD_DIR)"; do \
		case "$$path" in /dev/shm|/dev/shm/*) \
			echo "rented-GPU storage must be persistent disk, not $$path" >&2; exit 2;; esac; \
	done
	@if command -v findmnt >/dev/null 2>&1; then \
		fs_type=$$(findmnt -n -o FSTYPE -T "$(PERSIST_ROOT)"); \
		case "$$fs_type" in tmpfs|ramfs) \
			echo "PERSIST_ROOT is on $$fs_type, not persistent disk" >&2; exit 2;; esac; \
	fi
	mkdir -p "$(HF_HUB_CACHE)" "$(XDG_CACHE_HOME)" "$(TMPDIR)" \
		"$(DATA_DIR)" "$(ARTIFACTS_DIR)" "$(OFFLOAD_DIR)"

check: lint test profiles

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

test:
	$(PYTHON) -m pytest -q

profiles:
	$(PYTHON) -c "from routeaudit import config; [config.load(p) for p in config.list_models()]; print('profiles: ok')"

.PHONY: run thinking-check disk-dirs check lint test profiles
