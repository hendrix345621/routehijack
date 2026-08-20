PYTHON ?= python

check: lint test profiles

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

test:
	$(PYTHON) -m pytest -q

profiles:
	$(PYTHON) -c "from routeaudit import config; [config.load(p) for p in config.list_models()]; print('profiles: ok')"

.PHONY: check lint test profiles
