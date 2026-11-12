# WattBench developer targets. `make install` creates .venv and installs the package.
PYTHON ?= python3.11
PY := .venv/bin/python
WB := .venv/bin/wattbench
CONFIG ?= config/rig.yaml
DEMO_CONFIG ?= config/ci_smoke.yaml
OUT ?= runs/demo
RUN ?= $(shell ls -td runs/2* 2>/dev/null | head -1)
DATE := $(shell date -u +%Y%m%dT%H%M%SZ)

.PHONY: install lint typecheck test sweep report demo clean

install:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

lint:
	.venv/bin/ruff check .

typecheck:
	.venv/bin/mypy src/wattbench

test:
	.venv/bin/pytest -q --cov=wattbench

sweep:
	$(WB) sweep --config $(CONFIG) --out runs/$(DATE)

report:
	$(WB) report $(RUN)

# Fast but complete pipeline: devices -> sweep -> analyze -> report -> compare.
# Uses the small CI matrix so it finishes in well under a minute.
demo:
	$(WB) devices
	$(WB) sweep --config $(DEMO_CONFIG) --out $(OUT)
	$(WB) analyze $(OUT) --out $(OUT)/results
	$(WB) report $(OUT) --no-mirror
	$(WB) compare --baseline $(OUT) --candidate $(OUT)

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml
