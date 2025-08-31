.PHONY: setup dev test topo energy lint

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -e .[dev]

dev:
	. .venv/bin/activate && pip install -e .[topology,causality,dev]

test:
	. .venv/bin/activate && pytest -q

topo:
	. .venv/bin/activate && aps fit-topo --latent 2 --epochs 80 --topo-k 8 --topo-weight 1.0

energy:
	. .venv/bin/activate && python scripts/run_energy_demo.py

lint:
	. .venv/bin/activate && ruff check .
