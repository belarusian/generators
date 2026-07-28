VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python

$(VENV):
	python3 -m venv $(VENV)

install: $(VENV)
	$(PIP) install -e ".[dev]"

test: install
	$(PYTHON) -m pytest tests/ -v

.PHONY: install test
