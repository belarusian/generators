VENV := .venv
PYTHON := $(VENV)/bin/python
SETUP := ./setup.sh

install:
	$(SETUP)

test: install
	$(PYTHON) -m pytest tests/ -v

.PHONY: install test
