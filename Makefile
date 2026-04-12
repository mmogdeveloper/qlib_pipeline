PYTHON ?= python3

.PHONY: test test-all coverage

test:
	$(PYTHON) -m pytest tests/unit/ -v

test-all:
	$(PYTHON) -m pytest tests/ -v

coverage:
	$(PYTHON) -m pytest tests/unit/ --cov=. --cov-report=term-missing --cov-report=html -v
