.PHONY: help test lint format

help:
	@echo "make test    Run unit tests"
	@echo "make lint    Check Python formatting and lint"
	@echo "make format  Format Python and apply safe lint fixes"

test:
	uv run --group test pytest

lint:
	uv run --group test ruff check .
	uv run --group test ruff format --check .

format:
	uv run --group test ruff format .
	uv run --group test ruff check --fix .
