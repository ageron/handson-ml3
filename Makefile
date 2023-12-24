# Copyright 2023 O1 Software Network. MIT licensed.

SHELL := bash -e -o pipefail

lint:
	black constant/
	isort constant/
	ruff .

COVERAGE = --cov --cov-report=term-missing

test:
	python -W error -m unittest constant/*/*/*_test.py
	pytest $(COVERAGE) constant/

MYPY = mypy --ignore-missing-imports --no-namespace-packages

type: typecheck
typecheck:
	$(MYPY) constant/
