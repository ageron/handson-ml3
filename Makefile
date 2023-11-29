# Copyright 2023 O1 Software Network. MIT licensed.

SHELL := bash -e -o pipefail

lint:
	black constant/
	isort constant/
	ruff .

MYPY = mypy --no-namespace-packages

type: typecheck
typecheck:
	$(MYPY) constant/
