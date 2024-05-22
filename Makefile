.PHONY: help install dev lint format test docs deploy_package deploy_docs


REMOVE = rm -rfv
PYTHON = python3
PIP = $(PYTHON) -m pip install
TOX = $(PYTHON) -m tox -e


# About -----------------------------------------------------------------------
help:
	@echo "usage:"
	@echo "  make install:    install the package locally from source"
	@echo "  make dev:        install the package locally in development mode"
	@echo "  make format:     (FUTURE FEATURE) run black formatter"
	@echo "  make lint:       check style with flake8"
	@echo "  make test:       run unit tests via pytest"
	@echo "  make docs:       build jupyter-book documentation"


clean_docs:
	$(REMOVE) "docs/_build"
	find docs -type d -name "_autosummaries" | xargs $(REMOVE)


# Installation ----------------------------------------------------------------
install:
	$(PIP) .


dev:
	$(PIP) -e ".[dev]"


# Testing and documentation ---------------------------------------------------
format:
	# $(TOX) format


lint:
	$(TOX) lint


test:
	$(TOX) lint,format,py39,py310,py311,py312


docs:
	$(TOX) literature,docs


# Deployment (ADMINISTRATORS ONLY) --------------------------------------------
deploy_package: test
	git checkout main
	$(PIP) --upgrade build
	$(PIP) --upgrade twine
	$(PYTHON) -m build --sdist --wheel
	$(PYTHON) -m twine check dist/*
	$(PYTHON) -m twine upload dist/*


deploy_docs: docs  # clean_docs
	$(PYTHON) -m pip install --upgrade ghp-import
	git checkout main
	ghp-import --remote upstream --no-jekyll --push --force --no-history docs/_build/html
