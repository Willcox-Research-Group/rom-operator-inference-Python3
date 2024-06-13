.PHONY: help clean_docs install dev format style test docs deploy_package deploy_docs


REMOVE = rm -rfv
PYTHON = python3
PIP = $(PYTHON) -m pip install
TOX = $(PYTHON) -m tox


# About -----------------------------------------------------------------------
help:
	@echo "make recipes"
	@echo "------------"
	@echo "make install -> install the package locally from source"
	@echo "make dev     -> install the package locally in development mode"
	@echo "make format  -> format code with black"
	@echo "make style   -> check style with black and flake8"
	@echo "make test    -> run unit tests via pytest"
	@echo "make docs    -> build jupyter-book documentation"
	@echo "make all     -> check style, run tests, and build docs"
	@echo " "
	@echo "tox environments (tox -e <env>)"
	@echo "-------------------------------"
	@$(TOX) list


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
	$(TOX) -e format


style:
	$(TOX) -e style


test: style
	$(TOX)


docs:
	$(TOX) -e literature,docs

all: test docs


# Deployment (ADMINISTRATORS ONLY) --------------------------------------------
deploy_package:
	git checkout main
	$(TOX)
	$(PIP) --upgrade build
	$(PIP) --upgrade twine
	$(REMOVE) dist/
	$(PYTHON) -m build --sdist --wheel
	$(PYTHON) -m twine check dist/*
	$(PYTHON) -m twine upload dist/*


deploy_docs: # clean_docs
	git checkout main
	$(TOX) -e literature,docs
	$(PYTHON) -m pip install --upgrade ghp-import
	ghp-import --remote upstream --no-jekyll --push --force --no-history docs/_build/html
