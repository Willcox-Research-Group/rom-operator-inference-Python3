.PHONY: help clean install lint test docs docs_all deploy_package deploy_docs


REMOVE = rm -rfv
PYTHON = python3
PYTEST = pytest --cov opinf tests --cov-report html


# About -----------------------------------------------------------------------
help:
	@echo "usage:"
	@echo "  make clean: remove all build, test, and Python artifacts"
	@echo "  make install: install the package locally from source"
	@echo "  make lint: check style with flake8"
	@echo "  make test: run unit tests via pytest"
	@echo "  make test_light: run unit tests without reinstalling package"
	@echo "  make docs: build jupyter-book documentation"
	@echo "  make docs_light: build documentation without cleaning files"


# Cleanup ---------------------------------------------------------------------
clean:
	find . -type d -name "build" | xargs $(REMOVE)
	find . -type d -name "dist" | xargs $(REMOVE)
	find . -type d -name "*.egg*" | xargs $(REMOVE)
	find . -type f -name ".coverage*" | xargs $(REMOVE)
	find . -type d -name ".pytest_cache" | xargs $(REMOVE)
	find . -type d -name "__pycache__" | xargs $(REMOVE)
	find . -type d -name ".ipynb_checkpoints" | xargs $(REMOVE)
	find . -type d -name "htmlcov" | xargs $(REMOVE)
	find docs -type d -name "_build" | xargs $(REMOVE)
	find docs -type d -name "_autosummaries" | xargs $(REMOVE)


# Installation ----------------------------------------------------------------
install:
	$(PYTHON) -m pip install .


install_tests:
	$(PYTHON) -m pip install ".[tests]"


install_docs:
	$(PYTHON) -m pip install ".[docs]"


install_all:
	$(PYTHON) -m pip install ".[tests,docs]"


# Testing ---------------------------------------------------------------------
lint:
	$(PYTHON) -m flake8 src
	$(PYTHON) -m flake8 tests


# Run tests as is (no cleanup / installation).
test_light: lint
	$(PYTHON) -m $(PYTEST)
	# open htmlcov/index.html


# Clean everything, re-install package, and run tests.
test: clean install_tests test_light


# Documentation ---------------------------------------------------------------
# No cleaning, take advantage of caching.
docs_light:
	jupyter-book build --nitpick docs

# Clean everything, re-install package, and build docs.
docs: clean install_docs docs_light

# Clean everything, re-install package, and build docs with error detection.
docs_all: clean install_docs
	jupyter-book build --nitpick --warningiserror --all docs


# Deployment (ADMINISTRATORS ONLY) --------------------------------------------
deploy_package: test
	git checkout main
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m pip install --upgrade twine
	$(PYTHON) -m build --sdist --wheel
	$(PYTHON) -m twine check dist/*
	$(PYTHON) -m twine upload dist/*


deploy_docs: docs
	$(PYTHON) -m pip install --upgrade ghp-import
	git checkout main
	ghp-import --remote upstream --no-jekyll --push --force --no-history docs/_build/html
