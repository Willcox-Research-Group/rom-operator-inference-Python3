.PHONY: clean lint install test docs docs_all deploy


REMOVE = rm -rfv
PYTHON = python3
PYTEST = pytest --cov opinf tests --cov-report html


help:
	@echo "usage:"
	@echo "  make clean: remove all build, test, coverage and Python artifacts"
	@echo "  make lint: check style with flake8"
	@echo "  make install: install the package locally from source"
	@echo "  make test: run unit tests via pytest"
	@echo "  make docs: build jupyter-book documentation"


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


lint:
	$(PYTHON) -m flake8 src
	$(PYTHON) -m flake8 tests


install: clean
	$(PYTHON) -m pip install --use-feature=in-tree-build .


test: lint install
	$(PYTHON) -m $(PYTEST)
	# open htmlcov/index.html


docs:
	jupyter-book build docs -n -W --keep-going


docs_all:
	jupyter-book build --all docs


deploy: test docs_all
	git checkout main
	$(PYTHON) -m pip install build
	$(PYTHON) -m build --sdist --wheel
	$(PYTHON) -m twine check dist/*
	$(PYTHON) -m twine upload dist/*
