.PHONY: clean lint install test deploy


REMOVE = rm -rfv
PYTHON = python3
PYTEST = pytest --cov rom_operator_inference tests --cov-report html


clean:
	find . -type d -name "build" | xargs $(REMOVE)
	find . -type d -name "dist" | xargs $(REMOVE)
	find . -type d -name "*.egg*" | xargs $(REMOVE)
	find . -type f -name ".coverage*" | xargs $(REMOVE)
	find . -type d -name ".pytest_cache" | xargs $(REMOVE)
	find . -type d -name "__pycache__" | xargs $(REMOVE)
	find . -type d -name ".ipynb_checkpoints" | xargs $(REMOVE)
	find . -type d -name "htmlcov" | xargs $(REMOVE)


lint:
	$(PYTHON) -m flake8 src
	$(PYTHON) -m flake8 tests


install: clean
	$(PYTHON) -m pip install --use-feature=in-tree-build .


test: lint install
	$(PYTHON) -m $(PYTEST)
	# open htmlcov/index.html


deploy: test
	git checkout master
	$(PYTHON) setup.py sdist bdist_wheel
	$(PYTHON) -m twine upload dist/*
