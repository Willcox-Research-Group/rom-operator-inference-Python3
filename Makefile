.PHONY: test clean deploy

REMOVE = rm -rfv
PYTHON = python3
PYTEST = pytest --cov --cov-report html
TARGET = tests/*.py


test:
	$(PYTHON) check_docs.py
	$(PYTEST) $(TARGET)
	open htmlcov/index.html

clean:
	find . -type d -name "build" | xargs $(REMOVE)
	find . -type d -name "dist" | xargs $(REMOVE)
	find . -type d -name "*.egg*" | xargs $(REMOVE)
	find . -type f -name ".coverage*" | xargs $(REMOVE)
	find . -type d -name ".pytest_cache" | xargs $(REMOVE)
	find . -type d -name "__pycache__" | xargs $(REMOVE)
	find . -type d -name ".ipynb_checkpoints" | xargs $(REMOVE)
	find . -type d -name "htmlcov" | xargs $(REMOVE)

deploy: test clean
	git checkout master
	$(PYTHON) setup.py sdist bdist_wheel
	$(PYTHON) -m twine upload dist/*

