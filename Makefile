.PHONY: test clean

REMOVE = rm -rfv
TARGET = tests/*.py # test_core.py

test:
	python check_docs.py
	pytest --cov --cov-report html $(TARGET)
	open htmlcov/index.html

clean:
	find . -type d -name "*.egg*" | xargs $(REMOVE)
	find . -type f -name ".coverage*" | xargs $(REMOVE)
	find . -type d -name ".pytest_cache" | xargs $(REMOVE)
	find . -type d -name "__pycache__" | xargs $(REMOVE)
	find . -type d -name ".ipynb_checkpoints" | xargs $(REMOVE)
	find . -type d -name "htmlcov" | xargs $(REMOVE)
