.PHONY: install clean test

install:
	pip install -r requirements.txt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	python scripts/xor_problem.py
	python scripts/preprocess_mnist.py
