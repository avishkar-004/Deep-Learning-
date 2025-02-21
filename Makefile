.PHONY: install clean test setup-colab check-gpu

install:
	pip install -r requirements.txt

setup-colab:
	python scripts/colab_setup.py

setup-nltk:
	python scripts/setup_nltk.py

check-gpu:
	python utils/gpu_check.py

test:
	python scripts/xor_problem.py
	python scripts/preprocess_mnist.py
	python scripts/tfidf_vectorizer.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
