"""Google Colab environment setup script."""

import subprocess
import sys


def install_packages():
    """Install required packages in Colab."""
    packages = [
        'ultralytics',
        'rouge-score',
        'sacrebleu',
        'nltk',
        'seaborn'
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
    print("All packages installed.")


def setup_nltk():
    """Download NLTK data."""
    import nltk
    for pkg in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
        nltk.download(pkg, quiet=True)
    print("NLTK data downloaded.")


if __name__ == "__main__":
    install_packages()
    setup_nltk()
    print("Colab setup complete.")
