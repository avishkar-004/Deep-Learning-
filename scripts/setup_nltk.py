"""Download required NLTK data packages."""

import nltk

packages = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'movie_reviews',
    'punkt_tab'
]

for package in packages:
    print(f"Downloading {package}...")
    nltk.download(package, quiet=True)

print("All NLTK packages downloaded.")
