"""NLP text preprocessing for IMDB classification."""

import re
import string


def preprocess_review(text):
    """Preprocess a movie review text."""
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_stem(text):
    """Tokenize and apply basic stemming."""
    tokens = text.split()
    # Simple suffix removal
    stemmed = []
    for token in tokens:
        if token.endswith('ing'):
            token = token[:-3]
        elif token.endswith('ed'):
            token = token[:-2]
        elif token.endswith('ly'):
            token = token[:-2]
        if len(token) > 2:
            stemmed.append(token)
    return stemmed


if __name__ == "__main__":
    sample = "<p>This movie was absolutely amazing! I loved the acting.</p>"
    cleaned = preprocess_review(sample)
    tokens = tokenize_and_stem(cleaned)
    print(f"Original: {sample}")
    print(f"Cleaned: {cleaned}")
    print(f"Tokens: {tokens}")
