"""Text preprocessing utilities for NLP assignments."""

import re
import string


def clean_text(text):
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(tokens, stopwords_set):
    """Remove stopwords from token list."""
    return [t for t in tokens if t not in stopwords_set]


def simple_tokenize(text):
    """Simple whitespace tokenizer."""
    return text.lower().split()
