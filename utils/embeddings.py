"""Word embedding utilities."""

import numpy as np


def load_glove_embeddings(filepath, embedding_dim=100):
    """Load GloVe pretrained embeddings."""
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) == embedding_dim:
                embeddings[word] = vector
    return embeddings


def create_embedding_matrix(word_index, embeddings, embedding_dim=100):
    """Create embedding matrix from pretrained embeddings."""
    matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, idx in word_index.items():
        if word in embeddings:
            matrix[idx] = embeddings[word]
    coverage = sum(1 for w in word_index if w in embeddings) / len(word_index)
    print(f"Embedding coverage: {coverage:.2%}")
    return matrix
