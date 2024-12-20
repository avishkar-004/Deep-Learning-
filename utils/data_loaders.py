"""Common data loading utilities for assignments."""

import numpy as np
import os


def load_text_data(filepath, encoding='utf-8'):
    """Load text data from file."""
    with open(filepath, 'r', encoding=encoding) as f:
        return f.read()


def create_sequences(data, seq_length):
    """Create input-output sequences for sequence modeling."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_test_split_sequential(X, y, test_ratio=0.2):
    """Split sequential data maintaining temporal order."""
    split_idx = int(len(X) * (1 - test_ratio))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
