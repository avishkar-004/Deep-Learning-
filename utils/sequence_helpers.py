"""Helper functions for sequence modeling tasks."""

import numpy as np


def create_time_series_sequences(data, seq_length, target_col=None):
    """Create sequences for time series forecasting."""
    if target_col is not None:
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, target_col])
        return np.array(X), np.array(y)
    else:
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)


def char_to_index(text):
    """Create character to index mapping."""
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    return char2idx, idx2char


def encode_text(text, char2idx, seq_length):
    """Encode text into sequences for character-level modeling."""
    X, y = [], []
    for i in range(len(text) - seq_length):
        seq = [char2idx[c] for c in text[i:i + seq_length]]
        target = char2idx[text[i + seq_length]]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)


def generate_text(model, seed_text, char2idx, idx2char, length=100, temperature=1.0):
    """Generate text using trained character-level model."""
    import tensorflow as tf

    generated = seed_text
    for _ in range(length):
        encoded = [char2idx.get(c, 0) for c in generated[-len(seed_text):]]
        input_seq = np.array(encoded).reshape(1, -1)
        pred = model.predict(input_seq, verbose=0)[0]

        # Apply temperature
        pred = np.log(pred + 1e-10) / temperature
        pred = np.exp(pred) / np.sum(np.exp(pred))

        next_idx = np.random.choice(len(pred), p=pred)
        generated += idx2char[next_idx]

    return generated
