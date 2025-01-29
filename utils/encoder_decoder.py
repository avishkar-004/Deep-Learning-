"""Base encoder-decoder architecture components."""

import tensorflow as tf
from tensorflow.keras import layers, Model


class Encoder(Model):
    """LSTM-based encoder."""

    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(hidden_units, return_sequences=True, return_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x)
        return output, state_h, state_c


class Decoder(Model):
    """LSTM-based decoder with optional attention."""

    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(hidden_units, return_sequences=True, return_state=True)
        self.fc = layers.Dense(vocab_size)

    def call(self, x, initial_state):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=initial_state)
        output = self.fc(output)
        return output, state_h, state_c
