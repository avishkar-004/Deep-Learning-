"""Attention mechanism implementations for encoder-decoder models."""

import tensorflow as tf
from tensorflow.keras import layers


class BahdanauAttention(layers.Layer):
    """Bahdanau (additive) attention mechanism."""

    def __init__(self, units):
        super().__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class ScaledDotProductAttention(layers.Layer):
    """Scaled dot-product attention for transformers."""

    def call(self, query, key, value, mask=None):
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
        scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(d_k)
        if mask is not None:
            scores += (mask * -1e9)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights
