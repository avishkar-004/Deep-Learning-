"""Helper functions for neural network implementations."""

import numpy as np


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    """Derivative of sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU function."""
    return (x > 0).astype(float)


def softmax(x):
    """Softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    """Compute cross-entropy loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred))


def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)


def initialize_weights(layer_sizes, seed=42):
    """Initialize weights using Xavier initialization."""
    np.random.seed(seed)
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
        w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
        b = np.zeros((1, layer_sizes[i + 1]))
        weights.append(w)
        biases.append(b)
    return weights, biases
