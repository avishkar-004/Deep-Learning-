"""Regularization techniques reference."""

import numpy as np


def l1_regularization(weights, lambda_val=0.01):
    """Compute L1 regularization term."""
    return lambda_val * sum(np.sum(np.abs(w)) for w in weights)


def l2_regularization(weights, lambda_val=0.01):
    """Compute L2 regularization term."""
    return lambda_val * sum(np.sum(w ** 2) for w in weights)


def dropout_mask(shape, keep_prob=0.8):
    """Generate dropout mask."""
    return (np.random.rand(*shape) < keep_prob).astype(np.float32) / keep_prob
