"""Learning rate finder for optimal LR selection."""

import numpy as np


def lr_range_test(model, train_data, start_lr=1e-7, end_lr=10, num_steps=100):
    """Run learning rate range test."""
    lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_steps)
    losses = []
    
    for lr in lrs:
        # Simulate training step with learning rate
        losses.append(np.random.exponential(1.0 / (lr * 100 + 0.01)))
    
    return lrs, losses


if __name__ == "__main__":
    lrs, losses = lr_range_test(None, None)
    print(f"LR range: {lrs[0]:.2e} to {lrs[-1]:.2e}")
    print(f"Best LR (lowest loss): {lrs[np.argmin(losses)]:.2e}")
