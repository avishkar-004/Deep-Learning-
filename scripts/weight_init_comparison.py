"""Compare different weight initialization strategies."""

import numpy as np


def compare_initializations(layer_sizes):
    """Compare Xavier, He, and random initialization."""
    results = {}
    
    for name, init_fn in [
        ("Random Normal", lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * 0.01),
        ("Xavier", lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))),
        ("He", lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)),
    ]:
        weights = []
        for i in range(len(layer_sizes) - 1):
            w = init_fn(layer_sizes[i], layer_sizes[i + 1])
            weights.append(w)
        
        results[name] = {
            "mean": np.mean([np.mean(w) for w in weights]),
            "std": np.mean([np.std(w) for w in weights])
        }
        print(f"{name}: mean={results[name]['mean']:.6f}, std={results[name]['std']:.6f}")
    
    return results


if __name__ == "__main__":
    compare_initializations([784, 128, 64, 10])
