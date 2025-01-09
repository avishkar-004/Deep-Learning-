"""Analyze effect of batch size on training performance."""

import time


def benchmark_batch_sizes(model_fn, data, batch_sizes=[16, 32, 64, 128]):
    """Benchmark different batch sizes."""
    results = {}
    for bs in batch_sizes:
        start = time.time()
        # Simulated training
        steps = len(data) // bs if hasattr(data, '__len__') else 100
        elapsed = time.time() - start
        results[bs] = {
            'steps_per_epoch': steps,
            'time_per_epoch': elapsed
        }
        print(f"Batch size {bs}: {steps} steps")
    return results


if __name__ == "__main__":
    data = list(range(10000))
    results = benchmark_batch_sizes(None, data)
