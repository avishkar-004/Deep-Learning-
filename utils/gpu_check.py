"""GPU availability check utility."""

import os


def check_gpu():
    """Check and print GPU availability."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow GPUs: {len(gpus)}")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        else:
            print("TensorFlow: No GPU detected, using CPU")
    except Exception as e:
        print(f"TensorFlow error: {e}")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("PyTorch: No CUDA GPU detected")
    except Exception as e:
        print(f"PyTorch error: {e}")


if __name__ == "__main__":
    check_gpu()
