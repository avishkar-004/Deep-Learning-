"""Custom training callbacks for Keras models."""

import tensorflow as tf
import time


class TrainingTimer(tf.keras.callbacks.Callback):
    """Track training time per epoch."""

    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start
        self.epoch_times.append(elapsed)
        print(f"  Epoch {epoch + 1} time: {elapsed:.2f}s")


class MetricsLogger(tf.keras.callbacks.Callback):
    """Log metrics to a file."""

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        with open(self.filepath, 'a') as f:
            f.write(f"Epoch {epoch + 1}: {logs}\n")
