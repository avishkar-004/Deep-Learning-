"""MNIST data preprocessing utilities."""

import numpy as np


def load_and_preprocess_mnist():
    """Load and preprocess MNIST dataset."""
    from tensorflow.keras.datasets import mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten images
    X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0

    # One-hot encode labels
    n_classes = 10
    y_train_oh = np.zeros((y_train.shape[0], n_classes))
    y_train_oh[np.arange(y_train.shape[0]), y_train] = 1
    y_test_oh = np.zeros((y_test.shape[0], n_classes))
    y_test_oh[np.arange(y_test.shape[0]), y_test] = 1

    return (X_train, y_train_oh), (X_test, y_test_oh)


def load_and_preprocess_iris():
    """Load and preprocess Iris dataset."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # One-hot encode
    n_classes = 3
    y_oh = np.zeros((y.shape[0], n_classes))
    y_oh[np.arange(y.shape[0]), y] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_oh, test_size=0.2, random_state=42
    )

    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    print("Loading MNIST...")
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_mnist()
    print(f"MNIST Train: {X_train.shape}, Test: {X_test.shape}")

    print("\nLoading Iris...")
    (X_train_i, y_train_i), (X_test_i, y_test_i) = load_and_preprocess_iris()
    print(f"Iris Train: {X_train_i.shape}, Test: {X_test_i.shape}")
