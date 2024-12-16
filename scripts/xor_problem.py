"""Solve XOR problem with feedforward neural network from scratch."""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize weights
np.random.seed(42)
w_hidden = np.random.randn(2, 4)
b_hidden = np.zeros((1, 4))
w_output = np.random.randn(4, 1)
b_output = np.zeros((1, 1))

learning_rate = 0.5
epochs = 10000
losses = []

for epoch in range(epochs):
    # Forward pass
    hidden = sigmoid(X @ w_hidden + b_hidden)
    output = sigmoid(hidden @ w_output + b_output)

    # Loss
    loss = np.mean((y - output) ** 2)
    losses.append(loss)

    # Backward pass
    d_output = (y - output) * sigmoid_derivative(output)
    d_hidden = d_output @ w_output.T * sigmoid_derivative(hidden)

    # Update weights
    w_output += hidden.T @ d_output * learning_rate
    b_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    w_hidden += X.T @ d_hidden * learning_rate
    b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

print("Final predictions:")
for i in range(len(X)):
    hidden = sigmoid(X[i:i+1] @ w_hidden + b_hidden)
    output = sigmoid(hidden @ w_output + b_output)
    print(f"  {X[i]} -> {output[0][0]:.4f} (expected: {y[i][0]})")

print(f"\nFinal loss: {losses[-1]:.6f}")
