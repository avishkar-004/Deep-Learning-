"""Visualize results from deep learning experiments."""

import matplotlib.pyplot as plt
import numpy as np


def plot_per_class_metrics(class_names, precision, recall, f1, save_path=None):
    """Plot per-class metrics as grouped bar chart."""
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#4CAF50')
    ax.bar(x, recall, width, label='Recall', color='#2196F3')
    ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Classification Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_attention_heatmap(attention_weights, src_tokens, tgt_tokens, save_path=None):
    """Plot attention weights as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_weights, cmap='viridis')
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha='right')
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens)
    ax.set_xlabel('Source')
    ax.set_ylabel('Target')
    ax.set_title('Attention Weights')
    plt.colorbar(im)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    print("Visualization utilities loaded successfully.")
