"""Plotting utilities for deep learning experiments."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_loss(losses, title="Training Loss", save_path=None):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_accuracy_comparison(models, accuracies, title="Model Comparison", save_path=None):
    """Plot bar chart comparing model accuracies."""
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    bars = plt.bar(models, accuracies, color=colors)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history_dict, metrics=None, save_path=None):
    """Plot training history with multiple metrics."""
    if metrics is None:
        metrics = list(history_dict.keys())

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in history_dict:
            ax.plot(history_dict[metric], label=metric)
        val_metric = f'val_{metric}'
        if val_metric in history_dict:
            ax.plot(history_dict[val_metric], label=val_metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Training {metric.capitalize()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
