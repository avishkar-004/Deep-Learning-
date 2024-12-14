"""Evaluation metrics for deep learning models."""

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)


def compute_metrics(y_true, y_pred, class_names=None):
    """Compute comprehensive classification metrics."""
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true, y_pred, target_names=class_names
    )

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }


def print_metrics(metrics):
    """Pretty print evaluation metrics."""
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"\n{metrics['classification_report']}")
