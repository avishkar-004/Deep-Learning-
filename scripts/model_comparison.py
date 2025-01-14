"""Utility for comparing ML model performance."""

import json
import os


class ModelComparison:
    def __init__(self):
        self.results = []

    def add_result(self, model_name, accuracy, precision, recall, f1):
        self.results.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

    def get_best_model(self, metric="accuracy"):
        if not self.results:
            return None
        return max(self.results, key=lambda x: x[metric])

    def summary(self):
        print(f"{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 78)
        for r in self.results:
            print(f"{r['model']:<30} {r['accuracy']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1_score']:<12.4f}")

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)


if __name__ == "__main__":
    comp = ModelComparison()
    comp.add_result("Logistic Regression", 0.89, 0.88, 0.89, 0.88)
    comp.add_result("Naive Bayes", 0.85, 0.86, 0.85, 0.85)
    comp.summary()
    best = comp.get_best_model()
    print(f"\nBest model: {best['model']}")
