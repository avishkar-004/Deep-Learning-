"""Simple experiment tracking for deep learning assignments."""

import json
import os
from datetime import datetime


class ExperimentTracker:
    def __init__(self, log_file="results/experiments.jsonl"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, assignment, model_name, params, metrics):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "assignment": assignment,
            "model": model_name,
            "params": params,
            "metrics": metrics
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def get_best(self, assignment, metric="accuracy"):
        if not os.path.exists(self.log_file):
            return None
        best = None
        best_val = -float("inf")
        with open(self.log_file) as f:
            for line in f:
                entry = json.loads(line)
                if entry["assignment"] == assignment:
                    val = entry["metrics"].get(metric, 0)
                    if val > best_val:
                        best_val = val
                        best = entry
        return best
