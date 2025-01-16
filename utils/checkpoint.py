"""Model checkpointing utilities."""

import os
import json


def save_checkpoint(model_state, metrics, filepath):
    """Save model checkpoint with metadata."""
    metadata = {
        'metrics': metrics,
        'filepath': filepath
    }
    meta_path = filepath + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Checkpoint saved: {filepath}")


def load_best_checkpoint(checkpoint_dir, metric='val_accuracy'):
    """Load the best checkpoint from a directory."""
    best = None
    best_val = -float('inf')
    
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.meta.json'):
            with open(os.path.join(checkpoint_dir, f)) as fh:
                meta = json.load(fh)
                val = meta['metrics'].get(metric, 0)
                if val > best_val:
                    best_val = val
                    best = meta
    return best
