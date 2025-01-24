"""Evaluation metrics for text generation tasks."""

import numpy as np
from collections import Counter


def compute_bleu_score(reference, hypothesis, max_n=4):
    """Compute BLEU score between reference and hypothesis."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])

        matches = sum((hyp_ngrams & ref_ngrams).values())
        total = max(sum(hyp_ngrams.values()), 1)
        scores.append(matches / total)

    if 0 in scores:
        return 0.0

    # Geometric mean
    log_avg = sum(np.log(s) for s in scores) / len(scores)

    # Brevity penalty
    bp = min(1, np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    return bp * np.exp(log_avg)


if __name__ == "__main__":
    ref = "the cat sat on the mat"
    hyp = "the cat is on the mat"
    score = compute_bleu_score(ref, hyp)
    print(f"BLEU score: {score:.4f}")
