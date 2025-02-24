"""Simple tokenizer utilities for NLP tasks."""

import re
from collections import Counter


class SimpleTokenizer:
    def __init__(self, num_words=None):
        self.num_words = num_words
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts):
        word_counts = Counter()
        for text in texts:
            tokens = re.findall(r'\b\w+\b', text.lower())
            word_counts.update(tokens)

        most_common = word_counts.most_common(self.num_words)
        self.word_index = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}
        self.index_word = {idx: word for word, idx in self.word_index.items()}

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            tokens = re.findall(r'\b\w+\b', text.lower())
            seq = [self.word_index.get(t, 0) for t in tokens]
            sequences.append(seq)
        return sequences

    @property
    def vocab_size(self):
        return len(self.word_index) + 1
