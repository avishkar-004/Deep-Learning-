"""Simple TF-IDF vectorizer implementation."""

import numpy as np
from collections import Counter
import math


class SimpleTFIDF:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf = {}

    def fit(self, documents):
        """Fit TF-IDF on document collection."""
        doc_freq = Counter()
        n_docs = len(documents)

        for doc in documents:
            unique_words = set(doc.split())
            for word in unique_words:
                doc_freq[word] += 1

        # Select top features by document frequency
        top_words = sorted(doc_freq.items(), key=lambda x: -x[1])[:self.max_features]
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(top_words)}

        # Compute IDF
        for word, freq in doc_freq.items():
            if word in self.vocabulary:
                self.idf[word] = math.log(n_docs / (1 + freq))

        return self

    def transform(self, documents):
        """Transform documents to TF-IDF vectors."""
        matrix = np.zeros((len(documents), len(self.vocabulary)))

        for i, doc in enumerate(documents):
            word_counts = Counter(doc.split())
            total_words = sum(word_counts.values())

            for word, count in word_counts.items():
                if word in self.vocabulary:
                    tf = count / total_words
                    idf = self.idf.get(word, 0)
                    matrix[i, self.vocabulary[word]] = tf * idf

        return matrix

    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)


if __name__ == "__main__":
    docs = [
        "the cat sat on the mat",
        "the dog chased the cat",
        "the mat was on the floor"
    ]
    vectorizer = SimpleTFIDF(max_features=100)
    matrix = vectorizer.fit_transform(docs)
    print(f"TF-IDF matrix shape: {matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary)}")
