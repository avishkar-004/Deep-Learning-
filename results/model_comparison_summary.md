# Model Comparison Summary

## Assignment 4 - NLP Classification
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Logistic Regression (TF-IDF) | 89.2% | 0.89 | 0.89 | 0.89 |
| Logistic Regression (Count) | 87.8% | 0.88 | 0.88 | 0.88 |
| Naive Bayes (TF-IDF) | 85.4% | 0.86 | 0.85 | 0.85 |
| Naive Bayes (Count) | 84.1% | 0.84 | 0.84 | 0.84 |

## Assignment 5 - LSTM
| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Energy Forecasting | LSTM | RMSE | 0.042 |
| Spam Detection | LSTM | Accuracy | 98.2% |
| Text Generation | Char-LSTM | Perplexity | ~2.1 |

## Assignment 6 - Encoder-Decoder
| Model | BLEU | ROUGE-L | METEOR |
|-------|------|---------|--------|
| LSTM (no attention) | 0.12 | 0.31 | 0.28 |
| LSTM + Bahdanau | 0.18 | 0.38 | 0.34 |
| Transformer | 0.24 | 0.44 | 0.41 |
