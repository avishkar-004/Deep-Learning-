# Performance Benchmarks

## Training Times (Google Colab T4 GPU)

| Assignment | Model | Epochs | Training Time |
|-----------|-------|--------|---------------|
| 1 - XOR | Custom NN | 10,000 | ~2s (CPU) |
| 2 - MNIST | Custom NN | 50 | ~30s (CPU) |
| 3 - YOLO | YOLOv11n | 50 | ~15 min (GPU) |
| 4 - NLP | LogReg | N/A | ~5s (CPU) |
| 5 - LSTM | LSTM-128 | 50 | ~10 min (GPU) |
| 6 - Seq2Seq | Transformer | 30 | ~20 min (GPU) |

## Memory Usage

- Assignment 3 (YOLO): ~4GB GPU memory
- Assignment 5 (LSTM): ~2GB GPU memory
- Assignment 6 (Transformer): ~3GB GPU memory
