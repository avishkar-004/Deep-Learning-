# Deep Learning Lab Assignments

A collection of deep learning lab assignments covering fundamental to advanced topics including feedforward neural networks, CNNs, NLP text classification, YOLO object detection, LSTM-based sequence modeling, and encoder-decoder architectures with attention mechanisms.

## Assignments Overview

### Assignment 1 - Feedforward Neural Network from Scratch
- Implemented a feedforward neural network using only NumPy
- Solved the XOR problem with sigmoid activation and backpropagation
- Visualized training loss convergence over 10,000 epochs

### Assignment 2 - Neural Network on Iris and MNIST Datasets
- **Iris Dataset:** Built a neural network from scratch with ReLU activation and softmax output, achieving 100% test accuracy
- **MNIST Dataset:** Scaled the same architecture to 784-dimensional input with 128 hidden neurons, achieving 95.24% test accuracy
- Used cross-entropy loss and gradient descent optimization

### Assignment 3 - YOLOv11 Object Detection
- Trained YOLOv8/YOLOv11 on a custom Stationary Items dataset from Roboflow
- Performed dataset preparation, model training (50 epochs), and inference
- Evaluated with mAP, precision, recall, and F1 score per class
- Visualized training loss curves and per-class evaluation metrics

### Assignment 4 - NLP Text Classification
- Preprocessed the IMDB movie reviews dataset (50,000 reviews)
- Applied tokenization, stopword removal, lemmatization
- Implemented TF-IDF and CountVectorizer feature extraction
- Compared Logistic Regression and Naive Bayes classifiers
- Achieved up to 89% accuracy with confusion matrix visualization

### Assignment 5 - LSTM-Based Sequence Modeling
- **Time Series Forecasting:** Predicted energy consumption using LSTM on smart home data
- **Text Generation:** Trained character-level LSTM on Shakespeare text corpus
- **Text Classification:** Built LSTM model for SMS spam detection with 98%+ accuracy

### Assignment 6 - Encoder-Decoder with Attention Mechanisms
- Implemented paraphrase generation using three architectures:
  - LSTM Encoder-Decoder (no attention)
  - LSTM with Bahdanau Attention
  - Transformer with Self-Attention
- Evaluated using BLEU, ROUGE-L, and METEOR scores
- Compared all three approaches with detailed performance analysis

## Tech Stack

- **Languages:** Python 3.x
- **Deep Learning:** TensorFlow, Keras, PyTorch
- **Computer Vision:** OpenCV, Ultralytics (YOLO)
- **NLP:** NLTK, scikit-learn
- **Data Science:** NumPy, Pandas, Matplotlib, Seaborn
- **Platform:** Google Colab

## Project Structure

```
Deep-Learning-/
├── Assignments/
│   ├── Avishkar_Pawar_DeepLearningLabAssignment.ipynb    # Assignment 1
│   ├── Avishkar_Pawar_DeepLearningLabAssignment_2.ipynb  # Assignment 2
│   ├── Avishkar_Pawar_DeepLearningLabAssignment_3.ipynb  # Assignment 3
│   ├── Avishkar_Pawar_DeepLearningLabAssignment_4.ipynb  # Assignment 4
│   ├── Avishkar_Pawar_DeepLearningLabAssignment_5.ipynb  # Assignment 5
│   └── Avishkar_Pawar_DeepLearningLabAssignment_6.ipynb  # Assignment 6
├── scripts/                # Utility scripts
├── utils/                  # Helper modules
├── results/                # Experiment results
├── requirements.txt
├── Makefile
└── README.md
```

## How to Run

1. Open any notebook in [Google Colab](https://colab.research.google.com/)
2. Upload the required datasets (links provided within each notebook)
3. Run all cells sequentially

## Local Setup

```bash
git clone https://github.com/avishkar-004/Deep-Learning-.git
cd Deep-Learning-
pip install -r requirements.txt
```

## Author

Avishkar Pawar (202201040040)
