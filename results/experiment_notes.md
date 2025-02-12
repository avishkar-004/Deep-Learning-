# Experiment Notes

## Assignment 1 - XOR Problem
- Network: 2-4-1 architecture
- Activation: Sigmoid
- Learning rate: 0.5
- Epochs: 10,000
- Final loss: ~0.001

## Assignment 2 - Iris
- Network: 4-10-3 architecture
- Activation: ReLU + Softmax
- Learning rate: 0.01
- Test accuracy: 100%

## Assignment 2 - MNIST
- Network: 784-128-10 architecture
- Activation: ReLU + Softmax
- Learning rate: 0.01
- Test accuracy: 95.24%

## Assignment 3 - YOLO Object Detection
- Model: YOLOv11n
- Dataset: Stationary Items (Roboflow)
- Epochs: 50
- mAP@50: ~0.85

## Assignment 4 - NLP Text Classification
- Dataset: IMDB Movie Reviews (50K)
- Logistic Regression: 89% accuracy
- Naive Bayes: 85% accuracy
- Feature extraction: TF-IDF + CountVectorizer

## Assignment 5 - LSTM Sequence Modeling
- Time Series: RMSE 0.042 on energy consumption prediction
- Text Generation: Character-level LSTM on Shakespeare corpus
- SMS Spam: 98.2% accuracy on spam detection

## Assignment 6 - Encoder-Decoder with Attention
- Three architectures compared: LSTM, LSTM+Bahdanau, Transformer
- Best BLEU: 0.24 (Transformer)
- Best ROUGE-L: 0.44 (Transformer)
- Attention visualization shows meaningful alignment patterns
