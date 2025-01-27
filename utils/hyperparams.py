"""Hyperparameter configuration templates for each assignment."""

ASSIGNMENT_CONFIGS = {
    "assignment_1": {
        "hidden_size": 4,
        "learning_rate": 0.5,
        "epochs": 10000,
        "activation": "sigmoid"
    },
    "assignment_2_iris": {
        "hidden_size": 10,
        "learning_rate": 0.01,
        "epochs": 1000,
        "activation": "relu"
    },
    "assignment_2_mnist": {
        "hidden_size": 128,
        "learning_rate": 0.01,
        "epochs": 50,
        "activation": "relu"
    },
    "assignment_3_yolo": {
        "model": "yolov11n",
        "epochs": 50,
        "img_size": 640,
        "batch_size": 16
    },
    "assignment_5_lstm": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001
    }
}


def get_config(assignment_name):
    return ASSIGNMENT_CONFIGS.get(assignment_name, {})
