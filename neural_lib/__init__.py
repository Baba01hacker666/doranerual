"""neural_lib: A lightweight, modular neural network library built purely with NumPy.

Designed for educational hackability and low-power CPU environments.
"""

from neural_lib.base import Layer
from neural_lib.layers import (
    Dense,
    Dropout,
    LayerNorm,
    Flatten,
    Conv2D,
    MaxPool2D,
)
from neural_lib.activations import ReLU, Sigmoid, Softmax
from neural_lib.losses import Loss, BinaryCrossEntropy, CategoricalCrossEntropy
from neural_lib.optimizers import Optimizer, SGD, Adam, RMSprop
from neural_lib.metrics import Metric, Accuracy, accuracy_score
from neural_lib.model import Sequential, History, EvaluationResult
from neural_lib.serialization import save_model, load_model
from neural_lib.utils import (
    set_seed,
    train_test_split,
    one_hot_encode,
    to_categorical,
    batch_iterator,
    make_moons,
    make_blobs,
    make_digits,
)

__version__ = "0.2.0"

__all__ = [
    # Base
    "Layer",
    # Layers
    "Dense",
    "Dropout",
    "LayerNorm",
    "Flatten",
    "Conv2D",
    "MaxPool2D",
    # Activations
    "ReLU",
    "Sigmoid",
    "Softmax",
    # Losses
    "Loss",
    "BinaryCrossEntropy",
    "CategoricalCrossEntropy",
    # Optimizers
    "Optimizer",
    "SGD",
    "Adam",
    "RMSprop",
    # Metrics
    "Metric",
    "Accuracy",
    "accuracy_score",
    # Model & History
    "Sequential",
    "History",
    "EvaluationResult",
    # Serialization
    "save_model",
    "load_model",
    # Utilities
    "set_seed",
    "train_test_split",
    "one_hot_encode",
    "to_categorical",
    "batch_iterator",
    "make_moons",
    "make_blobs",
    "make_digits",
]
