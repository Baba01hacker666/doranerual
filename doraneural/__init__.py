"""doraneural: A lightweight, modular neural network library & CLI built purely with NumPy.

Designed for educational hackability and low-power CPU environments.
"""

from .base import Layer
from .layers import (
    Dense,
    Dropout,
    LayerNorm,
    Flatten,
    Conv2D,
    MaxPool2D,
)
from .activations import ReLU, Sigmoid, Softmax
from .losses import Loss, BinaryCrossEntropy, CategoricalCrossEntropy
from .optimizers import Optimizer, SGD, Adam, RMSprop
from .metrics import Metric, Accuracy, accuracy_score
from .model import Sequential, History, EvaluationResult
from .serialization import save_model, load_model
from .utils import (
    set_seed,
    train_test_split,
    one_hot_encode,
    to_categorical,
    batch_iterator,
    make_moons,
    make_blobs,
    make_digits,
)
from .easy import create, quick_train
from .teach import explain
from .errors import DoraneuralError, ShapeMismatchError, ModelNotCompiledError

__version__ = "1.0.0"

__all__ = [
    # High-level beginner one-liners
    "create",
    "quick_train",
    "explain",
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
    # Errors
    "DoraneuralError",
    "ShapeMismatchError",
    "ModelNotCompiledError",
]
