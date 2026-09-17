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
from .losses import (
    Loss,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    MeanSquaredError,
    MSELoss,
)
from .optimizers import Optimizer, SGD, Adam, RMSprop
from .metrics import Metric, Accuracy, accuracy_score, MSE, MAE, get_metric
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
from .data import load_csv, create_sample_classification_csv, create_sample_regression_csv
from .plot import plot_ascii_curve, plot_history
from .export import export_to_standalone_python
from .errors import DoraneuralError, ShapeMismatchError, ModelNotCompiledError

__version__ = "1.0.0"

__all__ = [
    # High-level beginner one-liners
    "create",
    "quick_train",
    "explain",
    # Data & Preprocessing
    "load_csv",
    "create_sample_classification_csv",
    "create_sample_regression_csv",
    # Visual ASCII plotting
    "plot_ascii_curve",
    "plot_history",
    # Zero-dependency Exporter
    "export_to_standalone_python",
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
    "MeanSquaredError",
    "MSELoss",
    # Optimizers
    "Optimizer",
    "SGD",
    "Adam",
    "RMSprop",
    # Metrics
    "Metric",
    "Accuracy",
    "accuracy_score",
    "MSE",
    "MAE",
    "get_metric",
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
