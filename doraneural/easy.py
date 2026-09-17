"""Simplified beginner-friendly API for creating and training neural networks.

Hides boilerplate math and activation wiring behind simple, intuitive one-liners.
"""

from typing import List, Union, Optional, Tuple
import numpy as np

from .model import Sequential, History
from .layers import Dense
from .activations import ReLU, Sigmoid, Softmax
from .losses import BinaryCrossEntropy, CategoricalCrossEntropy, MeanSquaredError
from .optimizers import Adam, SGD, RMSprop
from .utils import one_hot_encode, train_test_split
from .errors import ShapeMismatchError, TargetShapeMismatchError


def create(
    inputs: int,
    hidden: Union[int, List[int]] = 16,
    outputs: int = 1,
    task: str = "auto",
    optimizer: str = "adam",
    lr: float = 0.01,
) -> Sequential:
    """Create a fully compiled neural network in a single line.

    Automatically connects Dense layers, adds ReLU activations between hidden layers,
    and configures the correct output activation (Sigmoid or Softmax) and loss function.

    Args:
        inputs (int): Number of input features per sample.
        hidden (Union[int, List[int]]): Hidden layer sizes, e.g. 16 or [32, 16].
        outputs (int): Number of output units (1 for binary, >1 for multiclass).
        task (str): 'auto', 'binary', or 'multiclass'.
        optimizer (str): 'adam', 'sgd', or 'rmsprop'.
        lr (float): Learning rate.

    Returns:
        Sequential: A fully configured and compiled neural network.

    Example:
        >>> import doraneural as dn
        >>> model = dn.create(inputs=4, hidden=[16, 8], outputs=3)
        >>> model.fit(X_train, y_train)
    """
    if inputs <= 0 or outputs <= 0:
        raise ValueError("inputs and outputs must be positive integers.")

    hidden_sizes: List[int] = [hidden] if isinstance(hidden, int) else list(hidden)

    # Determine task mode: 'regression', 'binary', or 'multiclass'
    task_clean = task.lower().strip()
    if task_clean in ("regression", "reg"):
        task_mode = "regression"
    elif task_clean in ("binary", "bin"):
        task_mode = "binary"
    elif task_clean in ("multiclass", "multi"):
        task_mode = "multiclass"
    else:
        # Auto-detect from output count
        task_mode = "binary" if outputs == 1 else "multiclass"

    layers = []
    prev_dim = inputs

    # Add hidden layers with ReLU activations
    for h_size in hidden_sizes:
        if h_size <= 0:
            raise ValueError(f"Hidden layer size must be > 0, got {h_size}")
        layers.append(Dense(in_features=prev_dim, out_features=h_size, weight_init="he"))
        layers.append(ReLU())
        prev_dim = h_size

    # Configure output layer and loss function based on task
    if task_mode == "regression":
        # Pure linear output layer (no squashing activation)
        layers.append(Dense(in_features=prev_dim, out_features=outputs, weight_init="xavier"))
        loss_fn = MeanSquaredError()
        metrics = ["mse"]
    elif task_mode == "binary":
        layers.append(Dense(in_features=prev_dim, out_features=1, weight_init="xavier"))
        layers.append(Sigmoid())
        loss_fn = BinaryCrossEntropy()
        metrics = ["accuracy"]
    else:
        layers.append(Dense(in_features=prev_dim, out_features=outputs, weight_init="xavier"))
        layers.append(Softmax())
        loss_fn = CategoricalCrossEntropy()
        metrics = ["accuracy"]

    model = Sequential(layers)

    # Configure optimizer
    opt_name = optimizer.lower().strip()
    if opt_name == "adam":
        opt = Adam(lr=lr)
    elif opt_name in ("sgd", "momentum"):
        opt = SGD(lr=lr, momentum=0.9)
    elif opt_name == "rmsprop":
        opt = RMSprop(lr=lr)
    else:
        opt = Adam(lr=lr)

    model.compile(loss=loss_fn, optimizer=opt, metrics=metrics)
    return model


def quick_train(
    X: np.ndarray,
    y: np.ndarray,
    hidden: Union[int, List[int]] = (16, 8),
    epochs: int = 25,
    lr: float = 0.01,
    test_size: float = 0.2,
    verbose: int = 1,
) -> Tuple[Sequential, History]:
    """Inspect data, construct the ideal network, train it, and return the model.

    Args:
        X (np.ndarray): Training inputs, shape (samples, features).
        y (np.ndarray): Target labels (classes or 1/0).
        hidden (Union[int, List[int]]): Hidden layer sizes.
        epochs (int): Training epochs.
        lr (float): Learning rate.
        test_size (float): Proportion of data to hold out for evaluation.
        verbose (int): Verbosity level (1 = print progress, 0 = silent).

    Returns:
        Tuple[Sequential, History]: Trained model and recorded history.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y)

    if X_arr.ndim != 2:
        raise ValueError(f"X must be a 2D array (samples, features), got shape {X_arr.shape}")

    n_inputs = X_arr.shape[1]

    # Detect task and classes
    if y_arr.ndim == 2 and y_arr.shape[1] > 1:
        # Already one-hot
        n_outputs = y_arr.shape[1]
        task = "multiclass"
        y_proc = y_arr
    else:
        unique_vals = np.unique(y_arr)
        if len(unique_vals) <= 2:
            n_outputs = 1
            task = "binary"
            y_proc = y_arr.reshape(-1, 1).astype(np.float32)
        else:
            n_outputs = len(unique_vals)
            task = "multiclass"
            y_proc = one_hot_encode(y_arr, num_classes=n_outputs)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_proc, test_size=test_size, shuffle=True
    )

    if verbose:
        print(f"🚀 doraneural Quick-Train: Detected {n_inputs} inputs, {n_outputs} outputs ({task}).")
        print(f"   Splitting {len(X_arr)} samples into {len(X_train)} train / {len(X_test)} test.")

    model = create(inputs=n_inputs, hidden=hidden, outputs=n_outputs, task=task, lr=lr)

    if verbose:
        model.summary()

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        verbose=verbose,
        validation_data=(X_test, y_test),
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    if verbose:
        print(f"\n✨ Training Complete! Final Test Accuracy: {test_acc * 100:.2f}% (Loss: {test_loss:.4f})")

    return model, history
