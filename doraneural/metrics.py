"""Evaluation metrics for classification tasks.

Provides accuracy calculation for both binary and multi-class classification.
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the metric score.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted probabilities or class labels.

        Returns:
            float: Metric score.
        """
        pass

    @property
    def name(self) -> str:
        """Name of the metric."""
        return self.__class__.__name__.lower()


class Accuracy(Metric):
    """Classification accuracy metric.

    Computes the proportion of correct predictions:
        Accuracy = (Number of Correct Predictions) / (Total Predictions)

    Supports both binary classification (probabilities or 0/1 labels) and
    multiclass classification (one-hot matrices or class indices).
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate classification accuracy.

        Args:
            y_true (np.ndarray): Ground truth labels. Shape (N,), (N, 1), or (N, C) one-hot.
            y_pred (np.ndarray): Predictions. Shape (N,), (N, 1), or (N, C) probabilities.

        Returns:
            float: Fraction of accurately predicted samples in range [0.0, 1.0].
        """
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)

        # Handle ground truth labels
        if yt.ndim == 2 and yt.shape[1] > 1:
            # One-hot encoded ground truth
            true_classes = np.argmax(yt, axis=-1)
        else:
            true_classes = yt.ravel().astype(np.int64)

        # Handle prediction labels
        if yp.ndim == 2 and yp.shape[1] > 1:
            # Multiclass probabilities
            pred_classes = np.argmax(yp, axis=-1)
        elif yp.ndim == 2 and yp.shape[1] == 1:
            # Binary probability or binary label
            if np.issubdtype(yp.dtype, np.floating) and (yp.min() >= 0.0 and yp.max() <= 1.0):
                pred_classes = (yp.ravel() >= 0.5).astype(np.int64)
            else:
                pred_classes = yp.ravel().astype(np.int64)
        else:
            # 1D array
            if np.issubdtype(yp.dtype, np.floating) and (yp.min() >= 0.0 and yp.max() <= 1.0):
                pred_classes = (yp.ravel() >= 0.5).astype(np.int64)
            else:
                pred_classes = yp.ravel().astype(np.int64)

        if len(true_classes) != len(pred_classes):
            raise ValueError(
                f"Sample count mismatch in Accuracy: y_true has {len(true_classes)} samples, "
                f"y_pred has {len(pred_classes)} samples."
            )

        return float(np.mean(true_classes == pred_classes))


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Helper function to compute classification accuracy directly.

    Args:
        y_true (np.ndarray): Target ground-truth labels.
        y_pred (np.ndarray): Predicted labels or probabilities.

    Returns:
        float: Accuracy score in [0.0, 1.0].
    """
    return Accuracy()(y_true, y_pred)


class MSE(Metric):
    """Mean Squared Error metric for regression."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yp_raw = np.asarray(y_pred)
        dtype = yp_raw.dtype if np.issubdtype(yp_raw.dtype, np.floating) else np.dtype(np.float32)
        yt = np.asarray(y_true, dtype=dtype)
        yp = np.asarray(y_pred, dtype=dtype)
        return float(np.mean((yt - yp) ** 2))

    @property
    def name(self) -> str:
        return "mse"


class MAE(Metric):
    """Mean Absolute Error metric for regression."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yp_raw = np.asarray(y_pred)
        dtype = yp_raw.dtype if np.issubdtype(yp_raw.dtype, np.floating) else np.dtype(np.float32)
        yt = np.asarray(y_true, dtype=dtype)
        yp = np.asarray(y_pred, dtype=dtype)
        return float(np.mean(np.abs(yt - yp)))

    @property
    def name(self) -> str:
        return "mae"


def get_metric(metric: Union[str, Metric]) -> Metric:
    """Resolve a metric name or instance to a Metric object."""
    if isinstance(metric, Metric):
        return metric
    if isinstance(metric, str):
        normalized = metric.strip().lower()
        if normalized in ("accuracy", "acc"):
            return Accuracy()
        elif normalized in ("mse", "mean_squared_error"):
            return MSE()
        elif normalized in ("mae", "mean_absolute_error"):
            return MAE()
        raise ValueError(f"Unknown metric name: '{metric}'. Supported: 'accuracy', 'mse', 'mae'.")
    raise TypeError(f"Expected str or Metric instance, got {type(metric)}")
