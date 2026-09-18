"""Loss functions for training neural networks.

Includes Binary Cross-Entropy, Categorical Cross-Entropy, and Mean Squared Error.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


def _as_float_array(value: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Convert values to floating point without discarding float64 precision."""
    arr = np.asarray(value, dtype=dtype)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    return arr


def _loss_pair(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    yp = _as_float_array(y_pred)
    yt = _as_float_array(y_true, dtype=yp.dtype)
    return yp, yt


class Loss(ABC):
    """Abstract base class for loss functions.

    Calculates scalar loss value during the forward pass and upstream gradients
    with respect to network predictions during backpropagation.
    """

    def __init__(self) -> None:
        self._y_pred_cache: Optional[np.ndarray] = None
        self._y_true_cache: Optional[np.ndarray] = None

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate the scalar loss value.

        Args:
            y_pred (np.ndarray): Model predictions (e.g. probabilities).
            y_true (np.ndarray): Target ground-truth values.

        Returns:
            float: Mean loss across all samples in batch.
        """
        pass

    @abstractmethod
    def backward(
        self, y_pred: Optional[np.ndarray] = None, y_true: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute the derivative of the loss with respect to predictions dL/dy_pred.

        Args:
            y_pred (Optional[np.ndarray]): Model predictions. If None, uses cached predictions.
            y_true (Optional[np.ndarray]): Ground truth. If None, uses cached targets.

        Returns:
            np.ndarray: Gradient tensor matching the shape of y_pred, normalized by batch size.
        """
        pass

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Shorthand alias for forward()."""
        return self.forward(y_pred, y_true)


class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy loss for binary classification.

    Loss formula:
        L = - (1/N) * sum( y * log(p) + (1 - y) * log(1 - p) )

    Gradient with respect to p (y_pred):
        dL/dp = (1/N) * ( (p - y) / (p * (1 - p)) )

    Attributes:
        eps (float): Small epsilon for numerical clipping to prevent log(0) or division by zero.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps: float = float(eps)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate binary cross-entropy loss.

        Args:
            y_pred (np.ndarray): Predicted probabilities in [0, 1], shape (batch_size, 1) or (batch_size,).
            y_true (np.ndarray): Binary targets {0, 1}, matching batch size.

        Returns:
            float: Scalar mean binary cross-entropy loss.
        """
        yp, yt = _loss_pair(y_pred, y_true)

        if yp.ndim == 1:
            yp = yp.reshape(-1, 1)
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)

        if yp.shape != yt.shape:
            raise ValueError(
                f"Shape mismatch in BinaryCrossEntropy: y_pred shape {yp.shape} vs y_true shape {yt.shape}."
            )

        self._y_pred_cache = yp
        self._y_true_cache = yt

        # Numerically stable clipping
        yp_clipped = np.clip(yp, self.eps, 1.0 - self.eps)
        sample_loss = -(yt * np.log(yp_clipped) + (1.0 - yt) * np.log(1.0 - yp_clipped))
        return float(np.mean(sample_loss))

    def backward(
        self, y_pred: Optional[np.ndarray] = None, y_true: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate gradient dL/dy_pred.

        Returns:
            np.ndarray: Upstream gradient dL/dy_pred with shape (batch_size, 1).
        """
        yp = (
            _as_float_array(y_pred)
            if y_pred is not None
            else self._y_pred_cache
        )
        yt = (
            _as_float_array(y_true, dtype=yp.dtype)
            if y_true is not None
            else self._y_true_cache
        )

        if yp is None or yt is None:
            raise RuntimeError("BinaryCrossEntropy.backward called before forward pass.")

        if yp.ndim == 1:
            yp = yp.reshape(-1, 1)
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)

        batch_size = yp.shape[0]
        yp_clipped = np.clip(yp, self.eps, 1.0 - self.eps)

        # Vectorized gradient normalized by sample count
        grad = ((yp_clipped - yt) / (yp_clipped * (1.0 - yp_clipped))) / batch_size
        return grad


class CategoricalCrossEntropy(Loss):
    """Categorical Cross-Entropy loss for multi-class classification.

    Loss formula:
        L = - (1/N) * sum_i( sum_c( y_ic * log(p_ic) ) )

    Gradient with respect to probabilities p:
        dL/dp_ic = - (1/N) * (y_ic / p_ic)

    Note:
        When paired with a preceding Softmax layer, the combined gradient
        through Softmax simplifies to (1/N) * (p - y). Our modular design
        computes this naturally through the Softmax vector-Jacobian product.

    Attributes:
        eps (float): Small epsilon to avoid log(0) and division by zero.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps: float = float(eps)

    def _format_targets(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Convert integer class indices to one-hot format if needed."""
        yt = np.asarray(y_true)
        num_classes = y_pred.shape[1]

        # If y_true is 1D integer labels, or 2D (N, 1), convert to one-hot
        if yt.ndim == 1 or (yt.ndim == 2 and yt.shape[1] == 1):
            indices = yt.ravel().astype(int)
            if np.any((indices < 0) | (indices >= num_classes)):
                raise ValueError(
                    f"Class indices in y_true must be between 0 and {num_classes - 1}, "
                    f"found min={indices.min()}, max={indices.max()}."
                )
            one_hot = np.zeros((len(indices), num_classes), dtype=y_pred.dtype)
            one_hot[np.arange(len(indices)), indices] = 1.0
            return one_hot

        if yt.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch in CategoricalCrossEntropy: y_pred is {y_pred.shape}, "
                f"y_true is {yt.shape}."
            )
        return yt.astype(y_pred.dtype, copy=False)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate categorical cross-entropy loss.

        Args:
            y_pred (np.ndarray): Predicted class probabilities, shape (batch_size, num_classes).
            y_true (np.ndarray): Ground truth labels (one-hot (batch_size, num_classes) or
                class indices (batch_size,)).

        Returns:
            float: Scalar mean categorical cross-entropy loss.
        """
        yp = _as_float_array(y_pred)
        if yp.ndim != 2:
            raise ValueError(f"CategoricalCrossEntropy expects 2D y_pred, got shape {yp.shape}.")

        yt = self._format_targets(yp, y_true)

        self._y_pred_cache = yp
        self._y_true_cache = yt

        # Numerically stable clipping
        yp_clipped = np.clip(yp, self.eps, 1.0)
        sample_losses = -np.sum(yt * np.log(yp_clipped), axis=-1)
        return float(np.mean(sample_losses))

    def backward(
        self, y_pred: Optional[np.ndarray] = None, y_true: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate gradient dL/dy_pred.

        Returns:
            np.ndarray: Gradient tensor matching shape (batch_size, num_classes).
        """
        yp = (
            _as_float_array(y_pred)
            if y_pred is not None
            else self._y_pred_cache
        )
        yt = (
            _as_float_array(y_true, dtype=yp.dtype)
            if y_true is not None
            else self._y_true_cache
        )

        if yp is None or yt is None:
            raise RuntimeError("CategoricalCrossEntropy.backward called before forward pass.")

        if yp.shape != yt.shape:
            yt = self._format_targets(yp, yt)

        batch_size = yp.shape[0]
        yp_clipped = np.clip(yp, self.eps, 1.0)

        # Gradient normalized by batch size
        grad = (-yt / yp_clipped) / batch_size
        return grad


class MeanSquaredError(Loss):
    """Mean Squared Error (MSE) loss function for continuous regression tasks.

    Formula:
        L = (1/N) * sum( (y_pred - y_true)^2 )

    Gradient:
        dL/dy_pred = (2/N) * (y_pred - y_true)
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        yp, yt = _loss_pair(y_pred, y_true)

        if yp.ndim == 1:
            yp = yp.reshape(-1, 1)
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)

        if yp.shape != yt.shape:
            raise ValueError(f"Shape mismatch in MeanSquaredError: y_pred {yp.shape} vs y_true {yt.shape}")

        self._y_pred_cache = yp
        self._y_true_cache = yt

        return float(np.mean((yp - yt) ** 2))

    def backward(
        self, y_pred: Optional[np.ndarray] = None, y_true: Optional[np.ndarray] = None
    ) -> np.ndarray:
        yp = _as_float_array(y_pred) if y_pred is not None else self._y_pred_cache
        yt = _as_float_array(y_true, dtype=yp.dtype) if y_true is not None else self._y_true_cache

        if yp is None or yt is None:
            raise RuntimeError("MeanSquaredError.backward called before forward pass.")

        if yp.ndim == 1:
            yp = yp.reshape(-1, 1)
        if yt.ndim == 1:
            yt = yt.reshape(-1, 1)

        batch_size = yp.shape[0]
        # Factor of 2 / N
        return (2.0 * (yp - yt)) / batch_size


# Aliases
MSELoss = MeanSquaredError
