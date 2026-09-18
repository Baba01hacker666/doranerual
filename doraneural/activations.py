"""Activation function layers for neural networks.

Implements non-linear activation transformations (ReLU, Sigmoid, Softmax) conforming
to the standard Layer interface.
"""

from typing import Dict, Any, Optional
import numpy as np

from .base import Layer


# Keep activations on the incoming floating-point dtype.  The previous
# implementation coerced every tensor to float32, which silently defeated
# ``model.to_precision("float64")`` and added a cast at every layer.
def _as_float_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    return arr if np.issubdtype(arr.dtype, np.floating) else arr.astype(np.float32)


def _exp_limit(dtype: np.dtype) -> float:
    """Largest safe magnitude for a stable ``exp(-x)`` calculation."""
    return float(np.log(np.finfo(dtype).max) - 1.0)


class ReLU(Layer):
    """Rectified Linear Unit activation function.

    Applies the element-wise function:
        f(x) = max(0, x)

    Derivative:
        f'(x) = 1 if x > 0 else 0
    """

    def __init__(self) -> None:
        super().__init__()
        self.trainable: bool = False
        self._input_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply ReLU activation forward pass.

        Args:
            x (np.ndarray): Input tensor of any shape (batch_size, ...).

        Returns:
            np.ndarray: Output tensor with values clamped at minimum 0.
        """
        x_arr = _as_float_array(x)
        self._input_cache = x_arr
        return np.maximum(0.0, x_arr)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute backward gradient through ReLU.

        Args:
            grad_output (np.ndarray): Incoming gradient of loss with respect to
                output, matching the shape of x in forward().

        Returns:
            np.ndarray: Downstream gradient, passing through where x > 0 and 0 elsewhere.
        """
        if self._input_cache is None:
            raise RuntimeError("ReLU.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self._input_cache.dtype if self._input_cache is not None else np.float32)
        if grad_out.shape != self._input_cache.shape:
            raise ValueError(
                f"Gradient shape {grad_out.shape} does not match cached input shape {self._input_cache.shape}."
            )

        return grad_out * (self._input_cache > 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ReLU configuration for JSON export."""
        return {"type": "ReLU"}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ReLU":
        """Reconstruct ReLU from configuration."""
        return cls()


class Sigmoid(Layer):
    """Sigmoid activation function.

    Applies the logistic sigmoid transformation element-wise:
        f(x) = 1 / (1 + exp(-x))

    Derivative:
        f'(x) = f(x) * (1 - f(x))

    Includes numerical clipping to prevent overflow in exp(-x).
    """

    def __init__(self) -> None:
        super().__init__()
        self.trainable: bool = False
        self._output_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply Sigmoid activation forward pass.

        Args:
            x (np.ndarray): Input array of arbitrary shape.

        Returns:
            np.ndarray: Transformed probabilities in range (0, 1).
        """
        x_arr = _as_float_array(x)
        # Clip to prevent float32 overflow in exp(-x)
        x_clipped = np.clip(x_arr, -_exp_limit(x_arr.dtype), _exp_limit(x_arr.dtype))
        out = 1.0 / (1.0 + np.exp(-x_clipped))
        self._output_cache = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute backward gradient through Sigmoid.

        Args:
            grad_output (np.ndarray): Incoming upstream gradient.

        Returns:
            np.ndarray: Downstream gradient: grad_output * s * (1 - s).
        """
        if self._output_cache is None:
            raise RuntimeError("Sigmoid.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self._output_cache.dtype if self._output_cache is not None else np.float32)
        if grad_out.shape != self._output_cache.shape:
            raise ValueError(
                f"Gradient shape {grad_out.shape} does not match cached output shape {self._output_cache.shape}."
            )

        s = self._output_cache
        return grad_out * s * (1.0 - s)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Sigmoid configuration for JSON export."""
        return {"type": "Sigmoid"}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Sigmoid":
        """Reconstruct Sigmoid from configuration."""
        return cls()


class Softmax(Layer):
    """Numerically stable Softmax activation along the last axis.

    Applies:
        f(x_i) = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))

    Derivative with incoming upstream gradient dL/dout:
        dL/dx_i = out_i * (dL/dout_i - sum_k(dL/dout_k * out_k))
    """

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis: int = axis
        self.trainable: bool = False
        self._output_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply Softmax normalization along the specified axis (default: last axis).

        Args:
            x (np.ndarray): Input tensor, shape (batch_size, num_classes).

        Returns:
            np.ndarray: Probability distribution across classes summing to 1.
        """
        x_arr = _as_float_array(x)
        # Shift x by subtracting max along axis for numerical stability
        shifted_x = x_arr - np.max(x_arr, axis=self.axis, keepdims=True)
        exp_x = np.exp(shifted_x)
        out = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        self._output_cache = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute backward gradient through Softmax.

        Implements the vectorized vector-Jacobian product:
            dL/dx = out * (grad_output - sum(grad_output * out, axis=-1, keepdims=True))

        Args:
            grad_output (np.ndarray): Incoming gradient of loss with respect to probabilities.

        Returns:
            np.ndarray: Gradient of loss with respect to unnormalized logits.
        """
        if self._output_cache is None:
            raise RuntimeError("Softmax.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self._output_cache.dtype if self._output_cache is not None else np.float32)
        if grad_out.shape != self._output_cache.shape:
            raise ValueError(
                f"Gradient shape {grad_out.shape} does not match cached output shape {self._output_cache.shape}."
            )

        out = self._output_cache
        # Vectorized batch product with Jacobian:
        sum_grad_times_out = np.sum(grad_out * out, axis=self.axis, keepdims=True)
        return out * (grad_out - sum_grad_times_out)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Softmax configuration for JSON export."""
        return {"type": "Softmax", "axis": self.axis}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Softmax":
        """Reconstruct Softmax from configuration."""
        return cls(axis=config.get("axis", -1))


class Tanh(Layer):
    """Hyperbolic Tangent activation function.

    Applies the element-wise function:
        f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Derivative:
        f'(x) = 1 - tanh(x)^2
    """

    def __init__(self) -> None:
        super().__init__()
        self.trainable: bool = False
        self._output_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply Tanh activation forward pass."""
        x_arr = _as_float_array(x)
        out = np.tanh(x_arr)
        self._output_cache = out
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute backward gradient through Tanh."""
        if self._output_cache is None:
            raise RuntimeError("Tanh.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self._output_cache.dtype if self._output_cache is not None else np.float32)
        if grad_out.shape != self._output_cache.shape:
            raise ValueError(
                f"Gradient shape {grad_out.shape} does not match cached output shape {self._output_cache.shape}."
            )

        return grad_out * (1.0 - self._output_cache ** 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Tanh configuration for JSON export."""
        return {"type": "Tanh"}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Tanh":
        """Reconstruct Tanh from configuration."""
        return cls()


class SiLU(Layer):
    """Sigmoid Linear Unit (SiLU / Swish) activation function.

    Applies the element-wise function:
        f(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Derivative:
        f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
              = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    """

    def __init__(self) -> None:
        super().__init__()
        self.trainable: bool = False
        self._input_cache: Optional[np.ndarray] = None
        self._sig_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply SiLU activation forward pass."""
        x_arr = _as_float_array(x)
        x_clipped = np.clip(x_arr, -_exp_limit(x_arr.dtype), _exp_limit(x_arr.dtype))
        sig = 1.0 / (1.0 + np.exp(-x_clipped))
        self._input_cache = x_arr
        self._sig_cache = sig
        return x_arr * sig

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Compute backward gradient through SiLU."""
        if self._input_cache is None or self._sig_cache is None:
            raise RuntimeError("SiLU.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self._input_cache.dtype if self._input_cache is not None else np.float32)
        if grad_out.shape != self._input_cache.shape:
            raise ValueError(
                f"Gradient shape {grad_out.shape} does not match cached input shape {self._input_cache.shape}."
            )

        x = self._input_cache
        sig = self._sig_cache
        dsilu_dx = sig * (1.0 + x * (1.0 - sig))
        return grad_out * dsilu_dx

    def to_dict(self) -> Dict[str, Any]:
        """Serialize SiLU configuration for JSON export."""
        return {"type": "SiLU"}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SiLU":
        """Reconstruct SiLU from configuration."""
        return cls()


class Inverter(Layer):
    """Signal Polarity Inverter Activation Layer.

    Applies element-wise sign inversion:
        f(x) = -x

    Derivative:
        f'(x) = -1
    """

    def __init__(self) -> None:
        super().__init__()
        self.trainable = False
        self._input_shape = None
        self._input_dtype = np.dtype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Invert signal polarity forward pass."""
        x_arr = _as_float_array(x)
        self._input_shape = x_arr.shape
        self._input_dtype = x_arr.dtype
        return -x_arr

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Invert signal polarity backward pass."""
        grad_out = np.asarray(grad_output, dtype=self._input_dtype)
        return -grad_out

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Inverter configuration for JSON export."""
        return {"type": "Inverter"}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Inverter":
        """Reconstruct Inverter from configuration."""
        return cls()


