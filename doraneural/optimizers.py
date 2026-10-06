"""Numerically careful optimization algorithms for NumPy neural networks.

The built-in optimizers support global gradient clipping, decoupled weight
decay, and per-layer state. Adam also supports AMSGrad and keeps a separate
bias-correction step for each parameter, which is important when gradients are
missing or layers are trained intermittently.
"""

from abc import ABC, abstractmethod
import math
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .base import Layer


GradientSource = Union[
    Iterable[Layer],
    Iterable[np.ndarray],
    Dict[str, np.ndarray],
]
ParameterGradient = Tuple[Layer, str, np.ndarray, np.ndarray]


def _finite_float(name: str, value: float) -> float:
    """Convert a scalar hyperparameter to float and reject NaN/infinity."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return result


def _positive_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return result


def _non_negative_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return result


def _unique_gradient_arrays(source: GradientSource) -> List[np.ndarray]:
    """Extract each gradient buffer once from a supported public input."""
    if isinstance(source, dict):
        items = source.values()
    else:
        try:
            items = iter(source)
        except TypeError as exc:
            raise TypeError(
                "Expected layers, gradient arrays, or a dictionary of gradients"
            ) from exc

    gradients: List[np.ndarray] = []
    seen = set()
    for item in items:
        if item is None:
            continue
        if isinstance(item, np.ndarray):
            candidates = (item,)
        elif hasattr(item, "get_grads"):
            if not getattr(item, "trainable", False):
                continue
            layer_grads = item.get_grads()
            if not isinstance(layer_grads, dict):
                raise TypeError("Layer.get_grads() must return a dictionary")
            candidates = layer_grads.values()
        else:
            raise TypeError(
                "Gradient input items must be NumPy arrays or layers with get_grads()"
            )

        for grad in candidates:
            if grad is None or id(grad) in seen:
                continue
            if not isinstance(grad, np.ndarray):
                raise TypeError("Gradient buffers must be NumPy arrays")
            if not np.issubdtype(grad.dtype, np.floating):
                raise TypeError(
                    f"Gradient buffers must have a floating dtype, got {grad.dtype}"
                )
            if not np.all(np.isfinite(grad)):
                raise FloatingPointError("Gradient contains NaN or infinity")
            seen.add(id(grad))
            gradients.append(grad)
    return gradients


def _global_norm_parts(gradients: List[np.ndarray]) -> Tuple[float, float, float]:
    """Return (norm, max_abs, scaled_norm) without squaring large raw values.

    Scaling each gradient by the largest absolute entry prevents avoidable
    overflow when a float32 gradient is large. ``norm`` may still be ``inf``
    when the mathematical result is beyond float64's representable range; the
    other two values are sufficient to calculate a safe clipping operation.
    """
    max_abs = 0.0
    for grad in gradients:
        if grad.size:
            max_abs = max(max_abs, float(np.max(np.abs(grad))))

    if max_abs == 0.0:
        return 0.0, 0.0, 0.0

    squared_norm = 0.0
    for grad in gradients:
        if grad.size:
            scaled = grad / max_abs
            squared_norm += float(np.sum(scaled * scaled, dtype=np.float64))

    scaled_norm = math.sqrt(squared_norm)
    total_norm = max_abs * scaled_norm
    return total_norm, max_abs, scaled_norm


def _clip_gradient_arrays(gradients: GradientSource, max_norm: float) -> float:
    """Clip a set of arrays by one stable global L2 norm and return that norm."""
    gradients = _unique_gradient_arrays(gradients)
    if not gradients:
        return 0.0

    total_norm, max_abs, scaled_norm = _global_norm_parts(gradients)
    if max_abs == 0.0:
        return 0.0

    # Comparing against this threshold avoids overflowing while computing the
    # norm and also lets us clip a finite gradient whose true norm exceeds
    # float64's range.
    if max_abs > max_norm / scaled_norm:
        # Normalize first rather than forming max_norm / total_norm directly;
        # the latter can underflow to zero for extremely large finite values.
        target_max = max_norm / scaled_norm
        if any(not grad.flags.writeable for grad in gradients):
            raise ValueError("Cannot clip a read-only gradient array")
        for grad in gradients:
            np.divide(grad, max_abs, out=grad)
            np.multiply(grad, target_max, out=grad)

    return total_norm


def _collect_parameter_gradients(
    layers: Iterable[Layer],
    validate_gradient_finite: bool = True,
) -> List[ParameterGradient]:
    """Validate a training step before any parameter or optimizer state changes."""
    try:
        layer_iter = iter(layers)
    except TypeError as exc:
        raise TypeError("layers must be an iterable of Layer objects") from exc

    updates: List[ParameterGradient] = []
    for layer in layer_iter:
        if not hasattr(layer, "get_params") or not hasattr(layer, "get_grads"):
            raise TypeError(f"Expected a layer with parameters and gradients, got {type(layer)}")
        if not getattr(layer, "trainable", False):
            continue

        params = layer.get_params()
        grads = layer.get_grads()
        if not isinstance(params, dict) or not isinstance(grads, dict):
            raise TypeError("Layer.get_params() and get_grads() must return dictionaries")

        for name, param in params.items():
            if param is None:
                continue
            grad = grads.get(name)
            if grad is None:
                continue
            if not isinstance(param, np.ndarray) or not isinstance(grad, np.ndarray):
                raise TypeError(f"Parameter and gradient '{name}' must be NumPy arrays")
            if param.shape != grad.shape:
                raise ValueError(
                    f"Gradient shape mismatch for parameter '{name}': "
                    f"expected {param.shape}, got {grad.shape}"
                )
            if not np.issubdtype(param.dtype, np.floating):
                raise TypeError(
                    f"Trainable parameter '{name}' must have a floating dtype, got {param.dtype}"
                )
            if not np.issubdtype(grad.dtype, np.floating):
                raise TypeError(
                    f"Gradient for parameter '{name}' must have a floating dtype, got {grad.dtype}"
                )
            if not param.flags.writeable:
                raise ValueError(f"Trainable parameter '{name}' is read-only")
            if not np.all(np.isfinite(param)):
                raise FloatingPointError(f"Parameter '{name}' contains NaN or infinity")
            if validate_gradient_finite and not np.all(np.isfinite(grad)):
                raise FloatingPointError(f"Gradient for parameter '{name}' contains NaN or infinity")
            updates.append((layer, name, param, grad))
    return updates


class Optimizer(ABC):
    """Abstract base class for network optimizers."""

    @abstractmethod
    def step(self, layers: List[Layer]) -> None:
        """Update trainable parameters in the provided layers."""
        raise NotImplementedError

    def _prepare_step(self, layers: Iterable[Layer]) -> List[ParameterGradient]:
        """Validate parameter/gradient pairs and apply configured clipping."""
        lr = _finite_float("Learning rate", self.lr)
        if lr < 0.0:
            raise ValueError(f"Learning rate must be non-negative during training, got {lr}")

        clip_norm = getattr(self, "clip_norm", None)
        if clip_norm is not None:
            clip_norm = _positive_float("clip_norm", clip_norm)
        updates = _collect_parameter_gradients(
            layers,
            validate_gradient_finite=clip_norm is None,
        )
        if clip_norm is not None:
            _clip_gradient_arrays([grad for _, _, _, grad in updates], clip_norm)
        return updates

    def _clip_gradients(self, layers: List[Layer], clip_norm: Optional[float]) -> None:
        """Clip gradients by global L2 norm across all trainable parameters."""
        if clip_norm is not None:
            clip_grad_norm(layers, clip_norm)

    @staticmethod
    def _state_key(layer: Layer, param_name: str) -> Tuple[int, str]:
        """Use layer identity rather than list position so reordered layers keep state."""
        return id(layer), param_name

    @staticmethod
    def _state_array(
        state: Dict[Tuple[int, str], np.ndarray],
        key: Tuple[int, str],
        param: np.ndarray,
    ) -> np.ndarray:
        value = state.get(key)
        if value is None or value.shape != param.shape:
            value = np.zeros_like(param)
            state[key] = value
        elif value.dtype != param.dtype:
            value = value.astype(param.dtype, copy=False)
            state[key] = value
        return value


def clip_grad_norm(
    layers_or_grads: GradientSource,
    max_norm: float,
) -> float:
    """Clip the global L2 norm of gradients and return the norm before clipping.

    Accepts an iterable of layers, an iterable of NumPy gradient arrays, or a
    dictionary of gradient arrays. Duplicate references are counted once. The
    calculation is scaled to avoid overflow for large finite gradients; NaN
    and infinite gradients raise ``FloatingPointError`` instead of spreading.
    """
    max_norm = _positive_float("max_norm", max_norm)
    return _clip_gradient_arrays(layers_or_grads, max_norm)


def clip_grad_value(
    layers_or_grads: GradientSource,
    clip_value: float,
) -> None:
    """Clip every gradient component to ``[-clip_value, clip_value]`` in-place."""
    clip_value = _positive_float("clip_value", clip_value)
    gradients = _unique_gradient_arrays(layers_or_grads)
    if any(not grad.flags.writeable for grad in gradients):
        raise ValueError("Cannot clip a read-only gradient array")
    for grad in gradients:
        np.clip(grad, -clip_value, clip_value, out=grad)


class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum, Nesterov, and weight decay.

    Momentum buffers store gradients (rather than learning-rate-scaled
    updates), so learning-rate schedulers can change the learning rate without
    carrying the old rate into future steps. Weight decay is decoupled.
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        clip_norm: Optional[float] = None,
        nesterov: bool = False,
    ) -> None:
        lr = _positive_float("Learning rate", lr)
        momentum = _finite_float("Momentum", momentum)
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Momentum must be in [0.0, 1.0), got {momentum}")
        weight_decay = _non_negative_float("weight_decay", weight_decay)
        if clip_norm is not None:
            clip_norm = _positive_float("clip_norm", clip_norm)
        if nesterov and momentum <= 0.0:
            raise ValueError("Nesterov momentum requires momentum > 0")

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.nesterov = bool(nesterov)
        self._velocities: Dict[Tuple[int, str], np.ndarray] = {}

    def step(self, layers: List[Layer]) -> None:
        """Perform one SGD update across all trainable parameters."""
        for layer, name, param, grad in self._prepare_step(layers):
            key = self._state_key(layer, name)
            if self.weight_decay:
                param *= 1.0 - self.lr * self.weight_decay

            if self.momentum:
                velocity = self._state_array(self._velocities, key, param)
                velocity *= self.momentum
                velocity += grad
                if self.nesterov:
                    param -= self.lr * (grad + self.momentum * velocity)
                else:
                    param -= self.lr * velocity
            else:
                param -= self.lr * grad


class Adam(Optimizer):
    """Adam with bias correction, decoupled weight decay, and optional AMSGrad.

    ``weight_decay`` is decoupled (AdamW-style). Bias correction uses a
    parameter-specific step count, so a parameter with a missing gradient does
    not inherit correction steps from unrelated parameters.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_norm: Optional[float] = None,
        amsgrad: bool = False,
    ) -> None:
        lr = _positive_float("Learning rate", lr)
        beta1 = _finite_float("beta1", beta1)
        beta2 = _finite_float("beta2", beta2)
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0.0, 1.0), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0.0, 1.0), got {beta2}")
        eps = _positive_float("eps", eps)
        weight_decay = _non_negative_float("weight_decay", weight_decay)
        if clip_norm is not None:
            clip_norm = _positive_float("clip_norm", clip_norm)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.amsgrad = bool(amsgrad)

        # ``t`` remains a convenient count of step() calls. ``_steps`` is the
        # actual bias-correction counter for each parameter.
        self.t = 0
        self._steps: Dict[Tuple[int, str], int] = {}
        self._m: Dict[Tuple[int, str], np.ndarray] = {}
        self._v: Dict[Tuple[int, str], np.ndarray] = {}
        self._v_max: Dict[Tuple[int, str], np.ndarray] = {}

    def step(self, layers: List[Layer]) -> None:
        """Perform one Adam update across all trainable parameters."""
        updates = self._prepare_step(layers)
        self.t += 1

        for layer, name, param, grad in updates:
            key = self._state_key(layer, name)
            state_arrays = (self._m.get(key), self._v.get(key), self._v_max.get(key))
            if any(state is not None and state.shape != param.shape for state in state_arrays):
                # A layer may be rebuilt with a different parameter shape.
                # Discard its incompatible moments and bias-correction history.
                self._steps.pop(key, None)
                self._m.pop(key, None)
                self._v.pop(key, None)
                self._v_max.pop(key, None)
            m = self._state_array(self._m, key, param)
            v = self._state_array(self._v, key, param)
            step = self._steps.get(key, 0) + 1

            # Check the square before mutating either moment so an extreme but
            # finite gradient cannot leave half-updated optimizer state behind.
            with np.errstate(over="raise", invalid="raise"):
                grad_squared = grad * grad

            # Keep the moments in the parameter dtype and update in place.
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * grad_squared
            self._steps[key] = step

            second_moment = v
            if self.amsgrad:
                v_max = self._state_array(self._v_max, key, param)
                np.maximum(v_max, v, out=v_max)
                second_moment = v_max

            beta1_correction = 1.0 - self.beta1 ** step
            beta2_sqrt_correction = math.sqrt(1.0 - self.beta2 ** step)

            # Algebraically equivalent to m_hat / (sqrt(v_hat) + eps), with
            # epsilon scaled correctly when combining bias correction. Keeping
            # moments in-place avoids allocating two full-size corrected arrays.
            step_size = self.lr * beta2_sqrt_correction / beta1_correction
            denominator = np.sqrt(second_moment) + self.eps * beta2_sqrt_correction

            if self.weight_decay:
                param *= 1.0 - self.lr * self.weight_decay
            param -= step_size * m / denominator


class AdamW(Adam):
    """Explicit AdamW name for Adam's decoupled-weight-decay behavior.

    In this library ``Adam(weight_decay=...)`` already uses AdamW semantics;
    this class is provided as a clearer, familiar spelling and has the same
    options, including optional AMSGrad.
    """


class RMSprop(Optimizer):
    """RMSprop with optional momentum, decoupled weight decay, and clipping."""

    def __init__(
        self,
        lr: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        clip_norm: Optional[float] = None,
        weight_decay: float = 0.0,
    ) -> None:
        lr = _positive_float("Learning rate", lr)
        alpha = _finite_float("alpha", alpha)
        if not (0.0 <= alpha < 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0), got {alpha}")
        eps = _positive_float("eps", eps)
        momentum = _finite_float("momentum", momentum)
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"momentum must be in [0.0, 1.0), got {momentum}")
        if clip_norm is not None:
            clip_norm = _positive_float("clip_norm", clip_norm)
        weight_decay = _non_negative_float("weight_decay", weight_decay)

        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.weight_decay = weight_decay
        self._v: Dict[Tuple[int, str], np.ndarray] = {}
        self._buf: Dict[Tuple[int, str], np.ndarray] = {}

    def step(self, layers: List[Layer]) -> None:
        """Perform one RMSprop update across all trainable parameters."""
        for layer, name, param, grad in self._prepare_step(layers):
            key = self._state_key(layer, name)
            v = self._state_array(self._v, key, param)
            with np.errstate(over="raise", invalid="raise"):
                grad_squared = grad * grad
            v *= self.alpha
            v += (1.0 - self.alpha) * grad_squared

            update = grad / (np.sqrt(v) + self.eps)
            if self.weight_decay:
                param *= 1.0 - self.lr * self.weight_decay

            if self.momentum:
                buf = self._state_array(self._buf, key, param)
                buf *= self.momentum
                buf += update
                param -= self.lr * buf
            else:
                param -= self.lr * update
