"""Optimization algorithms for updating neural network parameters.

Provides:
- SGD (with optional Momentum and gradient clipping)
- Adam (with adaptive moments and decoupled weight decay)
- RMSprop (with adaptive root-mean-square scaling)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import numpy as np

from neural_lib.base import Layer


class Optimizer(ABC):
    """Abstract base class for network optimizers."""

    @abstractmethod
    def step(self, layers: List[Layer]) -> None:
        """Update trainable parameters in the provided layers.

        Args:
            layers (List[Layer]): Sequence of network layers.
        """
        pass

    def _clip_gradients(self, layers: List[Layer], clip_norm: Optional[float]) -> None:
        """Clip gradients by global L2 norm across all trainable parameters."""
        if clip_norm is None:
            return

        total_norm_sq = 0.0
        for layer in layers:
            if not layer.trainable:
                continue
            for grad in layer.get_grads().values():
                if grad is not None:
                    total_norm_sq += float(np.sum(grad * grad))

        total_norm = np.sqrt(total_norm_sq)
        if total_norm > clip_norm:
            scale = clip_norm / (total_norm + 1e-8)
            for layer in layers:
                if not layer.trainable:
                    continue
                for grad in layer.get_grads().values():
                    if grad is not None:
                        grad *= scale


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with optional Momentum.

    Updates parameters using the rule:
        If momentum > 0:
            velocity = momentum * velocity + lr * grad
            param = param - velocity
        Else:
            param = param - lr * grad
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.0,
        clip_norm: Optional[float] = None,
    ) -> None:
        """Initialize the SGD optimizer.

        Args:
            lr (float): Learning rate. Must be positive.
            momentum (float): Momentum constant in [0.0, 1.0).
            clip_norm (Optional[float]): Max global L2 norm for gradient clipping.
        """
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Momentum must be in [0.0, 1.0), got {momentum}")

        self.lr: float = float(lr)
        self.momentum: float = float(momentum)
        self.clip_norm: Optional[float] = float(clip_norm) if clip_norm is not None else None
        self._velocities: Dict[Tuple[int, str], np.ndarray] = {}

    def step(self, layers: List[Layer]) -> None:
        """Perform one SGD optimization step across all trainable layers."""
        self._clip_gradients(layers, self.clip_norm)

        for layer_idx, layer in enumerate(layers):
            if not layer.trainable:
                continue

            params = layer.get_params()
            grads = layer.get_grads()

            for param_name, param in params.items():
                if param is None:
                    continue
                grad = grads.get(param_name)
                if grad is None:
                    continue

                if self.momentum > 0.0:
                    key = (layer_idx, param_name)
                    if key not in self._velocities:
                        self._velocities[key] = np.zeros_like(param)

                    velocity = self._velocities[key]
                    velocity *= self.momentum
                    velocity += self.lr * grad
                    param -= velocity
                else:
                    param -= self.lr * grad


class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) optimizer.

    Maintains exponentially decaying averages of past gradients (m)
    and past squared gradients (v), with bias correction.

    Attributes:
        lr (float): Learning rate step size.
        beta1 (float): Exponential decay rate for first moment estimates (default: 0.9).
        beta2 (float): Exponential decay rate for second moment estimates (default: 0.999).
        eps (float): Small epsilon for numerical stability (default: 1e-8).
        weight_decay (float): Decoupled weight decay rate (AdamW-style) (default: 0.0).
        clip_norm (Optional[float]): Max global gradient norm.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_norm: Optional[float] = None,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0.0, 1.0), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0.0, 1.0), got {beta2}")

        self.lr: float = float(lr)
        self.beta1: float = float(beta1)
        self.beta2: float = float(beta2)
        self.eps: float = float(eps)
        self.weight_decay: float = float(weight_decay)
        self.clip_norm: Optional[float] = float(clip_norm) if clip_norm is not None else None

        self.t: int = 0
        self._m: Dict[Tuple[int, str], np.ndarray] = {}
        self._v: Dict[Tuple[int, str], np.ndarray] = {}

    def step(self, layers: List[Layer]) -> None:
        """Perform one Adam update step across all trainable parameters."""
        self._clip_gradients(layers, self.clip_norm)
        self.t += 1

        # Bias correction factors
        beta1_corr = 1.0 - (self.beta1 ** self.t)
        beta2_corr = 1.0 - (self.beta2 ** self.t)

        for layer_idx, layer in enumerate(layers):
            if not layer.trainable:
                continue

            params = layer.get_params()
            grads = layer.get_grads()

            for param_name, param in params.items():
                if param is None:
                    continue
                grad = grads.get(param_name)
                if grad is None:
                    continue

                key = (layer_idx, param_name)
                if key not in self._m:
                    self._m[key] = np.zeros_like(param)
                    self._v[key] = np.zeros_like(param)

                m = self._m[key]
                v = self._v[key]

                # Update biased 1st and 2nd moment estimates in-place
                m *= self.beta1
                m += (1.0 - self.beta1) * grad

                v *= self.beta2
                v += (1.0 - self.beta2) * (grad * grad)

                # Compute bias-corrected estimates
                m_hat = m / beta1_corr
                v_hat = v / beta2_corr

                # Decoupled weight decay (AdamW)
                if self.weight_decay > 0.0:
                    param -= self.lr * self.weight_decay * param

                # Parameter update
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSprop(Optimizer):
    """Root Mean Square Propagation (RMSprop) optimizer.

    Divides the gradient by a running average of its recent magnitude.

    Attributes:
        lr (float): Learning rate.
        alpha (float): Smoothing constant (default: 0.99).
        eps (float): Numerical stability term (default: 1e-8).
        momentum (float): Optional momentum factor (default: 0.0).
    """

    def __init__(
        self,
        lr: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        clip_norm: Optional[float] = None,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0.0 <= alpha < 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0), got {alpha}")

        self.lr: float = float(lr)
        self.alpha: float = float(alpha)
        self.eps: float = float(eps)
        self.momentum: float = float(momentum)
        self.clip_norm: Optional[float] = float(clip_norm) if clip_norm is not None else None

        self._v: Dict[Tuple[int, str], np.ndarray] = {}
        self._buf: Dict[Tuple[int, str], np.ndarray] = {}

    def step(self, layers: List[Layer]) -> None:
        """Perform one RMSprop update step across all trainable layers."""
        self._clip_gradients(layers, self.clip_norm)

        for layer_idx, layer in enumerate(layers):
            if not layer.trainable:
                continue

            params = layer.get_params()
            grads = layer.get_grads()

            for param_name, param in params.items():
                if param is None:
                    continue
                grad = grads.get(param_name)
                if grad is None:
                    continue

                key = (layer_idx, param_name)
                if key not in self._v:
                    self._v[key] = np.zeros_like(param)

                v = self._v[key]
                # v = alpha * v + (1 - alpha) * grad^2
                v *= self.alpha
                v += (1.0 - self.alpha) * (grad * grad)

                update_val = grad / (np.sqrt(v) + self.eps)

                if self.momentum > 0.0:
                    if key not in self._buf:
                        self._buf[key] = np.zeros_like(param)
                    buf = self._buf[key]
                    buf *= self.momentum
                    buf += update_val
                    param -= self.lr * buf
                else:
                    param -= self.lr * update_val
