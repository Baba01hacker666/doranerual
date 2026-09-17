"""Base abstractions for the neural network library.

Defines the common Layer and Module interface contract followed by all layers,
activations, and network components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class Layer(ABC):
    """Abstract base class representing a layer or transformation in a neural network.

    Every layer must implement the `forward` and `backward` methods, establishing
    a consistent interface contract across all building blocks.

    Attributes:
        trainable (bool): Flag indicating if this layer contains trainable parameters.
        _params (Dict[str, np.ndarray]): Dictionary mapping parameter names (e.g. 'weights')
            to their NumPy arrays.
        _grads (Dict[str, np.ndarray]): Dictionary mapping parameter names to their
            calculated gradients.
    """

    def __init__(self) -> None:
        self.trainable: bool = False
        self.training: bool = True
        self._params: Dict[str, np.ndarray] = {}
        self._grads: Dict[str, np.ndarray] = {}

    def train(self, mode: bool = True) -> "Layer":
        """Set training mode (True for training, False for evaluation)."""
        self.training = mode
        return self

    def eval(self) -> "Layer":
        """Set evaluation mode (disables dropout, fixes batch statistics)."""
        self.training = False
        return self

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward pass computation.

        Args:
            x (np.ndarray): Input tensor, typically with shape (batch_size, ...).

        Returns:
            np.ndarray: Output tensor produced by the layer.
        """
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Perform backpropagation through this layer.

        Computes gradients with respect to trainable parameters (if any) and
        returns the gradient with respect to the layer input for backpropagating
        to the upstream layer.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to this
                layer's output, with shape matching the layer's output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the layer's input,
                with shape matching the layer's input.
        """
        pass

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return references to the trainable parameters of this layer."""
        return self._params

    def get_grads(self) -> Dict[str, np.ndarray]:
        """Return references to the gradients of this layer's parameters."""
        return self._grads

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set or restore parameters from a dictionary of arrays.

        Args:
            params (Dict[str, np.ndarray]): Parameter mapping, e.g. {'weights': ..., 'biases': ...}.
        """
        for name, value in params.items():
            if name in self._params:
                np.copyto(self._params[name], value)
            else:
                self._params[name] = np.array(value, copy=True)

    def to_dict(self) -> dict:
        """Serialize layer configuration to a dictionary for JSON export."""
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, config: dict) -> "Layer":
        """Instantiate layer from configuration dictionary."""
        return cls()
