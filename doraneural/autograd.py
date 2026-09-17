"""Tensor-level automatic differentiation engine (Autograd) built purely on NumPy.

Provides:
- `Tensor`: Multi-dimensional tensor with reverse-mode automatic differentiation DAG.
- Automatic broadcasting resolution during backpropagation.
- Full mathematical operator overloading (+, -, *, /, @, **, reductions, activations).
- `no_grad`: Context manager to disable gradient tracking during inference.
- Micrograd-style topological sorting and reverse-mode gradient accumulation.
"""

from contextlib import contextmanager
from typing import Set, Tuple, List, Optional, Union, Callable, Any
import numpy as np


# Global gradient tracking flag
_GRAD_ENABLED = True


@contextmanager
def no_grad():
    """Context manager that disables gradient calculation for inference efficiency."""
    global _GRAD_ENABLED
    prev = _GRAD_ENABLED
    try:
        _GRAD_ENABLED = False
        yield
    finally:
        _GRAD_ENABLED = prev


def is_grad_enabled() -> bool:
    """Return True if autograd is currently recording operations."""
    return _GRAD_ENABLED


def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Sum out gradient dimensions that were broadcasted during forward pass.

    Args:
        grad: Incoming gradient array.
        target_shape: Target tensor shape to reduce gradient to.

    Returns:
        np.ndarray: Gradient collapsed to match target_shape.
    """
    grad_shape = grad.shape
    if grad_shape == target_shape:
        return grad

    # Sum along leading dimensions if target has fewer dimensions
    ndim_diff = len(grad_shape) - len(target_shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))

    # Sum along dimensions where target shape was 1 (broadcasted)
    keep_axes = []
    for idx, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            keep_axes.append(idx)

    if keep_axes:
        grad = grad.sum(axis=tuple(keep_axes), keepdims=True)

    return grad.reshape(target_shape)


class Tensor:
    """Multi-dimensional tensor supporting reverse-mode automatic differentiation.

    Args:
        data: Initial array or scalar.
        requires_grad: Whether to track operations for gradient computation.
        dtype: Numerical datatype (defaults to np.float32).
        _children: Internal DAG predecessor nodes.
        _op: String name of operation creating this tensor.
    """

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        dtype: Optional[Any] = None,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        if isinstance(data, Tensor):
            self.data: np.ndarray = data.data.copy()
        elif isinstance(data, np.ndarray):
            self.data = data if dtype is None else data.astype(dtype)
        else:
            dt = dtype if dtype is not None else np.float32
            self.data = np.asarray(data, dtype=dt)

        if dtype is not None and self.data.dtype != dtype:
            self.data = self.data.astype(dtype)

        self.requires_grad: bool = bool(requires_grad)
        self.grad: Optional[np.ndarray] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set["Tensor"] = set(_children) if _GRAD_ENABLED else set()
        self._op: str = _op

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> Union[int, float]:
        return self.data.item()

    def zero_grad(self) -> None:
        """Reset accumulated gradients to zero."""
        self.grad = None

    def detach(self) -> "Tensor":
        """Return a new Tensor detached from the computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)

    # ---------------------------------------------------------------------
    # Backward pass & Topological Sort
    # ---------------------------------------------------------------------

    def backward(self, gradient: Optional[Union[np.ndarray, "Tensor"]] = None) -> None:
        """Compute reverse-mode automatic differentiation gradients for all ancestors.

        Args:
            gradient: Upstream gradient dL/d(self). Defaults to 1.0 for scalar outputs.
        """
        if not self.requires_grad:
            raise RuntimeError("Called backward() on a Tensor that does not require gradients.")

        # Seed gradient
        if gradient is None:
            if self.data.size == 1:
                self.grad = np.ones_like(self.data, dtype=self.data.dtype)
            else:
                raise RuntimeError("Grad can only be created implicitly for scalar outputs. Provide gradient explicitly.")
        elif isinstance(gradient, Tensor):
            self.grad = gradient.data.copy().astype(self.data.dtype)
        else:
            self.grad = np.asarray(gradient, dtype=self.data.dtype)

        # Build topological sort of the computational DAG
        topo: List["Tensor"] = []
        visited: Set["Tensor"] = set()

        def build_topo(node: "Tensor"):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # Run reverse-mode autodiff in topological order
        for node in reversed(topo):
            node._backward()

    # ---------------------------------------------------------------------
    # Arithmetic Operators
    # ---------------------------------------------------------------------

    def __add__(self, other: Any) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        out_data = self.data + other_t.data
        req_grad = _GRAD_ENABLED and (self.requires_grad or other_t.requires_grad)
        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype, _children=(self, other_t), _op="+")

        if req_grad:
            def _backward():
                if self.requires_grad:
                    g = _unbroadcast(out.grad, self.shape)
                    self.grad = g if self.grad is None else self.grad + g
                if other_t.requires_grad:
                    g = _unbroadcast(out.grad, other_t.shape)
                    other_t.grad = g if other_t.grad is None else other_t.grad + g

            out._backward = _backward

        return out

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        out_data = self.data - other_t.data
        req_grad = _GRAD_ENABLED and (self.requires_grad or other_t.requires_grad)
        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype, _children=(self, other_t), _op="-")

        if req_grad:
            def _backward():
                if self.requires_grad:
                    g = _unbroadcast(out.grad, self.shape)
                    self.grad = g if self.grad is None else self.grad + g
                if other_t.requires_grad:
                    g = _unbroadcast(-out.grad, other_t.shape)
                    other_t.grad = g if other_t.grad is None else other_t.grad + g

            out._backward = _backward

        return out

    def __rsub__(self, other: Any) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        return other_t.__sub__(self)

    def __mul__(self, other: Any) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        out_data = self.data * other_t.data
        req_grad = _GRAD_ENABLED and (self.requires_grad or other_t.requires_grad)
        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype, _children=(self, other_t), _op="*")

        if req_grad:
            def _backward():
                if self.requires_grad:
                    g = _unbroadcast(out.grad * other_t.data, self.shape)
                    self.grad = g if self.grad is None else self.grad + g
                if other_t.requires_grad:
                    g = _unbroadcast(out.grad * self.data, other_t.shape)
                    other_t.grad = g if other_t.grad is None else other_t.grad + g

            out._backward = _backward

        return out

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        out_data = self.data / other_t.data
        req_grad = _GRAD_ENABLED and (self.requires_grad or other_t.requires_grad)
        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype, _children=(self, other_t), _op="/")

        if req_grad:
            def _backward():
                if self.requires_grad:
                    g = _unbroadcast(out.grad / other_t.data, self.shape)
                    self.grad = g if self.grad is None else self.grad + g
                if other_t.requires_grad:
                    g = _unbroadcast(-out.grad * self.data / (other_t.data ** 2), other_t.shape)
                    other_t.grad = g if other_t.grad is None else other_t.grad + g

            out._backward = _backward

        return out

    def __rtruediv__(self, other: Any) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other, dtype=self.dtype)
        return other_t.__truediv__(self)

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __pow__(self, power: Union[int, float]) -> "Tensor":
        power_val = float(power)
        out_data = self.data ** power_val
        req_grad = _GRAD_ENABLED and self.requires_grad
        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype, _children=(self,), _op=f"**{power_val}")

        if req_grad:
            def _backward():
                g = power_val * (self.data ** (power_val - 1.0)) * out.grad
                self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward

        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type for @: Tensor and {type(other)}")

        out_data = self.data @ other.data
        req_grad = _GRAD_ENABLED and (self.requires_grad or other.requires_grad)
        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype, _children=(self, other), _op="@")

        if req_grad:
            def _backward():
                if self.requires_grad:
                    # dL/dA = dL/dC @ B.T
                    g = out.grad @ other.data.swapaxes(-1, -2)
                    g = _unbroadcast(g, self.shape)
                    self.grad = g if self.grad is None else self.grad + g
                if other.requires_grad:
                    # dL/dB = A.T @ dL/dC
                    g = self.data.swapaxes(-1, -2) @ out.grad
                    g = _unbroadcast(g, other.shape)
                    other.grad = g if other.grad is None else other.grad + g

            out._backward = _backward

        return out

    # ---------------------------------------------------------------------
    # Non-linear activations & Elementwise functions
    # ---------------------------------------------------------------------

    def relu(self) -> "Tensor":
        """Rectified Linear Unit activation."""
        out_data = np.maximum(0, self.data)
        req_grad = _GRAD_ENABLED and self.requires_grad
        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype, _children=(self,), _op="ReLU")

        if req_grad:
            def _backward():
                g = (self.data > 0) * out.grad
                self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward

        return out

    def sigmoid(self) -> "Tensor":
        """Numerically stable logistic sigmoid function."""
        clipped = np.clip(self.data, -88.0, 88.0)
        sig = 1.0 / (1.0 + np.exp(-clipped))
        req_grad = _GRAD_ENABLED and self.requires_grad
        out = Tensor(sig, requires_grad=req_grad, dtype=self.dtype, _children=(self,), _op="Sigmoid")

        if req_grad:
            def _backward():
                g = sig * (1.0 - sig) * out.grad
                self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward

        return out

    def tanh(self) -> "Tensor":
        """Hyperbolic tangent activation."""
        th = np.tanh(self.data)
        req_grad = _GRAD_ENABLED and self.requires_grad
        out = Tensor(th, requires_grad=req_grad, dtype=self.dtype, _children=(self,), _op="Tanh")

        if req_grad:
            def _backward():
                g = (1.0 - th ** 2) * out.grad
                self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward

        return out

    def exp(self) -> "Tensor":
        """Elementwise natural exponential."""
        out_data = np.exp(np.clip(self.data, -88.0, 88.0))
        req_grad = _GRAD_ENABLED and self.requires_grad
        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype, _children=(self,), _op="exp")

        if req_grad:
            def _backward():
                g = out_data * out.grad
                self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward

        return out

    def log(self, eps: float = 1e-12) -> "Tensor":
        """Elementwise natural logarithm with stability epsilon."""
        clipped = np.clip(self.data, eps, None)
        out_data = np.log(clipped)
        req_grad = _GRAD_ENABLED and self.requires_grad
        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype, _children=(self,), _op="log")

        if req_grad:
            def _backward():
                g = (1.0 / clipped) * out.grad
                self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward

        return out

    # ---------------------------------------------------------------------
    # Reductions & Shape Operations
    # ---------------------------------------------------------------------

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        """Sum array elements over a given axis."""
        out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        req_grad = _GRAD_ENABLED and self.requires_grad
        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype, _children=(self,), _op="sum")

        if req_grad:
            def _backward():
                if axis is None:
                    g = np.broadcast_to(out.grad, self.shape)
                else:
                    expanded_shape = list(self.shape)
                    axes = [axis] if isinstance(axis, int) else list(axis)
                    for a in axes:
                        expanded_shape[a] = 1
                    g = np.broadcast_to(out.grad.reshape(expanded_shape), self.shape)

                self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward

        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        """Compute the arithmetic mean along the specified axis."""
        s = self.sum(axis=axis, keepdims=keepdims)
        num_elements = self.data.size if axis is None else (self.data.size // s.data.size)
        return s * (1.0 / num_elements)

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "Tensor":
        """Gives a new shape to a tensor without changing its data."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = tuple(shape)

        out_data = self.data.reshape(new_shape)
        req_grad = _GRAD_ENABLED and self.requires_grad
        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype, _children=(self,), _op="reshape")

        if req_grad:
            def _backward():
                g = out.grad.reshape(self.shape)
                self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward

        return out

    def transpose(self, *axes: int) -> "Tensor":
        """Reverse or permute the axes of a tensor."""
        if not axes:
            out_data = self.data.T
            inverse_axes = None
        else:
            out_data = np.transpose(self.data, axes)
            inverse_axes = np.argsort(axes)

        req_grad = _GRAD_ENABLED and self.requires_grad
        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype, _children=(self,), _op="transpose")

        if req_grad:
            def _backward():
                if inverse_axes is None:
                    g = out.grad.T
                else:
                    g = np.transpose(out.grad, inverse_axes)
                self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward

        return out

    def __getitem__(self, index: Any) -> "Tensor":
        """Slice indexing with reverse-mode gradient routing."""
        out_data = self.data[index]
        req_grad = _GRAD_ENABLED and self.requires_grad
        out = Tensor(out_data, requires_grad=req_grad, dtype=self.dtype, _children=(self,), _op="slice")

        if req_grad:
            def _backward():
                g = np.zeros_like(self.data)
                np.add.at(g, index, out.grad)
                self.grad = g if self.grad is None else self.grad + g

            out._backward = _backward

        return out

    def __repr__(self) -> str:
        grad_str = f", grad_fn=<{self._op}>" if self._op else ""
        req_str = f", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{req_str}{grad_str})"


def tensor(data: Any, requires_grad: bool = False, dtype: Optional[Any] = None) -> Tensor:
    """Factory helper to construct a new Tensor."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


# -------------------------------------------------------------------------
# Neural Network Autograd Modules & Parameter Abstractions
# -------------------------------------------------------------------------

class Parameter(Tensor):
    """A Tensor that is designated as a learnable model parameter."""

    def __init__(self, data: Any, dtype: Optional[Any] = None) -> None:
        super().__init__(data, requires_grad=True, dtype=dtype)


class Module:
    """Base class for all neural network modules in the autograd framework."""

    def __init__(self) -> None:
        self.training: bool = True

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for m in self.submodules():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def parameters(self) -> List[Parameter]:
        """Collect all Parameter instances from attributes and submodules."""
        params: List[Parameter] = []
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def submodules(self) -> List["Module"]:
        """Return all direct child Module instances."""
        mods: List["Module"] = []
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                mods.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        mods.append(item)
        return mods

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class Linear(Module):
    """Fully-connected linear transformation layer: y = x @ W + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: Any = np.float32) -> None:
        super().__init__()
        self.in_features: int = int(in_features)
        self.out_features: int = int(out_features)
        self.use_bias: bool = bool(bias)

        # He / Kaiming normal initialization
        std = np.sqrt(2.0 / self.in_features)
        w_data = np.random.randn(self.in_features, self.out_features) * std
        self.weight = Parameter(w_data, dtype=dtype)

        if self.use_bias:
            self.bias: Optional[Parameter] = Parameter(np.zeros((1, self.out_features)), dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        x_t = x if isinstance(x, Tensor) else Tensor(x, dtype=self.weight.dtype)
        out = x_t @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


def mse_loss(y_pred: Tensor, y_true: Union[Tensor, np.ndarray]) -> Tensor:
    """Mean Squared Error loss computed dynamically through the autograd DAG."""
    yt_t = y_true if isinstance(y_true, Tensor) else Tensor(y_true, dtype=y_pred.dtype)
    diff = y_pred - yt_t
    return (diff ** 2).mean()


def binary_cross_entropy(y_pred: Tensor, y_true: Union[Tensor, np.ndarray], eps: float = 1e-12) -> Tensor:
    """Binary cross entropy loss via autograd: -[y*log(p) + (1-y)*log(1-p)]."""
    yt_t = y_true if isinstance(y_true, Tensor) else Tensor(y_true, dtype=y_pred.dtype)
    loss = -1.0 * (yt_t * y_pred.log(eps=eps) + (1.0 - yt_t) * (1.0 - y_pred).log(eps=eps))
    return loss.mean()
