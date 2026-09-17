"""Neural network layer implementations.

Provides standard trainable and structural building blocks:
- Dense (fully connected)
- Dropout (regularization)
- LayerNorm (normalization)
- Flatten (spatial tensor reshaping)
- Conv2D (2D spatial convolutions via vectorized im2col)
- MaxPool2D (2D spatial downsampling)
"""

from typing import Dict, Any, Optional, Union, Tuple
import numpy as np

from .base import Layer


class Dense(Layer):
    """Fully connected (dense) linear layer.

    Performs the affine transformation:
        output = input @ weights + biases

    Attributes:
        in_features (int): Number of input features per sample.
        out_features (int): Number of output units (neurons) in this layer.
        weight_init (str): Initialization strategy: 'he', 'xavier', or 'small_random'.
        use_bias (bool): Whether to include an additive bias vector.
        weights (np.ndarray): Weight matrix of shape (in_features, out_features).
        biases (Optional[np.ndarray]): Bias vector of shape (1, out_features).
        dweights (np.ndarray): Gradient with respect to weights.
        dbiases (Optional[np.ndarray]): Gradient with respect to biases.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_init: str = "he",
        use_bias: bool = True,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"Dense layer dimensions must be positive integers. Got in_features={in_features}, out_features={out_features}"
            )

        self.in_features: int = int(in_features)
        self.out_features: int = int(out_features)
        self.weight_init: str = weight_init.lower()
        self.use_bias: bool = bool(use_bias)
        self.l1_reg: float = float(l1_reg)
        self.l2_reg: float = float(l2_reg)
        self.dtype = dtype
        self.trainable: bool = True

        # Initialize weights
        self.weights: np.ndarray = self._init_weights()
        self.biases: Optional[np.ndarray] = (
            np.zeros((1, self.out_features), dtype=self.dtype) if self.use_bias else None
        )

        # Gradient buffers
        self.dweights: np.ndarray = np.zeros_like(self.weights)
        self.dbiases: Optional[np.ndarray] = (
            np.zeros_like(self.biases) if self.use_bias else None
        )

        self._params["weights"] = self.weights
        if self.use_bias:
            self._params["biases"] = self.biases

        self._grads["weights"] = self.dweights
        if self.use_bias:
            self._grads["biases"] = self.dbiases

        self._input_cache: Optional[np.ndarray] = None

    def _init_weights(self) -> np.ndarray:
        if self.weight_init in ("he", "kaiming"):
            std = np.sqrt(2.0 / self.in_features)
            w = np.random.randn(self.in_features, self.out_features) * std
        elif self.weight_init in ("xavier", "glorot"):
            std = np.sqrt(2.0 / (self.in_features + self.out_features))
            w = np.random.randn(self.in_features, self.out_features) * std
        elif self.weight_init == "small_random":
            w = np.random.randn(self.in_features, self.out_features) * 0.01
        else:
            raise ValueError(
                f"Unknown weight_init '{self.weight_init}'. Supported: 'he', 'xavier', 'small_random'."
            )
        return w.astype(self.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.ndim < 2:
            raise ValueError(
                f"Dense layer expected array with at least 2 dimensions (*, {self.in_features}), "
                f"but received array with shape {x_arr.shape} ({x_arr.ndim} dimensions)."
            )
        if x_arr.shape[-1] != self.in_features:
            raise ValueError(
                f"Dense layer expected {self.in_features} features (last dim), "
                f"but received input with {x_arr.shape[-1]} features (shape {x_arr.shape})."
            )

        self._input_cache = x_arr
        output = x_arr @ self.weights
        if self.use_bias and self.biases is not None:
            output = output + self.biases
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input_cache is None:
            raise RuntimeError("Dense.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype)
        orig_shape = self._input_cache.shape
        expected_shape = (*orig_shape[:-1], self.out_features)

        if grad_out.shape != expected_shape:
            raise ValueError(
                f"Gradient shape mismatch in Dense backward. Expected {expected_shape}, "
                f"got {grad_out.shape}."
            )

        x_flat = self._input_cache.reshape(-1, self.in_features)
        grad_flat = grad_out.reshape(-1, self.out_features)

        np.copyto(self.dweights, x_flat.T @ grad_flat)

        # L1 and L2 regularization gradients
        if self.l2_reg > 0:
            self.dweights += self.l2_reg * self.weights
        if self.l1_reg > 0:
            self.dweights += self.l1_reg * np.sign(self.weights)

        if self.use_bias and self.dbiases is not None:
            np.copyto(self.dbiases, np.sum(grad_flat, axis=0, keepdims=True))

        grad_input = (grad_flat @ self.weights.T).reshape(orig_shape)
        return grad_input

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Dense",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "weight_init": self.weight_init,
            "use_bias": self.use_bias,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Dense":
        return cls(
            in_features=config["in_features"],
            out_features=config["out_features"],
            weight_init=config.get("weight_init", "he"),
            use_bias=config.get("use_bias", True),
        )


class Dropout(Layer):
    """Inverted Dropout regularization layer.

    During training, randomly sets units to 0 with probability `drop_rate`
    and scales remaining units by 1 / (1 - drop_rate).
    During evaluation, behaves as an identity function.
    """

    def __init__(self, drop_rate: float = 0.5, rate: Optional[float] = None) -> None:
        super().__init__()
        effective_rate = rate if rate is not None else drop_rate
        if not (0.0 <= effective_rate < 1.0):
            raise ValueError(f"drop_rate must be in [0.0, 1.0), got {effective_rate}")
        self.drop_rate: float = float(effective_rate)
        self.trainable: bool = False
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if self.training and self.drop_rate > 0.0:
            keep_prob = 1.0 - self.drop_rate
            self._mask = (np.random.rand(*x_arr.shape) < keep_prob) / keep_prob
            return x_arr * self._mask
        self._mask = None
        return x_arr

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_out = np.asarray(grad_output, dtype=np.float32)
        if self.training and self._mask is not None:
            return grad_out * self._mask
        return grad_out

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Dropout", "drop_rate": self.drop_rate}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Dropout":
        return cls(drop_rate=config.get("drop_rate", 0.5))


class LayerNorm(Layer):
    """Layer Normalization across features dimension.

    Normalizes inputs across the last axis:
        y = (x - mean) / sqrt(var + eps) * gamma + beta
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape: int = int(normalized_shape)
        self.eps: float = float(eps)
        self.trainable: bool = True

        # Learnable scale (gamma) and shift (beta)
        self.gamma: np.ndarray = np.ones((1, self.normalized_shape), dtype=np.float32)
        self.beta: np.ndarray = np.zeros((1, self.normalized_shape), dtype=np.float32)

        self.dgamma: np.ndarray = np.zeros_like(self.gamma)
        self.dbeta: np.ndarray = np.zeros_like(self.beta)

        self._params["gamma"] = self.gamma
        self._params["beta"] = self.beta
        self._grads["gamma"] = self.dgamma
        self._grads["beta"] = self.dbeta

        self._cache_x_norm: Optional[np.ndarray] = None
        self._cache_std: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.shape[-1] != self.normalized_shape:
            raise ValueError(
                f"LayerNorm expected last dimension {self.normalized_shape}, got shape {x_arr.shape}"
            )

        mean = np.mean(x_arr, axis=-1, keepdims=True)
        var = np.var(x_arr, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)

        x_norm = (x_arr - mean) / std
        self._cache_x_norm = x_norm
        self._cache_std = std

        return x_norm * self.gamma + self.beta

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_out = np.asarray(grad_output, dtype=np.float32)
        x_norm = self._cache_x_norm
        std = self._cache_std

        if x_norm is None or std is None:
            raise RuntimeError("LayerNorm.backward called before forward pass.")

        # Gradients for gamma and beta (sum across all batch and sequence dimensions)
        reduce_axes = tuple(range(grad_out.ndim - 1))
        np.copyto(self.dgamma, np.sum(grad_out * x_norm, axis=reduce_axes, keepdims=False).reshape(1, -1))
        np.copyto(self.dbeta, np.sum(grad_out, axis=reduce_axes, keepdims=False).reshape(1, -1))

        # Gradient with respect to x
        d_xnorm = grad_out * self.gamma
        d = self.normalized_shape
        dx = (1.0 / std) * (
            d_xnorm
            - np.mean(d_xnorm, axis=-1, keepdims=True)
            - x_norm * np.mean(d_xnorm * x_norm, axis=-1, keepdims=True)
        )
        return dx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "LayerNorm",
            "normalized_shape": self.normalized_shape,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LayerNorm":
        return cls(
            normalized_shape=config["normalized_shape"],
            eps=config.get("eps", 1e-5),
        )


class Flatten(Layer):
    """Flattens a multi-dimensional tensor to 2D (batch_size, -1)."""

    def __init__(self) -> None:
        super().__init__()
        self.trainable: bool = False
        self._orig_shape: Optional[Tuple[int, ...]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        self._orig_shape = x_arr.shape
        batch_size = x_arr.shape[0]
        return x_arr.reshape(batch_size, -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._orig_shape is None:
            raise RuntimeError("Flatten.backward called before forward pass.")
        return np.asarray(grad_output, dtype=np.float32).reshape(self._orig_shape)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Flatten"}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Flatten":
        return cls()


def _resolve_padding(padding: Union[int, str], k_size: int, dilation: int = 1) -> int:
    """Resolve integer or string padding ('same' or 'valid') to pixel count."""
    if isinstance(padding, str):
        pad_str = padding.lower().strip()
        if pad_str == "same":
            k_eff = (k_size - 1) * dilation + 1
            return (k_eff - 1) // 2
        elif pad_str == "valid":
            return 0
        else:
            raise ValueError(f"Unknown padding mode: '{padding}'. Supported: 'same', 'valid', or int.")
    return int(padding)


from .accel import im2col_indices, col2im_indices

def _im2col_indices(
    x: np.ndarray,
    kh: int,
    kw: int,
    padding: int = 1,
    stride: int = 1,
    dilation: int = 1,
) -> Tuple[np.ndarray, int, int]:
    """Extract sliding patches using accelerated Numba JIT / vectorized NumPy backend."""
    return im2col_indices(x, kh, kw, padding=padding, stride=stride, dilation=dilation)


def _col2im_indices(
    cols: np.ndarray,
    x_shape: Tuple[int, int, int, int],
    kh: int,
    kw: int,
    padding: int = 1,
    stride: int = 1,
    dilation: int = 1,
) -> np.ndarray:
    """Accumulate gradient columns back into image tensor using accelerated backend."""
    return col2im_indices(cols, x_shape, kh, kw, padding=padding, stride=stride, dilation=dilation)


class Conv2D(Layer):
    """2D Spatial Convolution Layer.

    Computes 2D cross-correlation across (batch_size, in_channels, height, width).
    Supports customizable kernel size, stride, dilation, padding modes ('same', 'valid', or int),
    and L1/L2 regularization.

    Attributes:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output filter channels.
        kernel_size (int): Size of square kernel (k, k).
        stride (int): Stride step size along height and width.
        padding (Union[int, str]): Zero-padding ('same', 'valid', or integer count).
        dilation (int): Spacing between kernel points.
        l1_reg (float): L1 regularization factor.
        l2_reg (float): L2 regularization factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[int, str] = 0,
        dilation: int = 1,
        weight_init: str = "he",
        use_bias: bool = True,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.in_channels: int = int(in_channels)
        self.out_channels: int = int(out_channels)
        self.kernel_size: int = int(kernel_size)
        self.stride: int = int(stride)
        self.padding_raw: Union[int, str] = padding
        self.dilation: int = max(1, int(dilation))
        self.resolved_padding: int = _resolve_padding(padding, self.kernel_size, self.dilation)
        self.weight_init: str = weight_init.lower()
        self.use_bias: bool = bool(use_bias)
        self.l1_reg: float = float(l1_reg)
        self.l2_reg: float = float(l2_reg)
        self.dtype = dtype
        self.trainable: bool = True

        # Weight shape: (out_channels, in_channels, kernel_size, kernel_size)
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        std = np.sqrt(2.0 / fan_in) if self.weight_init == "he" else 0.01
        self.weights: np.ndarray = (
            np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size) * std
        ).astype(self.dtype)

        self.biases: Optional[np.ndarray] = (
            np.zeros((self.out_channels, 1), dtype=self.dtype) if self.use_bias else None
        )

        self.dweights: np.ndarray = np.zeros_like(self.weights)
        self.dbiases: Optional[np.ndarray] = (
            np.zeros_like(self.biases) if self.use_bias else None
        )

        self._params["weights"] = self.weights
        if self.use_bias:
            self._params["biases"] = self.biases

        self._grads["weights"] = self.dweights
        if self.use_bias:
            self._grads["biases"] = self.dbiases

        self._x_shape: Optional[Tuple[int, int, int, int]] = None
        self._x_cols: Optional[np.ndarray] = None
        self._out_h: int = 0
        self._out_w: int = 0

    @property
    def padding(self) -> int:
        return self.resolved_padding

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.ndim != 4:
            raise ValueError(f"Conv2D expects 4D input (N, C, H, W), got shape {x_arr.shape}")
        if x_arr.shape[1] != self.in_channels:
            raise ValueError(
                f"Conv2D expects {self.in_channels} input channels, got {x_arr.shape[1]}"
            )

        self._x_shape = x_arr.shape
        N = x_arr.shape[0]

        # im2col with dilation
        cols, out_h, out_w = _im2col_indices(
            x_arr,
            self.kernel_size,
            self.kernel_size,
            self.resolved_padding,
            self.stride,
            self.dilation,
        )
        self._x_cols = cols
        self._out_h = out_h
        self._out_w = out_w

        # (out_channels, in_channels * k * k)
        w_row = self.weights.reshape(self.out_channels, -1)

        # Output matrix: (N * out_h * out_w, out_channels)
        out = cols @ w_row.T
        if self.use_bias and self.biases is not None:
            out += self.biases.T

        # Reshape to (N, out_channels, out_h, out_w)
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x_cols is None or self._x_shape is None:
            raise RuntimeError("Conv2D.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype)
        N = self._x_shape[0]

        # Reshape grad_out to (N * out_h * out_w, out_channels)
        grad_flat = grad_out.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # dW = grad_flat.T @ cols
        dw = grad_flat.T @ self._x_cols
        np.copyto(self.dweights, dw.reshape(self.weights.shape))

        # L1 and L2 regularization gradients
        if self.l2_reg > 0:
            self.dweights += self.l2_reg * self.weights
        if self.l1_reg > 0:
            self.dweights += self.l1_reg * np.sign(self.weights)

        # db = sum across spatial and batch
        if self.use_bias and self.dbiases is not None:
            db = np.sum(grad_flat, axis=0, keepdims=True).T
            np.copyto(self.dbiases, db)

        # dX_cols = grad_flat @ W
        w_row = self.weights.reshape(self.out_channels, -1)
        dx_cols = grad_flat @ w_row

        # col2im with dilation
        dx = _col2im_indices(
            dx_cols,
            self._x_shape,
            self.kernel_size,
            self.kernel_size,
            self.resolved_padding,
            self.stride,
            self.dilation,
        )
        return dx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Conv2D",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding_raw,
            "dilation": self.dilation,
            "weight_init": self.weight_init,
            "use_bias": self.use_bias,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Conv2D":
        return cls(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_size=config.get("kernel_size", 3),
            stride=config.get("stride", 1),
            padding=config.get("padding", 0),
            dilation=config.get("dilation", 1),
            weight_init=config.get("weight_init", "he"),
            use_bias=config.get("use_bias", True),
            l1_reg=config.get("l1_reg", 0.0),
            l2_reg=config.get("l2_reg", 0.0),
        )


class MaxPool2D(Layer):
    """2D Spatial Max Pooling layer."""

    def __init__(self, pool_size: int = 2, stride: Optional[int] = None) -> None:
        super().__init__()
        self.pool_size: int = int(pool_size)
        self.stride: int = int(stride) if stride is not None else self.pool_size
        self.trainable: bool = False
        self._x_cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim != 4:
            raise ValueError(f"MaxPool2D expects 4D input (N, C, H, W), got shape {x_arr.shape}")

        self._x_cache = x_arr
        N, C, H, W = x_arr.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        out = np.zeros((N, C, out_h, out_w), dtype=np.float32)
        for i in range(out_h):
            h_start = i * self.stride
            h_end = h_start + self.pool_size
            for j in range(out_w):
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                out[:, :, i, j] = np.max(x_arr[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x_cache is None:
            raise RuntimeError("MaxPool2D.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=np.float32)
        x = self._x_cache
        N, C, H, W = x.shape
        out_h, out_w = grad_out.shape[2], grad_out.shape[3]

        dx = np.zeros_like(x)
        for i in range(out_h):
            h_start = i * self.stride
            h_end = h_start + self.pool_size
            for j in range(out_w):
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                patch = x[:, :, h_start:h_end, w_start:w_end]
                max_val = np.max(patch, axis=(2, 3), keepdims=True)
                mask = (patch == max_val)
                # Normalize mask if multiple positions share the exact max
                mask_norm = mask / np.sum(mask, axis=(2, 3), keepdims=True)
                dx[:, :, h_start:h_end, w_start:w_end] += mask_norm * grad_out[:, :, i:i+1, j:j+1]

        return dx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "MaxPool2D",
            "pool_size": self.pool_size,
            "stride": self.stride,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MaxPool2D":
        return cls(
            pool_size=config.get("pool_size", 2),
            stride=config.get("stride", 2),
        )


class DendriticDense(Layer):
    """Multi-Compartment Dendritic Pyramidal Neuron Layer.

    Biological pyramidal neurons process incoming signals through multiple distinct
    dendritic branches before integrating them at the soma (cell body). Each dendritic
    branch computes localized non-linear multiplicative gating:
        branch_k = (x @ W_signal[k] + b_signal[k]) * SiLU(x @ W_gate[k] + b_gate[k])
        soma = sum_{k=1}^K branch_k + b_soma

    This architecture allows a SINGLE neuron layer to solve complex non-linear problems
    (such as XOR, circular/donut boundaries, and higher-order multiplicative interactions)
    without requiring deep multi-layer neural network stacks.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output units (neurons).
        num_branches (int): Number of dendritic branches per neuron (default: 2).
        use_bias (bool): Whether to include soma and branch additive biases.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_branches: int = 2,
        use_bias: bool = True,
        weight_init: str = "he",
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or num_branches <= 0:
            raise ValueError("Dimensions and num_branches must be positive integers.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_branches = int(num_branches)
        self.use_bias = bool(use_bias)
        self.weight_init = weight_init.lower()
        self.dtype = dtype
        self.trainable = True

        # Initialize branch signal and gating weights
        std = np.sqrt(2.0 / (self.in_features * self.num_branches))
        self.w_signal = (np.random.randn(self.num_branches, self.in_features, self.out_features) * std).astype(self.dtype)
        self.w_gate = (np.random.randn(self.num_branches, self.in_features, self.out_features) * std).astype(self.dtype)
        self.b_signal = np.zeros((self.num_branches, 1, self.out_features), dtype=self.dtype) if self.use_bias else None
        self.b_gate = np.zeros((self.num_branches, 1, self.out_features), dtype=self.dtype) if self.use_bias else None
        self.b_soma = np.zeros((1, self.out_features), dtype=self.dtype) if self.use_bias else None

        # Gradients
        self.dw_signal = np.zeros_like(self.w_signal)
        self.dw_gate = np.zeros_like(self.w_gate)
        self.db_signal = np.zeros_like(self.b_signal) if self.use_bias else None
        self.db_gate = np.zeros_like(self.b_gate) if self.use_bias else None
        self.db_soma = np.zeros_like(self.b_soma) if self.use_bias else None

        self._params["w_signal"] = self.w_signal
        self._params["w_gate"] = self.w_gate
        self._grads["w_signal"] = self.dw_signal
        self._grads["w_gate"] = self.dw_gate

        if self.use_bias:
            self._params["b_signal"] = self.b_signal
            self._params["b_gate"] = self.b_gate
            self._params["b_soma"] = self.b_soma
            self._grads["b_signal"] = self.db_signal
            self._grads["b_gate"] = self.db_gate
            self._grads["b_soma"] = self.db_soma

        self._x_cache = None
        self._signal_cache = None
        self._gate_raw_cache = None
        self._gate_act_cache = None
        self._orig_shape = None

    @staticmethod
    def _silu(z: np.ndarray) -> np.ndarray:
        return z / (1.0 + np.exp(-np.clip(z, -88.0, 88.0)))

    @staticmethod
    def _silu_deriv(z: np.ndarray) -> np.ndarray:
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -88.0, 88.0)))
        return s * (1.0 + z * (1.0 - s))

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.ndim < 2:
            raise ValueError(f"DendriticDense layer expected input with >= 2 dimensions, got shape {x_arr.shape}")
        if x_arr.shape[-1] != self.in_features:
            raise ValueError(
                f"DendriticDense layer expected {self.in_features} features, but got input with shape {x_arr.shape}"
            )

        self._orig_shape = x_arr.shape
        x_flat = x_arr.reshape(-1, self.in_features)
        self._x_cache = x_flat

        batch_size = x_flat.shape[0]
        signals = np.zeros((self.num_branches, batch_size, self.out_features), dtype=self.dtype)
        gate_raws = np.zeros((self.num_branches, batch_size, self.out_features), dtype=self.dtype)
        gate_acts = np.zeros((self.num_branches, batch_size, self.out_features), dtype=self.dtype)

        soma = np.zeros((batch_size, self.out_features), dtype=self.dtype)
        if self.use_bias and self.b_soma is not None:
            soma += self.b_soma

        for k in range(self.num_branches):
            sig = x_flat @ self.w_signal[k]
            gt = x_flat @ self.w_gate[k]
            if self.use_bias:
                sig = sig + self.b_signal[k]
                gt = gt + self.b_gate[k]

            act_g = self._silu(gt)
            signals[k] = sig
            gate_raws[k] = gt
            gate_acts[k] = act_g

            # Non-linear dendritic branch integration
            branch_out = sig * act_g
            soma += branch_out

        self._signal_cache = signals
        self._gate_raw_cache = gate_raws
        self._gate_act_cache = gate_acts

        out_shape = list(self._orig_shape[:-1]) + [self.out_features]
        return soma.reshape(out_shape)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x_cache is None:
            raise RuntimeError("DendriticDense.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype).reshape(-1, self.out_features)
        x_flat = self._x_cache

        # Reset parameter gradients
        self.dw_signal.fill(0.0)
        self.dw_gate.fill(0.0)
        if self.use_bias:
            self.db_signal.fill(0.0)
            self.db_gate.fill(0.0)
            self.db_soma[:] = np.sum(grad_out, axis=0, keepdims=True)

        dx = np.zeros_like(x_flat)

        for k in range(self.num_branches):
            sig = self._signal_cache[k]
            gt_raw = self._gate_raw_cache[k]
            act_g = self._gate_act_cache[k]

            d_sig = grad_out * act_g
            d_act_g = grad_out * sig
            d_gt = d_act_g * self._silu_deriv(gt_raw)

            self.dw_signal[k] += x_flat.T @ d_sig
            self.dw_gate[k] += x_flat.T @ d_gt

            if self.use_bias:
                self.db_signal[k] += np.sum(d_sig, axis=0, keepdims=True)
                self.db_gate[k] += np.sum(d_gt, axis=0, keepdims=True)

            dx += d_sig @ self.w_signal[k].T + d_gt @ self.w_gate[k].T

        return dx.reshape(self._orig_shape)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "DendriticDense",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "num_branches": self.num_branches,
            "use_bias": self.use_bias,
            "weight_init": self.weight_init,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DendriticDense":
        return cls(
            in_features=config["in_features"],
            out_features=config["out_features"],
            num_branches=config.get("num_branches", 2),
            use_bias=config.get("use_bias", True),
            weight_init=config.get("weight_init", "he"),
        )


class ChebyshevKAN(Layer):
    """Kolmogorov-Arnold Network (KAN) layer with orthogonal Chebyshev polynomial basis.

    Replaces fixed node activations with learnable continuous non-linear functions on every
    synaptic connection using orthogonal Chebyshev polynomials of degree K:
        y_j = sum_{i=1}^{in_features} phi_{ij}(x_i) + b_j
        phi_{ij}(x_i) = w_base_{ij} * SiLU(x_i) + sum_{k=0}^degree c_{ijk} * T_k(tanh(x_i))

    Attributes:
        in_features (int): Input feature dimensionality.
        out_features (int): Output feature dimensionality.
        degree (int): Maximum Chebyshev polynomial degree (default: 4).
        use_bias (bool): Whether to include an additive bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 4,
        use_bias: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or degree <= 0:
            raise ValueError("Dimensions and degree must be positive integers.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.degree = int(degree)
        self.use_bias = bool(use_bias)
        self.dtype = dtype
        self.trainable = True

        # Base linear projection: (in_features, out_features)
        std_base = np.sqrt(2.0 / (self.in_features + self.out_features))
        self.w_base = (np.random.randn(self.in_features, self.out_features) * std_base).astype(self.dtype)

        # Polynomial coefficients: (in_features, out_features, degree + 1)
        std_poly = 1.0 / (np.sqrt(self.in_features) * (self.degree + 1))
        self.c_poly = (np.random.randn(self.in_features, self.out_features, self.degree + 1) * std_poly).astype(self.dtype)

        self.biases = np.zeros((1, self.out_features), dtype=self.dtype) if self.use_bias else None

        # Gradients
        self.dw_base = np.zeros_like(self.w_base)
        self.dc_poly = np.zeros_like(self.c_poly)
        self.dbiases = np.zeros_like(self.biases) if self.use_bias else None

        self._params["w_base"] = self.w_base
        self._params["c_poly"] = self.c_poly
        self._grads["w_base"] = self.dw_base
        self._grads["c_poly"] = self.dc_poly

        if self.use_bias:
            self._params["biases"] = self.biases
            self._grads["biases"] = self.dbiases

        self._x_cache = None
        self._u_cache = None
        self._t_cache = None
        self._orig_shape = None

    @staticmethod
    def _silu(z: np.ndarray) -> np.ndarray:
        return z / (1.0 + np.exp(-np.clip(z, -88.0, 88.0)))

    @staticmethod
    def _silu_deriv(z: np.ndarray) -> np.ndarray:
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -88.0, 88.0)))
        return s * (1.0 + z * (1.0 - s))

    def _compute_chebyshev_basis(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Chebyshev polynomials T_k(u) and their derivatives T'_k(u) up to degree."""
        B, D = u.shape
        K = self.degree
        T = np.empty((B, D, K + 1), dtype=self.dtype)
        T_prime = np.empty((B, D, K + 1), dtype=self.dtype)

        # T_0(u) = 1, T'_0(u) = 0
        T[:, :, 0] = 1.0
        T_prime[:, :, 0] = 0.0

        if K >= 1:
            # T_1(u) = u, T'_1(u) = 1
            T[:, :, 1] = u
            T_prime[:, :, 1] = 1.0

        # Recurrence: T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
        # Derivative: T'_k(u) = 2*T_{k-1}(u) + 2*u*T'_{k-1}(u) - T'_{k-2}(u)
        for k in range(2, K + 1):
            T[:, :, k] = 2.0 * u * T[:, :, k - 1] - T[:, :, k - 2]
            T_prime[:, :, k] = 2.0 * T[:, :, k - 1] + 2.0 * u * T_prime[:, :, k - 1] - T_prime[:, :, k - 2]

        return T, T_prime

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.ndim < 2:
            raise ValueError(f"ChebyshevKAN layer expected input with >= 2 dimensions, got shape {x_arr.shape}")
        if x_arr.shape[-1] != self.in_features:
            raise ValueError(
                f"ChebyshevKAN layer expected {self.in_features} features, got {x_arr.shape}"
            )

        self._orig_shape = x_arr.shape
        x_flat = x_arr.reshape(-1, self.in_features)
        self._x_cache = x_flat

        # 1. Base branch: SiLU(x) @ w_base
        x_base = self._silu(x_flat)
        base_out = x_base @ self.w_base

        # 2. Chebyshev polynomial branch: normalize to [-1, 1] via tanh
        u = np.tanh(x_flat)
        self._u_cache = u
        T, T_prime = self._compute_chebyshev_basis(u)
        self._t_cache = (T, T_prime)

        # Tensor contraction: y_poly[b, j] = sum_{i, k} T[b, i, k] * c_poly[i, j, k]
        poly_out = np.einsum("bik,ijk->bj", T, self.c_poly)

        out = base_out + poly_out
        if self.use_bias and self.biases is not None:
            out += self.biases

        out_shape = list(self._orig_shape[:-1]) + [self.out_features]
        return out.reshape(out_shape)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x_cache is None:
            raise RuntimeError("ChebyshevKAN.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype).reshape(-1, self.out_features)
        x_flat = self._x_cache
        u = self._u_cache
        T, T_prime = self._t_cache

        # Bias gradient
        if self.use_bias and self.dbiases is not None:
            self.dbiases[:] = np.sum(grad_out, axis=0, keepdims=True)

        # Base branch gradients
        x_base = self._silu(x_flat)
        self.dw_base[:] = x_base.T @ grad_out
        dx_base = (grad_out @ self.w_base.T) * self._silu_deriv(x_flat)

        # Polynomial branch gradients
        self.dc_poly[:] = np.einsum("bik,bj->ijk", T, grad_out)
        dT = np.einsum("bj,ijk->bik", grad_out, self.c_poly)
        du = np.sum(dT * T_prime, axis=-1)
        dx_poly = du * (1.0 - u * u)

        dx = dx_base + dx_poly
        return dx.reshape(self._orig_shape)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ChebyshevKAN",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "degree": self.degree,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ChebyshevKAN":
        return cls(
            in_features=config["in_features"],
            out_features=config["out_features"],
            degree=config.get("degree", 4),
            use_bias=config.get("use_bias", True),
        )


class BifurcatedDense(Layer):
    """Bifurcated Threshold-Gated Dynamic Routing Layer.

    Applies dynamic neuron-level conditional routing based on signal thresholding.
    If the signal is sub-threshold, computation routes through Pathway B (low-intensity regime);
    if supra-threshold, computation routes through Pathway A (high-intensity regime).

    Formula:
        g = Sigmoid((x @ W_gate + b_gate) / temperature)
        h_A = x @ W_high + b_high
        h_B = x @ W_low + b_low
        y = g * h_A + (1 - g) * h_B

    Args:
        in_features (int): Input dimensionality.
        out_features (int): Output dimensionality.
        temperature (float): Softness temperature for gate transitions (default: 1.0).
        use_bias (bool): Whether to include bias terms.
        dtype: Data type (default: float32).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        temperature: float = 1.0,
        use_bias: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Dimensions must be positive integers.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.use_bias = bool(use_bias)
        self.dtype = dtype
        self.trainable = True

        std = np.sqrt(2.0 / (self.in_features + self.out_features))
        self.w_high = (np.random.randn(self.in_features, self.out_features) * std).astype(self.dtype)
        self.w_low = (np.random.randn(self.in_features, self.out_features) * std).astype(self.dtype)
        self.w_gate = (np.random.randn(self.in_features, self.out_features) * std).astype(self.dtype)

        self.b_high = np.zeros((1, self.out_features), dtype=self.dtype) if self.use_bias else None
        self.b_low = np.zeros((1, self.out_features), dtype=self.dtype) if self.use_bias else None
        self.b_gate = np.zeros((1, self.out_features), dtype=self.dtype) if self.use_bias else None

        self.dw_high = np.zeros_like(self.w_high)
        self.dw_low = np.zeros_like(self.w_low)
        self.dw_gate = np.zeros_like(self.w_gate)

        self.db_high = np.zeros_like(self.b_high) if self.use_bias else None
        self.db_low = np.zeros_like(self.b_low) if self.use_bias else None
        self.db_gate = np.zeros_like(self.b_gate) if self.use_bias else None

        self._params["w_high"] = self.w_high
        self._params["w_low"] = self.w_low
        self._params["w_gate"] = self.w_gate
        self._grads["w_high"] = self.dw_high
        self._grads["w_low"] = self.dw_low
        self._grads["w_gate"] = self.dw_gate

        if self.use_bias:
            self._params["b_high"] = self.b_high
            self._params["b_low"] = self.b_low
            self._params["b_gate"] = self.b_gate
            self._grads["b_high"] = self.db_high
            self._grads["b_low"] = self.db_low
            self._grads["b_gate"] = self.db_gate

        self._x_cache = None
        self._g_cache = None
        self._h_high_cache = None
        self._h_low_cache = None
        self._orig_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.ndim < 2:
            raise ValueError(f"BifurcatedDense expects input with >= 2 dims, got {x_arr.shape}")
        if x_arr.shape[-1] != self.in_features:
            raise ValueError(f"BifurcatedDense expected {self.in_features} features, got {x_arr.shape}")

        self._orig_shape = x_arr.shape
        x_flat = x_arr.reshape(-1, self.in_features)
        self._x_cache = x_flat

        z_gate = x_flat @ self.w_gate
        if self.use_bias and self.b_gate is not None:
            z_gate += self.b_gate
        scaled_gate = z_gate / self.temperature
        g = 1.0 / (1.0 + np.exp(-np.clip(scaled_gate, -88.0, 88.0)))
        self._g_cache = g

        h_high = x_flat @ self.w_high
        if self.use_bias and self.b_high is not None:
            h_high += self.b_high
        self._h_high_cache = h_high

        h_low = x_flat @ self.w_low
        if self.use_bias and self.b_low is not None:
            h_low += self.b_low
        self._h_low_cache = h_low

        out = g * h_high + (1.0 - g) * h_low
        out_shape = list(self._orig_shape[:-1]) + [self.out_features]
        return out.reshape(out_shape)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x_cache is None:
            raise RuntimeError("BifurcatedDense.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype).reshape(-1, self.out_features)
        x_flat = self._x_cache
        g = self._g_cache
        h_high = self._h_high_cache
        h_low = self._h_low_cache

        dh_high = grad_out * g
        dh_low = grad_out * (1.0 - g)
        dg = grad_out * (h_high - h_low)
        dz_gate = (dg * g * (1.0 - g)) / self.temperature

        self.dw_high[:] = x_flat.T @ dh_high
        self.dw_low[:] = x_flat.T @ dh_low
        self.dw_gate[:] = x_flat.T @ dz_gate

        if self.use_bias:
            self.db_high[:] = np.sum(dh_high, axis=0, keepdims=True)
            self.db_low[:] = np.sum(dh_low, axis=0, keepdims=True)
            self.db_gate[:] = np.sum(dz_gate, axis=0, keepdims=True)

        dx = dh_high @ self.w_high.T + dh_low @ self.w_low.T + dz_gate @ self.w_gate.T
        return dx.reshape(self._orig_shape)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "BifurcatedDense",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "temperature": self.temperature,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BifurcatedDense":
        return cls(
            in_features=config["in_features"],
            out_features=config["out_features"],
            temperature=config.get("temperature", 1.0),
            use_bias=config.get("use_bias", True),
        )


class ReflectiveDense(Layer):
    """Iterative Cortical Feedback & Reflection Layer.

    Models biological top-down reflection / recurrent predictive coding.
    The feedforward representation is iteratively reflected back to the input
    space via a feedback projection to refine the hypothesis before final emission.

    Algorithm (for step k = 0 ... K-1):
        1. h_raw^(k) = x^(k) @ W_fwd + b_fwd
        2. h^(k) = SiLU(h_raw^(k))
        3. If k < K - 1:
           r_raw^(k) = h^(k) @ W_ref + b_ref
           r^(k) = Tanh(r_raw^(k))
           x^(k+1) = x^(0) + alpha * r^(k)
        Output = h^(K-1)

    Args:
        in_features (int): Input dimensionality.
        out_features (int): Output dimensionality.
        reflection_steps (int): Number of iterative reflection passes K (default: 2).
        alpha (float): Reflection feedback blending rate (default: 0.5).
        use_bias (bool): Whether to include additive bias vectors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        reflection_steps: int = 2,
        alpha: float = 0.5,
        use_bias: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Dimensions must be positive integers.")
        if reflection_steps < 1:
            raise ValueError("reflection_steps must be at least 1.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.reflection_steps = int(reflection_steps)
        self.alpha = float(alpha)
        self.use_bias = bool(use_bias)
        self.dtype = dtype
        self.trainable = True

        std_fwd = np.sqrt(2.0 / (self.in_features + self.out_features))
        std_ref = np.sqrt(2.0 / (self.out_features + self.in_features))

        self.w_fwd = (np.random.randn(self.in_features, self.out_features) * std_fwd).astype(self.dtype)
        self.w_ref = (np.random.randn(self.out_features, self.in_features) * std_ref).astype(self.dtype)

        self.b_fwd = np.zeros((1, self.out_features), dtype=self.dtype) if self.use_bias else None
        self.b_ref = np.zeros((1, self.in_features), dtype=self.dtype) if self.use_bias else None

        self.dw_fwd = np.zeros_like(self.w_fwd)
        self.dw_ref = np.zeros_like(self.w_ref)
        self.db_fwd = np.zeros_like(self.b_fwd) if self.use_bias else None
        self.db_ref = np.zeros_like(self.b_ref) if self.use_bias else None

        self._params["w_fwd"] = self.w_fwd
        self._params["w_ref"] = self.w_ref
        self._grads["w_fwd"] = self.dw_fwd
        self._grads["w_ref"] = self.dw_ref

        if self.use_bias:
            self._params["b_fwd"] = self.b_fwd
            self._params["b_ref"] = self.b_ref
            self._grads["b_fwd"] = self.db_fwd
            self._grads["b_ref"] = self.db_ref

        self._cache = None
        self._orig_shape = None

    @staticmethod
    def _silu(z: np.ndarray) -> np.ndarray:
        return z / (1.0 + np.exp(-np.clip(z, -88.0, 88.0)))

    @staticmethod
    def _silu_deriv(z: np.ndarray) -> np.ndarray:
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -88.0, 88.0)))
        return s * (1.0 + z * (1.0 - s))

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.ndim < 2:
            raise ValueError(f"ReflectiveDense expects input with >= 2 dims, got {x_arr.shape}")
        if x_arr.shape[-1] != self.in_features:
            raise ValueError(f"ReflectiveDense expected {self.in_features} features, got {x_arr.shape}")

        self._orig_shape = x_arr.shape
        x_flat = x_arr.reshape(-1, self.in_features)

        x_states = [x_flat]
        h_raw_states = []
        h_states = []
        r_raw_states = []
        r_states = []

        curr_x = x_flat
        h = None
        for k in range(self.reflection_steps):
            h_raw = curr_x @ self.w_fwd
            if self.use_bias and self.b_fwd is not None:
                h_raw += self.b_fwd
            h = self._silu(h_raw)

            h_raw_states.append(h_raw)
            h_states.append(h)

            if k < self.reflection_steps - 1:
                r_raw = h @ self.w_ref
                if self.use_bias and self.b_ref is not None:
                    r_raw += self.b_ref
                r = np.tanh(r_raw)

                r_raw_states.append(r_raw)
                r_states.append(r)

                next_x = x_flat + self.alpha * r
                x_states.append(next_x)
                curr_x = next_x

        self._cache = {
            "x_states": x_states,
            "h_raw_states": h_raw_states,
            "h_states": h_states,
            "r_raw_states": r_raw_states,
            "r_states": r_states,
        }

        out_shape = list(self._orig_shape[:-1]) + [self.out_features]
        return h.reshape(out_shape)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("ReflectiveDense.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype).reshape(-1, self.out_features)

        self.dw_fwd.fill(0.0)
        self.dw_ref.fill(0.0)
        if self.use_bias:
            self.db_fwd.fill(0.0)
            self.db_ref.fill(0.0)

        K = self.reflection_steps
        x_states = self._cache["x_states"]
        h_raw_states = self._cache["h_raw_states"]
        h_states = self._cache["h_states"]
        r_raw_states = self._cache["r_raw_states"]
        r_states = self._cache["r_states"]

        dh = grad_out.copy()
        dx_accum = np.zeros_like(x_states[0])

        for k in reversed(range(K)):
            # Backprop through SiLU at step k
            dh_raw = dh * self._silu_deriv(h_raw_states[k])

            # Accumulate fwd parameter gradients
            self.dw_fwd += x_states[k].T @ dh_raw
            if self.use_bias:
                self.db_fwd += np.sum(dh_raw, axis=0, keepdims=True)

            # Gradient to x_states[k]
            dx_k = dh_raw @ self.w_fwd.T

            if k == 0:
                dx_accum += dx_k
            else:
                # x_states[k] = x_flat + alpha * r_states[k-1]
                dx_accum += dx_k
                dr = self.alpha * dx_k
                dr_raw = dr * (1.0 - r_states[k - 1] ** 2)

                self.dw_ref += h_states[k - 1].T @ dr_raw
                if self.use_bias:
                    self.db_ref += np.sum(dr_raw, axis=0, keepdims=True)

                dh = dr_raw @ self.w_ref.T

        return dx_accum.reshape(self._orig_shape)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ReflectiveDense",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "reflection_steps": self.reflection_steps,
            "alpha": self.alpha,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ReflectiveDense":
        return cls(
            in_features=config["in_features"],
            out_features=config["out_features"],
            reflection_steps=config.get("reflection_steps", 2),
            alpha=config.get("alpha", 0.5),
            use_bias=config.get("use_bias", True),
        )


class InvertedDense(Layer):
    """Synaptic Inverter Layer with Learnable Polarity Inversion.

    Allows each individual neuron to dynamically and differentiably invert
    its effective synaptic weights and biases between standard (+W) and inverted (-W) polarities.

    Formula:
        g_inv = Sigmoid(inversion_logits)  # Range (0, 1)
        scale = 1.0 - 2.0 * g_inv          # Range (+1 to -1)
        output = (x @ weights) * scale + biases * scale

    When inversion_logits << 0: scale -> +1 (Standard Excitatory Mode)
    When inversion_logits == 0: scale -> 0  (Pruned / Inactive Mode)
    When inversion_logits >> 0: scale -> -1 (Completely Inverted Inhibitory Mode)

    Args:
        in_features (int): Input dimensionality.
        out_features (int): Output dimensionality.
        init_inverted (bool): Whether to initialize in inverted mode (default: False).
        use_bias (bool): Whether to include bias terms.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_inverted: bool = False,
        use_bias: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Dimensions must be positive integers.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.init_inverted = bool(init_inverted)
        self.use_bias = bool(use_bias)
        self.dtype = dtype
        self.trainable = True

        std = np.sqrt(2.0 / (self.in_features + self.out_features))
        self.weights = (np.random.randn(self.in_features, self.out_features) * std).astype(self.dtype)
        self.biases = np.zeros((1, self.out_features), dtype=self.dtype) if self.use_bias else None

        init_val = 3.0 if self.init_inverted else -3.0
        self.inversion_logits = (np.ones((1, self.out_features), dtype=self.dtype) * init_val)

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases) if self.use_bias else None
        self.dinversion_logits = np.zeros_like(self.inversion_logits)

        self._params["weights"] = self.weights
        self._params["inversion_logits"] = self.inversion_logits
        self._grads["weights"] = self.dweights
        self._grads["inversion_logits"] = self.dinversion_logits

        if self.use_bias:
            self._params["biases"] = self.biases
            self._grads["biases"] = self.dbiases

        self._x_cache = None
        self._g_inv_cache = None
        self._scale_cache = None
        self._h_raw_cache = None
        self._orig_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.ndim < 2:
            raise ValueError(f"InvertedDense expects input with >= 2 dims, got {x_arr.shape}")
        if x_arr.shape[-1] != self.in_features:
            raise ValueError(f"InvertedDense expected {self.in_features} features, got {x_arr.shape}")

        self._orig_shape = x_arr.shape
        x_flat = x_arr.reshape(-1, self.in_features)
        self._x_cache = x_flat

        clipped_logits = np.clip(self.inversion_logits, -88.0, 88.0)
        g_inv = 1.0 / (1.0 + np.exp(-clipped_logits))
        scale = 1.0 - 2.0 * g_inv

        h_raw = x_flat @ self.weights
        if self.use_bias and self.biases is not None:
            h_raw += self.biases

        out = h_raw * scale

        self._g_inv_cache = g_inv
        self._scale_cache = scale
        self._h_raw_cache = h_raw

        out_shape = list(self._orig_shape[:-1]) + [self.out_features]
        return out.reshape(out_shape)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x_cache is None:
            raise RuntimeError("InvertedDense.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype).reshape(-1, self.out_features)
        x_flat = self._x_cache
        g_inv = self._g_inv_cache
        scale = self._scale_cache
        h_raw = self._h_raw_cache

        dh_raw = grad_out * scale
        dscale = np.sum(grad_out * h_raw, axis=0, keepdims=True)
        dg_inv = dscale * (-2.0)

        self.dinversion_logits[:] = dg_inv * g_inv * (1.0 - g_inv)
        self.dweights[:] = x_flat.T @ dh_raw
        if self.use_bias:
            self.dbiases[:] = np.sum(dh_raw, axis=0, keepdims=True)

        dx = dh_raw @ self.weights.T
        return dx.reshape(self._orig_shape)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "InvertedDense",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "init_inverted": self.init_inverted,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "InvertedDense":
        return cls(
            in_features=config["in_features"],
            out_features=config["out_features"],
            init_inverted=config.get("init_inverted", False),
            use_bias=config.get("use_bias", True),
        )


class Tensor4DDense(Layer):
    """4-Dimensional Hyper-Tensor Spacetime Dense Layer.

    Processes 4D hypercube tensors of shape (Batch, Dim1, Dim2, In_Channels).
    Maintains 4-dimensional geometric and spatio-temporal structure, performing
    affine transformation and feature channel contraction across the hyper-manifold.

    Formula:
        Y_{b, i, j, c_out} = sum_{c_in} X_{b, i, j, c_in} * W_{c_in, c_out} + Bias_{c_out}

    Args:
        in_channels (int): Dimensionality of the 4th tensor axis (input channels/features).
        out_channels (int): Output feature dimensionality on the 4th tensor axis.
        use_bias (bool): Whether to include additive bias vector on the feature axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bias: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive integers.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.use_bias = bool(use_bias)
        self.dtype = dtype
        self.trainable = True

        std = np.sqrt(2.0 / (self.in_channels + self.out_channels))
        self.weights = (np.random.randn(self.in_channels, self.out_channels) * std).astype(self.dtype)
        self.biases = np.zeros((1, 1, 1, self.out_channels), dtype=self.dtype) if self.use_bias else None

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases) if self.use_bias else None

        self._params["weights"] = self.weights
        self._grads["weights"] = self.dweights
        if self.use_bias:
            self._params["biases"] = self.biases
            self._grads["biases"] = self.dbiases

        self._x_cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.ndim != 4:
            raise ValueError(f"Tensor4DDense requires a 4D tensor (Batch, Dim1, Dim2, Channels), got shape {x_arr.shape}")
        if x_arr.shape[-1] != self.in_channels:
            raise ValueError(f"Tensor4DDense expected {self.in_channels} channels, got {x_arr.shape[-1]}")

        self._x_cache = x_arr
        B, D1, D2, C = x_arr.shape

        x_flat = x_arr.reshape(-1, self.in_channels)
        out_flat = x_flat @ self.weights
        out = out_flat.reshape(B, D1, D2, self.out_channels)

        if self.use_bias and self.biases is not None:
            out += self.biases

        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x_cache is None:
            raise RuntimeError("Tensor4DDense.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype)
        if grad_out.shape != (self._x_cache.shape[0], self._x_cache.shape[1], self._x_cache.shape[2], self.out_channels):
            raise ValueError(f"Gradient shape {grad_out.shape} does not match expected output shape.")

        x_flat = self._x_cache.reshape(-1, self.in_channels)
        grad_flat = grad_out.reshape(-1, self.out_channels)

        self.dweights[:] = x_flat.T @ grad_flat
        if self.use_bias:
            self.dbiases[:] = np.sum(grad_out, axis=(0, 1, 2), keepdims=True)

        dx_flat = grad_flat @ self.weights.T
        return dx_flat.reshape(self._x_cache.shape)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Tensor4DDense",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Tensor4DDense":
        return cls(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            use_bias=config.get("use_bias", True),
        )


class ComplexWaveDense(Layer):
    """Complex-Valued Neural Layer with Wave Phase & Interference Dynamics.

    Implements complex arithmetic: Z_out = Z_in @ W + B where W = W_r + i*W_i.
    Natural destructive phase interference emerges from:
        Y_real = X_real @ W_real - X_imag @ W_imag + B_real
        Y_imag = X_real @ W_imag + X_imag @ W_real + B_imag

    Applies the modReLU non-linearity: preserves rotational phase while gating magnitude.
    By default (return_complex=False), outputs the wave magnitude energy |Y| for direct
    compatibility with standard real-valued loss functions and downstream layers.

    Args:
        in_features (int): Number of complex input channels.
        out_features (int): Number of complex output channels.
        return_complex (bool): If True, returns (Batch, out_features, 2). If False, returns (Batch, out_features).
        use_bias (bool): Whether to include complex bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        return_complex: bool = False,
        use_bias: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Dimensions must be positive integers.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.return_complex = bool(return_complex)
        self.use_bias = bool(use_bias)
        self.dtype = dtype
        self.trainable = True

        std = np.sqrt(1.0 / (self.in_features + self.out_features))
        self.w_real = (np.random.randn(self.in_features, self.out_features) * std).astype(self.dtype)
        self.w_imag = (np.random.randn(self.in_features, self.out_features) * std).astype(self.dtype)

        self.b_real = np.zeros((1, self.out_features), dtype=self.dtype) if self.use_bias else None
        self.b_imag = np.zeros((1, self.out_features), dtype=self.dtype) if self.use_bias else None
        self.b_mod = np.zeros((1, self.out_features), dtype=self.dtype)

        self.dw_real = np.zeros_like(self.w_real)
        self.dw_imag = np.zeros_like(self.w_imag)
        self.db_real = np.zeros_like(self.b_real) if self.use_bias else None
        self.db_imag = np.zeros_like(self.b_imag) if self.use_bias else None
        self.db_mod = np.zeros_like(self.b_mod)

        self._params["w_real"] = self.w_real
        self._params["w_imag"] = self.w_imag
        self._params["b_mod"] = self.b_mod
        self._grads["w_real"] = self.dw_real
        self._grads["w_imag"] = self.dw_imag
        self._grads["b_mod"] = self.db_mod

        if self.use_bias:
            self._params["b_real"] = self.b_real
            self._params["b_imag"] = self.b_imag
            self._grads["b_real"] = self.db_real
            self._grads["b_imag"] = self.db_imag

        self._cache = None
        self._orig_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        self._orig_shape = x_arr.shape

        is_complex_input = (x_arr.ndim >= 3 and x_arr.shape[-1] == 2)
        if is_complex_input:
            # Explicit complex input: [..., in_features, 2]
            batch_prefix = list(self._orig_shape[:-2])
            x_flat = x_arr.reshape(-1, self.in_features, 2)
            x_r = x_flat[:, :, 0]
            x_i = x_flat[:, :, 1]
        else:
            # Real input: [..., in_features], imag = 0
            batch_prefix = list(self._orig_shape[:-1])
            x_r = x_arr.reshape(-1, self.in_features)
            x_i = np.zeros_like(x_r)

        # Complex matrix multiplication with wave interference
        y_r = x_r @ self.w_real - x_i @ self.w_imag
        y_i = x_r @ self.w_imag + x_i @ self.w_real

        if self.use_bias:
            if self.b_real is not None:
                y_r += self.b_real
            if self.b_imag is not None:
                y_i += self.b_imag

        # ModReLU activation: |Y| = sqrt(y_r^2 + y_i^2 + eps)
        eps = 1e-7
        r = np.sqrt(y_r ** 2 + y_i ** 2 + eps)
        r_shifted = r + self.b_mod
        mask = (r_shifted > 0.0).astype(self.dtype)
        act_mag = np.maximum(0.0, r_shifted)

        scale = (act_mag / r)
        out_r = y_r * scale
        out_i = y_i * scale

        self._cache = {
            "x_r": x_r,
            "x_i": x_i,
            "y_r": y_r,
            "y_i": y_i,
            "r": r,
            "mask": mask,
            "scale": scale,
        }

        if self.return_complex:
            out_complex = np.stack([out_r, out_i], axis=-1)
            out_shape = batch_prefix + [self.out_features, 2]
            return out_complex.reshape(out_shape)
        else:
            out_shape = batch_prefix + [self.out_features]
            return act_mag.reshape(out_shape)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("ComplexWaveDense.backward called before forward pass.")

        c = self._cache
        x_r, x_i = c["x_r"], c["x_i"]
        y_r, y_i = c["y_r"], c["y_i"]
        r = c["r"]
        mask = c["mask"]

        if self.return_complex:
            grad_out = np.asarray(grad_output, dtype=self.dtype).reshape(-1, self.out_features, 2)
            d_out_r = grad_out[:, :, 0]
            d_out_i = grad_out[:, :, 1]

            # Backprop through modReLU
            scale = c["scale"]
            d_act_mag = (d_out_r * y_r + d_out_i * y_i) / r
            dr = d_act_mag * mask

            dy_r = d_out_r * scale + (dr - scale * d_act_mag) * (y_r / r)
            dy_i = d_out_i * scale + (dr - scale * d_act_mag) * (y_i / r)
        else:
            grad_out = np.asarray(grad_output, dtype=self.dtype).reshape(-1, self.out_features)
            # Output was act_mag
            dr = grad_out * mask
            dy_r = dr * (y_r / r)
            dy_i = dr * (y_i / r)

        self.db_mod[:] = np.sum(dr, axis=0, keepdims=True)

        if self.use_bias:
            self.db_real[:] = np.sum(dy_r, axis=0, keepdims=True)
            self.db_imag[:] = np.sum(dy_i, axis=0, keepdims=True)

        # Wirtinger complex gradients
        self.dw_real[:] = x_r.T @ dy_r + x_i.T @ dy_i
        self.dw_imag[:] = x_r.T @ dy_i - x_i.T @ dy_r

        dx_r = dy_r @ self.w_real.T + dy_i @ self.w_imag.T
        dx_i = -dy_r @ self.w_imag.T + dy_i @ self.w_real.T

        if self._orig_shape is not None and len(self._orig_shape) >= 3 and self._orig_shape[-1] == 2:
            dx = np.stack([dx_r, dx_i], axis=-1)
            return dx.reshape(self._orig_shape)
        else:
            return dx_r.reshape(self._orig_shape)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ComplexWaveDense",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "return_complex": self.return_complex,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ComplexWaveDense":
        return cls(
            in_features=config["in_features"],
            out_features=config["out_features"],
            return_complex=config.get("return_complex", False),
            use_bias=config.get("use_bias", True),
        )


class FractalChaosDense(Layer):
    """Chaotic Attractor & Bifurcation Layer.

    Embeds deterministic non-linear chaos dynamics (logistic bifurcation map)
    inside each neuron, allowing networks to navigate fractal energy landscapes
    and escape flat saddle points.

    Formula:
        z_raw = x @ W + b
        u = Sigmoid(z_raw)
        c = r * u * (1 - u)           # Logistic chaotic map
        c_scaled = (c - 0.5) * 4.0
        coupling = Sigmoid(coupling_logits)
        y = (1 - coupling) * z_raw + coupling * c_scaled

    Args:
        in_features (int): Input dimensionality.
        out_features (int): Output dimensionality.
        r_init (float): Initial bifurcation parameter (default: 3.65, chaotic edge).
        use_bias (bool): Whether to include bias vector.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r_init: float = 3.65,
        use_bias: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Dimensions must be positive integers.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.r_init = float(r_init)
        self.use_bias = bool(use_bias)
        self.dtype = dtype
        self.trainable = True

        std = np.sqrt(2.0 / (self.in_features + self.out_features))
        self.weights = (np.random.randn(self.in_features, self.out_features) * std).astype(self.dtype)
        self.biases = np.zeros((1, self.out_features), dtype=self.dtype) if self.use_bias else None

        # Bifurcation parameter r per neuron
        self.r_param = (np.ones((1, self.out_features), dtype=self.dtype) * self.r_init)
        # Chaos coupling logit (init at -1.0 so coupling ~ 0.27)
        self.coupling_logits = (np.ones((1, self.out_features), dtype=self.dtype) * -1.0)

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases) if self.use_bias else None
        self.dr_param = np.zeros_like(self.r_param)
        self.dcoupling_logits = np.zeros_like(self.coupling_logits)

        self._params["weights"] = self.weights
        self._params["r_param"] = self.r_param
        self._params["coupling_logits"] = self.coupling_logits
        self._grads["weights"] = self.dweights
        self._grads["r_param"] = self.dr_param
        self._grads["coupling_logits"] = self.dcoupling_logits

        if self.use_bias:
            self._params["biases"] = self.biases
            self._grads["biases"] = self.dbiases

        self._cache = None
        self._orig_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        self._orig_shape = x_arr.shape
        x_flat = x_arr.reshape(-1, self.in_features)

        z_raw = x_flat @ self.weights
        if self.use_bias and self.biases is not None:
            z_raw += self.biases

        u = 1.0 / (1.0 + np.exp(-np.clip(z_raw, -88.0, 88.0)))
        r = np.clip(self.r_param, 2.5, 4.0)
        c = r * u * (1.0 - u)
        c_scaled = (c - 0.5) * 4.0

        coup = 1.0 / (1.0 + np.exp(-np.clip(self.coupling_logits, -88.0, 88.0)))
        out = (1.0 - coup) * z_raw + coup * c_scaled

        self._cache = {
            "x_flat": x_flat,
            "z_raw": z_raw,
            "u": u,
            "r": r,
            "c_scaled": c_scaled,
            "coup": coup,
        }

        out_shape = list(self._orig_shape[:-1]) + [self.out_features]
        return out.reshape(out_shape)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("FractalChaosDense.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype).reshape(-1, self.out_features)
        c = self._cache
        x_flat = c["x_flat"]
        z_raw = c["z_raw"]
        u = c["u"]
        r = c["r"]
        c_scaled = c["c_scaled"]
        coup = c["coup"]

        # Gradients for coupling logits
        dcoup = np.sum(grad_out * (c_scaled - z_raw), axis=0, keepdims=True)
        self.dcoupling_logits[:] = dcoup * coup * (1.0 - coup)

        # Gradient through chaotic attractor
        dc_scaled = grad_out * coup
        dc = dc_scaled * 4.0

        # Gradient for r_param
        self.dr_param[:] = np.sum(dc * u * (1.0 - u), axis=0, keepdims=True)

        # Gradient with respect to u
        du = dc * r * (1.0 - 2.0 * u)
        dz_from_u = du * u * (1.0 - u)

        # Total gradient with respect to z_raw
        dz_raw = grad_out * (1.0 - coup) + dz_from_u

        self.dweights[:] = x_flat.T @ dz_raw
        if self.use_bias:
            self.dbiases[:] = np.sum(dz_raw, axis=0, keepdims=True)

        dx = dz_raw @ self.weights.T
        return dx.reshape(self._orig_shape)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "FractalChaosDense",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "r_init": self.r_init,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FractalChaosDense":
        return cls(
            in_features=config["in_features"],
            out_features=config["out_features"],
            r_init=config.get("r_init", 3.65),
            use_bias=config.get("use_bias", True),
        )


class TunnelingDense(Layer):
    """Quantum-Inspired Potential Barrier Tunneling Layer.

    Solves the "Dead Neuron" problem by providing a non-zero gradient transmission
    channel even when inputs are deeply beneath the activation threshold.

    Formula:
        z = x @ W + b
        delta = max(0, V - z)
        P_tunnel = exp(-delta / tau)
        y = z * P_tunnel

    When z >= V: Classical regime, y = z, dy/dz = 1
    When z < V:  Quantum tunneling regime, y = z * exp(-(V - z)/tau),
                 dy/dz = P_tunnel * (1 + z/tau) != 0!

    Args:
        in_features (int): Input dimensionality.
        out_features (int): Output dimensionality.
        tau (float): Tunneling decay length / permeability (default: 1.0).
        use_bias (bool): Whether to include bias vector.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 1.0,
        use_bias: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Dimensions must be positive integers.")
        if tau <= 0:
            raise ValueError("tau must be positive.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.tau = float(tau)
        self.use_bias = bool(use_bias)
        self.dtype = dtype
        self.trainable = True

        std = np.sqrt(2.0 / (self.in_features + self.out_features))
        self.weights = (np.random.randn(self.in_features, self.out_features) * std).astype(self.dtype)
        self.biases = np.zeros((1, self.out_features), dtype=self.dtype) if self.use_bias else None

        # Learnable potential barrier height V per neuron
        self.barrier_v = (np.ones((1, self.out_features), dtype=self.dtype) * 0.5)

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases) if self.use_bias else None
        self.dbarrier_v = np.zeros_like(self.barrier_v)

        self._params["weights"] = self.weights
        self._params["barrier_v"] = self.barrier_v
        self._grads["weights"] = self.dweights
        self._grads["barrier_v"] = self.dbarrier_v

        if self.use_bias:
            self._params["biases"] = self.biases
            self._grads["biases"] = self.dbiases

        self._cache = None
        self._orig_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        self._orig_shape = x_arr.shape
        x_flat = x_arr.reshape(-1, self.in_features)

        z = x_flat @ self.weights
        if self.use_bias and self.biases is not None:
            z += self.biases

        delta = np.maximum(0.0, self.barrier_v - z)
        p_tunnel = np.exp(-delta / self.tau)
        out = z * p_tunnel

        self._cache = {
            "x_flat": x_flat,
            "z": z,
            "delta": delta,
            "p_tunnel": p_tunnel,
        }

        out_shape = list(self._orig_shape[:-1]) + [self.out_features]
        return out.reshape(out_shape)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("TunnelingDense.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype).reshape(-1, self.out_features)
        c = self._cache
        x_flat = c["x_flat"]
        z = c["z"]
        p_tunnel = c["p_tunnel"]

        sub_barrier_mask = (z < self.barrier_v).astype(self.dtype)

        # dy / dz
        dy_dz = np.where(sub_barrier_mask > 0.0, p_tunnel * (1.0 + z / self.tau), 1.0)
        # dy / dV
        dy_dv = np.where(sub_barrier_mask > 0.0, - (z * p_tunnel) / self.tau, 0.0)

        dz = grad_out * dy_dz
        self.dbarrier_v[:] = np.sum(grad_out * dy_dv, axis=0, keepdims=True)

        self.dweights[:] = x_flat.T @ dz
        if self.use_bias:
            self.dbiases[:] = np.sum(dz, axis=0, keepdims=True)

        dx = dz @ self.weights.T
        return dx.reshape(self._orig_shape)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "TunnelingDense",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "tau": self.tau,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TunnelingDense":
        return cls(
            in_features=config["in_features"],
            out_features=config["out_features"],
            tau=config.get("tau", 1.0),
            use_bias=config.get("use_bias", True),
        )



