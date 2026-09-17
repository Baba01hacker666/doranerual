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

from neural_lib.base import Layer


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
        if x_arr.ndim != 2:
            raise ValueError(
                f"Dense layer expected 2D input array of shape (batch_size, {self.in_features}), "
                f"but received array with shape {x_arr.shape} ({x_arr.ndim} dimensions)."
            )
        if x_arr.shape[1] != self.in_features:
            raise ValueError(
                f"Dense layer expected {self.in_features} features (dim 1), "
                f"but received input with {x_arr.shape[1]} features (shape {x_arr.shape})."
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
        batch_size = self._input_cache.shape[0]

        if grad_out.shape != (batch_size, self.out_features):
            raise ValueError(
                f"Gradient shape mismatch in Dense backward. Expected ({batch_size}, {self.out_features}), "
                f"got {grad_out.shape}."
            )

        np.copyto(self.dweights, self._input_cache.T @ grad_out)

        if self.use_bias and self.dbiases is not None:
            np.copyto(self.dbiases, np.sum(grad_out, axis=0, keepdims=True))

        grad_input = grad_out @ self.weights.T
        return grad_input

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Dense",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "weight_init": self.weight_init,
            "use_bias": self.use_bias,
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

    def __init__(self, drop_rate: float = 0.5) -> None:
        super().__init__()
        if not (0.0 <= drop_rate < 1.0):
            raise ValueError(f"drop_rate must be in [0.0, 1.0), got {drop_rate}")
        self.drop_rate: float = float(drop_rate)
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

        # Gradients for gamma and beta
        np.copyto(self.dgamma, np.sum(grad_out * x_norm, axis=0, keepdims=True))
        np.copyto(self.dbeta, np.sum(grad_out, axis=0, keepdims=True))

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


def _im2col_indices(
    x: np.ndarray,
    kh: int,
    kw: int,
    padding: int = 1,
    stride: int = 1,
) -> Tuple[np.ndarray, int, int]:
    """Efficient vectorized im2col transformation for 2D convolutions."""
    N, C, H, W = x.shape
    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1

    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )

    # Compute index slices
    cols = np.zeros((N, C, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = x_padded[:, :, i:i_max:stride, j:j_max:stride]

    # Reshape to (N * out_h * out_w, C * kh * kw)
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, C * kh * kw)
    return cols, out_h, out_w


def _col2im_indices(
    cols: np.ndarray,
    x_shape: Tuple[int, int, int, int],
    kh: int,
    kw: int,
    padding: int = 1,
    stride: int = 1,
) -> np.ndarray:
    """Accumulates column patches back into image tensor."""
    N, C, H, W = x_shape
    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1

    x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols.dtype)
    cols_reshaped = cols.reshape(N, out_h, out_w, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)

    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            x_padded[:, :, i:i_max:stride, j:j_max:stride] += cols_reshaped[:, :, i, j, :, :]

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


class Conv2D(Layer):
    """2D Spatial Convolution Layer.

    Computes 2D cross-correlation across (batch_size, in_channels, height, width).

    Attributes:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output filter channels.
        kernel_size (int): Size of square kernel (k, k).
        stride (int): Stride step size along height and width.
        padding (int): Zero-padding applied to borders.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        weight_init: str = "he",
        use_bias: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.in_channels: int = int(in_channels)
        self.out_channels: int = int(out_channels)
        self.kernel_size: int = int(kernel_size)
        self.stride: int = int(stride)
        self.padding: int = int(padding)
        self.weight_init: str = weight_init.lower()
        self.use_bias: bool = bool(use_bias)
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

        # im2col
        cols, out_h, out_w = _im2col_indices(
            x_arr, self.kernel_size, self.kernel_size, self.padding, self.stride
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

        # db = sum across spatial and batch
        if self.use_bias and self.dbiases is not None:
            db = np.sum(grad_flat, axis=0, keepdims=True).T
            np.copyto(self.dbiases, db)

        # dX_cols = grad_flat @ W
        w_row = self.weights.reshape(self.out_channels, -1)
        dx_cols = grad_flat @ w_row

        # col2im
        dx = _col2im_indices(
            dx_cols,
            self._x_shape,
            self.kernel_size,
            self.kernel_size,
            self.padding,
            self.stride,
        )
        return dx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Conv2D",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "weight_init": self.weight_init,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Conv2D":
        return cls(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_size=config.get("kernel_size", 3),
            stride=config.get("stride", 1),
            padding=config.get("padding", 0),
            weight_init=config.get("weight_init", "he"),
            use_bias=config.get("use_bias", True),
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
