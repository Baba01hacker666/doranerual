"""Attention mechanisms and Transformer components built from scratch with pure NumPy.

Provides:
- `PositionalEncoding`: Sinusoidal positional encodings for sequence inputs.
- `MultiHeadAttention`: Multi-head scaled dot-product attention with analytical backward pass.
- `TransformerBlock`: Complete transformer encoder block combining LayerNorm, MultiHeadAttention,
  residual connections, Dropout, and Feed-Forward networks.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from .base import Layer
from .layers import LayerNorm, Dropout, Dense, DendriticDense, ChebyshevKAN
from .activations import ReLU


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class PositionalEncoding(Layer):
    """Sinusoidal Positional Encoding for sequence inputs.

    Formula:
        PE(pos, 2i)   = sin(pos / (10000 ** (2i / d_model)))
        PE(pos, 2i+1) = cos(pos / (10000 ** (2i / d_model)))

    Args:
        d_model (int): Embedding dimensionality.
        max_len (int): Maximum sequence length to precompute.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dtype: np.dtype = np.float32) -> None:
        super().__init__()
        self.d_model: int = int(d_model)
        self.max_len: int = int(max_len)
        self.dtype = dtype
        self.trainable: bool = False

        pe = np.zeros((self.max_len, self.d_model), dtype=self.dtype)
        position = np.arange(0, self.max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2, dtype=np.float32) * -(np.log(10000.0) / self.d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe: np.ndarray = pe[np.newaxis, :, :]  # Shape: (1, max_len, d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.ndim != 3:
            raise ValueError(f"PositionalEncoding expects 3D input (batch, seq_len, d_model), got {x_arr.shape}")
        seq_len = x_arr.shape[1]
        if seq_len > self.max_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds max_len {self.max_len}")
        return x_arr + self.pe[:, :seq_len, :]

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Identity gradient since positional encoding has no learnable parameters
        return np.asarray(grad_output, dtype=self.dtype)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "PositionalEncoding", "d_model": self.d_model, "max_len": self.max_len}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PositionalEncoding":
        return cls(d_model=config["d_model"], max_len=config.get("max_len", 5000))


class MultiHeadAttention(Layer):
    """Multi-Head Scaled Dot-Product Attention layer with exact analytical BPTT gradients.

    Args:
        d_model (int): Total dimensionality of input and output features.
        num_heads (int): Number of parallel attention heads.
        causal (bool): If True, applies an upper-triangular autoregressive mask.
        use_bias (bool): Whether to include linear projection bias terms.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        causal: bool = False,
        use_bias: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model: int = int(d_model)
        self.num_heads: int = int(num_heads)
        self.d_k: int = self.d_model // self.num_heads
        self.causal: bool = bool(causal)
        self.use_bias: bool = bool(use_bias)
        self.dtype = dtype
        self.trainable: bool = True

        std = np.sqrt(2.0 / (self.d_model + self.d_model))
        self.W_q = (np.random.randn(self.d_model, self.d_model) * std).astype(self.dtype)
        self.W_k = (np.random.randn(self.d_model, self.d_model) * std).astype(self.dtype)
        self.W_v = (np.random.randn(self.d_model, self.d_model) * std).astype(self.dtype)
        self.W_o = (np.random.randn(self.d_model, self.d_model) * std).astype(self.dtype)

        self.b_q = np.zeros((1, self.d_model), dtype=self.dtype) if self.use_bias else None
        self.b_k = np.zeros((1, self.d_model), dtype=self.dtype) if self.use_bias else None
        self.b_v = np.zeros((1, self.d_model), dtype=self.dtype) if self.use_bias else None
        self.b_o = np.zeros((1, self.d_model), dtype=self.dtype) if self.use_bias else None

        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)
        self.dW_o = np.zeros_like(self.W_o)

        self.db_q = np.zeros_like(self.b_q) if self.use_bias else None
        self.db_k = np.zeros_like(self.b_k) if self.use_bias else None
        self.db_v = np.zeros_like(self.b_v) if self.use_bias else None
        self.db_o = np.zeros_like(self.b_o) if self.use_bias else None

        for name in ["W_q", "W_k", "W_v", "W_o"]:
            self._params[name] = getattr(self, name)
            self._grads[name] = getattr(self, f"d{name}")
        if self.use_bias:
            for name in ["b_q", "b_k", "b_v", "b_o"]:
                self._params[name] = getattr(self, name)
                self._grads[name] = getattr(self, f"d{name}")

        self._cache: Optional[Dict[str, Any]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.ndim != 3:
            raise ValueError(f"MultiHeadAttention expects 3D input (batch, seq_len, d_model), got {x_arr.shape}")

        B, T, D = x_arr.shape
        H = self.num_heads
        d_k = self.d_k

        # 1. Linear projections
        q = x_arr @ self.W_q
        k = x_arr @ self.W_k
        v = x_arr @ self.W_v
        if self.use_bias:
            q += self.b_q
            k += self.b_k
            v += self.b_v

        # 2. Reshape into heads: (B, H, T, d_k)
        Q = q.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        K = k.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        V = v.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)

        # 3. Scaled dot-product scores: (B, H, T, T)
        scale = 1.0 / np.sqrt(d_k)
        scores = (Q @ K.transpose(0, 1, 3, 2)) * scale

        # Optional causal autoregressive mask
        if self.causal:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores[:, :, mask] = -1e9

        attn_weights = _softmax(scores, axis=-1)

        # 4. Context values: (B, H, T, d_k)
        context = attn_weights @ V

        # 5. Concatenate heads and final projection
        context_concat = context.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = context_concat @ self.W_o
        if self.use_bias:
            out += self.b_o

        self._cache = {
            "x": x_arr,
            "Q": Q,
            "K": K,
            "V": V,
            "scale": scale,
            "attn_weights": attn_weights,
            "context_concat": context_concat,
        }
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache is None:
            raise RuntimeError("MultiHeadAttention.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype)
        x = self._cache["x"]
        Q = self._cache["Q"]
        K = self._cache["K"]
        V = self._cache["V"]
        scale = self._cache["scale"]
        A = self._cache["attn_weights"]
        context_concat = self._cache["context_concat"]

        B, T, D = x.shape
        H = self.num_heads
        d_k = self.d_k

        # Reset gradients
        for g in self._grads.values():
            if g is not None:
                g.fill(0.0)

        # 1. Output projection gradients
        for b in range(B):
            self.dW_o += context_concat[b].T @ grad_out[b]
        if self.use_bias and self.db_o is not None:
            self.db_o += np.sum(grad_out, axis=(0, 1), keepdims=False).reshape(1, -1)

        d_context_concat = grad_out @ self.W_o.T
        d_context = d_context_concat.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)

        # 2. Gradients through attention values
        dV = A.transpose(0, 1, 3, 2) @ d_context
        dA = d_context @ V.transpose(0, 1, 3, 2)

        # 3. Softmax gradient: dS = A * (dA - sum(A * dA)) * scale
        dS = A * (dA - np.sum(A * dA, axis=-1, keepdims=True)) * scale
        if self.causal:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            dS[:, :, mask] = 0.0

        dQ = dS @ K
        dK = dS.transpose(0, 1, 3, 2) @ Q

        # 4. Reshape head gradients back to (B, T, D)
        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(B, T, D)
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(B, T, D)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(B, T, D)

        # 5. Projection parameter gradients
        for b in range(B):
            self.dW_q += x[b].T @ dQ_flat[b]
            self.dW_k += x[b].T @ dK_flat[b]
            self.dW_v += x[b].T @ dV_flat[b]

        if self.use_bias:
            self.db_q += np.sum(dQ_flat, axis=(0, 1), keepdims=False).reshape(1, -1)
            self.db_k += np.sum(dK_flat, axis=(0, 1), keepdims=False).reshape(1, -1)
            self.db_v += np.sum(dV_flat, axis=(0, 1), keepdims=False).reshape(1, -1)

        # 6. Downstream gradient to input x
        dx = dQ_flat @ self.W_q.T + dK_flat @ self.W_k.T + dV_flat @ self.W_v.T
        return dx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "MultiHeadAttention",
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "causal": self.causal,
            "use_bias": self.use_bias,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MultiHeadAttention":
        return cls(
            d_model=config["d_model"],
            num_heads=config.get("num_heads", 4),
            causal=config.get("causal", False),
            use_bias=config.get("use_bias", True),
        )


class TransformerBlock(Layer):
    """Complete Transformer Block layer combining LayerNorm, MultiHeadAttention,

    residual skip connections, Dropout, and Feed-Forward Networks.

    Args:
        d_model (int): Embedding and attention dimension.
        num_heads (int): Number of attention heads.
        d_ff (int): Dimension of inner feed-forward layer.
        dropout (float): Dropout probability.
        causal (bool): Whether to use causal autoregressive attention mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_ff: int = 64,
        dropout: float = 0.0,
        causal: bool = False,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.d_model: int = int(d_model)
        self.num_heads: int = int(num_heads)
        self.d_ff: int = int(d_ff)
        self.dropout_rate: float = float(dropout)
        self.causal: bool = bool(causal)
        self.dtype = dtype
        self.trainable: bool = True

        # Pre-LN Transformer architecture
        self.ln1 = LayerNorm(normalized_shape=self.d_model)
        self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads, causal=self.causal)
        self.drop1 = Dropout(drop_rate=self.dropout_rate)

        self.ln2 = LayerNorm(normalized_shape=self.d_model)
        self.ff1 = Dense(in_features=self.d_model, out_features=self.d_ff, weight_init="he")
        self.act = ReLU()
        self.ff2 = Dense(in_features=self.d_ff, out_features=self.d_model, weight_init="xavier")
        self.drop2 = Dropout(drop_rate=self.dropout_rate)

        # Expose sublayer parameters and gradients
        self._sublayers = [self.ln1, self.mha, self.drop1, self.ln2, self.ff1, self.act, self.ff2, self.drop2]
        for sub_idx, sub in enumerate(self._sublayers):
            for p_name, param in sub.get_params().items():
                self._params[f"sub_{sub_idx}_{p_name}"] = param
            for g_name, grad in sub.get_grads().items():
                self._grads[f"sub_{sub_idx}_{g_name}"] = grad

        self._cache_residual: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def train(self, mode: bool = True) -> "TransformerBlock":
        super().train(mode)
        for sub in self._sublayers:
            sub.train(mode)
        return self

    def eval(self) -> "TransformerBlock":
        return self.train(False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Sublayer 1: Pre-LN Attention + Residual
        norm1 = self.ln1.forward(x)
        attn_out = self.mha.forward(norm1)
        x1 = x + self.drop1.forward(attn_out)

        # Sublayer 2: Pre-LN Feed-Forward + Residual
        norm2 = self.ln2.forward(x1)
        ff_out = self.ff2.forward(self.act.forward(self.ff1.forward(norm2)))
        out = x1 + self.drop2.forward(ff_out)

        self._cache_residual = (x, x1)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache_residual is None:
            raise RuntimeError("TransformerBlock.backward called before forward pass.")

        x, x1 = self._cache_residual
        d_out = np.asarray(grad_output, dtype=self.dtype)

        # Sublayer 2 backward (FF + Residual)
        d_ff2 = self.drop2.backward(d_out)
        d_act = self.ff2.backward(d_ff2)
        d_ff1 = self.act.backward(d_act)
        d_norm2 = self.ff1.backward(d_ff1)
        d_x1_branch = self.ln2.backward(d_norm2)

        # Residual skip addition
        d_x1 = d_out + d_x1_branch

        # Sublayer 1 backward (Attn + Residual)
        d_drop1 = self.drop1.backward(d_x1)
        d_mha = self.mha.backward(d_drop1)
        d_norm1 = self.ln1.backward(d_mha)

        # Residual skip addition
        d_x = d_x1 + d_norm1
        return d_x

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "TransformerBlock",
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout_rate,
            "causal": self.causal,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TransformerBlock":
        return cls(
            d_model=config["d_model"],
            num_heads=config.get("num_heads", 4),
            d_ff=config.get("d_ff", 64),
            dropout=config.get("dropout", 0.0),
            causal=config.get("causal", False),
        )


class KANTransformerBlock(Layer):
    """Transformer Block utilizing Kolmogorov-Arnold Network (KAN) feedforward layers.

    Replaces the standard fixed-activation MLP with Chebyshev polynomial basis expansion layers:
        x1 = x + Dropout(MultiHeadAttention(LayerNorm(x)))
        out = x1 + Dropout(ChebyshevKAN_2(ChebyshevKAN_1(LayerNorm(x1))))

    Args:
        d_model (int): Embedding and attention dimension.
        num_heads (int): Number of attention heads.
        d_ff (int): Hidden dimension of inner KAN layers.
        degree (int): Chebyshev polynomial degree on synaptic edges (default: 3).
        dropout (float): Dropout probability.
        causal (bool): Whether to use causal autoregressive attention mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_ff: int = 32,
        degree: int = 3,
        dropout: float = 0.0,
        causal: bool = False,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.degree = int(degree)
        self.dropout_rate = float(dropout)
        self.causal = bool(causal)
        self.dtype = dtype
        self.trainable = True

        self.ln1 = LayerNorm(normalized_shape=self.d_model)
        self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads, causal=self.causal)
        self.drop1 = Dropout(drop_rate=self.dropout_rate)

        self.ln2 = LayerNorm(normalized_shape=self.d_model)
        self.kan1 = ChebyshevKAN(in_features=self.d_model, out_features=self.d_ff, degree=self.degree)
        self.kan2 = ChebyshevKAN(in_features=self.d_ff, out_features=self.d_model, degree=self.degree)
        self.drop2 = Dropout(drop_rate=self.dropout_rate)

        self._sublayers = [self.ln1, self.mha, self.drop1, self.ln2, self.kan1, self.kan2, self.drop2]
        for sub_idx, sub in enumerate(self._sublayers):
            for p_name, param in sub.get_params().items():
                self._params[f"sub_{sub_idx}_{p_name}"] = param
            for g_name, grad in sub.get_grads().items():
                self._grads[f"sub_{sub_idx}_{g_name}"] = grad

        self._cache_residual: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def train(self, mode: bool = True) -> "KANTransformerBlock":
        super().train(mode)
        for sub in self._sublayers:
            sub.train(mode)
        return self

    def eval(self) -> "KANTransformerBlock":
        return self.train(False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Sublayer 1: Pre-LN Attention + Residual
        norm1 = self.ln1.forward(x)
        attn_out = self.mha.forward(norm1)
        x1 = x + self.drop1.forward(attn_out)

        # Sublayer 2: Pre-LN KAN Feed-Forward + Residual
        norm2 = self.ln2.forward(x1)
        kan_out = self.kan2.forward(self.kan1.forward(norm2))
        out = x1 + self.drop2.forward(kan_out)

        self._cache_residual = (x, x1)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache_residual is None:
            raise RuntimeError("KANTransformerBlock.backward called before forward pass.")

        x, x1 = self._cache_residual
        d_out = np.asarray(grad_output, dtype=self.dtype)

        # Sublayer 2 backward (KAN + Residual)
        d_drop2 = self.drop2.backward(d_out)
        d_kan2 = self.kan2.backward(d_drop2)
        d_kan1 = self.kan1.backward(d_kan2)
        d_norm2 = self.ln2.backward(d_kan1)

        d_x1 = d_out + d_norm2

        # Sublayer 1 backward (Attn + Residual)
        d_drop1 = self.drop1.backward(d_x1)
        d_mha = self.mha.backward(d_drop1)
        d_norm1 = self.ln1.backward(d_mha)

        d_x = d_x1 + d_norm1
        return d_x

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "KANTransformerBlock",
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "degree": self.degree,
            "dropout": self.dropout_rate,
            "causal": self.causal,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "KANTransformerBlock":
        return cls(
            d_model=config["d_model"],
            num_heads=config.get("num_heads", 4),
            d_ff=config.get("d_ff", 32),
            degree=config.get("degree", 3),
            dropout=config.get("dropout", 0.0),
            causal=config.get("causal", False),
        )


class DendriticTransformerBlock(Layer):
    """Transformer Block utilizing Multi-Compartment Pyramidal Dendritic feedforward layers.

    Replaces the standard MLP with multi-branch multiplicative dendritic gating:
        x1 = x + Dropout(MultiHeadAttention(LayerNorm(x)))
        out = x1 + Dropout(DendriticDense_2(DendriticDense_1(LayerNorm(x1))))

    Args:
        d_model (int): Embedding and attention dimension.
        num_heads (int): Number of attention heads.
        d_ff (int): Hidden dimension of inner dendritic layers.
        num_branches (int): Number of dendritic compartments per neuron (default: 2).
        dropout (float): Dropout probability.
        causal (bool): Whether to use causal autoregressive attention mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_ff: int = 32,
        num_branches: int = 2,
        dropout: float = 0.0,
        causal: bool = False,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.num_branches = int(num_branches)
        self.dropout_rate = float(dropout)
        self.causal = bool(causal)
        self.dtype = dtype
        self.trainable = True

        self.ln1 = LayerNorm(normalized_shape=self.d_model)
        self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads, causal=self.causal)
        self.drop1 = Dropout(drop_rate=self.dropout_rate)

        self.ln2 = LayerNorm(normalized_shape=self.d_model)
        self.dend1 = DendriticDense(in_features=self.d_model, out_features=self.d_ff, num_branches=self.num_branches)
        self.dend2 = DendriticDense(in_features=self.d_ff, out_features=self.d_model, num_branches=self.num_branches)
        self.drop2 = Dropout(drop_rate=self.dropout_rate)

        self._sublayers = [self.ln1, self.mha, self.drop1, self.ln2, self.dend1, self.dend2, self.drop2]
        for sub_idx, sub in enumerate(self._sublayers):
            for p_name, param in sub.get_params().items():
                self._params[f"sub_{sub_idx}_{p_name}"] = param
            for g_name, grad in sub.get_grads().items():
                self._grads[f"sub_{sub_idx}_{g_name}"] = grad

        self._cache_residual: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def train(self, mode: bool = True) -> "DendriticTransformerBlock":
        super().train(mode)
        for sub in self._sublayers:
            sub.train(mode)
        return self

    def eval(self) -> "DendriticTransformerBlock":
        return self.train(False)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Sublayer 1: Pre-LN Attention + Residual
        norm1 = self.ln1.forward(x)
        attn_out = self.mha.forward(norm1)
        x1 = x + self.drop1.forward(attn_out)

        # Sublayer 2: Pre-LN Dendritic Feed-Forward + Residual
        norm2 = self.ln2.forward(x1)
        dend_out = self.dend2.forward(self.dend1.forward(norm2))
        out = x1 + self.drop2.forward(dend_out)

        self._cache_residual = (x, x1)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache_residual is None:
            raise RuntimeError("DendriticTransformerBlock.backward called before forward pass.")

        x, x1 = self._cache_residual
        d_out = np.asarray(grad_output, dtype=self.dtype)

        # Sublayer 2 backward (Dendritic + Residual)
        d_drop2 = self.drop2.backward(d_out)
        d_dend2 = self.dend2.backward(d_drop2)
        d_dend1 = self.dend1.backward(d_dend2)
        d_norm2 = self.ln2.backward(d_dend1)

        d_x1 = d_out + d_norm2

        # Sublayer 1 backward (Attn + Residual)
        d_drop1 = self.drop1.backward(d_x1)
        d_mha = self.mha.backward(d_drop1)
        d_norm1 = self.ln1.backward(d_mha)

        d_x = d_x1 + d_norm1
        return d_x

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "DendriticTransformerBlock",
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "num_branches": self.num_branches,
            "dropout": self.dropout_rate,
            "causal": self.causal,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DendriticTransformerBlock":
        return cls(
            d_model=config["d_model"],
            num_heads=config.get("num_heads", 4),
            d_ff=config.get("d_ff", 32),
            num_branches=config.get("num_branches", 2),
            dropout=config.get("dropout", 0.0),
            causal=config.get("causal", False),
        )

