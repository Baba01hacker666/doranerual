"""Dora-Norm: Stabilized & Normalized Bio-Reflective KAN Transformer Architecture.

Addresses the failure modes of deep novel neuron stacking:
1. Stabilized Dendritic Gating: Centered at 1.0 with bounded perturbation [0.9, 1.1]
   dend_gate = 1.0 + 0.1 * tanh(norm_ffn @ w_dend)
   Eliminates exponential activation compounding (2^L) across deep layers.

2. Normalized Chebyshev KAN: Pre-RMSNorm before tanh ensures inputs remain within
   the high-gradient linear regime:
   u = tanh(RMSNorm(hidden))
   p = 0.1 * sum(c_k * T_k(u))
   hidden = hidden + p
   Eliminates derivative vanishing and polynomial divergence.

3. Additive Cortical Reflection:
   down_ref = 0.1 * tanh(RMSNorm(down) @ w_ref)
   down = down + down_ref
   Preserves 100% of the primary representation highway, preventing the 97% signal
   loss caused by the legacy 0.75 multiplier across 12 layers (0.75^12 ~ 0.031).

4. Layer-Scale Normalization:
   Residual branches scaled by 1 / sqrt(2 * n_layers) to guarantee constant feature
   variance across arbitrary depths.
"""

from typing import Any, List, Optional, Sequence, Tuple
import numpy as np

from doraneural.autograd import Module, Parameter, Tensor
from doraneural.transformer import LoRAAdapter, TensorAdamW


def _normal(in_features: int, out_features: int, dtype: Any = np.float32) -> np.ndarray:
    return (np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)).astype(dtype)


class DoraNormBlock(Module):
    """Stabilized Bio-Reflective KAN Transformer Block."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        n_layers: int = 12,
        kan_degree: int = 3,
        dtype: Any = np.float32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.head_size = self.dim // self.n_heads
        self.n_layers = max(1, int(n_layers))
        self.kan_degree = int(kan_degree)
        self.layer_scale = 1.0 / np.sqrt(2.0 * self.n_layers)

        # Standard Attention parameters
        self.rms_att = Parameter(np.ones((dim,), dtype=dtype), dtype=dtype)
        self.wq = Parameter(_normal(dim, dim, dtype), dtype=dtype)
        self.wk = Parameter(_normal(dim, (dim * n_kv_heads) // n_heads, dtype), dtype=dtype)
        self.wv = Parameter(_normal(dim, (dim * n_kv_heads) // n_heads, dtype), dtype=dtype)
        self.wo = Parameter(_normal(dim, dim, dtype), dtype=dtype)

        # SwiGLU FeedForward parameters
        self.rms_ffn = Parameter(np.ones((dim,), dtype=dtype), dtype=dtype)
        self.w1 = Parameter(_normal(dim, hidden_dim, dtype), dtype=dtype)
        self.w2 = Parameter(_normal(hidden_dim, dim, dtype), dtype=dtype)
        self.w3 = Parameter(_normal(dim, hidden_dim, dtype), dtype=dtype)

        # Stabilized Novel Neurons
        # 1. Bounded Dendritic Gating (dim -> hidden_dim)
        dend_std = 0.02 / np.sqrt(dim)
        self.w_dend = Parameter((np.random.randn(dim, hidden_dim) * dend_std).astype(dtype), dtype=dtype)

        # 2. Normalized Chebyshev KAN Coefficients (degree, hidden_dim)
        kan_std = 0.02 / (np.sqrt(hidden_dim) * (self.kan_degree + 1))
        self.c_poly = Parameter((np.random.randn(self.kan_degree, hidden_dim) * kan_std).astype(dtype), dtype=dtype)
        self.rms_kan = Parameter(np.ones((hidden_dim,), dtype=dtype), dtype=dtype)

        # 3. Additive Cortical Reflection (dim -> dim)
        ref_std = 0.02 / np.sqrt(dim)
        self.w_ref = Parameter((np.random.randn(dim, dim) * ref_std).astype(dtype), dtype=dtype)
        self.rms_ref = Parameter(np.ones((dim,), dtype=dtype), dtype=dtype)

    @staticmethod
    def _rmsnorm(x: Tensor, weight: Tensor) -> Tensor:
        rms = ((x ** 2).mean(axis=-1, keepdims=True) + 1e-5).__pow__(0.5)
        return (x / rms) * weight

    def _rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        first = x[:, :, 0::2]
        second = x[:, :, 1::2]
        rotated_first = first * cos - second * sin
        rotated_second = first * sin + second * cos
        pieces: List[Tensor] = []
        for idx in range(self.head_size // 2):
            pieces.extend((rotated_first[:, :, idx : idx + 1], rotated_second[:, :, idx : idx + 1]))
        return Tensor.concatenate(tuple(pieces), axis=-1)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        # 1. Multi-Head Self-Attention with pre-RMSNorm
        norm = self._rmsnorm(x, self.rms_att)
        q = (norm @ self.wq).reshape(x.shape[0], self.n_heads, self.head_size).transpose(1, 0, 2)
        k = (norm @ self.wk).reshape(x.shape[0], self.n_kv_heads, self.head_size).transpose(1, 0, 2)
        v = (norm @ self.wv).reshape(x.shape[0], self.n_kv_heads, self.head_size).transpose(1, 0, 2)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)

        if self.n_kv_heads != self.n_heads:
            kv_map = np.arange(self.n_heads) // (self.n_heads // self.n_kv_heads)
            k = k[kv_map]
            v = v[kv_map]

        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(self.head_size)
        scores = scores + mask
        scores_shifted = scores - Tensor(np.max(scores.data, axis=-1, keepdims=True), dtype=scores.dtype)
        exp_scores = scores_shifted.exp()
        probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        attn = (probs @ v).transpose(1, 0, 2).reshape(x.shape[0], self.dim)
        
        # Residual connection with layer scale
        x = x + (attn @ self.wo) * self.layer_scale

        # 2. Stabilized FeedForward + Novel Neurons
        norm_ffn = self._rmsnorm(x, self.rms_ffn)
        gate = norm_ffn @ self.w1
        up = norm_ffn @ self.w3

        # (a) Stabilized Dendritic Gating (centered at 1.0, bounded perturbation)
        dend_mod = (norm_ffn @ self.w_dend).tanh() * 0.1
        dend_gate = dend_mod + 1.0
        up = up * dend_gate

        # (b) SwiGLU non-linearity
        silu = gate.sigmoid() * gate
        hidden = silu * up

        # (c) Normalized Chebyshev KAN expansion
        norm_kan = self._rmsnorm(hidden, self.rms_kan)
        u = norm_kan.tanh()
        t1 = u
        t2 = (u * u) * 2.0 - 1.0
        t3 = (u * u * u) * 4.0 - u * 3.0
        p_out = (t1 * self.c_poly[0] + t2 * self.c_poly[1] + t3 * self.c_poly[2]) * 0.1
        hidden = hidden + p_out

        # (d) Down projection
        down = hidden @ self.w2

        # (e) Additive Cortical Reflection (preserves 100% primary highway)
        norm_ref = self._rmsnorm(down, self.rms_ref)
        down_ref = (norm_ref @ self.w_ref).tanh() * 0.1
        down = down + down_ref

        # Final FeedForward residual connection
        x = x + down * self.layer_scale
        return x


class DoraNormDecoderLM(Module):
    """Full causal language model using Dora-Norm blocks."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        vocab_size: int,
        seq_len: int,
        dtype: Any = np.float32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.head_size = self.dim // self.n_heads

        self.tok_emb = Parameter((np.random.randn(vocab_size, dim) * 0.02).astype(dtype), dtype=dtype)
        self.layers = [
            DoraNormBlock(
                dim=dim,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                n_layers=n_layers,
                dtype=dtype,
            )
            for _ in range(n_layers)
        ]
        self.rms_final = Parameter(np.ones((dim,), dtype=dtype), dtype=dtype)
        self.lm_head = Parameter((np.random.randn(dim, vocab_size) * 0.02).astype(dtype), dtype=dtype)

    def _rope_frequencies(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        head_dim = self.head_size
        theta = 10000.0 ** (-np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
        t = np.arange(seq_len, dtype=np.float32)
        angles = np.outer(t, theta)
        cos = Tensor(np.cos(angles), requires_grad=False)
        sin = Tensor(np.sin(angles), requires_grad=False)
        return cos, sin

    def forward(self, token_ids: Sequence[int]) -> Tensor:
        t = len(token_ids)
        if t == 0:
            raise ValueError("Empty token sequence")
        if t > self.seq_len:
            raise ValueError(f"Sequence length {t} exceeds max seq_len {self.seq_len}")

        x = self.tok_emb[token_ids]
        cos, sin = self._rope_frequencies(t)
        mask = Tensor(np.triu(np.full((t, t), -1e9, dtype=np.float32), k=1), requires_grad=False)

        for layer in self.layers:
            x = layer.forward(x, cos, sin, mask)

        rms = ((x ** 2).mean(axis=-1, keepdims=True) + 1e-5).__pow__(0.5)
        norm_final = (x / rms) * self.rms_final
        return norm_final @ self.lm_head

    def loss(self, token_ids: Sequence[int], target_ids: Sequence[int]) -> Tensor:
        targets = np.asarray(target_ids, dtype=np.int64)
        logits = self.forward(token_ids)
        if targets.shape != (len(token_ids),):
            raise ValueError("target_ids must have one target per input token")
        max_logits = Tensor(np.max(logits.data, axis=-1, keepdims=True), dtype=logits.dtype)
        log_norm = (logits - max_logits).exp().sum(axis=-1, keepdims=True).log() + max_logits
        active = (targets >= 0).astype(logits.dtype)
        safe_targets = np.clip(targets, 0, self.vocab_size - 1)
        selected = logits[np.arange(len(targets)), safe_targets]
        nll = (log_norm.reshape(-1) - selected) * Tensor(active, dtype=logits.dtype)
        return nll.sum() / float(max(1, int(active.sum())))

    def train_batch(
        self,
        token_ids: Sequence[int],
        target_ids: Sequence[int],
        optimizer: TensorAdamW,
    ) -> float:
        optimizer.zero_grad()
        loss = self.loss(token_ids, target_ids)
        loss.backward()
        optimizer.step()
        return float(loss.item())
