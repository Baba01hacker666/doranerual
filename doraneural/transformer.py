"""Trainable decoder-only transformer components built on doraneural autograd.

This module provides a compact, CPU-friendly full-backpropagation path for the
LLaMA-shaped Zexo checkpoints. Unlike the historical lightweight fine-tuner,
which only changed the embedding/classifier path, ``TransformerDecoderLM``
propagates gradients through RMSNorm, RoPE, causal grouped-query attention,
SwiGLU, every projection matrix, the embeddings, and the tied classifier.

It is intentionally a reference/training implementation. The existing NumPy
and C++ engines remain the fast inference paths.
"""

from typing import Any, List, Optional, Sequence, Tuple
import numpy as np

from .autograd import Module, Parameter, Tensor


class TensorAdamW:
    """Small AdamW optimizer for autograd ``Parameter`` objects."""

    def __init__(
        self,
        params: Sequence[Parameter],
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        self.params = list(params)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_count = 0
        self._m = {id(param): np.zeros_like(param.data) for param in self.params}
        self._v = {id(param): np.zeros_like(param.data) for param in self.params}

    def step(self) -> None:
        self.step_count += 1
        b1_corr = 1.0 - self.beta1 ** self.step_count
        b2_corr = 1.0 - self.beta2 ** self.step_count
        step_size = self.lr * np.sqrt(b2_corr) / b1_corr
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad
            m = self._m[id(param)]
            v = self._v[id(param)]
            m *= self.beta1
            m += (1.0 - self.beta1) * grad
            v *= self.beta2
            v += (1.0 - self.beta2) * (grad * grad)
            if self.weight_decay:
                param.data -= self.lr * self.weight_decay * param.data
            param.data -= step_size * m / (np.sqrt(v) + self.eps)

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()


class TransformerDecoderBlock(Module):
    """Pre-norm causal self-attention plus SwiGLU feed-forward block."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        weights: Optional[dict] = None,
        dtype: Any = np.float32,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0 or n_heads % n_kv_heads != 0:
            raise ValueError("dim must divide n_heads and n_heads must divide n_kv_heads")
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.head_size = self.dim // self.n_heads
        w = weights or {}
        self.rms_att = Parameter(w.get("rms_att", np.ones((dim,), dtype=dtype)), dtype=dtype)
        self.wq = Parameter(w.get("wq", _normal(dim, dim, dtype)), dtype=dtype)
        self.wk = Parameter(w.get("wk", _normal(dim, (dim * n_kv_heads) // n_heads, dtype)), dtype=dtype)
        self.wv = Parameter(w.get("wv", _normal(dim, (dim * n_kv_heads) // n_heads, dtype)), dtype=dtype)
        self.wo = Parameter(w.get("wo", _normal(dim, dim, dtype)), dtype=dtype)
        self.rms_ffn = Parameter(w.get("rms_ffn", np.ones((dim,), dtype=dtype)), dtype=dtype)
        self.w1 = Parameter(w.get("w1", _normal(dim, hidden_dim, dtype)), dtype=dtype)
        self.w2 = Parameter(w.get("w2", _normal(hidden_dim, dim, dtype)), dtype=dtype)
        self.w3 = Parameter(w.get("w3", _normal(dim, hidden_dim, dtype)), dtype=dtype)

    def _rope(self, x: Tensor, cos: Tensor, sin: Tensor, interleaved: bool) -> Tensor:
        # x is (heads, time, head_size), while cos/sin are (time, head_size/2).
        if interleaved:
            first = x[:, :, 0::2]
            second = x[:, :, 1::2]
            rotated_first = first * cos - second * sin
            rotated_second = first * sin + second * cos
            pieces: List[Tensor] = []
            for idx in range(self.head_size // 2):
                pieces.extend((rotated_first[:, :, idx : idx + 1], rotated_second[:, :, idx : idx + 1]))
            return Tensor.concatenate(tuple(pieces), axis=-1)
        # HuggingFace LLaMA stores RoPE pairs in the first and second halves.
        first = x[:, :, : self.head_size // 2]
        second = x[:, :, self.head_size // 2 :]
        return Tensor.concatenate((first * cos - second * sin, first * sin + second * cos), axis=-1)

    @staticmethod
    def _rmsnorm(x: Tensor, weight: Tensor) -> Tensor:
        # Match LlamaLLM's RMSNorm: sqrt(mean(x^2) + epsilon).
        rms = ((x ** 2).mean(axis=-1, keepdims=True) + 1e-5).__pow__(0.5)
        return (x / rms) * weight

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor, interleaved: bool = True) -> Tensor:
        # RMSNorm: broadcasting the one-dimensional scale over (time, dim).
        norm = self._rmsnorm(x, self.rms_att)
        q = (norm @ self.wq).reshape(x.shape[0], self.n_heads, self.head_size).transpose(1, 0, 2)
        k = (norm @ self.wk).reshape(x.shape[0], self.n_kv_heads, self.head_size).transpose(1, 0, 2)
        v = (norm @ self.wv).reshape(x.shape[0], self.n_kv_heads, self.head_size).transpose(1, 0, 2)
        q = self._rope(q, cos, sin, interleaved)
        k = self._rope(k, cos, sin, interleaved)

        # Expand grouped K/V heads using an indexed view; this keeps GQA's
        # memory advantage in the parameters while matching all query heads.
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
        x = x + (attn @ self.wo)

        norm_ffn = self._rmsnorm(x, self.rms_ffn)
        gate = norm_ffn @ self.w1
        up = norm_ffn @ self.w3
        silu = gate.sigmoid() * gate
        x = x + ((silu * up) @ self.w2)
        return x


def _normal(in_features: int, out_features: int, dtype: Any) -> np.ndarray:
    return (np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)).astype(dtype)


class TransformerDecoderLM(Module):
    """A fully trainable causal decoder language model.

    The constructor can create a fresh model for experiments. Use
    :meth:`from_llama` to attach it to a loaded ``LlamaLLM`` checkpoint so full
    gradients update the existing Zexo weights in-place.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        vocab_size: int,
        seq_len: int,
        rope_type: str = "interleaved",
        weights: Optional[dict] = None,
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
        self.rope_type = rope_type
        w = weights or {}
        self.token_embedding = Parameter(w.get("token_embedding", _normal(vocab_size, dim, dtype)), dtype=dtype)
        layer_weights = w.get("layers") or [{} for _ in range(n_layers)]
        self.layers = [
            TransformerDecoderBlock(
                dim, hidden_dim, n_heads, n_kv_heads,
                weights=layer_weights[idx], dtype=dtype,
            )
            for idx in range(n_layers)
        ]
        self.rms_final = Parameter(w.get("rms_final", np.ones((dim,), dtype=dtype)), dtype=dtype)
        # Weight tying is the default in the llama2.c/Zexo checkpoint format.
        self.lm_head = self.token_embedding if w.get("lm_head") is None else Parameter(w["lm_head"], dtype=dtype)

    @classmethod
    def from_llama(cls, llm: Any) -> "TransformerDecoderLM":
        """Create a trainable view over an existing ``LlamaLLM`` checkpoint."""
        p = llm.config
        layers = []
        for idx in range(p.n_layers):
            # LlamaLLM follows llama2.c/HuggingFace's (out, in) matrix
            # convention. The autograd module uses row-major (in, out)
            # matrices, so transpose projections at the boundary.
            layers.append({
                "rms_att": llm.rms_att[idx], "wq": llm.wq[idx].T,
                "wk": llm.wk[idx].T, "wv": llm.wv[idx].T, "wo": llm.wo[idx].T,
                "rms_ffn": llm.rms_ffn[idx], "w1": llm.w1[idx].T,
                "w2": llm.w2[idx].T, "w3": llm.w3[idx].T,
            })
        weights = {
            "token_embedding": llm.tok_emb,
            "layers": layers,
            "rms_final": llm.rms_final,
        }
        if not llm.shared_weights:
            weights["lm_head"] = llm.wcls
        model = cls(
            dim=p.dim, hidden_dim=p.hidden_dim, n_layers=p.n_layers,
            n_heads=p.n_heads, n_kv_heads=p.n_kv_heads, vocab_size=p.vocab_size,
            seq_len=p.seq_len, rope_type=p.rope_type, weights=weights,
            dtype=llm.tok_emb.dtype,
        )
        if not llm.shared_weights:
            # The classifier is stored as (vocab, dim), already matching the
            # model's output convention. Tied embeddings are handled above.
            model.lm_head.data[...] = llm.wcls
        model._llm = llm
        return model

    def _tables(self, length: int) -> Tuple[Tensor, Tensor, Tensor]:
        half = self.dim // self.n_heads // 2
        positions = np.arange(length, dtype=np.float32)[:, None]
        exponent = np.arange(half, dtype=np.float32)[None, :]
        inv_freq = 1.0 / (10000.0 ** (2.0 * exponent / (self.dim // self.n_heads)))
        angles = positions * inv_freq
        cos = Tensor(np.cos(angles).astype(self.token_embedding.dtype), dtype=self.token_embedding.dtype)
        sin = Tensor(np.sin(angles).astype(self.token_embedding.dtype), dtype=self.token_embedding.dtype)
        mask = np.triu(np.full((length, length), -np.inf, dtype=self.token_embedding.dtype), 1)
        return cos, sin, Tensor(mask, dtype=self.token_embedding.dtype)

    def forward(self, token_ids: Sequence[int]) -> Tensor:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1 or len(ids) < 1 or len(ids) > self.seq_len:
            raise ValueError(f"token_ids must be 1D with length in [1, {self.seq_len}]")
        x = self.token_embedding[ids]
        cos, sin, mask = self._tables(len(ids))
        for layer in self.layers:
            x = layer.forward(x, cos, sin, mask, interleaved=self.rope_type != "hf")
        x = TransformerDecoderBlock._rmsnorm(x, self.rms_final)
        return x @ self.lm_head.T

    def loss(self, token_ids: Sequence[int], target_ids: Sequence[int]) -> Tensor:
        targets = np.asarray(target_ids, dtype=np.int64)
        logits = self.forward(token_ids)
        if targets.shape != (len(token_ids),):
            raise ValueError("target_ids must have one target per input token")
        max_logits = Tensor(np.max(logits.data, axis=-1, keepdims=True), dtype=logits.dtype)
        log_norm = (logits - max_logits).exp().sum(axis=-1, keepdims=True).log() + max_logits
        selected = logits[np.arange(len(targets)), targets]
        return (log_norm.reshape(-1) - selected).mean()

    def train_batch(self, token_ids: Sequence[int], target_ids: Sequence[int], optimizer: TensorAdamW) -> float:
        optimizer.zero_grad()
        loss = self.loss(token_ids, target_ids)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def copy_to_llama(self, llm: Optional[Any] = None) -> None:
        """Copy row-major autograd weights back to LlamaLLM's checkpoint arrays."""
        target = llm or getattr(self, "_llm", None)
        if target is None:
            return
        target.tok_emb[...] = self.token_embedding.data
        for idx, layer in enumerate(self.layers):
            target.rms_att[idx][...] = layer.rms_att.data
            target.wq[idx][...] = layer.wq.data.T
            target.wk[idx][...] = layer.wk.data.T
            target.wv[idx][...] = layer.wv.data.T
            target.wo[idx][...] = layer.wo.data.T
            target.rms_ffn[idx][...] = layer.rms_ffn.data
            target.w1[idx][...] = layer.w1.data.T
            target.w2[idx][...] = layer.w2.data.T
            target.w3[idx][...] = layer.w3.data.T
        target.rms_final[...] = self.rms_final.data
        if not target.shared_weights:
            target.wcls[...] = self.lm_head.data
        target.reset_cache()

    def fit_tokens(
        self,
        tokens: Sequence[int],
        epochs: int = 1,
        seq_len: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        stride: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        eval_tokens: Optional[Sequence[int]] = None,
        max_eval_steps: Optional[int] = None,
        verbose: int = 1,
    ) -> dict:
        """Train all transformer parameters on next-token sequences."""
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 1 <= seq_len < self.seq_len:
            raise ValueError(f"seq_len must be in [1, {self.seq_len - 1}]")
        if stride is not None and not 1 <= stride <= seq_len:
            raise ValueError(f"stride must be in [1, seq_len], got {stride}")
        if len(tokens) < seq_len + 1:
            raise ValueError("tokens must contain at least seq_len + 1 tokens")
        step = stride or seq_len
        starts = list(range(0, len(tokens) - seq_len, step))
        if not starts:
            raise ValueError("no training windows available")
        optimizer = TensorAdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        rng = np.random.default_rng(seed)
        history = {"loss": []}
        if eval_tokens is not None:
            history["val_loss"] = []
        for epoch in range(epochs):
            order = starts.copy()
            if shuffle:
                rng.shuffle(order)
            total = 0.0
            for start in order:
                total += self.train_batch(tokens[start : start + seq_len], tokens[start + 1 : start + seq_len + 1], optimizer)
            history["loss"].append(total / len(order))
            if eval_tokens is not None:
                eval_starts = list(range(0, len(eval_tokens) - seq_len, step))
                if max_eval_steps is not None:
                    eval_starts = eval_starts[:max_eval_steps]
                values = []
                for start in eval_starts:
                    values.append(float(self.loss(eval_tokens[start : start + seq_len], eval_tokens[start + 1 : start + seq_len + 1]).item()))
                history["val_loss"].append(float(np.mean(values)) if values else float("nan"))
            if verbose:
                msg = f"[Full BP] Epoch {epoch + 1}/{epochs} loss={history['loss'][-1]:.4f}"
                if "val_loss" in history:
                    msg += f" val_loss={history['val_loss'][-1]:.4f}"
                print(msg, flush=True)
        self.copy_to_llama()
        return history
