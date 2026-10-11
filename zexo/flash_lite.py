"""A compact DeepSeek-V4.1-Flash-inspired reference architecture.

This is an educational, CPU/NumPy-scale model, not a reproduction of
DeepSeek-V4.1-Flash. It combines a causal encoder/decoder split, local sliding
window attention plus sparse global attention, CSA2-style cross-layer reuse of
global KV/top-k selections, and a shared-plus-routed top-k MoE feed-forward.
It deliberately omits the paper's multimodal stack, billion-parameter experts,
FP4 cache kernels, distributed training, Engram, mHC, and DSpark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from doraneural.autograd import Module, Parameter, Tensor, no_grad
from doraneural.llm import ByteTokenizer
from doraneural.transformer import TensorAdamW


def _normal(in_features: int, out_features: int, dtype: Any, scale: Optional[float] = None) -> np.ndarray:
    std = float(scale) if scale is not None else math.sqrt(2.0 / max(1, in_features))
    return (np.random.randn(in_features, out_features) * std).astype(dtype)


@dataclass
class FlashLiteConfig:
    """Hyperparameters for the small Flash-inspired reference model."""

    vocab_size: int = ByteTokenizer.VOCAB_SIZE
    dim: int = 64
    hidden_dim: int = 128
    n_layers: int = 6
    n_heads: int = 4
    n_kv_heads: int = 2
    seq_len: int = 256
    ced_split: Optional[int] = None
    local_only_encoder_layers: int = 2
    sliding_window: int = 64
    global_top_k: int = 32
    n_experts: int = 4
    experts_per_token: int = 2
    router_aux_loss_coef: float = 0.01
    rope_theta: float = 10000.0
    bos_token_id: int = ByteTokenizer.BOS_ID
    eos_token_id: int = ByteTokenizer.EOS_ID

    def __post_init__(self) -> None:
        self.vocab_size = int(self.vocab_size)
        self.dim = int(self.dim)
        self.hidden_dim = int(self.hidden_dim)
        self.n_layers = int(self.n_layers)
        self.n_heads = int(self.n_heads)
        self.n_kv_heads = int(self.n_kv_heads)
        self.seq_len = int(self.seq_len)
        if self.ced_split is None:
            self.ced_split = self.n_layers // 2
        self.ced_split = int(self.ced_split)
        self.local_only_encoder_layers = int(self.local_only_encoder_layers)
        self.sliding_window = int(self.sliding_window)
        self.global_top_k = int(self.global_top_k)
        self.n_experts = int(self.n_experts)
        self.experts_per_token = int(self.experts_per_token)
        if self.vocab_size <= 0 or self.dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("vocab_size, dim, and hidden_dim must be positive")
        if self.n_layers < 2 or not 1 <= self.ced_split < self.n_layers:
            raise ValueError("n_layers must be >= 2 and ced_split must leave encoder and decoder layers")
        if self.n_heads <= 0 or self.n_kv_heads <= 0 or self.dim % self.n_heads:
            raise ValueError("dim must be divisible by positive n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        if (self.dim // self.n_heads) % 2:
            raise ValueError("head_size must be even for RoPE")
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if not 1 <= self.sliding_window <= self.seq_len:
            raise ValueError("sliding_window must be in [1, seq_len]")
        if self.global_top_k <= 0:
            raise ValueError("global_top_k must be positive")
        if not 0 <= self.local_only_encoder_layers <= self.ced_split:
            raise ValueError("local_only_encoder_layers must be in [0, ced_split]")
        if self.n_experts < 2 or not 1 <= self.experts_per_token <= self.n_experts:
            raise ValueError("experts_per_token must be in [1, n_experts], with at least 2 experts")
        if not math.isfinite(self.router_aux_loss_coef) or self.router_aux_loss_coef < 0:
            raise ValueError("router_aux_loss_coef must be finite and non-negative")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not 0 <= self.bos_token_id < self.vocab_size or not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError("BOS/EOS token IDs must be inside the vocabulary")

    @classmethod
    def micro(cls, **overrides: Any) -> "FlashLiteConfig":
        """CPU smoke-test preset; pass overrides for smaller test contexts."""
        values: Dict[str, Any] = {
            "dim": 32,
            "hidden_dim": 64,
            "n_layers": 4,
            "n_heads": 4,
            "n_kv_heads": 2,
            "seq_len": 128,
            "local_only_encoder_layers": 1,
            "sliding_window": 32,
            "global_top_k": 16,
            "n_experts": 4,
            "experts_per_token": 2,
        }
        values.update(overrides)
        return cls(**values)

    @property
    def encoder_layers(self) -> int:
        return int(self.ced_split or 0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "FlashLiteConfig":
        return cls(**values)

    def decoder_mode(self, layer_index: int) -> str:
        """CSA2-lite mode schedule: Full -> Reindex -> Reuse, repeated."""
        if not self.encoder_layers <= layer_index < self.n_layers:
            return "full"
        return ("full", "reindex", "reuse")[(layer_index - self.encoder_layers) % 3]


class SwiGLUExpert(Module):
    """A single SwiGLU expert with row-major linear weights."""

    def __init__(self, dim: int, hidden_dim: int, dtype: Any = np.float32) -> None:
        super().__init__()
        self.w1 = Parameter(_normal(dim, hidden_dim, dtype), dtype=dtype)
        self.w3 = Parameter(_normal(dim, hidden_dim, dtype), dtype=dtype)
        self.w2 = Parameter(_normal(hidden_dim, dim, dtype), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        gate = x @ self.w1
        up = x @ self.w3
        return ((gate.sigmoid() * gate) * up) @ self.w2


class SharedRoutedMoE(Module):
    """Sparse top-k experts plus an always-active shared expert.

    Expert execution is genuinely sparse (only selected experts see each
    token). The balancing auxiliary loss is the standard differentiable
    importance/load product; routing decisions themselves are non-differentiable.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_experts: int,
        experts_per_token: int,
        dtype: Any = np.float32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_experts = int(n_experts)
        self.experts_per_token = int(experts_per_token)
        self.router = Parameter(_normal(dim, n_experts, dtype, scale=0.02), dtype=dtype)
        self.experts = [SwiGLUExpert(dim, hidden_dim, dtype) for _ in range(n_experts)]
        self.shared_expert = SwiGLUExpert(dim, hidden_dim, dtype)
        self.last_aux_loss = Tensor(np.asarray(0.0, dtype=dtype), dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.dim or x.shape[0] < 1:
            raise ValueError(f"MoE input must have shape (tokens, {self.dim})")
        logits = x @ self.router
        shifted = logits - Tensor(np.max(logits.data, axis=-1, keepdims=True), dtype=logits.dtype)
        probs = shifted.exp()
        probs = probs / probs.sum(axis=-1, keepdims=True)
        route_ids = np.argsort(-logits.data, axis=-1, kind="stable")[:, : self.experts_per_token]
        selected_probs = probs[np.arange(x.shape[0])[:, None], route_ids]
        selected_probs = selected_probs / (selected_probs.sum(axis=-1, keepdims=True) + 1e-9)

        # Dispatch token groups to each selected expert in one matrix operation,
        # then gather weighted rows back into their original token positions.
        routed_parts: List[List[Tensor]] = [[] for _ in range(x.shape[0])]
        for expert_index, expert in enumerate(self.experts):
            token_ids, slots = np.where(route_ids == expert_index)
            if token_ids.size == 0:
                continue
            expert_outputs = expert(x[token_ids])
            gate_weights = selected_probs[token_ids, slots].reshape(token_ids.size, 1)
            weighted_outputs = expert_outputs * gate_weights
            for row, token_id in enumerate(token_ids.tolist()):
                routed_parts[token_id].append(weighted_outputs[row : row + 1])

        shared_outputs = self.shared_expert(x)
        outputs: List[Tensor] = []
        for token_index, contributions in enumerate(routed_parts):
            if not contributions:
                raise RuntimeError("MoE routing failed to assign a token to any expert")
            routed = contributions[0]
            for contribution in contributions[1:]:
                routed = routed + contribution
            outputs.append(shared_outputs[token_index : token_index + 1] + routed)

        counts = np.bincount(route_ids.reshape(-1), minlength=self.n_experts).astype(logits.dtype)
        load = Tensor(counts / float(x.shape[0] * self.experts_per_token), dtype=logits.dtype)
        self.last_aux_loss = (probs.mean(axis=0) * load).sum() * float(self.n_experts)
        return Tensor.concatenate(tuple(outputs), axis=0)


class FlashLiteBlock(Module):
    """One block with local SWA, sparse global attention and a routed MoE."""

    def __init__(self, config: FlashLiteConfig, layer_index: int, dtype: Any = np.float32) -> None:
        super().__init__()
        self.config = config
        self.layer_index = int(layer_index)
        self.head_size = config.dim // config.n_heads
        self.global_mode = config.decoder_mode(layer_index)
        self.is_decoder = layer_index >= config.encoder_layers
        self.local_only = (
            not self.is_decoder and layer_index < config.local_only_encoder_layers
        )
        self.rms_att = Parameter(np.ones(config.dim, dtype=dtype), dtype=dtype)
        self.rms_global: Optional[Parameter] = (
            Parameter(np.ones(config.dim, dtype=dtype), dtype=dtype)
            if self.is_decoder and self.global_mode == "full" else None
        )
        self.wq = Parameter(_normal(config.dim, config.dim, dtype), dtype=dtype)
        # Every layer owns local-window KV and output projections.
        kv_dim = config.dim * config.n_kv_heads // config.n_heads
        self.wk_local = Parameter(_normal(config.dim, kv_dim, dtype), dtype=dtype)
        self.wv_local = Parameter(_normal(config.dim, kv_dim, dtype), dtype=dtype)
        self.wo_local = Parameter(_normal(config.dim, config.dim, dtype, scale=0.02 / math.sqrt(config.n_layers)), dtype=dtype)
        # Full layers produce global KV. Reindex/Reuse layers consume shared KV.
        self.wk_global: Optional[Parameter]
        self.wv_global: Optional[Parameter]
        if not self.local_only and self.global_mode == "full":
            self.wk_global = Parameter(_normal(config.dim, kv_dim, dtype), dtype=dtype)
            self.wv_global = Parameter(_normal(config.dim, kv_dim, dtype), dtype=dtype)
        else:
            self.wk_global = None
            self.wv_global = None
        self.wo_global: Optional[Parameter] = (
            Parameter(_normal(config.dim, config.dim, dtype, scale=0.02 / math.sqrt(config.n_layers)), dtype=dtype)
            if not self.local_only else None
        )
        self.rms_ffn = Parameter(np.ones(config.dim, dtype=dtype), dtype=dtype)
        self.moe = SharedRoutedMoE(
            config.dim, config.hidden_dim, config.n_experts,
            config.experts_per_token, dtype=dtype,
        )

    @staticmethod
    def _rmsnorm(x: Tensor, weight: Tensor) -> Tensor:
        rms = ((x ** 2).mean(axis=-1, keepdims=True) + 1e-5).__pow__(0.5)
        return (x / rms) * weight

    def _rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        first = x[:, :, 0::2]
        second = x[:, :, 1::2]
        parts: List[Tensor] = []
        rotated_first = first * cos - second * sin
        rotated_second = first * sin + second * cos
        for i in range(self.head_size // 2):
            parts.extend((rotated_first[:, :, i : i + 1], rotated_second[:, :, i : i + 1]))
        return Tensor.concatenate(tuple(parts), axis=-1)

    def _expand_kv(self, x: Tensor) -> Tensor:
        if self.config.n_kv_heads == self.config.n_heads:
            return x
        mapping = np.arange(self.config.n_heads) // (self.config.n_heads // self.config.n_kv_heads)
        return x[mapping]

    def _top_indices(self, q: Tensor, k: Tensor, causal_mask: Tensor) -> np.ndarray:
        # Routing indices are discrete and intentionally have no gradient. Do
        # their score pass on NumPy arrays; the selected attention pass below
        # recomputes differentiable scores only for the retained key/value rows.
        scores = np.matmul(q.data, k.data.transpose(0, 2, 1)) / math.sqrt(self.head_size)
        scores = scores + causal_mask.data[None, :, :]
        keep = min(self.config.global_top_k, scores.shape[-1])
        if keep == scores.shape[-1]:
            return np.broadcast_to(
                np.arange(scores.shape[-1], dtype=np.int64), scores.shape
            ).copy()
        return np.argpartition(scores, scores.shape[-1] - keep, axis=-1)[..., -keep:]

    def _attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor,
        selected_indices: Optional[np.ndarray] = None,
    ) -> Tensor:
        if selected_indices is None:
            scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.head_size)
            scores = scores + mask
            scores_for_softmax = scores
            values = v
            reduce_axis = 2
        else:
            heads, length, _ = q.shape
            head_ids = np.arange(heads)[:, None, None]
            query_ids = np.arange(length)[None, :, None]
            selected_k = k[head_ids, selected_indices]
            values = v[head_ids, selected_indices]
            query_vectors = q.reshape(heads, length, 1, self.head_size)
            scores_for_softmax = (query_vectors * selected_k).sum(axis=-1) / math.sqrt(self.head_size)
            is_causal = selected_indices <= query_ids
            selected_mask = Tensor(np.where(is_causal, 0.0, -np.inf).astype(q.dtype), dtype=q.dtype)
            scores_for_softmax = scores_for_softmax + selected_mask
            reduce_axis = 2
        row_max = Tensor(np.max(scores_for_softmax.data, axis=-1, keepdims=True), dtype=q.dtype)
        exp_scores = (scores_for_softmax - row_max).exp()
        probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        if selected_indices is None:
            return (probs @ values).transpose(1, 0, 2)
        return (probs.reshape(*probs.shape, 1) * values).sum(axis=reduce_axis).transpose(1, 0, 2)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        causal_mask: Tensor,
        local_mask: Tensor,
        encoder_state: Optional[Tensor],
        shared_kv: Optional[Tuple[Tensor, Tensor]],
        shared_indices: Optional[np.ndarray],
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Optional[np.ndarray]]:
        norm = self._rmsnorm(x, self.rms_att)
        q = (norm @ self.wq).reshape(x.shape[0], self.config.n_heads, self.head_size).transpose(1, 0, 2)
        q = self._rope(q, cos, sin)
        local_k = (norm @ self.wk_local).reshape(x.shape[0], self.config.n_kv_heads, self.head_size).transpose(1, 0, 2)
        local_v = (norm @ self.wv_local).reshape(x.shape[0], self.config.n_kv_heads, self.head_size).transpose(1, 0, 2)
        local_k = self._expand_kv(self._rope(local_k, cos, sin))
        local_v = self._expand_kv(local_v)
        local_out = self._attend(q, local_k, local_v, local_mask)
        x = x + local_out.reshape(x.shape[0], self.config.dim) @ self.wo_local

        current_kv = shared_kv
        current_indices = shared_indices
        if not self.local_only:
            if self.global_mode == "full":
                source = encoder_state if self.is_decoder else norm
                if source is None or self.wk_global is None or self.wv_global is None:
                    raise RuntimeError("Full attention layer is missing its CED source or global projections")
                if self.is_decoder:
                    if self.rms_global is None:
                        raise RuntimeError("CED decoder Full layer is missing its global normalization")
                    source = self._rmsnorm(source, self.rms_global)
                global_k = (source @ self.wk_global).reshape(
                    source.shape[0], self.config.n_kv_heads, self.head_size
                ).transpose(1, 0, 2)
                global_v = (source @ self.wv_global).reshape(
                    source.shape[0], self.config.n_kv_heads, self.head_size
                ).transpose(1, 0, 2)
                current_kv = (self._expand_kv(self._rope(global_k, cos, sin)), self._expand_kv(global_v))
            elif current_kv is None:
                raise RuntimeError(f"{self.global_mode} layer has no preceding Full layer KV to reuse")

            assert current_kv is not None
            global_k, global_v = current_kv
            if self.global_mode in ("full", "reindex") or current_indices is None:
                current_indices = self._top_indices(q, global_k, causal_mask)
            global_out = self._attend(q, global_k, global_v, causal_mask, current_indices)
            if self.wo_global is None:
                raise RuntimeError("global attention layer is missing its output projection")
            x = x + global_out.reshape(x.shape[0], self.config.dim) @ self.wo_global

        norm_ffn = self._rmsnorm(x, self.rms_ffn)
        x = x + self.moe(norm_ffn)
        return x, current_kv, current_indices


class DeepSeekFlashLiteLM(Module):
    """Small full-backprop reference model inspired by selected V4.1-Flash ideas."""

    def __init__(self, config: Optional[FlashLiteConfig] = None, dtype: Any = np.float32) -> None:
        super().__init__()
        self.config = config or FlashLiteConfig()
        self.dtype = np.dtype(dtype)
        if self.dtype.kind != "f":
            raise ValueError("FlashLite model dtype must be a floating-point type")
        # Mirror the common decoder attributes used by the preference trainer.
        self.dim = self.config.dim
        self.hidden_dim = self.config.hidden_dim
        self.n_layers = self.config.n_layers
        self.n_heads = self.config.n_heads
        self.n_kv_heads = self.config.n_kv_heads
        self.vocab_size = self.config.vocab_size
        self.seq_len = self.config.seq_len
        self.rope_type = "interleaved"
        self.rope_theta = self.config.rope_theta
        self.eos_token_id = self.config.eos_token_id
        self.token_embedding = Parameter(
            _normal(self.config.vocab_size, self.config.dim, dtype, scale=1.0 / math.sqrt(self.config.dim)),
            dtype=dtype,
        )
        self.layers = [FlashLiteBlock(self.config, index, dtype=dtype) for index in range(self.config.n_layers)]
        self.rms_final = Parameter(np.ones(self.config.dim, dtype=dtype), dtype=dtype)
        self.last_aux_loss = Tensor(np.asarray(0.0, dtype=dtype), dtype=dtype)
        self.last_lm_loss = Tensor(np.asarray(0.0, dtype=dtype), dtype=dtype)

    @property
    def parameter_count(self) -> int:
        return int(sum(parameter.data.size for parameter in self.parameters()))

    @property
    def architecture_signature(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple(sorted(self.config.to_dict().items()))

    @property
    def attention_modes(self) -> List[str]:
        modes = []
        for index in range(self.config.n_layers):
            if index < self.config.encoder_layers and index < self.config.local_only_encoder_layers:
                modes.append("swa-only")
            else:
                modes.append(self.config.decoder_mode(index))
        return modes

    def detach_metrics(self) -> None:
        """Replace stored loss tensors with scalar snapshots to release graphs."""
        for name in ("last_lm_loss", "last_aux_loss"):
            value = getattr(self, name)
            scalar = float(value.item())
            setattr(self, name, Tensor(np.asarray(scalar, dtype=self.dtype), dtype=self.dtype))
        for layer in self.layers:
            scalar = float(layer.moe.last_aux_loss.item())
            layer.moe.last_aux_loss = Tensor(np.asarray(scalar, dtype=self.dtype), dtype=self.dtype)

    def _position_tables(self, length: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        head_size = self.config.dim // self.config.n_heads
        half = head_size // 2
        positions = np.arange(length, dtype=np.float32)[:, None]
        exponent = np.arange(half, dtype=np.float32)[None, :]
        inv_freq = 1.0 / (self.config.rope_theta ** (2.0 * exponent / head_size))
        angles = positions * inv_freq
        cos = Tensor(np.cos(angles).astype(self.dtype), dtype=self.dtype)
        sin = Tensor(np.sin(angles).astype(self.dtype), dtype=self.dtype)
        rows = np.arange(length)[:, None]
        cols = np.arange(length)[None, :]
        causal = np.where(cols <= rows, 0.0, -np.inf).astype(self.dtype)
        local = np.where((cols <= rows) & (cols >= rows - self.config.sliding_window + 1), 0.0, -np.inf).astype(self.dtype)
        return cos, sin, Tensor(causal, dtype=self.dtype), Tensor(local, dtype=self.dtype)

    def forward(self, token_ids: Sequence[int]) -> Tensor:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1 or not 1 <= len(ids) <= self.config.seq_len:
            raise ValueError(f"token_ids must be 1D with length in [1, {self.config.seq_len}]")
        if np.any(ids < 0) or np.any(ids >= self.config.vocab_size):
            raise ValueError("token_ids contain values outside the vocabulary")
        x = self.token_embedding[ids]
        cos, sin, causal_mask, local_mask = self._position_tables(len(ids))
        encoder_state: Optional[Tensor] = None
        shared_kv: Optional[Tuple[Tensor, Tensor]] = None
        shared_indices: Optional[np.ndarray] = None
        aux_losses: List[Tensor] = []
        for index, layer in enumerate(self.layers):
            if index == self.config.encoder_layers:
                # CED boundary: the decoder's global KV is projected from the
                # final causal-encoder state; decoder SWA remains layer-local.
                encoder_state = x
                shared_kv = None
                shared_indices = None
            x, shared_kv, shared_indices = layer(
                x, cos, sin, causal_mask, local_mask, encoder_state,
                shared_kv, shared_indices,
            )
            aux_losses.append(layer.moe.last_aux_loss)
        x = FlashLiteBlock._rmsnorm(x, self.rms_final)
        self.last_aux_loss = aux_losses[0]
        for aux in aux_losses[1:]:
            self.last_aux_loss = self.last_aux_loss + aux
        self.last_aux_loss = self.last_aux_loss / float(max(1, len(aux_losses)))
        return x @ self.token_embedding.T

    def loss(self, token_ids: Sequence[int], target_ids: Sequence[int]) -> Tensor:
        targets = np.asarray(target_ids, dtype=np.int64)
        logits = self.forward(token_ids)
        if targets.shape != (len(token_ids),):
            raise ValueError("target_ids must contain one target per input token")
        if np.any(targets >= self.config.vocab_size):
            raise ValueError("target_ids contain values outside the vocabulary")
        active = targets >= 0
        safe_targets = np.clip(targets, 0, self.config.vocab_size - 1)
        row_max = Tensor(np.max(logits.data, axis=-1, keepdims=True), dtype=logits.dtype)
        log_norm = (logits - row_max).exp().sum(axis=-1, keepdims=True).log() + row_max
        selected = logits[np.arange(len(targets)), safe_targets]
        nll = ((log_norm.reshape(-1) - selected) * Tensor(active.astype(logits.dtype), dtype=logits.dtype)).sum()
        nll = nll / float(max(1, int(active.sum())))
        self.last_lm_loss = nll
        return nll + self.last_aux_loss * self.config.router_aux_loss_coef

    def fit_tokens(
        self,
        tokens: Sequence[int],
        epochs: int = 1,
        seq_len: Optional[int] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        stride: Optional[int] = None,
        max_steps: Optional[int] = None,
        eval_tokens: Optional[Sequence[int]] = None,
        max_eval_steps: Optional[int] = None,
        grad_clip: float = 1.0,
        seed: int = 42,
        shuffle: bool = True,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """Full-transformer next-token training with optional held-out selection."""
        if epochs <= 0 or not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("epochs must be positive and lr must be finite and positive")
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not math.isfinite(grad_clip) or grad_clip < 0.0:
            raise ValueError("grad_clip must be finite and non-negative")
        step_len = self.config.seq_len - 1 if seq_len is None else int(seq_len)
        if not 1 <= step_len < self.config.seq_len:
            raise ValueError(f"seq_len must be in [1, {self.config.seq_len - 1}]")
        step_stride = step_len if stride is None else int(stride)
        if not 1 <= step_stride <= step_len:
            raise ValueError("stride must be in [1, seq_len]")
        train_tokens = np.asarray(tokens, dtype=np.int64)
        if train_tokens.ndim != 1 or len(train_tokens) < step_len + 1:
            raise ValueError("tokens must be 1D and contain at least seq_len + 1 tokens")
        starts = list(range(0, len(train_tokens) - step_len, step_stride))
        if not starts:
            raise ValueError("no training windows available")
        eval_array = None if eval_tokens is None else np.asarray(eval_tokens, dtype=np.int64)
        if eval_array is not None and (eval_array.ndim != 1 or len(eval_array) < step_len + 1):
            raise ValueError("eval_tokens must contain at least seq_len + 1 tokens")
        eval_starts = list(range(0, len(eval_array) - step_len, step_stride)) if eval_array is not None else []
        if max_eval_steps is not None:
            if max_eval_steps <= 0:
                raise ValueError("max_eval_steps must be positive")
            eval_starts = eval_starts[:max_eval_steps]
        if eval_array is not None and not eval_starts:
            raise ValueError("no evaluation windows available")

        optimizer = TensorAdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        rng = np.random.default_rng(seed)
        steps_per_epoch = min(len(starts), max_steps) if max_steps is not None else len(starts)
        history: Dict[str, Any] = {"loss": [], "lm_loss": [], "router_aux_loss": [],
                                   "windows_per_epoch": steps_per_epoch, "total_steps": steps_per_epoch * epochs}
        if eval_array is not None:
            history["eval_loss"] = []
            best_eval_loss = float("inf")
            best_parameters: Optional[List[np.ndarray]] = None
            best_epoch = 0
        else:
            best_eval_loss = float("inf")
            best_parameters = None
            best_epoch = 0

        for epoch in range(epochs):
            order = np.asarray(starts, dtype=np.int64)
            if shuffle:
                rng.shuffle(order)
            if max_steps is not None:
                order = order[:max_steps]
            losses: List[float] = []
            lm_losses: List[float] = []
            aux_losses: List[float] = []
            for start in order.tolist():
                input_ids = train_tokens[start : start + step_len].tolist()
                target_ids = train_tokens[start + 1 : start + step_len + 1].tolist()
                optimizer.zero_grad()
                total_loss = self.loss(input_ids, target_ids)
                total_loss.backward()
                parameters = self.parameters()
                grad_norm = math.sqrt(sum(float(np.sum(p.grad * p.grad)) for p in parameters if p.grad is not None))
                if grad_clip > 0.0 and grad_norm > grad_clip:
                    scale = grad_clip / max(grad_norm, 1e-12)
                    for parameter in parameters:
                        if parameter.grad is not None:
                            parameter.grad *= scale
                optimizer.step()
                losses.append(float(total_loss.item()))
                lm_value = float(self.last_lm_loss.item())
                aux_value = float(self.last_aux_loss.item())
                lm_losses.append(lm_value)
                aux_losses.append(aux_value)
                # Do not keep the just-backpropagated activation graph alive
                # through module attributes between optimizer steps.
                self.detach_metrics()
                del total_loss
            history["loss"].append(float(np.mean(losses)))
            history["lm_loss"].append(float(np.mean(lm_losses)))
            history["router_aux_loss"].append(float(np.mean(aux_losses)))
            if eval_array is not None:
                eval_losses = []
                with no_grad():
                    for start in eval_starts:
                        ids = eval_array[start : start + step_len].tolist()
                        targets = eval_array[start + 1 : start + step_len + 1].tolist()
                        eval_losses.append(float(self.loss(ids, targets).item()))
                eval_loss = float(np.mean(eval_losses))
                history["eval_loss"].append(eval_loss)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_parameters = [p.data.copy() for p in self.parameters()]
                    best_epoch = epoch + 1
            if verbose:
                message = (f"[FlashLite] Epoch {epoch + 1}/{epochs} loss={history['loss'][-1]:.4f} "
                           f"lm={history['lm_loss'][-1]:.4f} aux={history['router_aux_loss'][-1]:.4f}")
                if eval_array is not None:
                    message += f" eval={history['eval_loss'][-1]:.4f}"
                print(message, flush=True)

        if best_parameters is not None:
            for parameter, values in zip(self.parameters(), best_parameters):
                parameter.data[...] = values
            history["best_epoch"] = best_epoch
            history["best_eval_loss"] = best_eval_loss
        return history

    def fit_text(
        self,
        text: str,
        tokenizer: Optional[ByteTokenizer] = None,
        eval_text: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Convenience wrapper for raw UTF-8 text and byte-tokenized pretraining."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("training text must be non-empty")
        tok = tokenizer or ByteTokenizer()
        if tok.vocab_size != self.config.vocab_size:
            raise ValueError("tokenizer vocabulary size does not match model config")
        train_tokens = tok.encode(text, bos=True) + [self.config.eos_token_id]
        validation_tokens = None
        if eval_text is not None:
            if not eval_text.strip():
                raise ValueError("eval_text must be non-empty when provided")
            validation_tokens = tok.encode(eval_text, bos=True) + [self.config.eos_token_id]
        return self.fit_tokens(train_tokens, eval_tokens=validation_tokens, **kwargs)

    def generate(
        self,
        prompt: str,
        tokenizer: Optional[ByteTokenizer] = None,
        max_tokens: int = 80,
        temperature: float = 0.7,
        top_k: int = 0,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a continuation by recomputing the bounded prefix each step."""
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        tok = tokenizer or ByteTokenizer()
        if tok.vocab_size != self.config.vocab_size:
            raise ValueError("tokenizer vocabulary size does not match model config")
        token_ids = tok.encode(prompt, bos=True)
        generated: List[int] = []
        rng = np.random.default_rng(seed)
        with no_grad():
            for _ in range(max_tokens):
                context = token_ids[-self.config.seq_len :]
                logits = self.forward(context).data[-1].astype(np.float64)
                if temperature == 0.0:
                    next_id = int(np.argmax(logits))
                else:
                    logits /= temperature
                    if top_k:
                        keep = min(top_k, self.config.vocab_size)
                        remove = np.argpartition(logits, -keep)[:-keep]
                        logits[remove] = -np.inf
                    logits -= np.max(logits)
                    probs = np.exp(logits)
                    probs /= probs.sum()
                    next_id = int(rng.choice(self.config.vocab_size, p=probs))
                if next_id == self.config.eos_token_id:
                    break
                token_ids.append(next_id)
                generated.append(next_id)
        return tok.decode(generated)

    def save(self, path: Union[str, Path]) -> Path:
        """Save config and all parameters in a compressed, non-pickle NPZ."""
        output = Path(path)
        if output.suffix != ".npz":
            output = output.with_suffix(".npz")
        output.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "config": np.asarray(json.dumps(self.config.to_dict())),
            "dtype": np.asarray(self.dtype.str),
        }
        for index, parameter in enumerate(self.parameters()):
            payload[f"parameter_{index:05d}"] = parameter.data
        np.savez_compressed(output, **payload)
        return output

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DeepSeekFlashLiteLM":
        """Load a model checkpoint previously written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as archive:
            if "config" not in archive:
                raise ValueError("checkpoint is missing FlashLite config metadata")
            config = FlashLiteConfig.from_dict(json.loads(str(archive["config"].item())))
            dtype = np.dtype(str(archive["dtype"].item())) if "dtype" in archive else np.float32
            model = cls(config, dtype=dtype)
            parameters = model.parameters()
            keys = sorted(key for key in archive.files if key.startswith("parameter_"))
            if len(keys) != len(parameters):
                raise ValueError(f"checkpoint parameter count mismatch: expected {len(parameters)}, got {len(keys)}")
            for key, parameter in zip(keys, parameters):
                values = archive[key]
                if values.shape != parameter.shape:
                    raise ValueError(f"checkpoint shape mismatch for {key}: {values.shape} != {parameter.shape}")
                parameter.data[...] = values
        return model


__all__ = ["DeepSeekFlashLiteLM", "FlashLiteConfig", "SharedRoutedMoE"]
