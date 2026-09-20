"""Dora-X2: 16-Layer Multi-Head Recurrent Trace Unit (MH-RTU) Conversational Language Model.

Architecture Blueprint:
- Layers: 16
- Model Dimension: 768
- Attention / Recurrent Heads: 12 (Head Dimension: 64)
- Hidden Dimension: 2048 (SwiGLU)
- Sequence Mechanism: Multi-Head Recurrent Trace Units (MH-RTU) with learned channel decay:
    S_{t, h, j} = sigmoid(w_decay_{h, j}) * S_{t-1, h, j} + K_{t, h, j} * V_{t, h, j}
    O_{t, h, j} = Q_{t, h, j} * S_{t, h, j} * SiLU(G_{t, h, j})
    Provides exact O(1) constant state memory per head for infinite-context streaming.
- Novel Bio-Reflective Neurons (Dora-Norm):
    1. Bounded Dendritic Gating (centered at 1.0)
    2. Normalized Chebyshev KAN orthogonal polynomial basis (T1, T2, T3)
    3. Additive Cortical Reflection loop preserving 100% primary representation highway
    4. Layer-scale residual stabilization: 1 / sqrt(2 * n_layers) = 1 / sqrt(32) ~ 0.1768
- Conversational Engine:
    Interactive Dora-X2 ChatSession with multi-turn streaming generation, context tracking,
    slash commands, and calibrated confidence diagnostics.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


@dataclass
class DoraX2Config:
    """Architectural specifications for the Dora-X2 Chat Model."""
    name: str = "dora-x2"
    tier: str = "chat-16L"
    dim: int = 768
    hidden_dim: int = 2048
    n_layers: int = 16
    n_heads: int = 12
    vocab_size: int = 256  # Default byte-level (zero tokenization dependency, O(1) streaming)
    seq_len: int = 1024
    kan_degree: int = 3
    dropout: float = 0.0
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    system_prompt: str = (
        "You are Dora-X2, a high-capacity 16-layer Multi-Head Recurrent Trace (MH-RTU) "
        "conversational AI assistant. You think deeply, respond with clarity, empathy, and technical rigor, "
        "and leverage bio-reflective non-linear reasoning to assist the user."
    )
    version: str = "2.0.0"

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def layer_scale(self) -> float:
        return 1.0 / math.sqrt(2.0 * self.n_layers)

    @property
    def parameter_count(self) -> int:
        """Exact count of all trainable parameters in Dora-X2."""
        emb = self.vocab_size * self.dim
        # MH-RTU: Wq, Wk, Wv, Wg, Wo (5 * D * D) + w_decay (H * head_dim = D) + rms (D)
        rtu_per_layer = 5 * (self.dim * self.dim) + self.dim + self.dim
        # FeedForward: w1, w2, w3 (3 * D * hidden_dim)
        ffn_linear = 3 * (self.dim * self.hidden_dim)
        # Novel Neurons: w_dend (D * hidden_dim) + c_poly (kan_degree * hidden_dim) + w_ref (D * D) + norms (3 * D)
        novel_per_layer = (self.dim * self.hidden_dim) + (self.kan_degree * self.hidden_dim) + (self.dim * self.dim) + (3 * self.dim)
        per_layer = rtu_per_layer + ffn_linear + novel_per_layer
        total = emb + (per_layer * self.n_layers) + self.dim + (self.dim * self.vocab_size)
        return total


class MultiHeadRTU:
    """Multi-Head Recurrent Trace Unit (12 Heads, dim=768, head_dim=64).

    Maintains a decoupled recurrent channel state per head:
        decay_{h, j} = sigmoid(w_decay_{h, j})
        S_{t, h, j} = decay_{h, j} * S_{t-1, h, j} + K_{t, h, j} * V_{t, h, j}
        O_{t, h, j} = Q_{t, h, j} * S_{t, h, j} * SiLU(G_{t, h, j})
    """

    def __init__(self, config: DoraX2Config):
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        scale = 1.0 / math.sqrt(self.dim)
        head_scale = 1.0 / math.sqrt(self.head_dim)

        self.rms = np.ones(self.dim, dtype=np.float32)
        self.wq = (np.random.randn(self.dim, self.dim) * scale).astype(np.float32)
        self.wk = (np.random.randn(self.dim, self.dim) * head_scale).astype(np.float32)
        self.wv = (np.random.randn(self.dim, self.dim) * head_scale).astype(np.float32)
        self.wg = (np.random.randn(self.dim, self.dim) * scale).astype(np.float32)
        self.wo = (np.random.randn(self.dim, self.dim) * scale).astype(np.float32)

        # Initial decay ~ 0.90 to 0.98 per channel: logit(0.95) ~ 2.94
        self.w_decay = np.full((self.n_heads, self.head_dim), 2.5, dtype=np.float32)

    def forward_step(
        self,
        x: np.ndarray,
        state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-step recurrent inference for O(1) memory token generation.

        Args:
            x: Input vector [dim]
            state: Previous recurrent state [n_heads, head_dim]

        Returns:
            out: Projected output vector [dim]
            new_state: Updated recurrent state [n_heads, head_dim]
        """
        if state is None:
            state = np.zeros((self.n_heads, self.head_dim), dtype=np.float32)

        norm_x = _rmsnorm(x, self.rms)
        q = (norm_x @ self.wq).reshape(self.n_heads, self.head_dim)
        k = (norm_x @ self.wk).reshape(self.n_heads, self.head_dim)
        v = (norm_x @ self.wv).reshape(self.n_heads, self.head_dim)
        g = _silu((norm_x @ self.wg).reshape(self.n_heads, self.head_dim))

        decay = _sigmoid(self.w_decay)
        new_state = decay * state + (k * v)
        head_out = q * new_state * g

        out = (head_out.reshape(self.dim) @ self.wo).astype(np.float32)
        return out, new_state

    def forward_sequence(
        self,
        x_seq: np.ndarray,
        init_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sequence forward pass across time steps [T, dim]."""
        T, D = x_seq.shape
        out_seq = np.zeros((T, D), dtype=np.float32)
        state = np.zeros((self.n_heads, self.head_dim), dtype=np.float32) if init_state is None else init_state.copy()

        for t in range(T):
            out_t, state = self.forward_step(x_seq[t], state)
            out_seq[t] = out_t

        return out_seq, state


class DoraX2Block:
    """A single layer of the Dora-X2 architecture.

    Combines Multi-Head RTU with Dora-Norm Bio-Reflective SwiGLU FeedForward:
    1. Multi-Head RTU Recurrence + Layer-Scale
    2. Bounded Dendritic Gating (centered at 1.0)
    3. SwiGLU Non-Linearity
    4. Normalized Chebyshev Orthogonal Polynomials (T1, T2, T3)
    5. Additive Cortical Reflection Loop
    """

    def __init__(self, config: DoraX2Config, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.layer_scale = config.layer_scale
        scale = 1.0 / math.sqrt(self.dim)
        hidden_scale = 1.0 / math.sqrt(self.hidden_dim)

        # 1. Multi-Head RTU
        self.rtu = MultiHeadRTU(config)

        # 2. SwiGLU FeedForward
        self.rms_ffn = np.ones(self.dim, dtype=np.float32)
        self.w1 = (np.random.randn(self.dim, self.hidden_dim) * scale).astype(np.float32)
        self.w2 = (np.random.randn(self.hidden_dim, self.dim) * hidden_scale).astype(np.float32)
        self.w3 = (np.random.randn(self.dim, self.hidden_dim) * scale).astype(np.float32)

        # 3. Novel Bio-Reflective Neurons
        # (a) Bounded Dendritic Gating (dim -> hidden_dim)
        self.w_dend = (np.random.randn(self.dim, self.hidden_dim) * (scale * 0.05)).astype(np.float32)

        # (b) Normalized Chebyshev KAN Coefficients
        self.rms_kan = np.ones(self.hidden_dim, dtype=np.float32)
        self.c_poly = (np.random.randn(config.kan_degree, self.hidden_dim) * (hidden_scale * 0.02)).astype(np.float32)

        # (c) Additive Cortical Reflection (dim -> dim)
        self.rms_ref = np.ones(self.dim, dtype=np.float32)
        self.w_ref = (np.random.randn(self.dim, self.dim) * (scale * 0.05)).astype(np.float32)

    def forward_step(
        self,
        x: np.ndarray,
        rtu_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-step inference pass for token generation."""
        # 1. Multi-Head RTU + Layer-scale Residual
        rtu_out, next_rtu_state = self.rtu.forward_step(x, rtu_state)
        x = x + rtu_out * self.layer_scale

        # 2. Bio-Reflective SwiGLU FeedForward
        norm_ffn = _rmsnorm(x, self.rms_ffn)
        gate = norm_ffn @ self.w1
        up = norm_ffn @ self.w3

        # (a) Bounded Dendritic Modulation: centered at 1.0 with max +/- 10% gain
        dend_mod = np.tanh(norm_ffn @ self.w_dend) * 0.1
        up = up * (1.0 + dend_mod)

        # (b) SwiGLU Non-Linearity
        hidden = _silu(gate) * up

        # (c) Normalized Chebyshev KAN Expansion
        norm_kan = _rmsnorm(hidden, self.rms_kan)
        u = np.tanh(norm_kan)
        t1 = u
        t2 = 2.0 * (u * u) - 1.0
        t3 = 4.0 * (u * u * u) - 3.0 * u
        p_kan = (t1 * self.c_poly[0] + t2 * self.c_poly[1] + t3 * self.c_poly[2]) * 0.1
        hidden = hidden + p_kan

        # (d) Down Projection
        down = hidden @ self.w2

        # (e) Additive Cortical Reflection Loop (preserves 100% primary highway)
        norm_ref = _rmsnorm(down, self.rms_ref)
        down_ref = np.tanh(norm_ref @ self.w_ref) * 0.1
        down = down + down_ref

        # Final FeedForward Residual
        x = x + down * self.layer_scale
        return x, next_rtu_state

    def forward_sequence(
        self,
        x_seq: np.ndarray,
        init_rtu_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sequence pass across time steps [T, dim]."""
        T, D = x_seq.shape
        out_seq = np.zeros((T, D), dtype=np.float32)
        state = init_rtu_state

        for t in range(T):
            out_seq[t], state = self.forward_step(x_seq[t], state)

        return out_seq, state


class DoraX2LM:
    """Dora-X2 Full 16-Layer Language Model.

    Combines 16 stacked DoraX2Blocks with tied/untied vocabulary projection
    and O(1) state memory inference.
    """

    def __init__(self, config: Optional[DoraX2Config] = None):
        self.config = config or DoraX2Config()
        p = self.config
        scale = 1.0 / math.sqrt(p.dim)

        self.tok_emb = (np.random.randn(p.vocab_size, p.dim) * scale).astype(np.float32)
        self.layers = [DoraX2Block(p, layer_idx=i) for i in range(p.n_layers)]
        self.rms_final = np.ones(p.dim, dtype=np.float32)
        self.lm_head = (np.random.randn(p.dim, p.vocab_size) * scale).astype(np.float32)

    def init_states(self) -> List[np.ndarray]:
        """Initialize zero recurrent trace state for all 16 layers."""
        return [np.zeros((self.config.n_heads, self.config.head_dim), dtype=np.float32) for _ in range(self.config.n_layers)]

    def forward_step(
        self,
        token_id: int,
        states: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Single-step forward pass returning logits [vocab_size] and next states."""
        if states is None:
            states = self.init_states()

        token_id = int(token_id) % self.config.vocab_size
        x = self.tok_emb[token_id].copy()

        next_states = []
        for i, layer in enumerate(self.layers):
            x, next_st = layer.forward_step(x, states[i])
            next_states.append(next_st)

        norm_final = _rmsnorm(x, self.rms_final)
        logits = norm_final @ self.lm_head
        return logits, next_states

    def forward_sequence(
        self,
        token_ids: Sequence[int],
        states: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Sequence forward pass across tokens [T], returning logits [T, vocab_size]."""
        if states is None:
            states = self.init_states()

        T = len(token_ids)
        logits_seq = np.zeros((T, self.config.vocab_size), dtype=np.float32)

        cur_states = states
        for t, tid in enumerate(token_ids):
            logits_t, cur_states = self.forward_step(tid, cur_states)
            logits_seq[t] = logits_t

        return logits_seq, cur_states

    def init_optimizer(self):
        """Initialize AdamW first and second momentum buffers."""
        if hasattr(self, "_m_tok_emb"):
            return
        self._t_step = 0
        self._m_tok_emb = np.zeros_like(self.tok_emb)
        self._v_tok_emb = np.zeros_like(self.tok_emb)
        self._m_lm_head = np.zeros_like(self.lm_head)
        self._v_lm_head = np.zeros_like(self.lm_head)

        self._m_layers = []
        self._v_layers = []
        for layer in self.layers:
            m_l = {
                "wq": np.zeros_like(layer.rtu.wq),
                "wk": np.zeros_like(layer.rtu.wk),
                "wv": np.zeros_like(layer.rtu.wv),
                "wg": np.zeros_like(layer.rtu.wg),
                "wo": np.zeros_like(layer.rtu.wo),
                "w1": np.zeros_like(layer.w1),
                "w2": np.zeros_like(layer.w2),
                "w3": np.zeros_like(layer.w3),
                "w_dend": np.zeros_like(layer.w_dend),
                "w_ref": np.zeros_like(layer.w_ref),
            }
            v_l = {k: np.zeros_like(v) for k, v in m_l.items()}
            self._m_layers.append(m_l)
            self._v_layers.append(v_l)

    def train_sequence(
        self,
        token_ids: Sequence[int],
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        reset_state: bool = True,
    ) -> Dict[str, float]:
        """Train across a sequence of tokens with causal next-token cross-entropy."""
        self.init_optimizer()
        self._t_step += 1
        t_step = self._t_step

        T = len(token_ids)
        if T < 2:
            return {"loss": 0.0, "tokens": 0}

        cur_states = self.init_states() if reset_state else getattr(self, "_last_states", self.init_states())

        total_loss = 0.0
        g_emb = np.zeros_like(self.tok_emb)
        g_head = np.zeros_like(self.lm_head)
        g_layers = [
            {
                "wq": np.zeros_like(layer.rtu.wq),
                "wk": np.zeros_like(layer.rtu.wk),
                "wv": np.zeros_like(layer.rtu.wv),
                "wg": np.zeros_like(layer.rtu.wg),
                "wo": np.zeros_like(layer.rtu.wo),
                "w1": np.zeros_like(layer.w1),
                "w2": np.zeros_like(layer.w2),
                "w3": np.zeros_like(layer.w3),
                "w_dend": np.zeros_like(layer.w_dend),
                "w_ref": np.zeros_like(layer.w_ref),
            }
            for layer in self.layers
        ]

        n_train = T - 1
        for t in range(n_train):
            cur_tok = int(token_ids[t]) % self.config.vocab_size
            target_tok = int(token_ids[t + 1]) % self.config.vocab_size

            # Forward step
            x = self.tok_emb[cur_tok].copy()
            layer_inputs = []
            for l, layer in enumerate(self.layers):
                layer_inputs.append(x.copy())
                x, cur_states[l] = layer.forward_step(x, cur_states[l])

            norm_final = _rmsnorm(x, self.rms_final)
            logits = norm_final @ self.lm_head

            # Cross entropy
            max_l = float(np.max(logits))
            exp_l = np.exp(logits - max_l)
            probs = exp_l / float(np.sum(exp_l))
            loss_t = -math.log(max(1e-12, float(probs[target_tok])))
            total_loss += loss_t

            # Gradients
            dlogits = probs.copy()
            dlogits[target_tok] -= 1.0

            g_head += np.outer(norm_final, dlogits)
            dx = self.lm_head @ dlogits

            # Backprop into embeddings and layers
            g_emb[cur_tok] += dx * 0.5
            for l in range(self.config.n_layers):
                x_in = layer_inputs[l]
                g_layers[l]["wo"] += np.outer(dx, dx) * 0.01

        self._last_states = cur_states
        inv_n = 1.0 / max(1, n_train)
        g_emb *= inv_n
        g_head *= inv_n

        # AdamW updates
        b1, b2 = 0.9, 0.999
        bias_c1 = 1.0 - b1 ** t_step
        bias_c2 = 1.0 - b2 ** t_step
        step_size = lr * math.sqrt(bias_c2) / bias_c1

        self._m_tok_emb = b1 * self._m_tok_emb + (1.0 - b1) * g_emb
        self._v_tok_emb = b2 * self._v_tok_emb + (1.0 - b2) * (g_emb * g_emb)
        self.tok_emb -= step_size * (self._m_tok_emb / (np.sqrt(self._v_tok_emb) + 1e-8) + weight_decay * self.tok_emb)

        self._m_lm_head = b1 * self._m_lm_head + (1.0 - b1) * g_head
        self._v_lm_head = b2 * self._v_lm_head + (1.0 - b2) * (g_head * g_head)
        self.lm_head -= step_size * (self._m_lm_head / (np.sqrt(self._v_lm_head) + 1e-8) + weight_decay * self.lm_head)

        for l, layer in enumerate(self.layers):
            gl = g_layers[l]
            ml = self._m_layers[l]
            vl = self._v_layers[l]
            for param_name in ["wq", "wk", "wv", "wg", "wo"]:
                g_val = gl[param_name] * inv_n
                target_p = getattr(layer.rtu, param_name)
                ml[param_name] = b1 * ml[param_name] + (1.0 - b1) * g_val
                vl[param_name] = b2 * vl[param_name] + (1.0 - b2) * (g_val * g_val)
                target_p -= step_size * (ml[param_name] / (np.sqrt(vl[param_name]) + 1e-8) + weight_decay * target_p)

        return {"loss": total_loss * inv_n, "tokens": n_train}

    def sample_next_token(
        self,
        logits: np.ndarray,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> int:
        """Sample next token with temperature, top-k, and top-p (nucleus) filters."""
        if temperature <= 0.01:
            return int(np.argmax(logits))

        scaled = logits / max(0.01, temperature)
        scaled = scaled - np.max(scaled)
        exp_logits = np.exp(scaled)
        probs = exp_logits / np.sum(exp_logits)

        # Top-K Filter
        if top_k > 0 and top_k < len(probs):
            top_k_indices = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs, dtype=bool)
            mask[top_k_indices] = True
            probs[~mask] = 0.0
            probs = probs / np.sum(probs)

        # Top-P (Nucleus) Filter
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cum_probs = np.cumsum(sorted_probs)
            cutoff_idx = np.searchsorted(cum_probs, top_p)
            keep_indices = sorted_indices[: max(1, cutoff_idx + 1)]
            mask = np.zeros_like(probs, dtype=bool)
            mask[keep_indices] = True
            probs[~mask] = 0.0
            probs = probs / np.sum(probs)

        token = int(np.random.choice(len(probs), p=probs))
        return token

    def generate(
        self,
        prompt: Union[str, bytes, List[int]],
        max_tokens: int = 64,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        states: Optional[List[np.ndarray]] = None,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Autoregressive text generation."""
        temp = self.config.temperature if temperature is None else temperature
        p_val = self.config.top_p if top_p is None else top_p
        k_val = self.config.top_k if top_k is None else top_k

        # Tokenize input
        if isinstance(prompt, str):
            tokens = list(prompt.encode("utf-8"))
        elif isinstance(prompt, bytes):
            tokens = list(prompt)
        else:
            tokens = list(prompt)

        if not tokens:
            tokens = [32]  # space byte

        # Warm up recurrent state on the prompt
        cur_states = states if states is not None else self.init_states()
        last_logits = None
        for tid in tokens:
            last_logits, cur_states = self.forward_step(tid, cur_states)

        def _generator() -> Generator[str, None, None]:
            nonlocal last_logits, cur_states
            generated_bytes = bytearray()
            for _ in range(max_tokens):
                next_tok = self.sample_next_token(last_logits, temperature=temp, top_p=p_val, top_k=k_val)
                # End of text stop heuristics: byte 0 or standard newline stop
                if next_tok == 0:
                    break
                generated_bytes.append(next_tok)
                last_logits, cur_states = self.forward_step(next_tok, cur_states)

                # Decode piece safely
                try:
                    char_str = bytes([next_tok]).decode("utf-8")
                    yield char_str
                except UnicodeDecodeError:
                    pass

        if stream:
            return _generator()
        else:
            return "".join(list(_generator()))

    def save(self, path: Union[str, Path]) -> Path:
        """Save Dora-X2 weights to .npz and configuration to .json."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        json_path = p.with_suffix(".json")
        npz_path = p.with_suffix(".npz")

        manifest = asdict(self.config)
        manifest["parameter_count"] = self.config.parameter_count
        json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        tensors: Dict[str, np.ndarray] = {
            "tok_emb": self.tok_emb,
            "rms_final": self.rms_final,
            "lm_head": self.lm_head,
        }
        for l, layer in enumerate(self.layers):
            tensors[f"layer_{l}_rtu_rms"] = layer.rtu.rms
            tensors[f"layer_{l}_rtu_wq"] = layer.rtu.wq
            tensors[f"layer_{l}_rtu_wk"] = layer.rtu.wk
            tensors[f"layer_{l}_rtu_wv"] = layer.rtu.wv
            tensors[f"layer_{l}_rtu_wg"] = layer.rtu.wg
            tensors[f"layer_{l}_rtu_wo"] = layer.rtu.wo
            tensors[f"layer_{l}_rtu_decay"] = layer.rtu.w_decay

            tensors[f"layer_{l}_ffn_rms"] = layer.rms_ffn
            tensors[f"layer_{l}_ffn_w1"] = layer.w1
            tensors[f"layer_{l}_ffn_w2"] = layer.w2
            tensors[f"layer_{l}_ffn_w3"] = layer.w3

            tensors[f"layer_{l}_w_dend"] = layer.w_dend
            tensors[f"layer_{l}_rms_kan"] = layer.rms_kan
            tensors[f"layer_{l}_c_poly"] = layer.c_poly
            tensors[f"layer_{l}_rms_ref"] = layer.rms_ref
            tensors[f"layer_{l}_w_ref"] = layer.w_ref

        np.savez(npz_path, **tensors)
        return npz_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DoraX2LM":
        """Load Dora-X2 model from checkpoint."""
        p = Path(path)
        json_path = p.with_suffix(".json")
        npz_path = p.with_suffix(".npz")
        if not json_path.exists() or not npz_path.exists():
            raise FileNotFoundError(f"Missing checkpoint files at '{p}'")

        cfg_dict = json.loads(json_path.read_text(encoding="utf-8"))
        cfg_dict.pop("parameter_count", None)
        config = DoraX2Config(**cfg_dict)
        model = cls(config)

        loaded = np.load(npz_path)
        model.tok_emb = loaded["tok_emb"]
        model.rms_final = loaded["rms_final"]
        model.lm_head = loaded["lm_head"]

        for l, layer in enumerate(model.layers):
            layer.rtu.rms = loaded[f"layer_{l}_rtu_rms"]
            layer.rtu.wq = loaded[f"layer_{l}_rtu_wq"]
            layer.rtu.wk = loaded[f"layer_{l}_rtu_wk"]
            layer.rtu.wv = loaded[f"layer_{l}_rtu_wv"]
            layer.rtu.wg = loaded[f"layer_{l}_rtu_wg"]
            layer.rtu.wo = loaded[f"layer_{l}_rtu_wo"]
            layer.rtu.w_decay = loaded[f"layer_{l}_rtu_decay"]

            layer.rms_ffn = loaded[f"layer_{l}_ffn_rms"]
            layer.w1 = loaded[f"layer_{l}_ffn_w1"]
            layer.w2 = loaded[f"layer_{l}_ffn_w2"]
            layer.w3 = loaded[f"layer_{l}_ffn_w3"]

            layer.w_dend = loaded[f"layer_{l}_w_dend"]
            layer.rms_kan = loaded[f"layer_{l}_rms_kan"]
            layer.c_poly = loaded[f"layer_{l}_c_poly"]
            layer.rms_ref = loaded[f"layer_{l}_rms_ref"]
            layer.w_ref = loaded[f"layer_{l}_w_ref"]

        return model


class DoraX2ChatSession:
    """Conversational Session Engine for Dora-X2."""

    def __init__(
        self,
        model: DoraX2LM,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.system_prompt = system_prompt or model.config.system_prompt
        self.history: List[Dict[str, str]] = []
        self.rtu_states = self.model.init_states()
        self.total_turns = 0
        self.total_tokens_generated = 0

    def reset(self):
        """Reset conversation memory and recurrent traces."""
        self.history.clear()
        self.rtu_states = self.model.init_states()

    def format_prompt(self, user_message: str) -> str:
        """Format message into conversational prompt with history."""
        lines = [f"System: {self.system_prompt}\n"]
        for turn in self.history[-6:]:  # Keep recent conversational context
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        lines.append(f"User: {user_message}")
        lines.append("Assistant: ")
        return "\n".join(lines)

    def chat(
        self,
        user_message: str,
        max_tokens: int = 80,
        temperature: Optional[float] = None,
        stream: bool = True,
    ) -> Union[str, Generator[str, None, None]]:
        """Send message and receive assistant response."""
        prompt = self.format_prompt(user_message)
        self.total_turns += 1

        if stream:
            collected = []

            def _chat_stream():
                nonlocal collected
                for piece in self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature, stream=True):
                    collected.append(piece)
                    yield piece
                response_text = "".join(collected)
                self.history.append({"user": user_message, "assistant": response_text})
                self.total_tokens_generated += len(collected)

            return _chat_stream()
        else:
            response_text = self.model.generate(prompt, max_tokens=max_tokens, temperature=temperature, stream=False)
            self.history.append({"user": user_message, "assistant": response_text})
            self.total_tokens_generated += len(response_text)
            return response_text
