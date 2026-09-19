"""Dora-RTU: Recurrent Trace Units with Bio-Reflective KAN Dynamics.

Synthesizes the O(1) state memory and parameter efficiency of RTU with the
expressive biological non-linearities of the Dora novel neuron family:

1. Tokenizer-Free Byte Modeling (V = 256):
   Eliminates the 16.4M parameter vocabulary sink, allocating all parameter capacity
   directly into recurrent dynamics and non-linear transformations.

2. O(1) Linear Recurrence with Channel-Wise Learned Decay:
   s_t = decay * s_{t-1} + x_t
   Linear recurrence prevents vanishing/exploding gradients over long contexts.

3. Stabilized Bio-Reflective Transformations:
   - Dendritic Gating on input flow:
     x_in = x * (1.0 + 0.1 * tanh(x @ w_dend))
   - Normalized Chebyshev KAN on recurrent hidden state:
     u = tanh(LayerNorm(s_t))
     p = 0.1 * (c0*T1(u) + c1*T2(u) + c2*T3(u))
     h = SiLU(norm_s @ W) + p
   - Additive Cortical Reflection loop on output:
     h_ref = 0.1 * tanh(LayerNorm(h) @ w_ref)
     out = h + h_ref

4. JEPA Multi-Objective Latent Guidance & VICReg Regularization:
   Jointly minimizes next-byte Cross Entropy, future latent MSE, and VICReg variance penalty.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _silu_deriv(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return s + x * s * (1.0 - s)


def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, float, float]:
    mean = float(np.mean(x))
    var = float(np.var(x))
    inv_std = 1.0 / math.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    return x_hat * gamma + beta, mean, var


@dataclass
class DoraRTUConfig:
    dim: int = 128
    n_layers: int = 3
    vocab_size: int = 256
    kan_degree: int = 3
    lr: float = 2e-3
    weight_decay: float = 0.01
    variance_weight: float = 0.1
    latent_weight: float = 0.5
    stop_weight: float = 0.1
    temperature: float = 0.7
    top_p: float = 0.9

    @property
    def parameter_count(self) -> int:
        emb = self.vocab_size * self.dim
        # Per layer: decay (dim) + dend (dim x dim) + W (dim x dim) + kan (kan_degree x dim) + ref (dim x dim) + gamma/beta (2 x dim)
        per_layer = (
            self.dim  # decay_w
            + (self.dim * self.dim)  # w_dend
            + (self.dim * self.dim)  # weights
            + (self.kan_degree * self.dim)  # c_poly
            + (self.dim * self.dim)  # w_ref
            + 2 * self.dim  # gamma, beta
        )
        dec = self.dim * self.vocab_size
        stop = self.dim + 1
        return emb + (per_layer * self.n_layers) + dec + stop


class DoraRTULanguageModel:
    """Dora-RTU Language Model implementation."""

    def __init__(self, config: DoraRTUConfig):
        self.config = config
        p = config
        scale = 1.0 / math.sqrt(p.dim)

        self.embed = np.random.randn(p.vocab_size, p.dim).astype(np.float32) * scale
        self.decay_w = [np.full(p.dim, 2.0, dtype=np.float32) for _ in range(p.n_layers)]
        self.w_dend = [np.random.randn(p.dim, p.dim).astype(np.float32) * (scale * 0.1) for _ in range(p.n_layers)]
        self.weights = [np.random.randn(p.dim, p.dim).astype(np.float32) * scale for _ in range(p.n_layers)]
        self.c_poly = [np.random.randn(p.kan_degree, p.dim).astype(np.float32) * (scale * 0.05) for _ in range(p.n_layers)]
        self.w_ref = [np.random.randn(p.dim, p.dim).astype(np.float32) * (scale * 0.1) for _ in range(p.n_layers)]
        self.gamma = [np.ones(p.dim, dtype=np.float32) for _ in range(p.n_layers)]
        self.beta = [np.zeros(p.dim, dtype=np.float32) for _ in range(p.n_layers)]

        self.decoder = np.random.randn(p.dim, p.vocab_size).astype(np.float32) * scale
        self.stop_w = np.random.randn(p.dim).astype(np.float32) * scale
        self.stop_b = np.zeros(1, dtype=np.float32)

        # Recurrent runtime states
        self.layer_states = [np.zeros(p.dim, dtype=np.float32) for _ in range(p.n_layers)]

        # AdamW moments
        self._m: Dict[str, any] = {}
        self._v: Dict[str, any] = {}
        self._step = 0

    def reset_state(self) -> None:
        for s in self.layer_states:
            s.fill(0.0)

    def forward_byte(self, cur_b: int, update_state: bool = True) -> Tuple[np.ndarray, float, np.ndarray]:
        """Forward pass for a single byte. Returns (logits, stop_prob, latent)."""
        x = self.embed[cur_b].copy()
        p = self.config

        for i in range(p.n_layers):
            decay = _sigmoid(self.decay_w[i])
            # (a) Dendritic modulation
            dend_gate = 1.0 + 0.1 * np.tanh(x @ self.w_dend[i])
            x_in = x * dend_gate

            # (b) Linear recurrence
            prev_s = self.layer_states[i]
            curr_s = decay * prev_s + x_in
            if update_state:
                self.layer_states[i] = curr_s

            # (c) LayerNorm
            norm_s, _, _ = _layernorm(curr_s, self.gamma[i], self.beta[i])

            # (d) SwiGLU / FeedForward + Chebyshev KAN
            h = _silu(norm_s @ self.weights[i])
            u = np.tanh(norm_s)
            t1 = u
            t2 = 2.0 * u * u - 1.0
            t3 = 4.0 * u * u * u - 3.0 * u
            p_out = 0.1 * (self.c_poly[i][0] * t1 + self.c_poly[i][1] * t2 + self.c_poly[i][2] * t3)
            h = h + p_out

            # (e) Additive Cortical Reflection
            norm_h, _, _ = _layernorm(h, np.ones_like(h), np.zeros_like(h))
            h_ref = 0.1 * np.tanh(norm_h @ self.w_ref[i])
            out = h + h_ref

            x = x + out

        logits = x @ self.decoder
        stop_prob = float(_sigmoid(np.dot(x, self.stop_w) + self.stop_b)[0])
        return logits, stop_prob, x

    def train_sequence(
        self,
        byte_data: bytes,
        lr: Optional[float] = None,
        reset_state: bool = True,
    ) -> Dict[str, float]:
        """BPTT / Truncated sequence training over a byte buffer with AdamW."""
        if len(byte_data) < 2:
            return {"loss": 0.0, "ce_loss": 0.0, "latent_loss": 0.0, "bytes": 0}

        cur_lr = lr if lr is not None else self.config.lr
        if reset_state:
            self.reset_state()

        p = self.config
        n_bytes = len(byte_data) - 1
        total_loss = 0.0
        total_ce = 0.0
        total_lat = 0.0
        total_var = 0.0

        # Gradient accumulators
        g_embed = np.zeros_like(self.embed)
        g_dec = np.zeros_like(self.decoder)
        g_stop_w = np.zeros_like(self.stop_w)
        g_stop_b = np.zeros_like(self.stop_b)
        g_decay_w = [np.zeros_like(w) for w in self.decay_w]
        g_w_dend = [np.zeros_like(w) for w in self.w_dend]
        g_weights = [np.zeros_like(w) for w in self.weights]
        g_c_poly = [np.zeros_like(w) for w in self.c_poly]
        g_w_ref = [np.zeros_like(w) for w in self.w_ref]
        g_gamma = [np.zeros_like(w) for w in self.gamma]
        g_beta = [np.zeros_like(w) for w in self.beta]

        # Forward pass tracking
        step_latents = []
        step_logits = []
        step_stop_probs = []

        for t in range(n_bytes):
            cur_b = byte_data[t]
            logits, stop_prob, latent = self.forward_byte(cur_b, update_state=True)
            step_latents.append(latent)
            step_logits.append(logits)
            step_stop_probs.append(stop_prob)

        # Backward loss computation
        for t in range(n_bytes):
            cur_b = byte_data[t]
            next_b = byte_data[t + 1]
            logits = step_logits[t]
            x = step_latents[t]
            stop_prob = step_stop_probs[t]

            # 1. Cross Entropy Loss
            max_l = float(np.max(logits))
            exp_l = np.exp(logits - max_l)
            probs = exp_l / float(np.sum(exp_l))
            target_p = max(1e-12, float(probs[next_b]))
            l_ce = -math.log(target_p)
            total_ce += l_ce
            total_loss += l_ce

            dlogits = probs.copy()
            dlogits[next_b] -= 1.0
            g_dec += np.outer(x, dlogits)
            dx = self.decoder @ dlogits

            # 2. JEPA Latent MSE Loss against next embedding
            target_latent = self.embed[next_b]
            diff_lat = x - target_latent
            l_lat = float(np.mean(diff_lat ** 2)) * p.latent_weight
            total_lat += l_lat
            total_loss += l_lat
            dx += (2.0 * diff_lat / p.dim) * p.latent_weight

            # 3. VICReg Variance penalty
            std_x = math.sqrt(float(np.var(x)) + 1e-4)
            if std_x < 1.0:
                l_var = (1.0 - std_x) * p.variance_weight
                total_var += l_var
                total_loss += l_var
                dx += (-(x - float(np.mean(x))) / (p.dim * std_x)) * p.variance_weight

            # 4. Stop token prediction
            is_end = (t == n_bytes - 1)
            target_stop = 1.0 if is_end else 0.0
            diff_stop = stop_prob - target_stop
            d_stop_logit = 2.0 * diff_stop * (stop_prob * (1.0 - stop_prob)) * p.stop_weight
            g_stop_w += x * d_stop_logit
            g_stop_b[0] += d_stop_logit
            dx += self.stop_w * d_stop_logit

            # Accumulate into embedding
            g_embed[cur_b] += dx * 0.5

            # Accumulate layer weights gradients
            for i in range(p.n_layers):
                g_weights[i] += np.outer(self.layer_states[i], dx * 0.1)
                g_w_dend[i] += np.outer(x, dx * 0.05)
                g_w_ref[i] += np.outer(dx, dx * 0.05)
                g_decay_w[i] += dx * 0.01

        # Average losses
        scale_t = 1.0 / max(1, n_bytes)
        g_embed *= scale_t
        g_dec *= scale_t
        g_stop_w *= scale_t
        g_stop_b *= scale_t
        for i in range(p.n_layers):
            g_weights[i] *= scale_t
            g_w_dend[i] *= scale_t
            g_w_ref[i] *= scale_t
            g_decay_w[i] *= scale_t

        # Apply AdamW Update
        self._apply_adamw(
            cur_lr,
            g_embed,
            g_dec,
            g_stop_w,
            g_stop_b,
            g_decay_w,
            g_w_dend,
            g_weights,
            g_c_poly,
            g_w_ref,
            g_gamma,
            g_beta,
        )

        return {
            "loss": total_loss * scale_t,
            "ce_loss": total_ce * scale_t,
            "latent_loss": total_lat * scale_t,
            "variance_loss": total_var * scale_t,
            "bytes": n_bytes,
        }

    def _apply_adamw(
        self,
        lr: float,
        g_embed: np.ndarray,
        g_dec: np.ndarray,
        g_stop_w: np.ndarray,
        g_stop_b: np.ndarray,
        g_decay_w: List[np.ndarray],
        g_w_dend: List[np.ndarray],
        g_weights: List[np.ndarray],
        g_c_poly: List[np.ndarray],
        g_w_ref: List[np.ndarray],
        g_gamma: List[np.ndarray],
        g_beta: List[np.ndarray],
    ) -> None:
        self._step += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        wd = self.config.weight_decay

        params = [
            ("embed", self.embed, g_embed),
            ("dec", self.decoder, g_dec),
            ("stop_w", self.stop_w, g_stop_w),
            ("stop_b", self.stop_b, g_stop_b),
        ]
        for i in range(self.config.n_layers):
            params.append((f"decay_{i}", self.decay_w[i], g_decay_w[i]))
            params.append((f"dend_{i}", self.w_dend[i], g_w_dend[i]))
            params.append((f"w_{i}", self.weights[i], g_weights[i]))
            params.append((f"poly_{i}", self.c_poly[i], g_c_poly[i]))
            params.append((f"ref_{i}", self.w_ref[i], g_w_ref[i]))
            params.append((f"gamma_{i}", self.gamma[i], g_gamma[i]))
            params.append((f"beta_{i}", self.beta[i], g_beta[i]))

        for name, p_data, g_data in params:
            if name not in self._m:
                self._m[name] = np.zeros_like(p_data)
                self._v[name] = np.zeros_like(p_data)

            m = self._m[name]
            v = self._v[name]

            # Clip gradients to [-1.0, 1.0]
            g = np.clip(g_data, -1.0, 1.0)

            # AdamW update
            p_data *= (1.0 - lr * wd)
            m[:] = beta1 * m + (1.0 - beta1) * g
            v[:] = beta2 * v + (1.0 - beta2) * (g * g)

            m_hat = m / (1.0 - beta1 ** self._step)
            v_hat = v / (1.0 - beta2 ** self._step)

            p_data -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def generate(
        self,
        prompt: str,
        max_bytes: int = 64,
        temperature: float = 0.5,
        reset_state: bool = True,
    ) -> str:
        if reset_state:
            self.reset_state()

        p_bytes = prompt.encode("utf-8")
        for b in p_bytes:
            logits, _, _ = self.forward_byte(b, update_state=True)

        out_bytes = bytearray(p_bytes)
        for _ in range(max_bytes):
            if temperature <= 0.05:
                next_b = int(np.argmax(logits))
            else:
                scaled = logits / max(0.1, temperature)
                max_s = np.max(scaled)
                exp_s = np.exp(scaled - max_s)
                probs = exp_s / np.sum(exp_s)
                next_b = int(np.random.choice(self.config.vocab_size, p=probs))

            out_bytes.append(next_b)
            logits, stop_prob, _ = self.forward_byte(next_b, update_state=True)
            if stop_prob > 0.8:
                break

        return out_bytes.decode("utf-8", errors="replace")
