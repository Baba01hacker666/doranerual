"""Recurrent Trace Unit (RTU) Language Model Architecture.

A byte-level recurrent latent model with:
- Byte-level tokenization (Vocabulary size = 256 bytes, zero tokenizer dependency)
- Recurrent Trace Units (RTUs) with learned channel-wise exponential decay (O(1) constant memory)
- Real-Time Recurrent Learning (RTRL) forward-mode eligibility traces for online test-time streaming
- Joint JEPA latent prediction + Cross-Entropy + VICReg variance regularization
- Full sequence batch training and autoregressive streaming text generation
"""

from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return x * _sigmoid(x)


def _silu_deriv(x: np.ndarray) -> np.ndarray:
    """Derivative of SiLU: s + x * s * (1 - s) where s = sigmoid(x)."""
    s = _sigmoid(x)
    return s + x * s * (1.0 - s)


def _layernorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, float, float]:
    """1D LayerNorm forward pass returning (normalized, mean, var)."""
    mean = float(np.mean(x))
    var = float(np.var(x))
    inv_std = 1.0 / math.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    out = x_hat * gamma + beta
    return out, mean, var


def _layernorm_backward(
    dout: np.ndarray,
    x: np.ndarray,
    gamma: np.ndarray,
    mean: float,
    var: float,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D LayerNorm backward pass returning (dx, dgamma, dbeta)."""
    dim = len(x)
    std = math.sqrt(var + eps)
    inv_std = 1.0 / std
    x_hat = (x - mean) * inv_std

    dgamma = dout * x_hat
    dbeta = dout.copy()

    dxhat = dout * gamma
    dvar = float(np.sum(dxhat * (x - mean) * -0.5 * (inv_std ** 3)))
    dmean = float(np.sum(dxhat * -inv_std)) + dvar * float(np.mean(-2.0 * (x - mean)))

    dx = dxhat * inv_std + dvar * 2.0 * (x - mean) / dim + dmean / dim
    return dx, dgamma, dbeta


@dataclass
class RTUConfig:
    """Hyperparameter blueprint for RTU Recurrent Latent Language Model."""
    dim: int = 256
    n_layers: int = 4
    vocab_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    variance_weight: float = 0.1
    latent_weight: float = 0.5
    stop_weight: float = 0.1
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42

    @property
    def parameter_count(self) -> int:
        """Total trainable parameter count."""
        # Embedding: 256 x dim
        emb = self.vocab_size * self.dim
        # Per layer: decay_w (dim) + weight (dim x dim) + gamma (dim) + beta (dim)
        per_layer = self.dim + (self.dim * self.dim) + 2 * self.dim
        # Decoder: dim x vocab_size
        dec = self.dim * self.vocab_size
        # Stop head: dim x 1 + 1
        stop = self.dim + 1
        return emb + (per_layer * self.n_layers) + dec + stop


class RTULayerState:
    """Mutable runtime state & eligibility traces for a single RTU layer."""

    def __init__(self, dim: int, vocab_size: int = 256):
        self.dim = dim
        self.vocab_size = vocab_size
        self.state = np.zeros(dim, dtype=np.float32)
        self.decaytrace = np.zeros(dim, dtype=np.float32)
        self.embedtrace = np.zeros((vocab_size, dim), dtype=np.float32)

    def reset(self) -> None:
        """Reset internal recurrence and eligibility traces to zero."""
        self.state.fill(0.0)
        self.decaytrace.fill(0.0)
        self.embedtrace.fill(0.0)


class RTULanguageModel:
    """Byte-level Recurrent Trace Unit (RTU) Language Model with Latent Prediction.

    Features:
    - O(1) constant inference and training memory.
    - Zero tokenization tokenizer: operates natively on raw bytes (0..255).
    - Real-Time Recurrent Learning (RTRL) forward eligibility traces for test-time streaming.
    - Multi-objective JEPA loss: Cross-entropy + Next-latent MSE + VICReg variance penalty.
    """

    def __init__(self, config: Optional[RTUConfig] = None):
        self.config = config or RTUConfig()
        p = self.config
        rng = np.random.default_rng(p.seed)

        # Variance scaling initialization
        emb_std = 1.0 / math.sqrt(p.dim)
        proj_std = 0.02 / math.sqrt(max(1, p.n_layers))

        # Weights
        self.embed = rng.normal(0.0, emb_std, (p.vocab_size, p.dim)).astype(np.float32)
        self.decoder = rng.normal(0.0, emb_std, (p.dim, p.vocab_size)).astype(np.float32)
        self.stop_w = rng.normal(0.0, emb_std, (p.dim,)).astype(np.float32)
        self.stop_b = np.zeros(1, dtype=np.float32)

        # Layer weights
        self.decay_w: List[np.ndarray] = []
        self.weights: List[np.ndarray] = []
        self.gamma: List[np.ndarray] = []
        self.beta: List[np.ndarray] = []
        self.layer_states: List[RTULayerState] = []

        for _ in range(p.n_layers):
            # Initial decay biased towards retaining memory (~0.90 to 0.98)
            init_decay = rng.uniform(2.0, 4.0, size=(p.dim,)).astype(np.float32)
            self.decay_w.append(init_decay)
            self.weights.append(rng.normal(0.0, proj_std, (p.dim, p.dim)).astype(np.float32))
            self.gamma.append(np.ones(p.dim, dtype=np.float32))
            self.beta.append(np.zeros(p.dim, dtype=np.float32))
            self.layer_states.append(RTULayerState(p.dim, p.vocab_size))

        # AdamW Optimizer moments
        self._adam_step = 0
        self._m_embed = np.zeros_like(self.embed)
        self._v_embed = np.zeros_like(self.embed)
        self._m_dec = np.zeros_like(self.decoder)
        self._v_dec = np.zeros_like(self.decoder)
        self._m_stop_w = np.zeros_like(self.stop_w)
        self._v_stop_w = np.zeros_like(self.stop_w)
        self._m_stop_b = np.zeros_like(self.stop_b)
        self._v_stop_b = np.zeros_like(self.stop_b)

        self._m_decay: List[np.ndarray] = [np.zeros_like(w) for w in self.decay_w]
        self._v_decay: List[np.ndarray] = [np.zeros_like(w) for w in self.decay_w]
        self._m_weights: List[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        self._v_weights: List[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        self._m_gamma: List[np.ndarray] = [np.zeros_like(w) for w in self.gamma]
        self._v_gamma: List[np.ndarray] = [np.zeros_like(w) for w in self.gamma]
        self._m_beta: List[np.ndarray] = [np.zeros_like(w) for w in self.beta]
        self._v_beta: List[np.ndarray] = [np.zeros_like(w) for w in self.beta]

    def reset_state(self) -> None:
        """Clear all recurrent state vectors and eligibility traces."""
        for ls in self.layer_states:
            ls.reset()

    def forward_byte(
        self,
        b: int,
        update_state: bool = True,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Forward a single byte through the RTU network.

        Args:
            b: Byte value in [0, 255].
            update_state: If True, mutates internal recurrent state vectors.

        Returns:
            Tuple of (logits (256,), stop_prob float, latent_vector (dim,)).
        """
        if not 0 <= b < self.config.vocab_size:
            raise ValueError(f"Byte {b} out of range [0, 255]")

        enc = self.embed[b]
        x = enc.copy()

        for i in range(self.config.n_layers):
            decay = _sigmoid(self.decay_w[i])
            prev_s = self.layer_states[i].state
            curr_s = decay * prev_s + enc
            if update_state:
                self.layer_states[i].state = curr_s

            norm_s, _, _ = _layernorm_forward(curr_s, self.gamma[i], self.beta[i])
            out = _silu(norm_s @ self.weights[i])
            x = x + out

        logits = x @ self.decoder
        stop_logit = float(np.dot(x, self.stop_w) + self.stop_b[0])
        stop_prob = float(_sigmoid(np.array([stop_logit]))[0])

        return logits, stop_prob, x

    def step_online_train(
        self,
        cur_b: int,
        next_b: Optional[int] = None,
        is_end: bool = False,
        lr: Optional[float] = None,
    ) -> Tuple[int, float, Dict[str, float]]:
        """Single-step online test-time training with RTRL eligibility traces.

        Updates parameters in-place using forward eligibility traces and
        returns (sampled_next_byte, stop_prob, loss_dict).
        """
        cur_lr = lr if lr is not None else self.config.lr
        p = self.config
        enc = self.embed[cur_b]
        x = enc.copy()

        # Cache forward activations for backward pass
        layer_decays = []
        layer_prev_states = []
        layer_curr_states = []
        layer_norm_states = []
        layer_means = []
        layer_vars = []
        layer_pre_acts = []

        for i in range(p.n_layers):
            decay = _sigmoid(self.decay_w[i])
            prev_s = self.layer_states[i].state.copy()
            curr_s = decay * prev_s + enc

            norm_s, mean, var = _layernorm_forward(curr_s, self.gamma[i], self.beta[i])
            pre_act = norm_s @ self.weights[i]
            out = _silu(pre_act)

            layer_decays.append(decay)
            layer_prev_states.append(prev_s)
            layer_curr_states.append(curr_s)
            layer_norm_states.append(norm_s)
            layer_means.append(mean)
            layer_vars.append(var)
            layer_pre_acts.append(pre_act)

            # Advance state
            self.layer_states[i].state = curr_s
            x = x + out

        # Output predictions
        logits = x @ self.decoder
        stop_logit = float(np.dot(x, self.stop_w) + self.stop_b[0])
        stop_prob = float(_sigmoid(np.array([stop_logit]))[0])

        losses = {"total": 0.0, "ce": 0.0, "latent": 0.0, "variance": 0.0, "stop": 0.0}

        # Gradients accumulator
        g_embed = np.zeros_like(self.embed)
        g_dec = np.zeros_like(self.decoder)
        g_stop_w = np.zeros_like(self.stop_w)
        g_stop_b = np.zeros_like(self.stop_b)
        g_decay_w: List[np.ndarray] = [np.zeros_like(w) for w in self.decay_w]
        g_weights: List[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        g_gamma: List[np.ndarray] = [np.zeros_like(w) for w in self.gamma]
        g_beta: List[np.ndarray] = [np.zeros_like(w) for w in self.beta]

        # 1. Variance loss: VICReg penalty max(0, 1 - std(x))
        x_var = float(np.var(x))
        x_std = math.sqrt(x_var + 1e-4)
        dx = np.zeros_like(x)
        if x_std < 1.0:
            loss_var = (1.0 - x_std) * p.variance_weight
            losses["variance"] = loss_var
            losses["total"] += loss_var
            dx += (-(x - float(np.mean(x))) / (p.dim * x_std)) * p.variance_weight

        # If supervised target byte is provided, compute CE, Latent MSE, and Stop losses
        if next_b is not None:
            # 2. Cross Entropy loss with numerically stable softmax
            max_l = float(np.max(logits))
            exp_l = np.exp(logits - max_l)
            probs = exp_l / float(np.sum(exp_l))
            target_prob = max(1e-12, float(probs[next_b]))
            loss_ce = -math.log(target_prob)
            losses["ce"] = loss_ce
            losses["total"] += loss_ce

            dlogits = probs.copy()
            dlogits[next_b] -= 1.0

            # 3. Latent prediction MSE loss against target embedding (stop_gradient)
            target_latent = self.embed[next_b].copy()
            diff_lat = x - target_latent
            loss_lat = float(np.mean(diff_lat ** 2)) * p.latent_weight
            losses["latent"] = loss_lat
            losses["total"] += loss_lat
            dx += (2.0 * diff_lat / p.dim) * p.latent_weight

            # 4. Stop token MSE loss
            target_stop = 1.0 if is_end else 0.0
            diff_stop = stop_prob - target_stop
            loss_stop = (diff_stop ** 2) * p.stop_weight
            losses["stop"] = loss_stop
            losses["total"] += loss_stop

            d_stop_logit = 2.0 * diff_stop * (stop_prob * (1.0 - stop_prob)) * p.stop_weight
            g_stop_w = x * d_stop_logit
            g_stop_b[0] = d_stop_logit

            # Gradients from decoder logits and stop head into x
            g_dec = np.outer(x, dlogits)
            dx += self.decoder @ dlogits
            dx += self.stop_w * d_stop_logit

            # Backprop through layers and eligibility traces
            for i in reversed(range(p.n_layers)):
                d_out = dx.copy()
                dz = d_out * _silu_deriv(layer_pre_acts[i])
                g_weights[i] = np.outer(layer_norm_states[i], dz)

                d_norm = dz @ self.weights[i].T
                d_state, d_gam, d_bet = _layernorm_backward(
                    d_norm,
                    layer_curr_states[i],
                    self.gamma[i],
                    layer_means[i],
                    layer_vars[i],
                )
                g_gamma[i] += d_gam
                g_beta[i] += d_bet

                # RTRL Eligibility Trace Update
                # embedtrace_t = decay * embedtrace_{t-1} + 1_{c}
                ls = self.layer_states[i]
                ls.embedtrace = ls.embedtrace * layer_decays[i]
                ls.embedtrace[cur_b] += 1.0
                g_embed += ls.embedtrace * d_state

                # decaytrace_t = decay * decaytrace_{t-1} + decay * (1 - decay) * state_{t-1}
                decay_deriv = layer_decays[i] * (1.0 - layer_decays[i])
                ls.decaytrace = layer_decays[i] * ls.decaytrace + decay_deriv * layer_prev_states[i]
                g_decay_w[i] += d_state * ls.decaytrace

            # Direct embedding path gradient
            g_embed[cur_b] += dx

            # Apply AdamW optimizer update
            self._apply_adamw(
                cur_lr,
                g_embed,
                g_dec,
                g_stop_w,
                g_stop_b,
                g_decay_w,
                g_weights,
                g_gamma,
                g_beta,
            )

        sampled_b = self.sample_byte(logits)
        return sampled_b, stop_prob, losses

    def _apply_adamw(
        self,
        lr: float,
        g_embed: np.ndarray,
        g_dec: np.ndarray,
        g_stop_w: np.ndarray,
        g_stop_b: np.ndarray,
        g_decay_w: List[np.ndarray],
        g_weights: List[np.ndarray],
        g_gamma: List[np.ndarray],
        g_beta: List[np.ndarray],
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Execute in-place AdamW parameter update with weight decay."""
        self._adam_step += 1
        t = self._adam_step
        corr1 = max(1e-7, 1.0 - beta1 ** t)
        corr2 = max(1e-7, 1.0 - beta2 ** t)
        wd = self.config.weight_decay

        def _update(param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray, decay: bool = True) -> None:
            # Gradient clipping per parameter block
            gnorm = float(np.linalg.norm(grad))
            if gnorm > 5.0:
                grad = grad * (5.0 / gnorm)
            m[:] = beta1 * m + (1.0 - beta1) * grad
            v[:] = beta2 * v + (1.0 - beta2) * (grad * grad)
            m_hat = m / corr1
            v_hat = v / corr2
            step = m_hat / (np.sqrt(v_hat) + eps)
            if decay:
                param -= lr * (step + wd * param)
            else:
                param -= lr * step

        _update(self.embed, g_embed, self._m_embed, self._v_embed)
        _update(self.decoder, g_dec, self._m_dec, self._v_dec)
        _update(self.stop_w, g_stop_w, self._m_stop_w, self._v_stop_w)
        _update(self.stop_b, g_stop_b, self._m_stop_b, self._v_stop_b, decay=False)

        for i in range(self.config.n_layers):
            _update(self.decay_w[i], g_decay_w[i], self._m_decay[i], self._v_decay[i], decay=False)
            _update(self.weights[i], g_weights[i], self._m_weights[i], self._v_weights[i])
            _update(self.gamma[i], g_gamma[i], self._m_gamma[i], self._v_gamma[i], decay=False)
            _update(self.beta[i], g_beta[i], self._m_beta[i], self._v_beta[i], decay=False)

    def sample_byte(self, logits: np.ndarray, temperature: Optional[float] = None, top_p: Optional[float] = None) -> int:
        """Sample next byte from logits using temperature and top-p (nucleus) filtering."""
        temp = temperature if temperature is not None else self.config.temperature
        p_val = top_p if top_p is not None else self.config.top_p

        if temp <= 0.0:
            return int(np.argmax(logits))

        scaled = logits / max(1e-4, temp)
        exp_l = np.exp(scaled - np.max(scaled))
        probs = exp_l / np.sum(exp_l)

        # Top-P Nucleus Filtering
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)

        cutoff_idx = int(np.searchsorted(cumsum, p_val)) + 1
        keep_indices = sorted_indices[:max(1, cutoff_idx)]
        keep_probs = probs[keep_indices]
        keep_probs /= np.sum(keep_probs)

        return int(np.random.choice(keep_indices, p=keep_probs))

    def train_sequence(
        self,
        byte_data: bytes,
        lr: Optional[float] = None,
        reset_state: bool = False,
    ) -> Dict[str, float]:
        """Train sequentially on a byte buffer.

        Args:
            byte_data: UTF-8 raw bytes sequence.
            lr: Optional learning rate override.
            reset_state: If True, resets recurrent states before the sequence.

        Returns:
            Dictionary containing average losses and total bytes trained.
        """
        if len(byte_data) < 2:
            return {"total_loss": 0.0, "bytes": 0}

        if reset_state:
            self.reset_state()

        n_bytes = len(byte_data) - 1
        total_loss = 0.0
        ce_loss = 0.0
        lat_loss = 0.0
        var_loss = 0.0

        for i in range(n_bytes):
            cur_b = byte_data[i]
            next_b = byte_data[i + 1]
            is_end = (i == n_bytes - 1)
            _, _, losses = self.step_online_train(cur_b, next_b, is_end=is_end, lr=lr)
            total_loss += losses["total"]
            ce_loss += losses["ce"]
            lat_loss += losses["latent"]
            var_loss += losses["variance"]

        return {
            "loss": total_loss / n_bytes,
            "ce_loss": ce_loss / n_bytes,
            "latent_loss": lat_loss / n_bytes,
            "variance_loss": var_loss / n_bytes,
            "bytes": n_bytes,
        }

    def train_text(
        self,
        text: str,
        epochs: int = 3,
        lr: Optional[float] = None,
        chunk_size: int = 128,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train model on standard text across multiple epochs."""
        raw_bytes = text.encode("utf-8")
        cur_lr = lr if lr is not None else self.config.lr
        history: Dict[str, List[float]] = {"loss": [], "ce_loss": []}

        if verbose:
            print(f"🚀 Training RTU Language Model ({len(raw_bytes):,} bytes, {epochs} epochs, lr={cur_lr})...")

        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            ep_ce = 0.0
            total_chunks = max(1, (len(raw_bytes) + chunk_size - 1) // chunk_size)
            t0 = time.perf_counter()

            for c_idx in range(0, len(raw_bytes) - 1, chunk_size):
                chunk = raw_bytes[c_idx : c_idx + chunk_size + 1]
                stats = self.train_sequence(chunk, lr=cur_lr, reset_state=(c_idx == 0))
                ep_loss += stats["loss"]
                ep_ce += stats["ce_loss"]

            avg_loss = ep_loss / total_chunks
            avg_ce = ep_ce / total_chunks
            history["loss"].append(avg_loss)
            history["ce_loss"].append(avg_ce)
            elapsed = time.perf_counter() - t0

            if verbose:
                tok_s = len(raw_bytes) / max(1e-4, elapsed)
                print(f"✓ Epoch {ep:2d}/{epochs} finished in {elapsed:.2f}s | Avg Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} ({tok_s:,.0f} byte/s)")

        return history

    def generate(
        self,
        prompt: str,
        max_bytes: int = 128,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_threshold: float = 0.8,
        reset_state: bool = True,
    ) -> str:
        """Autoregressively generate text from a string prompt.

        Args:
            prompt: Text prompt string.
            max_bytes: Upper limit on newly generated bytes.
            temperature: Sampling temperature (0 for greedy argmax).
            top_p: Nucleus sampling cutoff.
            stop_threshold: Stop token probability threshold to end output early.
            reset_state: Whether to reset recurrence before priming.

        Returns:
            Generated continuation string.
        """
        if reset_state:
            self.reset_state()

        p_bytes = prompt.encode("utf-8")
        if not p_bytes:
            p_bytes = b" "

        # 1. Prime the recurrent state with prompt bytes
        last_logits = None
        for b in p_bytes:
            last_logits, _, _ = self.forward_byte(b, update_state=True)

        # 2. Autoregressively generate next bytes
        generated_bytes = []
        cur_logits = last_logits

        for _ in range(max_bytes):
            next_b = self.sample_byte(cur_logits, temperature=temperature, top_p=top_p)
            cur_logits, stop_prob, _ = self.forward_byte(next_b, update_state=True)
            if stop_prob > stop_threshold:
                break
            generated_bytes.append(next_b)

        return bytes(generated_bytes).decode("utf-8", errors="replace")

    def save(self, filepath: Union[str, Path]) -> Path:
        """Save model parameters and configuration to a .npz file."""
        out_p = Path(filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, np.ndarray] = {
            "embed": self.embed,
            "decoder": self.decoder,
            "stop_w": self.stop_w,
            "stop_b": self.stop_b,
        }
        for i in range(self.config.n_layers):
            arrays[f"decay_w_{i}"] = self.decay_w[i]
            arrays[f"weights_{i}"] = self.weights[i]
            arrays[f"gamma_{i}"] = self.gamma[i]
            arrays[f"beta_{i}"] = self.beta[i]
            arrays[f"state_{i}"] = self.layer_states[i].state

        meta = json.dumps(asdict(self.config))
        np.savez_compressed(out_p, meta=meta, **arrays)
        return out_p

    def load(self, filepath: Union[str, Path]) -> None:
        """Load model parameters and state from a .npz file."""
        inp_p = Path(filepath)
        if not inp_p.exists():
            raise FileNotFoundError(f"Checkpoint file {inp_p} not found")

        data = np.load(inp_p, allow_pickle=True)
        self.embed[:] = data["embed"]
        self.decoder[:] = data["decoder"]
        self.stop_w[:] = data["stop_w"]
        self.stop_b[:] = data["stop_b"]

        for i in range(self.config.n_layers):
            self.decay_w[i][:] = data[f"decay_w_{i}"]
            self.weights[i][:] = data[f"weights_{i}"]
            self.gamma[i][:] = data[f"gamma_{i}"]
            self.beta[i][:] = data[f"beta_{i}"]
            if f"state_{i}" in data:
                self.layer_states[i].state[:] = data[f"state_{i}"]
