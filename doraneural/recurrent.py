"""Recurrent Neural Network layers for sequence modeling.

Provides pure NumPy implementations with analytical Backpropagation Through Time (BPTT):
- `RNN` / `SimpleRNN`: Elman recurrent network layer.
- `LSTM`: Long Short-Term Memory network layer with forget, input, output gates and cell state.
- `GRU`: Gated Recurrent Unit layer with reset and update gates.
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from .base import Layer


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


class SimpleRNN(Layer):
    """Elman Simple Recurrent Neural Network layer.

    Formula:
        h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)

    Args:
        in_features (int): Dimensionality of input vector at each time step.
        hidden_units (int): Dimensionality of hidden state vector.
        return_sequences (bool): If True, returns tensor of shape (batch, time, hidden_units).
            If False, returns final state of shape (batch, hidden_units).
    """

    def __init__(
        self,
        in_features: int,
        hidden_units: int,
        return_sequences: bool = False,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.in_features: int = int(in_features)
        self.hidden_units: int = int(hidden_units)
        self.return_sequences: bool = bool(return_sequences)
        self.dtype = dtype
        self.trainable: bool = True

        # Weights initialization (Xavier)
        std_x = np.sqrt(2.0 / (self.in_features + self.hidden_units))
        std_h = np.sqrt(2.0 / (self.hidden_units + self.hidden_units))
        self.W_xh: np.ndarray = (np.random.randn(self.in_features, self.hidden_units) * std_x).astype(self.dtype)
        self.W_hh: np.ndarray = (np.random.randn(self.hidden_units, self.hidden_units) * std_h).astype(self.dtype)
        self.b_h: np.ndarray = np.zeros((1, self.hidden_units), dtype=self.dtype)

        # Gradients
        self.dW_xh: np.ndarray = np.zeros_like(self.W_xh)
        self.dW_hh: np.ndarray = np.zeros_like(self.W_hh)
        self.db_h: np.ndarray = np.zeros_like(self.b_h)

        self._params["W_xh"] = self.W_xh
        self._params["W_hh"] = self.W_hh
        self._params["b_h"] = self.b_h

        self._grads["W_xh"] = self.dW_xh
        self._grads["W_hh"] = self.dW_hh
        self._grads["b_h"] = self.db_h

        self._cache_x: Optional[np.ndarray] = None
        self._cache_h: Optional[List[np.ndarray]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.ndim != 3:
            raise ValueError(f"SimpleRNN expects 3D input (batch, time, features), got shape {x_arr.shape}")
        if x_arr.shape[2] != self.in_features:
            raise ValueError(f"SimpleRNN expected {self.in_features} features, got {x_arr.shape[2]}")

        N, T, _ = x_arr.shape
        self._cache_x = x_arr
        self._cache_h = []

        h = np.zeros((N, self.hidden_units), dtype=self.dtype)
        h_all = np.zeros((N, T, self.hidden_units), dtype=self.dtype)

        for t in range(T):
            xt = x_arr[:, t]
            z = xt @ self.W_xh + h @ self.W_hh + self.b_h
            h = np.tanh(z)
            self._cache_h.append(h)
            h_all[:, t] = h

        return h_all if self.return_sequences else h

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache_x is None or self._cache_h is None:
            raise RuntimeError("SimpleRNN.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype)
        N, T, Din = self._cache_x.shape

        self.dW_xh.fill(0.0)
        self.dW_hh.fill(0.0)
        self.db_h.fill(0.0)
        dx = np.zeros_like(self._cache_x)

        dh_next = np.zeros((N, self.hidden_units), dtype=self.dtype)

        for t in reversed(range(T)):
            h_t = self._cache_h[t]
            h_prev = self._cache_h[t - 1] if t > 0 else np.zeros((N, self.hidden_units), dtype=self.dtype)

            dh_curr = grad_out[:, t] if self.return_sequences else (grad_out if t == T - 1 else 0.0)
            dh = dh_curr + dh_next

            # tanh derivative: dL/dz = dh * (1 - h^2)
            dz = dh * (1.0 - h_t * h_t)

            self.dW_xh += self._cache_x[:, t].T @ dz
            self.dW_hh += h_prev.T @ dz
            self.db_h += np.sum(dz, axis=0, keepdims=True)

            dx[:, t] = dz @ self.W_xh.T
            dh_next = dz @ self.W_hh.T

        return dx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "SimpleRNN",
            "in_features": self.in_features,
            "hidden_units": self.hidden_units,
            "return_sequences": self.return_sequences,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SimpleRNN":
        return cls(
            in_features=config["in_features"],
            hidden_units=config["hidden_units"],
            return_sequences=config.get("return_sequences", False),
        )


# Alias
RNN = SimpleRNN


class LSTM(Layer):
    """Long Short-Term Memory (LSTM) layer.

    Includes forget, input, output gates and cell memory state with exact BPTT gradients.

    Args:
        in_features (int): Input feature dimensions.
        hidden_units (int): Hidden state dimensions.
        return_sequences (bool): Return full sequence or final hidden state.
    """

    def __init__(
        self,
        in_features: int,
        hidden_units: int,
        return_sequences: bool = False,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.in_features: int = int(in_features)
        self.hidden_units: int = int(hidden_units)
        self.return_sequences: bool = bool(return_sequences)
        self.dtype = dtype
        self.trainable: bool = True

        # Packed 4-gate weight matrices [input, forget, candidate, output]
        std_x = np.sqrt(2.0 / (self.in_features + 4 * self.hidden_units))
        std_h = np.sqrt(2.0 / (self.hidden_units + 4 * self.hidden_units))

        self.W_x: np.ndarray = (np.random.randn(self.in_features, 4 * self.hidden_units) * std_x).astype(self.dtype)
        self.W_h: np.ndarray = (np.random.randn(self.hidden_units, 4 * self.hidden_units) * std_h).astype(self.dtype)
        self.b: np.ndarray = np.zeros((1, 4 * self.hidden_units), dtype=self.dtype)
        # Standard practice: initialize forget gate bias to 1.0 to help gradient flow early in training
        self.b[0, self.hidden_units : 2 * self.hidden_units] = 1.0

        self.dW_x: np.ndarray = np.zeros_like(self.W_x)
        self.dW_h: np.ndarray = np.zeros_like(self.W_h)
        self.db: np.ndarray = np.zeros_like(self.b)

        self._params["W_x"] = self.W_x
        self._params["W_h"] = self.W_h
        self._params["b"] = self.b

        self._grads["W_x"] = self.dW_x
        self._grads["W_h"] = self.dW_h
        self._grads["b"] = self.db

        self._cache: Optional[List[Any]] = None
        self._cache_x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.ndim != 3:
            raise ValueError(f"LSTM expects 3D input (batch, time, features), got shape {x_arr.shape}")
        if x_arr.shape[2] != self.in_features:
            raise ValueError(f"LSTM expected {self.in_features} features, got {x_arr.shape[2]}")

        N, T, _ = x_arr.shape
        self._cache_x = x_arr
        self._cache = []

        h = np.zeros((N, self.hidden_units), dtype=self.dtype)
        c = np.zeros((N, self.hidden_units), dtype=self.dtype)
        h_all = np.zeros((N, T, self.hidden_units), dtype=self.dtype)

        U = self.hidden_units
        for t in range(T):
            xt = x_arr[:, t]
            gates = xt @ self.W_x + h @ self.W_h + self.b
            i_gate = _sigmoid(gates[:, :U])
            f_gate = _sigmoid(gates[:, U : 2 * U])
            g_gate = np.tanh(gates[:, 2 * U : 3 * U])
            o_gate = _sigmoid(gates[:, 3 * U :])

            c_prev = c.copy()
            c = f_gate * c_prev + i_gate * g_gate
            tanh_c = np.tanh(c)
            h = o_gate * tanh_c

            self._cache.append((h, c, c_prev, i_gate, f_gate, g_gate, o_gate, tanh_c))
            h_all[:, t] = h

        return h_all if self.return_sequences else h

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache_x is None or self._cache is None:
            raise RuntimeError("LSTM.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype)
        N, T, Din = self._cache_x.shape
        U = self.hidden_units

        self.dW_x.fill(0.0)
        self.dW_h.fill(0.0)
        self.db.fill(0.0)
        dx = np.zeros_like(self._cache_x)

        dh_next = np.zeros((N, U), dtype=self.dtype)
        dc_next = np.zeros((N, U), dtype=self.dtype)

        for t in reversed(range(T)):
            h_t, c_t, c_prev, i_g, f_g, g_g, o_g, tanh_c = self._cache[t]
            h_prev = self._cache[t - 1][0] if t > 0 else np.zeros((N, U), dtype=self.dtype)

            dh_curr = grad_out[:, t] if self.return_sequences else (grad_out if t == T - 1 else 0.0)
            dh = dh_curr + dh_next

            do = dh * tanh_c
            dc = dc_next + dh * o_g * (1.0 - tanh_c * tanh_c)
            df = dc * c_prev
            di = dc * g_g
            dg = dc * i_g

            # Gate pre-activations
            dzi = di * i_g * (1.0 - i_g)
            dzf = df * f_g * (1.0 - f_g)
            dzg = dg * (1.0 - g_g * g_g)
            dzo = do * o_g * (1.0 - o_g)
            dz = np.hstack([dzi, dzf, dzg, dzo])

            self.dW_x += self._cache_x[:, t].T @ dz
            self.dW_h += h_prev.T @ dz
            self.db += np.sum(dz, axis=0, keepdims=True)

            dx[:, t] = dz @ self.W_x.T
            dh_next = dz @ self.W_h.T
            dc_next = dc * f_g

        return dx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "LSTM",
            "in_features": self.in_features,
            "hidden_units": self.hidden_units,
            "return_sequences": self.return_sequences,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LSTM":
        return cls(
            in_features=config["in_features"],
            hidden_units=config["hidden_units"],
            return_sequences=config.get("return_sequences", False),
        )


class GRU(Layer):
    """Gated Recurrent Unit (GRU) layer.

    Combines forget and input gates into an update gate, reducing parameter count compared to LSTM.

    Args:
        in_features (int): Input feature dimensions.
        hidden_units (int): Hidden state dimensions.
        return_sequences (bool): Return full sequence or final hidden state.
    """

    def __init__(
        self,
        in_features: int,
        hidden_units: int,
        return_sequences: bool = False,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.in_features: int = int(in_features)
        self.hidden_units: int = int(hidden_units)
        self.return_sequences: bool = bool(return_sequences)
        self.dtype = dtype
        self.trainable: bool = True

        U = self.hidden_units
        std_x = np.sqrt(2.0 / (self.in_features + U))
        std_h = np.sqrt(2.0 / (U + U))

        # Reset gate
        self.W_xr = (np.random.randn(self.in_features, U) * std_x).astype(self.dtype)
        self.W_hr = (np.random.randn(U, U) * std_h).astype(self.dtype)
        self.b_r = np.zeros((1, U), dtype=self.dtype)

        # Update gate
        self.W_xz = (np.random.randn(self.in_features, U) * std_x).astype(self.dtype)
        self.W_hz = (np.random.randn(U, U) * std_h).astype(self.dtype)
        self.b_z = np.zeros((1, U), dtype=self.dtype)

        # Candidate state
        self.W_xn = (np.random.randn(self.in_features, U) * std_x).astype(self.dtype)
        self.W_hn = (np.random.randn(U, U) * std_h).astype(self.dtype)
        self.b_n = np.zeros((1, U), dtype=self.dtype)

        # Gradients
        self.dW_xr = np.zeros_like(self.W_xr)
        self.dW_hr = np.zeros_like(self.W_hr)
        self.db_r = np.zeros_like(self.b_r)

        self.dW_xz = np.zeros_like(self.W_xz)
        self.dW_hz = np.zeros_like(self.W_hz)
        self.db_z = np.zeros_like(self.b_z)

        self.dW_xn = np.zeros_like(self.W_xn)
        self.dW_hn = np.zeros_like(self.W_hn)
        self.db_n = np.zeros_like(self.b_n)

        for name in ["W_xr", "W_hr", "b_r", "W_xz", "W_hz", "b_z", "W_xn", "W_hn", "b_n"]:
            self._params[name] = getattr(self, name)
            self._grads[name] = getattr(self, f"d{name}")

        self._cache: Optional[List[Any]] = None
        self._cache_x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=self.dtype)
        if x_arr.ndim != 3:
            raise ValueError(f"GRU expects 3D input (batch, time, features), got shape {x_arr.shape}")
        if x_arr.shape[2] != self.in_features:
            raise ValueError(f"GRU expected {self.in_features} features, got {x_arr.shape[2]}")

        N, T, _ = x_arr.shape
        self._cache_x = x_arr
        self._cache = []

        h = np.zeros((N, self.hidden_units), dtype=self.dtype)
        h_all = np.zeros((N, T, self.hidden_units), dtype=self.dtype)

        for t in range(T):
            xt = x_arr[:, t]
            h_prev = h.copy()

            r = _sigmoid(xt @ self.W_xr + h_prev @ self.W_hr + self.b_r)
            z = _sigmoid(xt @ self.W_xz + h_prev @ self.W_hz + self.b_z)
            n = np.tanh(xt @ self.W_xn + (r * h_prev) @ self.W_hn + self.b_n)
            h = (1.0 - z) * n + z * h_prev

            self._cache.append((h, h_prev, r, z, n, xt))
            h_all[:, t] = h

        return h_all if self.return_sequences else h

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._cache_x is None or self._cache is None:
            raise RuntimeError("GRU.backward called before forward pass.")

        grad_out = np.asarray(grad_output, dtype=self.dtype)
        N, T, Din = self._cache_x.shape
        U = self.hidden_units

        for g in self._grads.values():
            g.fill(0.0)

        dx = np.zeros_like(self._cache_x)
        dh_next = np.zeros((N, U), dtype=self.dtype)

        for t in reversed(range(T)):
            h_t, h_prev, r, z, n, xt = self._cache[t]
            dh_curr = grad_out[:, t] if self.return_sequences else (grad_out if t == T - 1 else 0.0)
            dh = dh_curr + dh_next

            dn = dh * (1.0 - z)
            dz_gate = dh * (h_prev - n)

            dn_raw = dn * (1.0 - n * n)
            dz_raw = dz_gate * z * (1.0 - z)

            self.dW_xn += xt.T @ dn_raw
            self.db_n += np.sum(dn_raw, axis=0, keepdims=True)

            dr_hprev = dn_raw @ self.W_hn.T
            dr = dr_hprev * h_prev
            dr_raw = dr * r * (1.0 - r)

            self.dW_xr += xt.T @ dr_raw
            self.db_r += np.sum(dr_raw, axis=0, keepdims=True)

            self.dW_xz += xt.T @ dz_raw
            self.db_z += np.sum(dz_raw, axis=0, keepdims=True)

            self.dW_hn += (r * h_prev).T @ dn_raw
            self.dW_hr += h_prev.T @ dr_raw
            self.dW_hz += h_prev.T @ dz_raw

            dx[:, t] = dn_raw @ self.W_xn.T + dr_raw @ self.W_xr.T + dz_raw @ self.W_xz.T
            dh_next = (
                dh * z
                + dr_hprev * r
                + dr_raw @ self.W_hr.T
                + dz_raw @ self.W_hz.T
            )

        return dx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "GRU",
            "in_features": self.in_features,
            "hidden_units": self.hidden_units,
            "return_sequences": self.return_sequences,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "GRU":
        return cls(
            in_features=config["in_features"],
            hidden_units=config["hidden_units"],
            return_sequences=config.get("return_sequences", False),
        )
