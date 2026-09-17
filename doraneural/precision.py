"""Precision and mixed-precision controls for memory efficiency and compute throughput.

Provides:
- Global precision toggles (`set_precision`, `get_precision`, `precision_scope`).
- Model & Layer precision casting (`to_precision` / `.to_precision("float32" | "float64")`).
- Memory profiler (`memory_summary`) reporting parameter/gradient RAM usage.
"""

from contextlib import contextmanager
from typing import Union, Dict, Any, Optional
import numpy as np

# Global default dtype (float32 is standard for speed and 50% memory savings)
_CURRENT_PRECISION = np.float32

PRECISION_MAP = {
    "float32": np.float32,
    "f32": np.float32,
    "single": np.float32,
    "float": np.float32,
    "float64": np.float64,
    "f64": np.float64,
    "double": np.float64,
    np.float32: np.float32,
    np.float64: np.float64,
    float: np.float64,
}


def canonicalize_dtype(dtype: Union[str, np.dtype, type]) -> np.dtype:
    """Normalize user dtype string or type to np.float32 or np.float64."""
    if isinstance(dtype, str):
        key = dtype.lower().strip()
    else:
        key = dtype

    if key in PRECISION_MAP:
        return PRECISION_MAP[key]

    try:
        dt = np.dtype(dtype)
        if dt in (np.float32, np.float64):
            return dt.type
    except Exception:
        pass

    raise ValueError(
        f"Unsupported precision '{dtype}'. Supported options: 'float32' (single), 'float64' (double)."
    )


def get_precision() -> np.dtype:
    """Get the current global default floating-point precision."""
    return _CURRENT_PRECISION


def set_precision(dtype: Union[str, np.dtype, type]) -> np.dtype:
    """Set the global default floating-point precision.

    Args:
        dtype: 'float32', 'float64', np.float32, or np.float64.

    Returns:
        np.dtype: The newly configured active precision.
    """
    global _CURRENT_PRECISION
    _CURRENT_PRECISION = canonicalize_dtype(dtype)
    return _CURRENT_PRECISION


@contextmanager
def precision_scope(dtype: Union[str, np.dtype, type]):
    """Context manager temporarily setting active numerical precision.

    Example:
        >>> with precision_scope("float64"):
        ...     # High precision gradient verification
        ...     model.fit(X, y)
    """
    global _CURRENT_PRECISION
    prev = _CURRENT_PRECISION
    try:
        _CURRENT_PRECISION = canonicalize_dtype(dtype)
        yield _CURRENT_PRECISION
    finally:
        _CURRENT_PRECISION = prev


def to_precision(model_or_layer: Any, dtype: Union[str, np.dtype, type]) -> Any:
    """Cast a Model, Layer, or Optimizer state in-place to the specified precision.

    Casts all learnable weights, biases, cached gradients, and optimizer moments.

    Args:
        model_or_layer: Sequential model or Layer object.
        dtype: Target precision ('float32' or 'float64').

    Returns:
        The casted object.
    """
    target_dt = canonicalize_dtype(dtype)

    # If it's a Sequential model
    if hasattr(model_or_layer, "layers"):
        model_or_layer.dtype = target_dt
        for layer in model_or_layer.layers:
            to_precision(layer, target_dt)

        # Cast optimizer state if model is compiled
        opt = getattr(model_or_layer, "optimizer", None)
        if opt is not None:
            # Adam state (m, v)
            if hasattr(opt, "m"):
                for k, v in list(opt.m.items()):
                    if v is not None:
                        opt.m[k] = v.astype(target_dt)
            if hasattr(opt, "v"):
                for k, v in list(opt.v.items()):
                    if v is not None:
                        opt.v[k] = v.astype(target_dt)
            # SGD / RMSprop velocity
            if hasattr(opt, "velocity"):
                for k, v in list(opt.velocity.items()):
                    if v is not None:
                        opt.velocity[k] = v.astype(target_dt)
            if hasattr(opt, "v_mean"):
                for k, v in list(opt.v_mean.items()):
                    if v is not None:
                        opt.v_mean[k] = v.astype(target_dt)

        return model_or_layer

    # If it's an individual Layer
    if hasattr(model_or_layer, "get_params"):
        model_or_layer.dtype = target_dt
        # Recast parameters in-place
        params = model_or_layer.get_params()
        for p_name, p_val in list(params.items()):
            if p_val is not None:
                p_cast = p_val.astype(target_dt)
                params[p_name] = p_cast
                setattr(model_or_layer, p_name, p_cast)

        # Recast gradients in-place
        grads = model_or_layer.get_grads()
        for g_name, g_val in list(grads.items()):
            if g_val is not None:
                g_cast = g_val.astype(target_dt)
                grads[g_name] = g_cast
                if hasattr(model_or_layer, f"d{g_name}"):
                    setattr(model_or_layer, f"d{g_name}", g_cast)
                elif hasattr(model_or_layer, g_name):
                    setattr(model_or_layer, g_name, g_cast)

        # If sublayers exist (e.g. in TransformerBlock)
        if hasattr(model_or_layer, "_sublayers"):
            for sub in model_or_layer._sublayers:
                to_precision(sub, target_dt)

    return model_or_layer


def memory_summary(model_or_layer: Any) -> Dict[str, Any]:
    """Calculate parameter count, byte consumption, and RAM metrics.

    Args:
        model_or_layer: Sequential model or Layer instance.

    Returns:
        Dict[str, Any]: Detailed breakdown with keys:
            - 'num_parameters': int
            - 'param_bytes': int
            - 'grad_bytes': int
            - 'total_bytes': int
            - 'dtype': str
            - 'formatted': str
    """
    total_params = 0
    param_bytes = 0
    grad_bytes = 0
    detected_dtype = None

    layers = getattr(model_or_layer, "layers", [model_or_layer])
    for layer in layers:
        if hasattr(layer, "get_params"):
            for p in layer.get_params().values():
                if p is not None:
                    total_params += p.size
                    param_bytes += p.nbytes
                    if detected_dtype is None:
                        detected_dtype = p.dtype
        if hasattr(layer, "get_grads"):
            for g in layer.get_grads().values():
                if g is not None:
                    grad_bytes += g.nbytes

    total_bytes = param_bytes + grad_bytes
    dt_name = str(detected_dtype) if detected_dtype is not None else "unknown"

    if total_bytes < 1024:
        size_str = f"{total_bytes} B"
    elif total_bytes < 1024 * 1024:
        size_str = f"{total_bytes / 1024:.2f} KB"
    else:
        size_str = f"{total_bytes / (1024 * 1024):.2f} MB"

    savings = "50% smaller than float64" if "32" in dt_name else "2x larger than float32"
    formatted = f"{total_params:,} parameters | {size_str} ({dt_name}) [{savings}]"

    return {
        "num_parameters": total_params,
        "param_bytes": param_bytes,
        "grad_bytes": grad_bytes,
        "total_bytes": total_bytes,
        "dtype": dt_name,
        "formatted": formatted,
    }
