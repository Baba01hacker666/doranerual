"""Lightweight model serialization and deserialization.

Saves network architecture as human-readable JSON and parameter arrays in NumPy
compressed .npz archives, enabling portable persistence without external dependencies.
"""

import json
import os
from pathlib import Path
from typing import Dict, Type, Any, Union
import numpy as np

from .base import Layer
from .layers import Dense, Dropout, LayerNorm, Flatten, Conv2D, MaxPool2D, DendriticDense, ChebyshevKAN, BifurcatedDense, ReflectiveDense, InvertedDense, Tensor4DDense, ComplexWaveDense, FractalChaosDense, TunnelingDense
from .activations import ReLU, Sigmoid, Softmax, Tanh, SiLU, Inverter
from .recurrent import SimpleRNN, RNN, LSTM, GRU
from .attention import PositionalEncoding, MultiHeadAttention, TransformerBlock, KANTransformerBlock, DendriticTransformerBlock

# Registry of serializable layer types
LAYER_REGISTRY: Dict[str, Type[Layer]] = {
    "Dense": Dense,
    "DendriticDense": DendriticDense,
    "ChebyshevKAN": ChebyshevKAN,
    "BifurcatedDense": BifurcatedDense,
    "ReflectiveDense": ReflectiveDense,
    "InvertedDense": InvertedDense,
    "Tensor4DDense": Tensor4DDense,
    "ComplexWaveDense": ComplexWaveDense,
    "FractalChaosDense": FractalChaosDense,
    "TunnelingDense": TunnelingDense,
    "Dropout": Dropout,
    "LayerNorm": LayerNorm,
    "Flatten": Flatten,
    "Conv2D": Conv2D,
    "MaxPool2D": MaxPool2D,
    "ReLU": ReLU,
    "Sigmoid": Sigmoid,
    "Softmax": Softmax,
    "Tanh": Tanh,
    "SiLU": SiLU,
    "Inverter": Inverter,
    "SimpleRNN": SimpleRNN,
    "RNN": RNN,
    "LSTM": LSTM,
    "GRU": GRU,
    "PositionalEncoding": PositionalEncoding,
    "MultiHeadAttention": MultiHeadAttention,
    "TransformerBlock": TransformerBlock,
    "KANTransformerBlock": KANTransformerBlock,
    "DendriticTransformerBlock": DendriticTransformerBlock,
}


def register_layer(name: str, layer_cls: Type[Layer]) -> None:
    """Register a custom layer class for serialization."""
    LAYER_REGISTRY[name] = layer_cls


def _resolve_paths(filepath: Union[str, Path]) -> tuple[Path, Path]:
    """Resolve paths for JSON architecture and NPZ weights files."""
    path_str = str(filepath)
    if path_str.endswith(".json"):
        base = path_str[:-5]
    elif path_str.endswith(".npz"):
        base = path_str[:-4]
    else:
        base = path_str

    json_path = Path(f"{base}.json")
    npz_path = Path(f"{base}.npz")
    return json_path, npz_path


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """Save model architecture and parameter weights to disk.

    Creates two files:
      1. `<filepath>.json` containing layer specifications and hyperparameters.
      2. `<filepath>.npz` containing NumPy parameter arrays for all trainable layers.

    Args:
        model (Sequential): The model to serialize.
        filepath (Union[str, Path]): Base file path or path to .json/.npz file.
    """
    json_path, npz_path = _resolve_paths(filepath)

    # Ensure parent directory exists
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Serialize architecture to JSON
    arch_config = {
        "model_type": "Sequential",
        "version": "1.0",
        "layers": [layer.to_dict() for layer in model.layers],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(arch_config, f, indent=2)

    # 2. Serialize weights to NPZ
    weights_dict: Dict[str, np.ndarray] = {}
    for idx, layer in enumerate(model.layers):
        if layer.trainable:
            params = layer.get_params()
            for param_name, param_array in params.items():
                if param_array is not None:
                    key = f"layer_{idx}_{param_name}"
                    weights_dict[key] = param_array

    np.savez(npz_path, **weights_dict)


def load_model(filepath: Union[str, Path]) -> Any:
    """Load model architecture and weights from disk.

    Args:
        filepath (Union[str, Path]): Base file path or path to .json/.npz file.

    Returns:
        Sequential: Reconstructed model with restored parameter weights.

    Raises:
        FileNotFoundError: If architecture or weight file does not exist.
        ValueError: If layer specification is unsupported.
    """
    # Import locally to prevent circular dependency
    from .model import Sequential

    json_path, npz_path = _resolve_paths(filepath)

    if not json_path.exists():
        raise FileNotFoundError(f"Model architecture file not found: {json_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Model weights file not found: {npz_path}")

    # 1. Load architecture
    with open(json_path, "r", encoding="utf-8") as f:
        arch_config = json.load(f)

    layers = []
    for layer_cfg in arch_config.get("layers", []):
        layer_type = layer_cfg.get("type")
        if layer_type not in LAYER_REGISTRY:
            raise ValueError(
                f"Unknown layer type '{layer_type}'. Registered types: {list(LAYER_REGISTRY.keys())}"
            )
        layer_cls = LAYER_REGISTRY[layer_type]
        layers.append(layer_cls.from_dict(layer_cfg))

    model = Sequential(layers)

    # 2. Restore weights
    with np.load(npz_path) as npz_data:
        for idx, layer in enumerate(model.layers):
            if layer.trainable:
                restored_params = {}
                for param_name in layer.get_params().keys():
                    key = f"layer_{idx}_{param_name}"
                    if key in npz_data:
                        restored_params[param_name] = npz_data[key]
                if restored_params:
                    layer.set_params(restored_params)

    return model
