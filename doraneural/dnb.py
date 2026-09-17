"""Doraneural Binary (.dnb) Format Specification (v1.0).

A portable, architecture-independent, versioned binary file format for neural models.
Does not use Python pickle or external dependencies.

Binary Layout:
+-------------------------------------------------------------------+
| 1. Header (32 bytes)                                              |
|    - Magic: b'DNB\\x01' (4 bytes)                                  |
|    - Version: uint16 (2 bytes)                                    |
|    - Flags: uint16 (2 bytes)                                      |
|    - Metadata Offset: uint64 (8 bytes)                            |
|    - Metadata Length: uint64 (8 bytes)                            |
|    - Tensor Count: uint64 (8 bytes)                               |
+-------------------------------------------------------------------+
| 2. Metadata Block (UTF-8 JSON)                                    |
|    - Model type, layer configurations, input/output schemas       |
+-------------------------------------------------------------------+
| 3. Tensors Block (8-byte aligned raw arrays with CRC32 checksums) |
|    - Name, dtype code, ndim, shape, byte length, payload, CRC32   |
+-------------------------------------------------------------------+
"""

import json
import struct
import zlib
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union, Optional
import numpy as np

from .base import Layer
from .model import Sequential
from .serialization import LAYER_REGISTRY


MAGIC_HEADER = b"DNB\x01"
CURRENT_FORMAT_VERSION = 1

# Dtype enumeration mapping
DTYPE_CODE_TO_NP = {
    1: np.dtype("float32"),
    2: np.dtype("float64"),
    3: np.dtype("int32"),
    4: np.dtype("int64"),
    5: np.dtype("uint8"),
    6: np.dtype("int8"),
}
NP_TO_DTYPE_CODE = {v: k for k, v in DTYPE_CODE_TO_NP.items()}


def _align_to(n: int, alignment: int = 8) -> int:
    """Calculate padding required to align offset to boundary."""
    remainder = n % alignment
    return 0 if remainder == 0 else (alignment - remainder)


def save_dnb(model: Any, filepath: Union[str, Path]) -> Path:
    """Save a model to the portable Doraneural Binary (.dnb) format.

    Args:
        model: Sequential model or Module instance.
        filepath: Target destination path (.dnb).

    Returns:
        Path: Path to written file.
    """
    path = Path(filepath)
    if not path.name.endswith(".dnb"):
        path = path.with_suffix(".dnb")
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Extract architecture metadata
    if hasattr(model, "layers"):
        layer_configs = [layer.to_dict() for layer in model.layers]
        model_type = "Sequential"
    else:
        layer_configs = []
        model_type = model.__class__.__name__

    # 2. Extract trainable parameter tensors
    tensor_records: List[Tuple[str, np.ndarray]] = []
    if hasattr(model, "layers"):
        for layer_idx, layer in enumerate(model.layers):
            if hasattr(layer, "get_params"):
                for p_name, p_val in layer.get_params().items():
                    if p_val is not None:
                        key = f"layer_{layer_idx}.{p_name}"
                        tensor_records.append((key, np.ascontiguousarray(p_val)))
    elif hasattr(model, "parameters"):
        for idx, param in enumerate(model.parameters()):
            tensor_records.append((f"param_{idx}", np.ascontiguousarray(param.data)))

    # 3. Prepare Metadata JSON
    metadata = {
        "format": "DNB",
        "version": CURRENT_FORMAT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_type": model_type,
        "layers": layer_configs,
        "tensor_count": len(tensor_records),
    }
    meta_bytes = json.dumps(metadata, indent=None).encode("utf-8")
    meta_length = len(meta_bytes)

    header_size = 32
    meta_offset = header_size

    # Calculate padding after metadata for 8-byte tensor alignment
    post_meta_pad = _align_to(header_size + meta_length, 8)
    first_tensor_offset = header_size + meta_length + post_meta_pad

    with open(path, "wb") as f:
        # Write 32-byte header placeholder
        flags = 0  # Bit flags
        header = struct.pack(
            "<4sHHQQQ",
            MAGIC_HEADER,
            CURRENT_FORMAT_VERSION,
            flags,
            meta_offset,
            meta_length,
            len(tensor_records),
        )
        f.write(header)

        # Write Metadata block
        f.write(meta_bytes)
        if post_meta_pad > 0:
            f.write(b"\x00" * post_meta_pad)

        # Write Tensors
        for name, arr in tensor_records:
            name_bytes = name.encode("ascii")
            name_len = len(name_bytes)
            dtype_code = NP_TO_DTYPE_CODE.get(arr.dtype, 1)
            ndim = arr.ndim
            shape = arr.shape
            raw_bytes = arr.tobytes()
            data_len = len(raw_bytes)
            crc = zlib.crc32(raw_bytes)

            # Record header:
            # name_len (uint16), dtype_code (uint8), ndim (uint8)
            f.write(struct.pack("<HBB", name_len, dtype_code, ndim))
            f.write(name_bytes)

            # Shape: ndim * uint64
            for dim in shape:
                f.write(struct.pack("<Q", dim))

            # Data length (uint64), CRC32 (uint32)
            f.write(struct.pack("<QI", data_len, crc))

            # Payload
            f.write(raw_bytes)

            # Align to 8 bytes for next tensor
            cur_pos = f.tell()
            pad = _align_to(cur_pos, 8)
            if pad > 0:
                f.write(b"\x00" * pad)

    return path


def inspect_dnb(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Inspect header, metadata, and tensor records in a .dnb file without loading arrays into RAM.

    Args:
        filepath: Path to .dnb binary file.

    Returns:
        Dict[str, Any]: Detailed inspection dictionary.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"DNB file not found: {path}")

    with open(path, "rb") as f:
        header_bytes = f.read(32)
        if len(header_bytes) < 32:
            raise ValueError("Corrupt DNB file: Header size is smaller than 32 bytes.")

        magic, version, flags, meta_offset, meta_len, tensor_count = struct.unpack("<4sHHQQQ", header_bytes)

        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}. Expected: {MAGIC_HEADER}")

        # Read metadata
        f.seek(meta_offset)
        meta_bytes = f.read(meta_len)
        metadata = json.loads(meta_bytes.decode("utf-8"))

        # Skip to first tensor aligned boundary
        post_meta_pad = _align_to(meta_offset + meta_len, 8)
        f.seek(meta_offset + meta_len + post_meta_pad)

        tensor_infos = []
        for _ in range(tensor_count):
            prefix = f.read(4)
            name_len, dtype_code, ndim = struct.unpack("<HBB", prefix)
            name = f.read(name_len).decode("ascii")

            shape = []
            for _ in range(ndim):
                dim = struct.unpack("<Q", f.read(8))[0]
                shape.append(dim)

            data_len, crc = struct.unpack("<QI", f.read(12))
            dt = DTYPE_CODE_TO_NP.get(dtype_code, np.float32)
            payload_offset = f.tell()

            tensor_infos.append({
                "name": name,
                "shape": tuple(shape),
                "dtype": str(dt),
                "bytes": data_len,
                "offset": payload_offset,
                "crc32": f"0x{crc:08X}",
            })

            # Skip payload + padding
            f.seek(data_len, 1)
            cur_pos = f.tell()
            pad = _align_to(cur_pos, 8)
            if pad > 0:
                f.seek(pad, 1)

    total_params = sum(int(np.prod(t["shape"])) for t in tensor_infos)
    return {
        "file": str(path),
        "magic": magic.decode("latin1", errors="replace"),
        "version": version,
        "format_version": version,
        "model_type": metadata.get("model_type", "Sequential"),
        "layers_count": len(metadata.get("layers", [])),
        "total_parameters": total_params,
        "tensors_count": tensor_count,
        "metadata": metadata,
        "tensors": tensor_infos,
    }


def load_dnb(filepath: Union[str, Path]) -> Sequential:
    """Load a model from a portable Doraneural Binary (.dnb) file with checksum verification.

    Args:
        filepath: Path to .dnb binary file.

    Returns:
        Sequential: Reconstructed model with restored parameters.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"DNB file not found: {path}")

    with open(path, "rb") as f:
        header_bytes = f.read(32)
        if len(header_bytes) < 32:
            raise ValueError("Corrupt DNB file: Header size is smaller than 32 bytes.")

        magic, version, flags, meta_offset, meta_len, tensor_count = struct.unpack("<4sHHQQQ", header_bytes)

        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic}. Expected: {MAGIC_HEADER}")

        if version > CURRENT_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported DNB format version {version}. Highest supported version is {CURRENT_FORMAT_VERSION}."
            )

        # Read and parse metadata
        f.seek(meta_offset)
        meta_bytes = f.read(meta_len)
        metadata = json.loads(meta_bytes.decode("utf-8"))

        # Reconstruct layers
        layers: List[Layer] = []
        for layer_cfg in metadata.get("layers", []):
            layer_type = layer_cfg["type"]
            if layer_type not in LAYER_REGISTRY:
                raise ValueError(f"Unsupported layer type '{layer_type}' in DNB file.")
            layer_cls = LAYER_REGISTRY[layer_type]
            layer = layer_cls.from_dict(layer_cfg)
            layers.append(layer)

        model = Sequential(layers)

        # Read tensors and verify CRC32
        post_meta_pad = _align_to(meta_offset + meta_len, 8)
        f.seek(meta_offset + meta_len + post_meta_pad)

        loaded_tensors: Dict[str, np.ndarray] = {}

        for _ in range(tensor_count):
            prefix = f.read(4)
            name_len, dtype_code, ndim = struct.unpack("<HBB", prefix)
            name = f.read(name_len).decode("ascii")

            shape = []
            for _ in range(ndim):
                dim = struct.unpack("<Q", f.read(8))[0]
                shape.append(dim)

            data_len, expected_crc = struct.unpack("<QI", f.read(12))
            dt = DTYPE_CODE_TO_NP.get(dtype_code, np.float32)

            raw_bytes = f.read(data_len)
            actual_crc = zlib.crc32(raw_bytes)
            if actual_crc != expected_crc:
                raise ValueError(
                    f"DNB Checksum Error in tensor '{name}': Expected CRC32 0x{expected_crc:08X}, got 0x{actual_crc:08X}."
                )

            arr = np.frombuffer(raw_bytes, dtype=dt).reshape(shape).copy()
            loaded_tensors[name] = arr

            # Align to 8 bytes
            cur_pos = f.tell()
            pad = _align_to(cur_pos, 8)
            if pad > 0:
                f.seek(pad, 1)

        # Assign weights back to model layers
        for layer_idx, layer in enumerate(model.layers):
            if hasattr(layer, "get_params"):
                params = layer.get_params()
                for p_name in list(params.keys()):
                    key = f"layer_{layer_idx}.{p_name}"
                    if key in loaded_tensors:
                        np.copyto(params[p_name], loaded_tensors[key])

    return model
