"""High-performance C++ backend interface for doraneural LLM engine.

Provides automatic JIT compilation and ctypes bindings to libdoraneural.so.
Falls back seamlessly to pure NumPy if a C++ compiler is not available.
"""

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LlamaCppConfigStruct(ctypes.Structure):
    _fields_ = [
        ("dim", ctypes.c_int),
        ("hidden_dim", ctypes.c_int),
        ("n_layers", ctypes.c_int),
        ("n_heads", ctypes.c_int),
        ("n_kv_heads", ctypes.c_int),
        ("vocab_size", ctypes.c_int),
        ("seq_len", ctypes.c_int),
        ("rope_type", ctypes.c_int),
    ]


class LlamaCppWeightsStruct(ctypes.Structure):
    _fields_ = [
        ("token_embedding_table", ctypes.POINTER(ctypes.c_float)),
        ("rms_att_weight", ctypes.POINTER(ctypes.c_float)),
        ("wq", ctypes.POINTER(ctypes.c_float)),
        ("wk", ctypes.POINTER(ctypes.c_float)),
        ("wv", ctypes.POINTER(ctypes.c_float)),
        ("wo", ctypes.POINTER(ctypes.c_float)),
        ("rms_ffn_weight", ctypes.POINTER(ctypes.c_float)),
        ("w1", ctypes.POINTER(ctypes.c_float)),
        ("w2", ctypes.POINTER(ctypes.c_float)),
        ("w3", ctypes.POINTER(ctypes.c_float)),
        ("rms_final_weight", ctypes.POINTER(ctypes.c_float)),
        ("wcls", ctypes.POINTER(ctypes.c_float)),
        ("shared_classifier", ctypes.c_int),
    ]


_LIB_HANDLE: Optional[ctypes.CDLL] = None
_INIT_ATTEMPTED: bool = False


def _find_compiler() -> Optional[str]:
    """Find an available C++ compiler."""
    for compiler in ["g++", "clang++", "c++"]:
        path = shutil.which(compiler)
        if path:
            return path
    return None


def build_cpp_library(force: bool = False) -> Optional[Path]:
    """Compile the C++ shared library if it does not already exist."""
    csrc_dir = Path(__file__).resolve().parent / "csrc"
    cpp_file = csrc_dir / "llm_engine.cpp"
    so_file = csrc_dir / "libdoraneural.so"

    if not cpp_file.exists():
        return None

    if (
        so_file.exists()
        and not force
        and so_file.stat().st_mtime_ns >= cpp_file.stat().st_mtime_ns
    ):
        return so_file

    compiler = _find_compiler()
    if not compiler:
        return None

    cmd = [
        compiler,
        "-O3",
        "-shared",
        "-fPIC",
        "-std=c++17",
        "-fopenmp",
        "-ffast-math",
        "-march=native",
        "-funroll-loops",
        str(cpp_file),
        "-o",
        str(so_file),
    ]

    try:
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if ret.returncode != 0:
            # Fallback without -march=native
            cmd_no_native = [compiler, "-O3", "-shared", "-fPIC", "-std=c++17", "-fopenmp", "-ffast-math", str(cpp_file), "-o", str(so_file)]
            ret = subprocess.run(cmd_no_native, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if ret.returncode != 0:
                # Fallback without -fopenmp
                cmd_no_omp = [compiler, "-O3", "-shared", "-fPIC", "-std=c++17", "-ffast-math", str(cpp_file), "-o", str(so_file)]
                ret = subprocess.run(cmd_no_omp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if ret.returncode != 0:
                    cmd_basic = [compiler, "-O3", "-shared", "-fPIC", "-std=c++17", str(cpp_file), "-o", str(so_file)]
                    ret = subprocess.run(cmd_basic, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    if ret.returncode != 0:
                        return None
        return so_file if so_file.exists() else None
    except Exception:
        return None


def get_cpp_library() -> Optional[ctypes.CDLL]:
    """Get the loaded ctypes CDLL handle to libdoraneural.so."""
    global _LIB_HANDLE, _INIT_ATTEMPTED
    if _LIB_HANDLE is not None:
        return _LIB_HANDLE
    if _INIT_ATTEMPTED:
        return None

    _INIT_ATTEMPTED = True
    so_path = build_cpp_library()
    if not so_path or not so_path.exists():
        return None

    try:
        lib = ctypes.CDLL(str(so_path))

        # Define argtypes and restypes
        lib.llama_create.argtypes = [ctypes.POINTER(LlamaCppConfigStruct), ctypes.POINTER(LlamaCppWeightsStruct)]
        lib.llama_create.restype = ctypes.c_void_p

        lib.llama_free.argtypes = [ctypes.c_void_p]
        lib.llama_free.restype = None

        lib.llama_reset_cache.argtypes = [ctypes.c_void_p]
        lib.llama_reset_cache.restype = None

        lib.llama_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        lib.llama_forward.restype = None

        lib.llama_generate.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.llama_generate.restype = ctypes.c_int

        lib.llama_train_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.llama_train_step.restype = ctypes.c_float

        lib.llama_full_train_step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.llama_full_train_step.restype = ctypes.c_float

        lib.llama_get_threads.argtypes = []
        lib.llama_get_threads.restype = ctypes.c_int

        lib.llama_set_threads.argtypes = [ctypes.c_int]
        lib.llama_set_threads.restype = None

        lib.llama_sample_token.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
        lib.llama_sample_token.restype = ctypes.c_int

        _LIB_HANDLE = lib
        return _LIB_HANDLE
    except Exception:
        return None


def is_cpp_available() -> bool:
    """Return True if C++ engine is compiled and available."""
    return get_cpp_library() is not None


class CppLlamaEngine:
    """Python wrapper for the compiled C++ LLaMA engine."""

    def __init__(self, config: Any, weights_dict: dict) -> None:
        self.lib = get_cpp_library()
        if not self.lib:
            raise RuntimeError("C++ library libdoraneural.so is not available.")

        self.config_struct = LlamaCppConfigStruct(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            rope_type=getattr(config, "rope_type_int", 0),
        )

        self._contiguous_refs: List[np.ndarray] = []

        def _ptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
            if arr.dtype != np.float32 or not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr, dtype=np.float32)
                # Keep converted storage alive for the entire native engine
                # lifetime; a temporary ctypes pointer is not sufficient.
                self._contiguous_refs.append(arr)
            return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._weights_ref = weights_dict  # Keep original Python arrays alive
        is_shared = int(weights_dict.get("shared_classifier", 1))

        self.weights_struct = LlamaCppWeightsStruct(
            token_embedding_table=_ptr(weights_dict["token_embedding_table"]),
            rms_att_weight=_ptr(weights_dict["rms_att_weight"]),
            wq=_ptr(weights_dict["wq"]),
            wk=_ptr(weights_dict["wk"]),
            wv=_ptr(weights_dict["wv"]),
            wo=_ptr(weights_dict["wo"]),
            rms_ffn_weight=_ptr(weights_dict["rms_ffn_weight"]),
            w1=_ptr(weights_dict["w1"]),
            w2=_ptr(weights_dict["w2"]),
            w3=_ptr(weights_dict["w3"]),
            rms_final_weight=_ptr(weights_dict["rms_final_weight"]),
            wcls=_ptr(weights_dict["wcls"]),
            shared_classifier=is_shared,
        )

        self.handle = self.lib.llama_create(
            ctypes.byref(self.config_struct),
            ctypes.byref(self.weights_struct),
        )
        if not self.handle:
            raise RuntimeError("Failed to create C++ LlamaEngine instance.")

        self.vocab_size = config.vocab_size
        self._logits_buf = np.empty(self.vocab_size, dtype=np.float32)
        requested_threads = os.environ.get("DORANEURAL_NUM_THREADS")
        if requested_threads:
            try:
                self.set_threads(int(requested_threads))
            except ValueError:
                pass

    def __del__(self) -> None:
        if hasattr(self, "handle") and self.handle and self.lib:
            self.lib.llama_free(self.handle)
            self.handle = None

    @property
    def threads(self) -> int:
        """Return the native OpenMP thread count."""
        return int(self.lib.llama_get_threads())

    def set_threads(self, num_threads: int) -> None:
        """Tune native inference/training parallelism for the current process."""
        if num_threads <= 0:
            raise ValueError(f"num_threads must be positive, got {num_threads}")
        self.lib.llama_set_threads(int(num_threads))

    def reset_cache(self) -> None:
        """Reset key-value cache arenas."""
        self.lib.llama_reset_cache(self.handle)

    def forward(self, token: int, pos: int, copy_logits: bool = True) -> np.ndarray:
        """Run forward pass for a single token using C++ OpenMP/SIMD engine."""
        ptr = self._logits_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.llama_forward(self.handle, int(token), int(pos), ptr)
        return self._logits_buf.copy() if copy_logits else self._logits_buf

    def sample(self, temperature: float = 0.7, top_p: float = 0.9) -> int:
        """Sample next token directly in C++ using fast partial-sort sampling."""
        return int(self.lib.llama_sample_token(self.handle, float(temperature), float(top_p)))

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[int]:
        """Generate tokens autoregressively in pure C++ without GIL or Python overhead."""
        if not prompt_tokens:
            raise ValueError("prompt_tokens must not be empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if len(prompt_tokens) >= self.config_struct.seq_len:
            raise ValueError("prompt_tokens must fit inside the model context window")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        p_arr = (ctypes.c_int * len(prompt_tokens))(*prompt_tokens)
        out_buf = (ctypes.c_int * max_new_tokens)()

        n_gen = self.lib.llama_generate(
            self.handle,
            p_arr,
            len(prompt_tokens),
            max_new_tokens,
            float(temperature),
            float(top_p),
            out_buf,
        )
        return [int(out_buf[i]) for i in range(n_gen)]

    def train_step(
        self,
        input_tokens: List[int],
        target_tokens: List[int],
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> float:
        """Perform a C++ training step computing cross-entropy loss and in-place weight updates."""
        seq_len = len(input_tokens)
        if seq_len != len(target_tokens):
            raise ValueError(f"input_tokens length ({seq_len}) must match target_tokens ({len(target_tokens)})")

        in_arr = (ctypes.c_int * seq_len)(*input_tokens)
        target_arr = (ctypes.c_int * seq_len)(*target_tokens)

        loss = self.lib.llama_train_step(
            self.handle,
            in_arr,
            target_arr,
            seq_len,
            float(lr),
            float(weight_decay),
            float(beta1),
            float(beta2),
            float(eps),
        )
        return float(loss)

    def full_train_step(
        self,
        input_tokens: List[int],
        target_tokens: List[int],
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> float:
        """Run native full-transformer backpropagation and AdamW update."""
        seq_len = len(input_tokens)
        if seq_len != len(target_tokens):
            raise ValueError(f"input_tokens length ({seq_len}) must match target_tokens ({len(target_tokens)})")
        if seq_len <= 0 or seq_len >= self.config_struct.seq_len:
            raise ValueError("sequence length must fit inside the model context window")
        in_arr = (ctypes.c_int * seq_len)(*input_tokens)
        target_arr = (ctypes.c_int * seq_len)(*target_tokens)
        loss = self.lib.llama_full_train_step(
            self.handle,
            in_arr,
            target_arr,
            seq_len,
            float(lr),
            float(weight_decay),
            float(beta1),
            float(beta2),
            float(eps),
        )
        return float(loss)
