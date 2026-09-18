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
from typing import Any, Dict, List, Optional, Tuple, Union

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
        ("eos_token_id", ctypes.c_int),
        ("rope_theta", ctypes.c_float),
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
    global _LIB_HANDLE, _INIT_ATTEMPTED
    if force:
        # Reset cached state so get_cpp_library will retry compilation
        _LIB_HANDLE = None
        _INIT_ATTEMPTED = False
        # Remove old .so to force rebuild
        try:
            so_path = Path(__file__).resolve().parent / "csrc" / "libdoraneural.so"
            if so_path.exists():
                so_path.unlink()
        except Exception:
            pass

    csrc_dir = Path(__file__).resolve().parent / "csrc"
    cpp_file = csrc_dir / "llm_engine.cpp"
    so_file = csrc_dir / "libdoraneural.so"

    if not cpp_file.exists():
        print(f"[cpp_backend] C++ source not found: {cpp_file}", file=sys.stderr)
        return None

    if (
        so_file.exists()
        and not force
        and so_file.stat().st_mtime_ns >= cpp_file.stat().st_mtime_ns
    ):
        return so_file

    compiler = _find_compiler()
    if not compiler:
        print("[cpp_backend] No C++ compiler found (tried g++, clang++, c++)", file=sys.stderr)
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
            print(f"[cpp_backend] Primary compile failed: {ret.stderr.decode()[:1000]}", file=sys.stderr)
            # Fallback without -march=native
            cmd_no_native = [compiler, "-O3", "-shared", "-fPIC", "-std=c++17", "-fopenmp", "-ffast-math", str(cpp_file), "-o", str(so_file)]
            ret = subprocess.run(cmd_no_native, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if ret.returncode != 0:
                print(f"[cpp_backend] Fallback (no native) failed: {ret.stderr.decode()[:1000]}", file=sys.stderr)
                # Fallback without -fopenmp
                cmd_no_omp = [compiler, "-O3", "-shared", "-fPIC", "-std=c++17", "-ffast-math", str(cpp_file), "-o", str(so_file)]
                ret = subprocess.run(cmd_no_omp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if ret.returncode != 0:
                    print(f"[cpp_backend] Fallback (no omp) failed: {ret.stderr.decode()[:1000]}", file=sys.stderr)
                    cmd_basic = [compiler, "-O3", "-shared", "-fPIC", "-std=c++17", str(cpp_file), "-o", str(so_file)]
                    ret = subprocess.run(cmd_basic, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    if ret.returncode != 0:
                        print(f"[cpp_backend] Basic compile failed: {ret.stderr.decode()[:1000]}", file=sys.stderr)
                        return None
        if not so_file.exists():
            print(f"[cpp_backend] Expected .so not found after compile: {so_file}", file=sys.stderr)
            return None
        # Quick sanity: file size >10KB
        if so_file.stat().st_size < 10*1024:
            print(f"[cpp_backend] Compiled .so suspiciously small: {so_file.stat().st_size} bytes", file=sys.stderr)
            return None
        return so_file
    except Exception as e:
        print(f"[cpp_backend] Exception during compile: {e}", file=sys.stderr)
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
        print(f"[cpp_backend] Library not available at {so_path}", file=sys.stderr)
        return None

    try:
        lib = ctypes.CDLL(str(so_path))
        # Verify required symbols exist
        required_symbols = ["llama_create", "llama_free", "llama_forward", "llama_generate", "llama_get_threads", "llama_set_threads", "llama_sample_token"]
        for sym in required_symbols:
            if not hasattr(lib, sym):
                print(f"[cpp_backend] Missing symbol in .so: {sym}", file=sys.stderr)
                return None

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

        lib.llama_full_train_step.argtypes = []
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
            ctypes.c_float,
        ]
        lib.llama_full_train_step.restype = ctypes.c_float

        lib.llama_get_threads.argtypes = []
        lib.llama_get_threads.restype = ctypes.c_int

        lib.llama_set_threads.argtypes = [ctypes.c_int]
        lib.llama_set_threads.restype = None

        lib.llama_sample_token.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
        lib.llama_sample_token.restype = ctypes.c_int

        if hasattr(lib, "llama_sample_token_ex"):
            lib.llama_sample_token_ex.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int]
            lib.llama_sample_token_ex.restype = ctypes.c_int

        if hasattr(lib, "llama_generate_ex"):
            lib.llama_generate_ex.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.llama_generate_ex.restype = ctypes.c_int

        if hasattr(lib, "llama_forward_argmax"):
            lib.llama_forward_argmax.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            lib.llama_forward_argmax.restype = ctypes.c_int

        if hasattr(lib, "llama_set_profile"):
            lib.llama_set_profile.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.llama_set_profile.restype = None

        if hasattr(lib, "llama_reset_profile"):
            lib.llama_reset_profile.argtypes = [ctypes.c_void_p]
            lib.llama_reset_profile.restype = None

        if hasattr(lib, "llama_get_profile"):
            lib.llama_get_profile.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
            lib.llama_get_profile.restype = None

        if hasattr(lib, "llama_get_cpu_arch"):
            lib.llama_get_cpu_arch.argtypes = []
            lib.llama_get_cpu_arch.restype = ctypes.c_char_p

        if hasattr(lib, "llama_get_cpu_backend"):
            lib.llama_get_cpu_backend.argtypes = []
            lib.llama_get_cpu_backend.restype = ctypes.c_char_p

        if hasattr(lib, "llama_print_cpu_features"):
            lib.llama_print_cpu_features.argtypes = []
            lib.llama_print_cpu_features.restype = None

        _LIB_HANDLE = lib
        return _LIB_HANDLE
    except Exception as e:
        print(f"[cpp_backend] Failed to load library: {e}", file=sys.stderr)
        return None


def is_cpp_available() -> bool:
    """Return True if C++ engine is compiled and available."""
    return get_cpp_library() is not None


def get_cpu_arch() -> str:
    """Return detected CPU architecture ('ARM64', 'x86_64', etc.)."""
    lib = get_cpp_library()
    if lib and hasattr(lib, "llama_get_cpu_arch"):
        res = lib.llama_get_cpu_arch()
        return res.decode("utf-8") if res else "Unknown"
    return "Unknown"


def get_cpu_backend() -> str:
    """Return active optimal kernel dispatch backend ('ARM_NEON', 'AVX512_VNNI', etc.)."""
    lib = get_cpp_library()
    if lib and hasattr(lib, "llama_get_cpu_backend"):
        res = lib.llama_get_cpu_backend()
        return res.decode("utf-8") if res else "SCALAR"
    return "SCALAR"


def print_cpu_features() -> None:
    """Print detected CPU hardware SIMD features and active dispatch."""
    lib = get_cpp_library()
    if lib and hasattr(lib, "llama_print_cpu_features"):
        lib.llama_print_cpu_features()


class CppLlamaEngine:
    """Python wrapper for the compiled C++ LLaMA engine."""

    # Required weight keys for a valid LLaMA checkpoint
    _REQUIRED_WEIGHT_KEYS = [
        "token_embedding_table", "rms_att_weight", "wq", "wk", "wv", "wo",
        "rms_ffn_weight", "w1", "w2", "w3", "rms_final_weight", "wcls"
    ]

    def __init__(self, config: Any, weights_dict: dict) -> None:
        self.lib = get_cpp_library()
        if not self.lib:
            raise RuntimeError("C++ library libdoraneural.so is not available. Tried to compile from doraneural/csrc/llm_engine.cpp but failed. Check compiler (g++/clang++) and OpenMP support.")

        # Validate config
        for attr in ["dim", "hidden_dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "seq_len"]:
            if not hasattr(config, attr):
                raise ValueError(f"Config missing attribute: {attr}")
            val = getattr(config, attr)
            if not isinstance(val, int) or val <= 0:
                raise ValueError(f"Config {attr} must be positive int, got {val}")
        if config.dim % config.n_heads != 0:
            raise ValueError(f"dim {config.dim} must be divisible by n_heads {config.n_heads}")
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError(f"n_heads {config.n_heads} must be divisible by n_kv_heads {config.n_kv_heads}")

        # Validate weights dict
        if not isinstance(weights_dict, dict):
            raise TypeError(f"weights_dict must be dict, got {type(weights_dict)}")
        missing = [k for k in self._REQUIRED_WEIGHT_KEYS if k not in weights_dict]
        if missing:
            raise KeyError(f"weights_dict missing required keys: {missing}")

        self.config_struct = LlamaCppConfigStruct(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            vocab_size=config.vocab_size,
            seq_len=config.seq_len,
            rope_type=getattr(config, "rope_type_int", 0),
            eos_token_id=int(getattr(config, "eos_token_id", 2)),
            rope_theta=float(getattr(config, "rope_theta", 10000.0)),
        )

        self._contiguous_refs: List[np.ndarray] = []

        def _ptr(arr: np.ndarray, name: str) -> ctypes.POINTER(ctypes.c_float):
            if arr is None:
                raise ValueError(f"Weight {name} is None")
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Weight {name} must be np.ndarray, got {type(arr)}")
            if arr.size == 0:
                raise ValueError(f"Weight {name} is empty")
            if arr.dtype != np.float32 or not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr, dtype=np.float32)
                # Keep converted storage alive for the entire native engine
                # lifetime; a temporary ctypes pointer is not sufficient.
                self._contiguous_refs.append(arr)
            return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._weights_ref = weights_dict  # Keep original Python arrays alive
        is_shared = int(weights_dict.get("shared_classifier", 1))

        try:
            self.weights_struct = LlamaCppWeightsStruct(
                token_embedding_table=_ptr(weights_dict["token_embedding_table"], "token_embedding_table"),
                rms_att_weight=_ptr(weights_dict["rms_att_weight"], "rms_att_weight"),
                wq=_ptr(weights_dict["wq"], "wq"),
                wk=_ptr(weights_dict["wk"], "wk"),
                wv=_ptr(weights_dict["wv"], "wv"),
                wo=_ptr(weights_dict["wo"], "wo"),
                rms_ffn_weight=_ptr(weights_dict["rms_ffn_weight"], "rms_ffn_weight"),
                w1=_ptr(weights_dict["w1"], "w1"),
                w2=_ptr(weights_dict["w2"], "w2"),
                w3=_ptr(weights_dict["w3"], "w3"),
                rms_final_weight=_ptr(weights_dict["rms_final_weight"], "rms_final_weight"),
                wcls=_ptr(weights_dict["wcls"], "wcls"),
                shared_classifier=is_shared,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to prepare weight pointers: {e}") from e

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
        try:
            if hasattr(self, "handle") and self.handle and self.lib:
                self.lib.llama_free(self.handle)
                self.handle = None
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.handle and self.lib:
                self.lib.llama_free(self.handle)
        except Exception:
            pass
        self.handle = None
        return False

    @property
    def threads(self) -> int:
        """Return the native OpenMP thread count."""
        try:
            return int(self.lib.llama_get_threads())
        except Exception as e:
            raise RuntimeError(f"Failed to get threads: {e}") from e

    def set_threads(self, num_threads: int) -> None:
        """Tune native inference/training parallelism for the current process."""
        if num_threads <= 0:
            raise ValueError(f"num_threads must be positive, got {num_threads}")
        if num_threads > 64:
            print(f"[cpp_backend] Warning: num_threads {num_threads} unusually high, may degrade performance", file=sys.stderr)
        try:
            self.lib.llama_set_threads(int(num_threads))
        except Exception as e:
            raise RuntimeError(f"Failed to set threads: {e}") from e

    def reset_cache(self) -> None:
        """Reset key-value cache arenas."""
        if not self.handle:
            raise RuntimeError("Engine handle is null, cannot reset cache")
        try:
            self.lib.llama_reset_cache(self.handle)
        except Exception as e:
            raise RuntimeError(f"Failed to reset cache: {e}") from e

    def forward(self, token: int, pos: int, copy_logits: bool = True) -> np.ndarray:
        """Run forward pass for a single token using C++ OpenMP/SIMD engine."""
        if not self.handle:
            raise RuntimeError("Engine handle is null")
        if not isinstance(token, int) or token < 0 or token >= self.vocab_size:
            raise ValueError(f"token {token} out of vocab range [0,{self.vocab_size})")
        if not isinstance(pos, int) or pos < 0 or pos >= self.config_struct.seq_len:
            raise ValueError(f"pos {pos} out of range [0,{self.config_struct.seq_len})")
        try:
            ptr = self._logits_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self.lib.llama_forward(self.handle, int(token), int(pos), ptr)
            return self._logits_buf.copy() if copy_logits else self._logits_buf
        except Exception as e:
            raise RuntimeError(f"Forward failed at pos {pos}: {e}") from e

    def forward_argmax(self, token: int, pos: int) -> int:
        """Forward single token with fused classifier argmax (zero logit memory write)."""
        if not self.handle:
            raise RuntimeError("Engine handle is null")
        if not isinstance(token, int) or token < 0 or token >= self.vocab_size:
            raise ValueError(f"token {token} out of vocab range [0,{self.vocab_size})")
        if not isinstance(pos, int) or pos < 0 or pos >= self.config_struct.seq_len:
            raise ValueError(f"pos {pos} out of range [0,{self.config_struct.seq_len})")
        try:
            if hasattr(self.lib, "llama_forward_argmax"):
                return int(self.lib.llama_forward_argmax(self.handle, int(token), int(pos)))
            logits = self.forward(token, pos, copy_logits=False)
            return int(np.argmax(logits))
        except Exception as e:
            raise RuntimeError(f"Forward argmax failed at pos {pos}: {e}") from e

    def sample(self, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 0) -> int:
        """Sample next token directly in C++ using fast partial-sort sampling."""
        if not self.handle:
            raise RuntimeError("Engine handle is null")
        if temperature < 0:
            raise ValueError(f"temperature must be >=0, got {temperature}")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0,1], got {top_p}")
        if top_k < 0:
            raise ValueError(f"top_k must be >=0, got {top_k}")
        try:
            if top_k > 0 and hasattr(self.lib, "llama_sample_token_ex"):
                return int(self.lib.llama_sample_token_ex(self.handle, float(temperature), float(top_p), int(top_k)))
            return int(self.lib.llama_sample_token(self.handle, float(temperature), float(top_p)))
        except Exception as e:
            raise RuntimeError(f"Sampling failed: {e}") from e

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        """Generate tokens autoregressively in pure C++ without GIL or Python overhead."""
        if not self.handle:
            raise RuntimeError("Engine handle is null")
        if not prompt_tokens:
            raise ValueError("prompt_tokens must not be empty")
        if not all(isinstance(t, int) for t in prompt_tokens):
            raise TypeError("prompt_tokens must be list of ints")
        if any(t < 0 or t >= self.vocab_size for t in prompt_tokens):
            raise ValueError(f"prompt_tokens contains out-of-vocab ids (vocab_size={self.vocab_size})")
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
        if max_new_tokens == 0:
            return []
        if len(prompt_tokens) >= self.config_struct.seq_len:
            raise ValueError(f"prompt_tokens len {len(prompt_tokens)} must fit inside context window {self.config_struct.seq_len}")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        if temperature < 0:
            raise ValueError(f"temperature must be >=0, got {temperature}")
        if top_k < 0:
            raise ValueError(f"top_k must be >=0, got {top_k}")
        if max_new_tokens > self.config_struct.seq_len:
            print(f"[cpp_backend] Warning: max_new_tokens {max_new_tokens} > seq_len {self.config_struct.seq_len}, will be truncated", file=sys.stderr)

        try:
            p_arr = (ctypes.c_int * len(prompt_tokens))(*prompt_tokens)
            out_buf = (ctypes.c_int * max_new_tokens)()
            eos_id = eos_token_id if eos_token_id is not None else -1

            if hasattr(self.lib, "llama_generate_ex"):
                n_gen = self.lib.llama_generate_ex(
                    self.handle,
                    p_arr,
                    len(prompt_tokens),
                    max_new_tokens,
                    float(temperature),
                    float(top_p),
                    int(top_k),
                    int(eos_id),
                    out_buf,
                )
            else:
                n_gen = self.lib.llama_generate(
                    self.handle,
                    p_arr,
                    len(prompt_tokens),
                    max_new_tokens,
                    float(temperature),
                    float(top_p),
                    out_buf,
                )
            if n_gen < 0:
                raise RuntimeError(f"llama_generate returned error code {n_gen}")
            if n_gen > max_new_tokens:
                print(f"[cpp_backend] Warning: n_gen {n_gen} > max_new_tokens {max_new_tokens}, truncating", file=sys.stderr)
                n_gen = max_new_tokens
            result = [int(out_buf[i]) for i in range(n_gen)]
            # Validate output tokens
            if any(t < 0 or t >= self.vocab_size for t in result):
                print(f"[cpp_backend] Warning: generated out-of-vocab tokens detected", file=sys.stderr)
            return result
        except Exception as e:
            if isinstance(e, (ValueError, TypeError, RuntimeError)):
                raise
            raise RuntimeError(f"Generation failed: {e}") from e

    def set_profile(self, enable: bool = True) -> None:
        """Enable or disable native performance profiling timers."""
        if not self.handle:
            raise RuntimeError("Engine handle is null")
        if hasattr(self.lib, "llama_set_profile"):
            self.lib.llama_set_profile(self.handle, 1 if enable else 0)

    def reset_profile(self) -> None:
        """Reset native performance profiling counters."""
        if not self.handle:
            raise RuntimeError("Engine handle is null")
        if hasattr(self.lib, "llama_reset_profile"):
            self.lib.llama_reset_profile(self.handle)

    def get_profile(self) -> Dict[str, Union[float, int]]:
        """Retrieve breakdown of native time spent in forward pass components."""
        if not self.handle:
            raise RuntimeError("Engine handle is null")
        if hasattr(self.lib, "llama_get_profile"):
            stats = (ctypes.c_double * 8)()
            self.lib.llama_get_profile(self.handle, stats)
            return {
                "rmsnorm_ms": float(stats[0]),
                "qkv_ms": float(stats[1]),
                "rope_ms": float(stats[2]),
                "attn_ms": float(stats[3]),
                "ffn_ms": float(stats[4]),
                "classifier_ms": float(stats[5]),
                "total_ms": float(stats[6]),
                "count": int(stats[7]),
            }
        return {}

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
        grad_clip: float = 0.0,
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
            float(grad_clip),
        )
        return float(loss)
