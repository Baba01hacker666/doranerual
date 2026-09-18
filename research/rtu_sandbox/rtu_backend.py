"""C++ acceleration backend for RTU language model sandbox."""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional, Tuple

import numpy as np

CSRC_DIR = Path(__file__).parent / "csrc"
LIB_PATH = CSRC_DIR / "librtu.so"
_LIB_HANDLE: Optional[ctypes.CDLL] = None


def compile_rtu_engine(force: bool = False) -> Optional[Path]:
    """Compiles librtu.so using g++ / clang++ with OpenMP and vectorization flags."""
    global _LIB_HANDLE
    if LIB_PATH.exists() and not force:
        return LIB_PATH

    compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    if not compiler:
        return None

    src_file = CSRC_DIR / "rtu_engine.cpp"
    if not src_file.exists():
        return None

    cmd = [
        compiler,
        "-O3",
        "-shared",
        "-fPIC",
        "-std=c++17",
        "-fopenmp",
        "-Wall",
        "-Wextra",
        str(src_file),
        "-o",
        str(LIB_PATH),
    ]

    # Platform-specific optimization flags
    import platform
    machine = platform.machine().lower()
    if "x86_64" in machine or "amd64" in machine:
        cmd.extend(["-mavx2", "-mfma"])
    elif "arm" in machine or "aarch64" in machine:
        cmd.extend(["-march=native"])

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        _LIB_HANDLE = None
        return LIB_PATH
    except subprocess.CalledProcessError as err:
        # Fallback to base compilation without extra SIMD flags
        base_cmd = [
            compiler,
            "-O3",
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-fopenmp",
            str(src_file),
            "-o",
            str(LIB_PATH),
        ]
        try:
            subprocess.run(base_cmd, check=True, capture_output=True)
            _LIB_HANDLE = None
            return LIB_PATH
        except Exception as e:
            print(f"Warning: Could not compile librtu.so: {e}")
            return None


def get_rtu_lib() -> Optional[ctypes.CDLL]:
    """Load or compile librtu.so."""
    global _LIB_HANDLE
    if _LIB_HANDLE is not None:
        return _LIB_HANDLE

    if not LIB_PATH.exists():
        compile_rtu_engine()

    if not LIB_PATH.exists():
        return None

    try:
        lib = ctypes.CDLL(str(LIB_PATH))
        _configure_ctypes_signatures(lib)
        _LIB_HANDLE = lib
        return lib
    except Exception as e:
        print(f"Warning: Could not load {LIB_PATH}: {e}")
        return None


def _configure_ctypes_signatures(lib: ctypes.CDLL):
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_float_pp = ctypes.POINTER(c_float_p)
    c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
    c_int_p = ctypes.POINTER(ctypes.c_int)

    lib.rtu_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.rtu_create.restype = ctypes.c_void_p

    lib.rtu_free.argtypes = [ctypes.c_void_p]
    lib.rtu_free.restype = None

    lib.rtu_reset_state.argtypes = [ctypes.c_void_p]
    lib.rtu_reset_state.restype = None

    lib.rtu_forward_byte.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        c_float_p,
        c_float_p,
        c_float_p,
    ]
    lib.rtu_forward_byte.restype = None

    lib.rtu_sample_byte.argtypes = [
        ctypes.c_void_p,
        c_float_p,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.rtu_sample_byte.restype = ctypes.c_int

    lib.rtu_step_online_train.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        c_int_p,
        c_float_p,
        c_float_p,
    ]
    lib.rtu_step_online_train.restype = None

    lib.rtu_train_sequence.argtypes = [
        ctypes.c_void_p,
        c_uint8_p,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        c_float_p,
        c_float_p,
        c_float_p,
        c_float_p,
    ]
    lib.rtu_train_sequence.restype = None

    lib.rtu_get_weights.argtypes = [
        ctypes.c_void_p,
        c_float_p,
        c_float_p,
        c_float_p,
        c_float_p,
        c_float_pp,
        c_float_pp,
        c_float_pp,
        c_float_pp,
        c_float_pp,
    ]
    lib.rtu_get_weights.restype = None

    lib.rtu_set_weights.argtypes = [
        ctypes.c_void_p,
        c_float_p,
        c_float_p,
        c_float_p,
        c_float_p,
        c_float_pp,
        c_float_pp,
        c_float_pp,
        c_float_pp,
        c_float_pp,
    ]
    lib.rtu_set_weights.restype = None
