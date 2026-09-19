"""Zexo-Pulse & Zexo-Xtra: Native High-Performance C++ Decision Engine.

Provides sub-millisecond, non-autoregressive System-1 intelligence with native
C++ acceleration for:
1. Category / Intent Domain Routing (ChoiceHead)
2. Safety & Prompt Injection Moderation (BooleanHead)
3. Prompt Technical Complexity Scoring (ScoreHead)
4. Over-parameterized Deep Architectures (Zexo-Xtra: 6+ Layers, dim=512)
"""

from __future__ import annotations
import ctypes
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from doraneural.cpp_backend import build_cpp_library
from doraneural.jev import DecisionOutput

REPO_ROOT = Path(__file__).resolve().parent.parent


class PulseConfigStruct(ctypes.Structure):
    _fields_ = [
        ("vocab_size", ctypes.c_int),
        ("dim", ctypes.c_int),
        ("hidden_dim", ctypes.c_int),
        ("n_layers", ctypes.c_int),
        ("n_classes", ctypes.c_int),
        ("score_min", ctypes.c_float),
        ("score_max", ctypes.c_float),
    ]


class ZexoPulse:
    """Native C++ Accelerated System-1 Decision Engine (Zexo-Pulse)."""

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        categories: Optional[List[str]] = None,
        score_min: float = 0.0,
        score_max: float = 10.0,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.categories = categories or [
            "programming", "math", "machine_learning", "systems",
            "reasoning", "safety", "creative_writing", "dialogue",
        ]
        self.cat_to_idx = {c: i for i, c in enumerate(self.categories)}
        self.n_classes = len(self.categories)
        self.score_min = score_min
        self.score_max = score_max

        self._lib = self._load_library()
        self._config = PulseConfigStruct(
            self.vocab_size,
            self.dim,
            self.hidden_dim,
            self.n_layers,
            self.n_classes,
            self.score_min,
            self.score_max,
        )
        self._engine = self._lib.pulse_create(ctypes.byref(self._config))
        if not self._engine:
            raise RuntimeError("Failed to create C++ PulseEngine instance.")

    def _load_library(self) -> ctypes.CDLL:
        so_path = REPO_ROOT / "doraneural" / "csrc" / "libdoraneural.so"
        if not so_path.exists():
            build_cpp_library(force=True)
        lib = ctypes.CDLL(str(so_path))

        lib.pulse_create.argtypes = [ctypes.POINTER(PulseConfigStruct)]
        lib.pulse_create.restype = ctypes.c_void_p

        lib.pulse_free.argtypes = [ctypes.c_void_p]
        lib.pulse_free.restype = None

        lib.pulse_forward.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.pulse_forward.restype = None

        lib.pulse_train_sample.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.pulse_train_sample.restype = ctypes.c_float

        lib.pulse_train_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
        ]
        lib.pulse_train_batch.restype = ctypes.c_float

        return lib

    def __del__(self):
        if hasattr(self, "_engine") and self._engine and hasattr(self, "_lib") and self._lib:
            try:
                self._lib.pulse_free(self._engine)
                self._engine = None
            except Exception:
                pass

    def decide(self, text: Union[str, bytes], head_name: str = "category_router") -> DecisionOutput:
        """Execute instantaneous C++ single-pass decision (< 0.05ms)."""
        raw_bytes = text.encode("utf-8") if isinstance(text, str) else text
        n = len(raw_bytes)
        if n == 0:
            raw_bytes = b" "
            n = 1

        tokens = (ctypes.c_uint8 * n)(*raw_bytes)
        out_choice = (ctypes.c_float * self.n_classes)()
        out_bool = (ctypes.c_float * 1)()
        out_score = (ctypes.c_float * 1)()

        t0 = time.perf_counter()
        self._lib.pulse_forward(
            self._engine,
            tokens,
            n,
            out_choice,
            out_bool,
            out_score,
        )
        lat_ms = (time.perf_counter() - t0) * 1000.0

        if head_name in ("category_router", "category", "choice"):
            probs = np.array(list(out_choice), dtype=np.float32)
            best_idx = int(np.argmax(probs))
            return DecisionOutput(
                action=self.categories[best_idx],
                confidence=float(probs[best_idx]),
                probabilities=probs,
                scores={c: float(p) for c, p in zip(self.categories, probs)},
                latency_ms=lat_ms,
            )

        if head_name in ("safety_flag", "safety", "bool"):
            p = float(out_bool[0])
            is_flagged = (p >= 0.5)
            conf = p if is_flagged else (1.0 - p)
            return DecisionOutput(
                action=is_flagged,
                confidence=float(conf),
                probabilities=np.array([1.0 - p, p], dtype=np.float32),
                latency_ms=lat_ms,
            )

        if head_name in ("complexity_score", "complexity", "score"):
            val = float(out_score[0])
            return DecisionOutput(
                action=round(val, 2),
                confidence=1.0,
                scores={"score": val},
                latency_ms=lat_ms,
            )

        raise ValueError(f"Unknown decision head: '{head_name}'. Expected category, safety, or complexity.")

    def train_batch(
        self,
        samples: List[Dict[str, any]],
        lr: float = 0.005,
        lambda_cal: float = 0.5,
        weight_decay: float = 0.001,
        num_threads: int = 4,
    ) -> float:
        """Execute C++ mini-batch update with RLCD loss and AdamW."""
        if not samples:
            return 0.0

        flat = bytearray()
        offsets = []
        lens = []
        target_cats = []
        target_safeties = []
        target_complexities = []

        for s in samples:
            text = s["text"].encode("utf-8") if isinstance(s["text"], str) else s["text"]
            if not text:
                text = b" "
            offsets.append(len(flat))
            lens.append(len(text))
            flat.extend(text)

            cat_idx = self.cat_to_idx.get(s.get("category", "general"), 0)
            target_cats.append(cat_idx)
            target_safeties.append(1 if s.get("is_safety", False) else 0)
            target_complexities.append(float(s.get("complexity", 5.0)))

        n = len(samples)
        c_flat = (ctypes.c_uint8 * len(flat)).from_buffer(flat)
        c_offsets = (ctypes.c_int * n)(*offsets)
        c_lens = (ctypes.c_int * n)(*lens)
        c_cats = (ctypes.c_int * n)(*target_cats)
        c_safes = (ctypes.c_int * n)(*target_safeties)
        c_comps = (ctypes.c_float * n)(*target_complexities)

        loss = self._lib.pulse_train_batch(
            self._engine,
            c_flat,
            c_offsets,
            c_lens,
            c_cats,
            c_safes,
            c_comps,
            n,
            float(lr),
            float(lambda_cal),
            float(weight_decay),
            int(num_threads),
        )
        return float(loss)


class ZexoXtra(ZexoPulse):
    """Over-parameterized Deep Decision Engine (6 Layers, dim=512, hidden_dim=1024).

    Designed for high-capacity semantic topological routing and multi-attribute calibration.
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        score_min: float = 0.0,
        score_max: float = 10.0,
    ):
        super().__init__(
            vocab_size=256,
            dim=512,
            hidden_dim=1024,
            n_layers=6,
            categories=categories,
            score_min=score_min,
            score_max=score_max,
        )


# Backward compatibility alias
JevDecisionModel = ZexoPulse
JevXtra = ZexoXtra
