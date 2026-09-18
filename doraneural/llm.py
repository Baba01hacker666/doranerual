"""Pure NumPy Pretrained LLaMA Language Model Engine.

Implements full autoregressive Transformer decoder inference in pure NumPy with:
- RoPE (Rotary Position Embeddings) — both llama2.c interleaved and HuggingFace split-half
- RMSNorm
- Multi-Head Attention with KV-Caching (including Grouped-Query Attention)
- SwiGLU Feed-Forward Network
- Heap-accelerated, cached SentencePiece Byte-Fallback BPE Tokenizer (binary .bin or HuggingFace tokenizer.json)
- SafeTensors loader for single-file, BF16, and sharded HuggingFace models (zero PyTorch dependency)
- Nucleus (Top-p) & Temperature Sampling
- Streaming Generation Generator
- Automatic download from Hugging Face Hub (any LlamaForCausalLM repo)
"""

import heapq
import json
import math
import os
import re
import struct
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np


HF_REPOSITORIES = {
    "stories260K": {
        "model_url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin",
        "tok_url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin",
        "model_file": "stories260K.bin",
        "tok_file": "tok512.bin",
        "description": "260K parameters, 5 layers, 512 vocab (~1 MB)",
    },
    "stories15M": {
        "model_url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
        "tok_url": "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin",
        "model_file": "stories15M.bin",
        "tok_file": "tokenizer.bin",
        "description": "15M parameters, 6 layers, 32K vocab (~60 MB)",
    },
}


@dataclass
class LlamaConfig:
    """Hyperparameters for LLaMA architecture."""
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    rope_type: str = "interleaved"  # "interleaved" (llama2.c) or "hf" (HuggingFace split-half)
    rope_theta: float = 10000.0
    eos_token_id: int = 2

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return (self.dim * self.n_kv_heads) // self.n_heads

    @property
    def rope_type_int(self) -> int:
        """0 = llama2.c interleaved, 1 = HuggingFace split-half."""
        return 1 if self.rope_type == "hf" else 0


class LlamaTokenizer:
    """Byte-pair encoding (BPE) SentencePiece tokenizer in pure Python."""

    def __init__(self, tokenizer_path: Union[str, Path], vocab_size: int) -> None:
        self.vocab_size: int = vocab_size
        self.vocab: List[str] = []
        self.scores: List[float] = []

        path = Path(tokenizer_path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        with open(path, "rb") as f:
            self.max_token_len = struct.unpack("<I", f.read(4))[0]
            for _ in range(vocab_size):
                score, tlen = struct.unpack("<fI", f.read(8))
                piece = f.read(tlen).decode("utf-8", errors="replace")
                self.vocab.append(piece)
                self.scores.append(score)

        self.str_to_id = {s: i for i, s in enumerate(self.vocab)}
        self._pair_cache: Dict[Tuple[int, int], int] = {}
        self._encode_cache: Dict[Tuple[str, bool], Tuple[int, ...]] = {}
        self._encode_cache_limit = 2048

    def _pair_id(self, left: int, right: int) -> int:
        """Look up a merge pair, caching hot adjacent-token lookups."""
        key = (left, right)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        candidate = self.str_to_id.get(self.vocab[left] + self.vocab[right], -1)
        if len(self._pair_cache) < 100_000:
            self._pair_cache[key] = candidate
        return candidate

    def _merge_tokens(self, tokens: List[int]) -> List[int]:
        """Merge SentencePiece tokens with a heap instead of repeated scans.

        The old implementation rescanned every adjacent pair after each merge,
        which is quadratic in the number of input symbols. A linked list plus a
        max-heap keeps the same leftmost tie-breaking behavior while reducing
        the common case to O(n log n).
        """
        if len(tokens) < 2:
            return tokens

        next_idx = list(range(1, len(tokens))) + [-1]
        prev_idx = [-1] + list(range(len(tokens) - 1))
        alive = [True] * len(tokens)
        heap: List[Tuple[float, int, int]] = []

        def push_pair(left: int) -> None:
            right = next_idx[left]
            if right < 0 or not alive[left] or not alive[right]:
                return
            candidate = self._pair_id(tokens[left], tokens[right])
            if candidate >= 0:
                # Higher SentencePiece score wins; smaller index preserves
                # the original left-to-right tie break.
                heapq.heappush(heap, (-self.scores[candidate], left, candidate))

        for idx in range(len(tokens) - 1):
            push_pair(idx)

        while heap:
            _, left, candidate = heapq.heappop(heap)
            right = next_idx[left]
            if right < 0 or not alive[left] or not alive[right]:
                continue
            if self._pair_id(tokens[left], tokens[right]) != candidate:
                continue

            tokens[left] = candidate
            next_right = next_idx[right]
            next_idx[left] = next_right
            if next_right >= 0:
                prev_idx[next_right] = left
            alive[right] = False
            push_pair(prev_idx[left])
            push_pair(left)

        merged: List[int] = []
        idx = 0
        while idx >= 0:
            if alive[idx]:
                merged.append(tokens[idx])
            idx = next_idx[idx]
        return merged

    def encode(self, text: str, bos: bool = True) -> List[int]:
        """Encode text with cached, heap-based SentencePiece BPE merges."""
        cache_key = (text, bool(bos))
        cached = self._encode_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        tokens: List[int] = [1] if bos else []  # BOS token
        if text:
            # llama2.c's tokenizer uses a leading space token when present.
            space_id = self.str_to_id.get(" ")
            if space_id is not None:
                tokens.append(space_id)

        for ch in text:
            token_id = self.str_to_id.get(ch)
            if token_id is not None:
                tokens.append(token_id)
            else:
                unknown_id = self.str_to_id.get("<unk>", 0)
                for byte in ch.encode("utf-8"):
                    fallback_id = byte + 3
                    tokens.append(fallback_id if fallback_id < len(self.vocab) else unknown_id)

        result = self._merge_tokens(tokens)
        if len(self._encode_cache) >= self._encode_cache_limit:
            self._encode_cache.pop(next(iter(self._encode_cache)))
        self._encode_cache[cache_key] = tuple(result)
        return result

    def encode_batch(self, texts: Sequence[str], bos: bool = True) -> List[List[int]]:
        """Encode multiple strings while reusing the tokenizer's merge caches."""
        return [self.encode(text, bos=bos) for text in texts]

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID into its string representation."""
        if token_id < 0 or token_id >= len(self.vocab):
            return ""
        s = self.vocab[token_id]
        if s.startswith("<0x") and s.endswith(">"):
            try:
                s = bytes([int(s[3:-1], 16)]).decode("utf-8", errors="replace")
            except Exception:
                pass
        return s

    def decode(self, tokens: List[int]) -> str:
        """Decode a sequence of tokens into a UTF-8 string."""
        pieces = []
        for t in tokens:
            if t in (1, 2):  # Skip BOS and EOS
                continue
            pieces.append(self.decode_token(t))
        return "".join(pieces)

    def decode_batch(self, batches: Sequence[List[int]]) -> List[str]:
        """Decode multiple token sequences."""
        return [self.decode(tokens) for tokens in batches]


class HFTokenizer:
    """HuggingFace tokenizer.json BPE tokenizer in pure Python.

    Reads the standard tokenizer.json format used by LlamaTokenizer
    models on HuggingFace Hub (SentencePiece BPE with ▁ word boundary).
    """

    SPIECE = chr(0x2581)  # ▁

    def __init__(self, tokenizer_path: Union[str, Path]) -> None:
        path = Path(tokenizer_path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)

        model_section = tok_data.get("model", {})
        self.vocab: Dict[str, int] = model_section.get("vocab", {})
        self.vocab_size: int = len(self.vocab)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        # Build BPE merge rank table from merge list
        merges = model_section.get("merges", [])
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            parts = m.split(" ")
            if len(parts) == 2:
                self.bpe_ranks[(parts[0], parts[1])] = i
        self._cache: Dict[str, Tuple[int, ...]] = {}
        self._encode_cache: Dict[Tuple[str, bool], Tuple[int, ...]] = {}
        self._encode_cache_limit = 2048
        self._chunk_pattern = re.compile(r"(\n+|" + re.escape(self.SPIECE) + r")")

    def _encode_piece(self, piece: str) -> List[int]:
        """Encode one piece using a heap-based BPE merge loop."""
        cached = self._cache.get(piece)
        if cached is not None:
            return list(cached)
        if piece in self.vocab:
            result = (self.vocab[piece],)
            self._cache[piece] = result
            return list(result)

        symbols = list(piece)
        if len(symbols) > 1:
            next_idx = list(range(1, len(symbols))) + [-1]
            prev_idx = [-1] + list(range(len(symbols) - 1))
            alive = [True] * len(symbols)
            heap: List[Tuple[int, int, str, str]] = []

            def push_pair(left: int) -> None:
                right = next_idx[left]
                if right < 0 or not alive[left] or not alive[right]:
                    return
                pair = (symbols[left], symbols[right])
                rank = self.bpe_ranks.get(pair)
                if rank is not None:
                    heapq.heappush(heap, (rank, left, pair[0], pair[1]))

            for idx in range(len(symbols) - 1):
                push_pair(idx)

            while heap:
                rank, left, first, second = heapq.heappop(heap)
                right = next_idx[left]
                if right < 0 or not alive[left] or not alive[right]:
                    continue
                if symbols[left] != first or symbols[right] != second:
                    continue
                if self.bpe_ranks.get((first, second)) != rank:
                    continue

                symbols[left] = first + second
                next_right = next_idx[right]
                next_idx[left] = next_right
                if next_right >= 0:
                    prev_idx[next_right] = left
                alive[right] = False
                push_pair(prev_idx[left])
                push_pair(left)

            merged: List[str] = []
            idx = 0
            while idx >= 0:
                if alive[idx]:
                    merged.append(symbols[idx])
                idx = next_idx[idx]
            symbols = merged

        result_ids: List[int] = []
        unk_id = self.vocab.get("<unk>", 0)
        for symbol in symbols:
            token_id = self.vocab.get(symbol)
            if token_id is not None:
                result_ids.append(token_id)
            else:
                for byte in symbol.encode("utf-8"):
                    result_ids.append(self.vocab.get(f"<0x{byte:02X}>", unk_id))
        result = tuple(result_ids)
        self._cache[piece] = result
        return list(result)

    def encode(self, text: str, bos: bool = True) -> List[int]:
        """Encode text with compiled chunking and cached BPE pieces."""
        cache_key = (text, bool(bos))
        cached = self._encode_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        tokens: List[int] = [1] if bos else []  # BOS = 1
        if text:
            norm = self.SPIECE + text.replace(" ", self.SPIECE)
            for chunk in self._chunk_pattern.split(norm):
                if chunk:
                    tokens.extend(self._encode_piece(chunk))

        if len(self._encode_cache) >= self._encode_cache_limit:
            self._encode_cache.pop(next(iter(self._encode_cache)))
        self._encode_cache[cache_key] = tuple(tokens)
        return tokens

    def encode_batch(self, texts: Sequence[str], bos: bool = True) -> List[List[int]]:
        """Encode multiple strings with shared piece and text caches."""
        return [self.encode(text, bos=bos) for text in texts]

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to its string piece."""
        s = self.id_to_token.get(token_id, "")
        if s.startswith("<0x") and s.endswith(">"):
            try:
                s = bytes([int(s[3:-1], 16)]).decode("utf-8", errors="replace")
            except Exception:
                return ""
        return s.replace(self.SPIECE, " ")

    def decode(self, tokens: List[int]) -> str:
        """Decode a sequence of token IDs into a UTF-8 string."""
        pieces = []
        for t in tokens:
            if t in (0, 1, 2):  # Skip UNK, BOS, EOS
                continue
            pieces.append(self.decode_token(t))
        text = "".join(pieces)
        if text.startswith(" "):
            text = text[1:]  # Strip leading space from SentencePiece prefix
        return text

    def decode_batch(self, batches: Sequence[List[int]]) -> List[str]:
        """Decode multiple token sequences."""
        return [self.decode(tokens) for tokens in batches]


def load_safetensors(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load a safetensors file or sharded ``*.index.json`` into NumPy arrays.

    The reader intentionally has no PyTorch dependency. Model shards are
    merged by tensor name, so ordinary HuggingFace sharded checkpoints can be
    passed directly to :class:`LlamaLLM`.
    """
    _DTYPE_MAP = {
        "F16": np.float16,
        "F32": np.float32,
        "F64": np.float64,
        "I8": np.int8,
        "I16": np.int16,
        "I32": np.int32,
        "I64": np.int64,
        "U8": np.uint8,
    }
    path = Path(path)
    if path.name.endswith(".index.json"):
        with path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        if not shard_names:
            raise ValueError(f"Safetensors index contains no weight_map: {path}")
        tensors: Dict[str, np.ndarray] = {}
        for shard_name in shard_names:
            shard_path = path.parent / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Safetensors shard listed by {path} is missing: {shard_path}")
            tensors.update(load_safetensors(shard_path))
        return tensors

    tensors = {}
    with path.open("rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len

        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype_str = info["dtype"]
            shape = info["shape"]
            start, end = info["data_offsets"]
            f.seek(data_start + start)
            raw_bytes = f.read(end - start)
            if dtype_str == "BF16":
                # Safetensors stores bfloat16 as the high 16 bits of an
                # IEEE float32 word; interpreting it as float16 is incorrect.
                words = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.uint32)
                arr = (words << 16).view(np.float32).reshape(shape)
            else:
                np_dtype = _DTYPE_MAP.get(dtype_str)
                if np_dtype is None:
                    raise ValueError(f"Unsupported safetensors dtype: {dtype_str}")
                arr = np.frombuffer(raw_bytes, dtype=np_dtype).astype(np.float32).reshape(shape)
            tensors[name] = arr
    return tensors


def load_hf_config(config_path: Union[str, Path]) -> dict:
    """Load a HuggingFace config.json and return it as a plain dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class LlamaLLM:
    """Pure NumPy LLaMA transformer model for autoregressive text generation."""

    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        backend: str = "auto",
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        is_safetensors = (
            str(self.model_path).endswith(".safetensors")
            or str(self.model_path).endswith(".safetensors.index.json")
        )

        if is_safetensors:
            self._load_safetensors(config_path)
        else:
            self._load_llama2c_bin()

        # Load Tokenizer (detect format by extension)
        tok_path = Path(self.tokenizer_path)
        if tok_path.name.endswith(".json"):
            self.tokenizer = HFTokenizer(tok_path)
        else:
            self.tokenizer = LlamaTokenizer(tok_path, vocab_size=self.config.vocab_size)

        # Allocate Key-Value Cache
        p = self.config
        self.key_cache = np.zeros((p.n_layers, p.seq_len, p.kv_dim), dtype=np.float32)
        self.val_cache = np.zeros((p.n_layers, p.seq_len, p.kv_dim), dtype=np.float32)

        # Precompute RoPE frequency tables for NumPy engine
        half = p.head_size // 2
        dim_idx = np.arange(half, dtype=np.float32)
        self._rope_inv_freq = 1.0 / (10000.0 ** (2.0 * dim_idx / p.head_size))
        dim_idx_inter = np.arange(0, p.head_size, 2, dtype=np.float32)
        self._rope_inv_freq_interleaved = 1.0 / (10000.0 ** (dim_idx_inter / p.head_size))

        # C++ Backend Integration
        self.weights_dict = {
            "token_embedding_table": self.tok_emb,
            "rms_att_weight": self.rms_att,
            "wq": self.wq,
            "wk": self.wk,
            "wv": self.wv,
            "wo": self.wo,
            "rms_ffn_weight": self.rms_ffn,
            "w1": self.w1,
            "w2": self.w2,
            "w3": self.w3,
            "rms_final_weight": self.rms_final,
            "wcls": self.wcls,
            "shared_classifier": int(self.shared_weights),
        }

        self.backend = "numpy"
        self.cpp_engine = None
        if backend in ("auto", "cpp"):
            try:
                from .cpp_backend import CppLlamaEngine, is_cpp_available
                if is_cpp_available():
                    self.cpp_engine = CppLlamaEngine(self.config, self.weights_dict)
                    self.backend = "cpp"
            except Exception:
                if backend == "cpp":
                    raise
                self.backend = "numpy"

    def _load_llama2c_bin(self) -> None:
        """Load weights from the llama2.c binary format (.bin)."""
        with open(self.model_path, "rb") as f:
            header = f.read(28)
            dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack("<7i", header)
            shared_weights = vocab_size > 0
            vocab_size = abs(vocab_size)

            self.config = LlamaConfig(
                dim=dim, hidden_dim=hidden_dim, n_layers=n_layers,
                n_heads=n_heads, n_kv_heads=n_kv_heads,
                vocab_size=vocab_size, seq_len=seq_len,
                rope_type="interleaved",
            )
            raw = np.fromfile(f, dtype=np.float32)

        offset = 0
        def take(shape: Tuple[int, ...]) -> np.ndarray:
            nonlocal offset
            sz = int(np.prod(shape))
            arr = raw[offset : offset + sz].reshape(shape)
            offset += sz
            return arr

        p = self.config
        self.tok_emb = take((p.vocab_size, p.dim))
        self.rms_att = take((p.n_layers, p.dim))
        self.wq = take((p.n_layers, p.dim, p.dim))
        self.wk = take((p.n_layers, p.kv_dim, p.dim))
        self.wv = take((p.n_layers, p.kv_dim, p.dim))
        self.wo = take((p.n_layers, p.dim, p.dim))
        self.rms_ffn = take((p.n_layers, p.dim))
        self.w1 = take((p.n_layers, p.hidden_dim, p.dim))
        self.w2 = take((p.n_layers, p.dim, p.hidden_dim))
        self.w3 = take((p.n_layers, p.hidden_dim, p.dim))
        self.rms_final = take((p.dim,))
        offset += p.seq_len * p.head_size  # Skip legacy frequency table
        if shared_weights or offset >= len(raw):
            self.wcls = self.tok_emb
        else:
            self.wcls = take((p.vocab_size, p.dim))
        self.shared_weights = bool(shared_weights)

    def _load_safetensors(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Load weights from HuggingFace SafeTensors format (.safetensors).

        Expects a config.json alongside the model file (or explicitly provided).
        """
        if config_path is None:
            config_path = self.model_path.parent / "config.json"
        cfg = load_hf_config(config_path)

        dim = cfg["hidden_size"]
        hidden_dim = cfg["intermediate_size"]
        n_layers = cfg["num_hidden_layers"]
        n_heads = cfg["num_attention_heads"]
        n_kv_heads = cfg.get("num_key_value_heads", n_heads)
        vocab_size = cfg["vocab_size"]
        seq_len = cfg.get("max_position_embeddings", 1024)
        tie_embeddings = cfg.get("tie_word_embeddings", True)

        self.config = LlamaConfig(
            dim=dim, hidden_dim=hidden_dim, n_layers=n_layers,
            n_heads=n_heads, n_kv_heads=n_kv_heads,
            vocab_size=vocab_size, seq_len=seq_len,
            rope_type="hf",
        )

        tensors = load_safetensors(self.model_path)
        p = self.config

        # Token embedding
        self.tok_emb = tensors["model.embed_tokens.weight"]

        # Per-layer weights stacked into (n_layers, ...) arrays
        rms_att_list, wq_list, wk_list, wv_list, wo_list = [], [], [], [], []
        rms_ffn_list, w1_list, w2_list, w3_list = [], [], [], []
        for l in range(n_layers):
            prefix = f"model.layers.{l}"
            rms_att_list.append(tensors[f"{prefix}.input_layernorm.weight"])
            wq_list.append(tensors[f"{prefix}.self_attn.q_proj.weight"])
            wk_list.append(tensors[f"{prefix}.self_attn.k_proj.weight"])
            wv_list.append(tensors[f"{prefix}.self_attn.v_proj.weight"])
            wo_list.append(tensors[f"{prefix}.self_attn.o_proj.weight"])
            rms_ffn_list.append(tensors[f"{prefix}.post_attention_layernorm.weight"])
            w1_list.append(tensors[f"{prefix}.mlp.gate_proj.weight"])
            w2_list.append(tensors[f"{prefix}.mlp.down_proj.weight"])
            w3_list.append(tensors[f"{prefix}.mlp.up_proj.weight"])

        self.rms_att = np.stack(rms_att_list)
        self.wq = np.stack(wq_list)
        self.wk = np.stack(wk_list)
        self.wv = np.stack(wv_list)
        self.wo = np.stack(wo_list)
        self.rms_ffn = np.stack(rms_ffn_list)
        self.w1 = np.stack(w1_list)
        self.w2 = np.stack(w2_list)
        self.w3 = np.stack(w3_list)

        # Final norm
        self.rms_final = tensors["model.norm.weight"]

        # Output classifier (lm_head)
        if tie_embeddings:
            self.wcls = self.tok_emb
            self.shared_weights = True
        elif "lm_head.weight" in tensors:
            self.wcls = tensors["lm_head.weight"]
            self.shared_weights = False
        else:
            self.wcls = self.tok_emb
            self.shared_weights = True

    def set_num_threads(self, num_threads: int) -> None:
        """Set native OpenMP threads when the C++ backend is active."""
        if num_threads <= 0:
            raise ValueError(f"num_threads must be positive, got {num_threads}")
        if self.cpp_engine is not None:
            self.cpp_engine.set_threads(num_threads)

    @property
    def num_threads(self) -> int:
        """Return native thread count, or one for the pure NumPy backend."""
        if self.cpp_engine is not None:
            return self.cpp_engine.threads
        return 1

    def reset_cache(self) -> None:
        """Clear key-value cache arenas."""
        if self.cpp_engine is not None:
            self.cpp_engine.reset_cache()
        self.key_cache.fill(0.0)
        self.val_cache.fill(0.0)

    @staticmethod
    def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        return (x / np.sqrt(np.mean(x ** 2) + eps)) * weight

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        # Numerically stable SwiGLU activation
        return x / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def forward(self, token: int, pos: int) -> np.ndarray:
        """Execute single-token forward pass with KV-caching. Returns logits (vocab_size,)."""
        if self.cpp_engine is not None:
            return self.cpp_engine.forward(token, pos)

        p = self.config
        x = self.tok_emb[token].copy()

        for l in range(p.n_layers):
            # 1. Pre-Attention RMSNorm
            xb = self._rmsnorm(x, self.rms_att[l])

            # 2. QKV Linear Projections
            q = self.wq[l] @ xb       # (dim,)
            k = self.wk[l] @ xb       # (kv_dim,)
            v = self.wv[l] @ xb       # (kv_dim,)

            # 3. RoPE Rotary Position Embeddings
            if p.rope_type == "hf":
                # HuggingFace split-half RoPE: [q_first_half, q_second_half]
                half = p.head_size // 2
                freqs = pos * self._rope_inv_freq
                cos_val, sin_val = np.cos(freqs), np.sin(freqs)

                q_heads = q.reshape(p.n_heads, p.head_size)
                q1, q2 = q_heads[:, :half], q_heads[:, half:]
                q_heads[:, :half] = q1 * cos_val - q2 * sin_val
                q_heads[:, half:] = q2 * cos_val + q1 * sin_val
                q = q_heads.reshape(p.dim)

                k_heads = k.reshape(p.n_kv_heads, p.head_size)
                k1, k2 = k_heads[:, :half], k_heads[:, half:]
                k_heads[:, :half] = k1 * cos_val - k2 * sin_val
                k_heads[:, half:] = k2 * cos_val + k1 * sin_val
                k = k_heads.reshape(p.kv_dim)
            else:
                # Standard llama2.c interleaved RoPE: [q0, q1, q2, q3, ...]
                freqs = pos * self._rope_inv_freq_interleaved
                fcr = np.tile(np.cos(freqs), p.n_heads)
                fci = np.tile(np.sin(freqs), p.n_heads)

                q0, q1 = q[0::2].copy(), q[1::2].copy()
                q[0::2] = q0 * fcr - q1 * fci
                q[1::2] = q0 * fci + q1 * fcr

                if p.kv_dim > 0:
                    fcr_k = np.tile(np.cos(freqs), p.n_kv_heads)
                    fci_k = np.tile(np.sin(freqs), p.n_kv_heads)
                    k0, k1 = k[0::2].copy(), k[1::2].copy()
                    k[0::2] = k0 * fcr_k - k1 * fci_k
                    k[1::2] = k0 * fci_k + k1 * fcr_k

            # Store into KV-cache
            self.key_cache[l, pos] = k
            self.val_cache[l, pos] = v

            # 4. Multi-Head Attention
            q_heads = q.reshape(p.n_heads, p.head_size)
            kv_mul = p.n_heads // p.n_kv_heads

            attn_out = np.zeros((p.n_heads, p.head_size), dtype=np.float32)
            for h in range(p.n_heads):
                kv_h = h // kv_mul
                k_past = self.key_cache[l, : pos + 1, kv_h * p.head_size : (kv_h + 1) * p.head_size]
                v_past = self.val_cache[l, : pos + 1, kv_h * p.head_size : (kv_h + 1) * p.head_size]

                scores = (k_past @ q_heads[h]) / np.sqrt(p.head_size)
                probs = self._softmax(scores)
                attn_out[h] = probs @ v_past

            # Project attention heads back to dim and add residual
            x += self.wo[l] @ attn_out.reshape(p.dim)

            # 5. Pre-FFN RMSNorm & SwiGLU MLP
            xb = self._rmsnorm(x, self.rms_ffn[l])
            gate = self._silu(self.w1[l] @ xb)
            up = self.w3[l] @ xb
            x += self.w2[l] @ (gate * up)

        # Final RMSNorm and Classifier projection
        x = self._rmsnorm(x, self.rms_final)
        self._last_x = x
        return self.wcls @ x

    def sample(self, logits: np.ndarray, temperature: float = 0.7, top_p: float = 0.9) -> int:
        """Sample next token from logits using nucleus (top-p) and temperature."""
        if temperature <= 0.0:
            return int(np.argmax(logits))

        logits = logits / max(1e-4, temperature)
        probs = self._softmax(logits)

        if top_p < 1.0:
            vocab_size = len(probs)
            if vocab_size > 256:
                # Fast top-K selection before sorting
                K = min(vocab_size, 64)
                part_idx = np.argpartition(probs, -K)[-K:]
                part_probs = probs[part_idx]
                sort_order = np.argsort(part_probs)[::-1]
                sorted_indices = part_idx[sort_order]
                sorted_probs = part_probs[sort_order]
                cumsum = np.cumsum(sorted_probs)
                if cumsum[-1] < top_p:
                    sorted_indices = np.argsort(probs)[::-1]
                    sorted_probs = probs[sorted_indices]
                    cumsum = np.cumsum(sorted_probs)
            else:
                sorted_indices = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                cumsum = np.cumsum(sorted_probs)

            # Mask out probabilities beyond top_p cutoff
            cutoff_mask = cumsum > top_p
            cutoff_mask[0] = False  # Keep at least the top-1 choice
            sorted_probs[cutoff_mask] = 0.0
            s = np.sum(sorted_probs)
            if s > 0:
                sorted_probs /= s
            else:
                sorted_probs[0] = 1.0

            selected_idx = np.random.choice(len(sorted_probs), p=sorted_probs)
            return int(sorted_indices[selected_idx])

        return int(np.random.choice(len(probs), p=probs))

    def generate(
        self,
        prompt: str = "Once upon a time",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Autoregressively generate text from a prompt.

        Args:
            prompt: Seed text string.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (>0.0 for creativity, 0.0 for deterministic greedy).
            top_p: Nucleus sampling probability cutoff (0.0 to 1.0).
            top_k: Top-K sampling cutoff (0 to disable).
            eos_token_id: End of sequence token ID (defaults to config.eos_token_id or 2).
            stream: If True, returns a generator yielding pieces as they are produced.

        Returns:
            Generated text string, or Generator of token strings.
        """
        if max_tokens < 0:
            raise ValueError(f"max_tokens must be non-negative, got {max_tokens}")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        if top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {top_k}")
        self.reset_cache()
        prompt_tokens = self.tokenizer.encode(prompt, bos=True)
        eos_id = eos_token_id if eos_token_id is not None else getattr(self.config, "eos_token_id", 2)

        if not prompt_tokens:
            prompt_tokens = [1]
        # Keep one context slot available for at least one generated token.
        # This prevents native cache writes past the end for long prompts.
        if len(prompt_tokens) >= self.config.seq_len:
            prompt_tokens = prompt_tokens[-(self.config.seq_len - 1):]

        # Fast path: C++ non-streaming generation executes entirely in native C++
        if not stream and self.cpp_engine is not None:
            gen_ids = self.cpp_engine.generate(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos_id,
            )
            return prompt + self.tokenizer.decode(gen_ids)

        def _generator() -> Generator[str, None, None]:
            pos = 0
            if self.cpp_engine is not None:
                # Fast streaming path via C++ engine (zero logits copying to Python)
                for tok in prompt_tokens:
                    self.cpp_engine.forward(tok, pos, copy_logits=False)
                    pos += 1

                for _ in range(max_tokens):
                    if pos >= self.config.seq_len - 1:
                        break

                    next_token = self.cpp_engine.sample(temperature=temperature, top_p=top_p, top_k=top_k)
                    if next_token == eos_id:
                        break

                    piece = self.tokenizer.decode_token(next_token)
                    yield piece

                    self.cpp_engine.forward(next_token, pos, copy_logits=False)
                    pos += 1
            else:
                # Pure NumPy streaming generator
                logits = None
                for tok in prompt_tokens:
                    logits = self.forward(tok, pos)
                    pos += 1

                for _ in range(max_tokens):
                    if pos >= self.config.seq_len - 1:
                        break

                    next_token = self.sample(logits, temperature=temperature, top_p=top_p)
                    if next_token == eos_id:
                        break

                    piece = self.tokenizer.decode_token(next_token)
                    yield piece

                    logits = self.forward(next_token, pos)
                    pos += 1

        if stream:
            return _generator()

        pieces = list(_generator())
        return prompt + "".join(pieces)

    def train_step(
        self,
        input_tokens: List[int],
        target_tokens: List[int],
        lr: float = 1e-4,
        weight_decay: float = 0.01,
    ) -> float:
        """Run a single training forward+backward step and update weights."""
        if self.cpp_engine is not None:
            return self.cpp_engine.train_step(
                input_tokens,
                target_tokens,
                lr=lr,
                weight_decay=weight_decay,
            )
        return self._train_step_numpy(
            input_tokens,
            target_tokens,
            lr=lr,
            weight_decay=weight_decay,
        )

    def _train_step_numpy(
        self,
        input_tokens: List[int],
        target_tokens: List[int],
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> float:
        """Pure NumPy fallback training step with SFT target masking and AdamW optimization."""
        seq_len = len(input_tokens)
        self.reset_cache()
        total_loss = 0.0

        active_targets = sum(1 for t in target_tokens if 0 <= t < self.config.vocab_size)
        if active_targets == 0:
            return 0.0

        inv_targets = 1.0 / active_targets

        # Initialize NumPy AdamW moments if not already present
        if not hasattr(self, "_adam_step"):
            self._adam_step = 0
            self._m_emb = np.zeros_like(self.tok_emb)
            self._v_emb = np.zeros_like(self.tok_emb)
            cls_target_w = self.wcls if not self.shared_weights else self.tok_emb
            self._m_cls = np.zeros_like(cls_target_w)
            self._v_cls = np.zeros_like(cls_target_w)

        self._adam_step += 1
        t_step = self._adam_step
        beta1_corr = max(1e-7, 1.0 - beta1 ** t_step)
        beta2_corr = max(1e-7, 1.0 - beta2 ** t_step)

        for pos in range(seq_len):
            in_tok = input_tokens[pos]
            target_tok = target_tokens[pos]
            logits = self.forward(in_tok, pos)

            if target_tok < 0 or target_tok >= self.config.vocab_size:
                continue

            e = np.exp(logits - np.max(logits))
            probs = e / np.sum(e)

            target_prob = max(1e-12, float(probs[target_tok]))
            total_loss += -np.log(target_prob)

            dlogits = probs.copy()
            dlogits[target_tok] -= 1.0
            dlogits *= inv_targets

            cls_w = self.wcls if not self.shared_weights else self.tok_emb
            g_emb = cls_w.T @ dlogits

            # AdamW on token embeddings
            self._m_emb[in_tok] = beta1 * self._m_emb[in_tok] + (1.0 - beta1) * g_emb
            self._v_emb[in_tok] = beta2 * self._v_emb[in_tok] + (1.0 - beta2) * (g_emb * g_emb)
            m_hat = self._m_emb[in_tok] / beta1_corr
            v_hat = self._v_emb[in_tok] / beta2_corr
            self.tok_emb[in_tok] -= lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * self.tok_emb[in_tok])

            # Gradient update for classifier weights
            last_x = getattr(self, "_last_x", None)
            if last_x is not None:
                g_cls_target = dlogits[target_tok] * last_x
                self._m_cls[target_tok] = beta1 * self._m_cls[target_tok] + (1.0 - beta1) * g_cls_target
                self._v_cls[target_tok] = beta2 * self._v_cls[target_tok] + (1.0 - beta2) * (g_cls_target * g_cls_target)
                m_hat_cls = self._m_cls[target_tok] / beta1_corr
                v_hat_cls = self._v_cls[target_tok] / beta2_corr
                cls_w[target_tok] -= lr * (m_hat_cls / (np.sqrt(v_hat_cls) + eps) + weight_decay * cls_w[target_tok])

                if self.config.vocab_size <= 1024:
                    for v in range(self.config.vocab_size):
                        if v != target_tok:
                            g_v = dlogits[v] * last_x
                            self._m_cls[v] = beta1 * self._m_cls[v] + (1.0 - beta1) * g_v
                            self._v_cls[v] = beta2 * self._v_cls[v] + (1.0 - beta2) * (g_v * g_v)
                            m_h = self._m_cls[v] / beta1_corr
                            v_h = self._v_cls[v] / beta2_corr
                            cls_w[v] -= lr * (m_h / (np.sqrt(v_h) + eps) + weight_decay * cls_w[v])

        return total_loss / active_targets

    def _prepare_training_batches(
        self,
        text: str,
        seq_len: int,
        mask_prompts: bool = True,
        max_batches: Optional[int] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        """Convert training text into paired (input_seq, target_seq) training chunks.

        If conversational dialogue format is detected (User:/Zexo: or User:/Assistant:),
        it applies SFT instruction masking so prompt tokens have target=-1 (ignored in loss),
        and responses have their true token targets.
        """
        batches = []
        is_dialogue = bool(re.search(r'(?:^|\n)User:\s*', text, re.IGNORECASE))

        if is_dialogue and mask_prompts:
            print(f"🔤 Parsing and tokenizing conversational dialogues ({len(text):,} chars)...", flush=True)
            raw_dialogues = [d.strip() for d in re.split(r'\n\s*\n(?=User:)', text, flags=re.IGNORECASE) if d.strip()]
            turn_pattern = re.compile(r'(User:\s*.*?\n(?:Zexo|Assistant):\s*)(.*?)(?=(?:\nUser:|$))', re.DOTALL | re.IGNORECASE)
            total_d = len(raw_dialogues)
            log_interval = max(500, total_d // 10)

            for d_idx, d in enumerate(raw_dialogues):
                if max_batches is not None and len(batches) >= max_batches:
                    break
                if total_d > 500 and (d_idx + 1) % log_interval == 0:
                    pct = (d_idx + 1) / total_d * 100.0
                    print(f"   Tokenized {d_idx + 1:,}/{total_d:,} dialogues ({pct:5.1f}%) -> {len(batches):,} batches packed", flush=True)

                matches = list(turn_pattern.finditer(d))
                if not matches:
                    continue
                d_inputs = []
                d_targets = []
                for m in matches:
                    prompt_text = m.group(1)
                    resp_text = m.group(2).strip() + "\n"
                    p_toks = self.tokenizer.encode(prompt_text, bos=False)
                    r_toks = self.tokenizer.encode(resp_text, bos=False)
                    turn_tokens = p_toks + r_toks
                    turn_targets = [-1] * len(p_toks) + r_toks
                    d_inputs.extend(turn_tokens[:-1])
                    d_targets.extend(turn_targets[1:])

                # Window dialogue into chunks of length seq_len
                for j in range(0, len(d_inputs), seq_len):
                    chunk_in = d_inputs[j : j + seq_len]
                    chunk_tgt = d_targets[j : j + seq_len]
                    if len(chunk_in) >= 4 and any(t >= 0 for t in chunk_tgt):
                        batches.append((chunk_in, chunk_tgt))
                        if max_batches is not None and len(batches) >= max_batches:
                            break
            print(f"📦 Packed {len(batches):,} training batches ({len(batches) * seq_len:,} sequence tokens).", flush=True)
        else:
            print(f"🔤 Tokenizing pretraining corpus ({len(text):,} chars)...", flush=True)
            tokens = self.tokenizer.encode(text, bos=False)
            if len(tokens) < seq_len + 1:
                tokens = tokens * ((seq_len + 2) // max(1, len(tokens)) + 1)
            for j in range(0, len(tokens) - seq_len, seq_len):
                batches.append((tokens[j : j + seq_len], tokens[j + 1 : j + seq_len + 1]))
                if max_batches is not None and len(batches) >= max_batches:
                    break
            print(f"📦 Packed {len(batches):,} training batches ({len(batches) * seq_len:,} sequence tokens).", flush=True)

        return batches

    def _evaluate_tokens(
        self,
        tokens: List[int],
        seq_len: int,
        stride: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> float:
        """Measure next-token cross-entropy without changing model weights."""
        if len(tokens) < seq_len + 1:
            return float("nan")
        step = stride or seq_len
        starts = list(range(0, len(tokens) - seq_len, step))
        if max_steps is not None:
            starts = starts[:max(0, max_steps)]
        total_nll = 0.0
        total_tokens = 0
        for start_idx in starts:
            self.reset_cache()
            inputs = tokens[start_idx : start_idx + seq_len]
            targets = tokens[start_idx + 1 : start_idx + seq_len + 1]
            for pos, (input_token, target_token) in enumerate(zip(inputs, targets)):
                logits = self.forward(input_token, pos)
                max_logit = float(np.max(logits))
                log_norm = max_logit + float(np.log(np.sum(np.exp(logits - max_logit))))
                total_nll += log_norm - float(logits[target_token])
                total_tokens += 1
        self.reset_cache()
        return total_nll / max(1, total_tokens)

    def full_backprop_model(self):
        """Return the CPU NumPy model that differentiates the whole transformer."""
        from .transformer import TransformerDecoderLM
        model = getattr(self, "_full_backprop_model", None)
        if model is None:
            model = TransformerDecoderLM.from_llama(self)
            self._full_backprop_model = model
        return model

    def _train_full_native(
        self,
        text: str,
        epochs: int,
        lr: float,
        seq_len: int,
        weight_decay: float,
        verbose: int,
        eval_text: Optional[str],
        validation_split: float,
        stride: Optional[int],
        shuffle: bool,
        seed: int,
        max_eval_steps: Optional[int],
    ) -> dict:
        """Native C++ full-transformer training loop."""
        if self.cpp_engine is None:
            raise RuntimeError("Native full backpropagation requires the C++ backend")
        if eval_text is not None and validation_split:
            raise ValueError("Pass either eval_text or validation_split, not both")

        def encode_nonempty(value: str, label: str) -> List[int]:
            encoded = self.tokenizer.encode(value, bos=False)
            if not encoded:
                raise ValueError(f"{label} produced no tokens")
            return encoded

        tokens = encode_nonempty(text, "training text")
        if eval_text is not None:
            eval_tokens = encode_nonempty(eval_text, "evaluation text")
        elif validation_split > 0.0:
            split_at = int(len(tokens) * (1.0 - validation_split))
            split_at = min(max(split_at, seq_len + 1), len(tokens) - 1)
            eval_tokens = tokens[split_at:]
            tokens = tokens[:split_at]
        else:
            eval_tokens = None
        if len(tokens) < seq_len + 1:
            tokens = tokens * (((seq_len + 1) // len(tokens)) + 1)
        step = stride or seq_len
        if not 1 <= step <= seq_len:
            raise ValueError(f"stride must be in [1, seq_len], got {step}")
        starts = list(range(0, len(tokens) - seq_len, step))
        if not starts:
            raise ValueError("training text does not contain a usable sequence")

        rng = np.random.default_rng(seed)
        history = {
            "loss": [],
            "tokens": len(tokens),
            "windows_per_epoch": len(starts),
            "total_steps": len(starts) * epochs,
            "backend": "native_cpp",
            "threads": self.num_threads,
        }
        if verbose:
            print(
                f"[Full BP/C++] workflow: {len(tokens):,} tokens -> "
                f"{len(starts):,} windows/epoch x {epochs:,} epoch(s) = "
                f"{len(starts) * epochs:,} native updates; threads={self.num_threads:,}",
                flush=True,
            )
        if eval_tokens is not None:
            history["val_loss"] = []
        training_started = time.perf_counter()
        for epoch in range(epochs):
            order = starts.copy()
            if shuffle:
                rng.shuffle(order)
            total = 0.0
            for start in order:
                total += self.cpp_engine.full_train_step(
                    tokens[start : start + seq_len],
                    tokens[start + 1 : start + seq_len + 1],
                    lr=lr,
                    weight_decay=weight_decay,
                )
            history["loss"].append(total / len(order))
            if eval_tokens is not None:
                history["val_loss"].append(self._evaluate_tokens(
                    eval_tokens, seq_len, stride=step, max_steps=max_eval_steps,
                ))
            if verbose:
                elapsed = max(time.perf_counter() - training_started, 1e-9)
                updates_done = (epoch + 1) * len(order)
                rate = updates_done / elapsed
                remaining = max((epochs * len(order) - updates_done) / max(rate, 1e-9), 0.0)
                message = (
                    f"[Full BP/C++] Epoch {epoch + 1}/{epochs} "
                    f"loss={history['loss'][-1]:.4f} "
                    f"speed={rate:.2f} updates/s ETA={remaining:.1f}s"
                )
                if "val_loss" in history:
                    message += f" val_loss={history['val_loss'][-1]:.4f}"
                print(message, flush=True)
        self.reset_cache()
        return history

    def train_full(
        self,
        text: str,
        epochs: int = 3,
        lr: float = 1e-4,
        seq_len: int = 32,
        weight_decay: float = 0.01,
        verbose: int = 1,
        eval_text: Optional[str] = None,
        validation_split: float = 0.0,
        stride: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        max_eval_steps: Optional[int] = None,
        native: bool = True,
    ) -> dict:
        """Fine-tune every transformer parameter using native C++ or NumPy autograd.

        This is deliberately separate from the historical head-only trainer:
        it is slower and intended for small CPU runs, validation, and research.
        """
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 1 <= seq_len < self.config.seq_len:
            raise ValueError(f"seq_len must be in [1, {self.config.seq_len - 1}], got {seq_len}")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1)")
        if eval_text is not None and validation_split:
            raise ValueError("Pass either eval_text or validation_split, not both")
        if native and self.cpp_engine is not None:
            return self._train_full_native(
                text, epochs, lr, seq_len, weight_decay, verbose,
                eval_text, validation_split, stride, shuffle, seed, max_eval_steps,
            )

        def encode_nonempty(value: str, label: str) -> List[int]:
            encoded = self.tokenizer.encode(value, bos=False)
            if not encoded:
                raise ValueError(f"{label} produced no tokens")
            return encoded

        train_tokens = encode_nonempty(text, "training text")
        if eval_text is not None:
            eval_tokens = encode_nonempty(eval_text, "evaluation text")
        elif validation_split > 0.0:
            split_at = int(len(train_tokens) * (1.0 - validation_split))
            split_at = min(max(split_at, seq_len + 1), len(train_tokens) - 1)
            eval_tokens = train_tokens[split_at:]
            train_tokens = train_tokens[:split_at]
        else:
            eval_tokens = None
        if len(train_tokens) < seq_len + 1:
            repeats = ((seq_len + 1) // len(train_tokens)) + 1
            train_tokens = train_tokens * repeats
        step = stride or seq_len
        if not 1 <= step <= seq_len:
            raise ValueError(f"stride must be in [1, seq_len], got {step}")

        history = self.full_backprop_model().fit_tokens(
            train_tokens,
            epochs=epochs,
            seq_len=seq_len,
            lr=lr,
            weight_decay=weight_decay,
            stride=step,
            shuffle=shuffle,
            seed=seed,
            eval_tokens=eval_tokens,
            max_eval_steps=max_eval_steps,
            verbose=verbose,
        )
        self.reset_cache()
        return history

    def train_lora(
        self,
        text: str,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: Sequence[str] = ("q", "v"),
        epochs: int = 3,
        lr: float = 1e-3,
        seq_len: int = 32,
        weight_decay: float = 0.0,
        verbose: int = 1,
        eval_text: Optional[str] = None,
        validation_split: float = 0.0,
        stride: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        max_eval_steps: Optional[int] = None,
        adapter_path: Optional[Union[str, Path]] = None,
    ) -> dict:
        """Fine-tune a pretrained checkpoint with adapter-only LoRA weights."""
        if not text:
            raise ValueError("training text must not be empty")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1)")
        if eval_text is not None and validation_split:
            raise ValueError("Pass either eval_text or validation_split, not both")
        model = self.full_backprop_model()
        model.enable_lora(rank=rank, alpha=alpha, target_modules=target_modules, freeze_base=True)
        tokens = self.tokenizer.encode(text, bos=False)
        eval_tokens = self.tokenizer.encode(eval_text, bos=False) if eval_text is not None else None
        if validation_split > 0.0:
            split_at = int(len(tokens) * (1.0 - validation_split))
            split_at = min(max(split_at, seq_len + 1), len(tokens) - 1)
            eval_tokens = tokens[split_at:]
            tokens = tokens[:split_at]
        if len(tokens) < seq_len + 1:
            repeats = ((seq_len + 1) // len(tokens)) + 1
            tokens = tokens * repeats
        history = model.fit_lora_tokens(
            tokens,
            epochs=epochs,
            seq_len=seq_len,
            lr=lr,
            weight_decay=weight_decay,
            stride=stride,
            shuffle=shuffle,
            seed=seed,
            eval_tokens=eval_tokens,
            max_eval_steps=max_eval_steps,
            verbose=verbose,
        )
        if adapter_path is not None:
            history["adapter_path"] = str(model.save_lora(adapter_path))
        # Keep chat/generation fast: fold the trained delta into the arrays
        # already owned by the native engine after saving the standalone adapter.
        model.merge_lora()
        self.reset_cache()
        return history

    def load_lora(self, adapter_path: Union[str, Path]):
        """Load and merge a saved adapter onto this checkpoint for fast inference."""
        model = self.full_backprop_model()
        model.load_lora(adapter_path)
        model.merge_lora()
        self.reset_cache()
        return model

    def train(
        self,
        text: str,
        epochs: int = 3,
        lr: float = 1e-4,
        seq_len: int = 32,
        weight_decay: float = 0.01,
        verbose: int = 1,
        mask_prompts: bool = True,
        max_batches: Optional[int] = None,
        eval_text: Optional[str] = None,
        validation_split: float = 0.0,
        stride: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        max_eval_steps: Optional[int] = None,
        full_backprop: bool = False,
        native_full: bool = True,
        lora_rank: Optional[int] = None,
        lora_alpha: float = 16.0,
        lora_targets: Optional[Sequence[str]] = None,
        adapter_path: Optional[Union[str, Path]] = None,
    ) -> dict:
        """Fine-tune the model on custom text with reproducible validation.

        Args:
            text: Training text corpus.
            epochs: Number of complete passes over the text.
            lr: Peak learning rate for parameter updates.
            seq_len: Chunk length for training sequences.
            weight_decay: L2 regularization factor.
            verbose: 1 to print epoch progress, 0 to silence.
            mask_prompts: If True and dialogue tags are present, mask prompt tokens.
            max_batches: Maximum batches/steps per epoch.
            eval_text: Optional held-out evaluation corpus.
            validation_split: Fraction of training text to reserve for validation if eval_text omitted.
            stride: Stride for sequence windowing.
            shuffle: Whether to shuffle sequence windows per epoch.
            seed: Random seed for shuffling.
            max_eval_steps: Maximum evaluation steps.
            full_backprop: If True, differentiate through all transformer layers.
            native_full: If True and full_backprop, use native C++ full backprop engine.
            lora_rank: If specified, train LoRA adapter of this rank.
            lora_alpha: LoRA scaling factor.
            lora_targets: Module names targeted by LoRA (e.g. ("q", "v")).
            adapter_path: Path to save trained LoRA adapter weights.

        Returns:
            Dictionary containing 'loss' and optionally 'val_loss', 'tokens', 'windows_per_epoch', 'total_steps'.
        """
        if lora_rank is not None:
            if full_backprop:
                raise ValueError("Choose either lora_rank or full_backprop, not both")
            return self.train_lora(
                text,
                rank=lora_rank,
                alpha=lora_alpha,
                target_modules=lora_targets or ("q", "v"),
                epochs=epochs,
                lr=lr,
                seq_len=seq_len,
                weight_decay=weight_decay,
                verbose=verbose,
                eval_text=eval_text,
                validation_split=validation_split,
                stride=stride,
                shuffle=shuffle,
                seed=seed,
                max_eval_steps=max_eval_steps,
                adapter_path=adapter_path,
            )
        if full_backprop:
            return self.train_full(
                text,
                epochs=epochs,
                lr=lr,
                seq_len=seq_len,
                weight_decay=weight_decay,
                verbose=verbose,
                eval_text=eval_text,
                validation_split=validation_split,
                stride=stride,
                shuffle=shuffle,
                seed=seed,
                max_eval_steps=max_eval_steps,
                native=native_full,
            )
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 1 <= seq_len < self.config.seq_len:
            raise ValueError(f"seq_len must be in [1, {self.config.seq_len - 1}], got {seq_len}")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1)")

        def encode_nonempty(value: str, label: str) -> List[int]:
            encoded = self.tokenizer.encode(value, bos=False)
            if not encoded:
                raise ValueError(f"{label} produced no tokens")
            return encoded

        is_dialogue = bool(re.search(r'(?:^|\n)User:\s*', text, re.IGNORECASE))
        if is_dialogue and mask_prompts:
            batches = self._prepare_training_batches(text, seq_len=seq_len, mask_prompts=mask_prompts, max_batches=max_batches)
            if not batches:
                raise ValueError("No training batches generated from corpus.")
            train_tokens = []
            starts = None
            total_tokens = len(batches) * seq_len
        else:
            train_tokens = encode_nonempty(text, "training text")
            total_tokens = len(train_tokens)
            batches = None

        if eval_text is not None and validation_split:
            raise ValueError("Pass either eval_text or validation_split, not both")

        if eval_text is not None:
            eval_tokens = encode_nonempty(eval_text, "evaluation text")
        elif validation_split > 0.0 and batches is None:
            split_at = int(len(train_tokens) * (1.0 - validation_split))
            split_at = min(max(split_at, seq_len + 1), len(train_tokens) - 1)
            eval_tokens = train_tokens[split_at:]
            train_tokens = train_tokens[:split_at]
            total_tokens = len(train_tokens)
        else:
            eval_tokens = None

        if batches is None:
            if len(train_tokens) < seq_len + 1:
                repeats = ((seq_len + 1) // len(train_tokens)) + 1
                train_tokens = train_tokens * repeats
            step = stride or seq_len
            if not 1 <= step <= seq_len:
                raise ValueError(f"stride must be in [1, seq_len], got {step}")
            starts = list(range(0, len(train_tokens) - seq_len, step))
            if not starts:
                raise ValueError("training text does not contain a usable sequence")
            if max_batches is not None:
                starts = starts[:max_batches]

        rng = np.random.default_rng(seed)
        history = {"loss": []}
        if eval_tokens is not None:
            history["val_loss"] = []

        total_steps_per_epoch = len(batches) if batches is not None else len(starts)
        total_steps = total_steps_per_epoch * epochs
        warmup_steps = max(1, int(0.05 * total_steps))
        min_lr = lr * 0.1
        log_step_cadence = min(25, max(1, total_steps_per_epoch // 20))

        history["tokens"] = total_tokens
        history["windows_per_epoch"] = total_steps_per_epoch
        history["total_steps"] = total_steps

        if verbose:
            print(f"🚀 Starting training: {epochs} epochs, {total_steps_per_epoch:,} steps/epoch ({total_steps:,} total steps)...", flush=True)

        global_step = 0
        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            steps = 0
            t_ep_start = time.perf_counter()
            last_print_time = t_ep_start

            if batches is not None:
                epoch_batches = list(batches)
                if shuffle:
                    rng.shuffle(epoch_batches)
                batch_iter = epoch_batches
            else:
                order = starts.copy()
                if shuffle:
                    rng.shuffle(order)
                batch_iter = [
                    (train_tokens[idx : idx + seq_len], train_tokens[idx + 1 : idx + seq_len + 1])
                    for idx in order
                ]

            for in_seq, target_seq in batch_iter:
                # Cosine learning rate schedule with linear warmup
                if global_step < warmup_steps:
                    cur_lr = lr * ((global_step + 1) / warmup_steps)
                else:
                    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                    cur_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))

                loss = self.train_step(in_seq, target_seq, lr=cur_lr, weight_decay=weight_decay)
                ep_loss += loss
                steps += 1
                global_step += 1

                now = time.perf_counter()
                should_log = (
                    steps in (1, 2, 3, 5, 10, 20, 50)
                    or (steps % log_step_cadence == 0)
                    or (steps == total_steps_per_epoch)
                    or (now - last_print_time >= 3.0)
                )

                if verbose and should_log:
                    last_print_time = now
                    elapsed = max(1e-4, now - t_ep_start)
                    tok_sec = (steps * seq_len) / elapsed
                    rem_steps = total_steps_per_epoch - steps
                    eta_sec = rem_steps / max(1e-4, steps / elapsed)
                    pct = (steps / total_steps_per_epoch) * 100.0
                    print(
                        f"  [Epoch {ep}/{epochs}] Step {steps:,}/{total_steps_per_epoch:,} ({pct:5.1f}%) "
                        f"| Loss: {loss:6.4f} | LR: {cur_lr:.2e} | {tok_sec:,.0f} tok/s | ETA: {eta_sec:.0f}s",
                        flush=True
                    )

            avg_loss = ep_loss / max(1, steps)
            history["loss"].append(avg_loss)
            if eval_tokens is not None:
                step = stride or seq_len
                val_loss = self._evaluate_tokens(eval_tokens, seq_len, stride=step, max_steps=max_eval_steps)
                history["val_loss"].append(val_loss)

            ep_time = time.perf_counter() - t_ep_start
            if verbose:
                suffix = ""
                if eval_tokens is not None:
                    suffix = f" | Val Loss: {history['val_loss'][-1]:.4f}"
                print(
                    f"✓ Epoch {ep:2d}/{epochs} finished in {ep_time:.1f}s | "
                    f"Avg Loss: {avg_loss:.4f}{suffix} (Engine: {self.backend.upper()})\n",
                    flush=True,
                )

        return history

    def save(self, filepath: Union[str, Path]) -> Path:
        """Save fine-tuned model checkpoint back to a .bin file."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        p = self.config

        vocab_sign = p.vocab_size if self.shared_weights else -p.vocab_size
        header = struct.pack(
            "<7i",
            p.dim,
            p.hidden_dim,
            p.n_layers,
            p.n_heads,
            p.n_kv_heads,
            vocab_sign,
            p.seq_len,
        )

        with open(out_path, "wb") as f:
            f.write(header)
            self.tok_emb.astype(np.float32).tofile(f)
            self.rms_att.astype(np.float32).tofile(f)
            self.wq.astype(np.float32).tofile(f)
            self.wk.astype(np.float32).tofile(f)
            self.wv.astype(np.float32).tofile(f)
            self.wo.astype(np.float32).tofile(f)
            self.rms_ffn.astype(np.float32).tofile(f)
            self.w1.astype(np.float32).tofile(f)
            self.w2.astype(np.float32).tofile(f)
            self.w3.astype(np.float32).tofile(f)
            self.rms_final.astype(np.float32).tofile(f)

            padding = np.zeros(p.seq_len * p.head_size, dtype=np.float32)
            padding.tofile(f)

            if not self.shared_weights and self.wcls is not self.tok_emb:
                self.wcls.astype(np.float32).tofile(f)

        return out_path

def _fetch_file(url: str, dest: Path) -> None:
    """Download a single file from a URL with progress display."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"Downloading {dest.name} from Hugging Face...")
    req = urllib.request.Request(url, headers={"User-Agent": "doraneural/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 65536
        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                print(f"\r  Progress: {pct:5.1f}% ({downloaded / 1024 / 1024:.2f} MB)", end="", flush=True)
    print(f"\nSaved to {dest}")


def download_hf_model(model_name: str = "stories260K", target_dir: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
    """Download pretrained weights and tokenizer from Hugging Face Hub.

    Supports two modes:
      1. Built-in registry names: "stories260K", "stories15M" (llama2.c .bin format)
      2. Any HuggingFace repo: "owner/model" (e.g. "arnir0/Tiny-LLM") — auto-detects
         safetensors + tokenizer.json format.

    Args:
        model_name: Built-in name or HuggingFace "owner/model" identifier.
        target_dir: Directory to save files. Defaults to ~/.cache/doraneural/models/<model_name>.

    Returns:
        Tuple of (model_path, tokenizer_path).
    """
    # Built-in registry path
    if model_name in HF_REPOSITORIES:
        meta = HF_REPOSITORIES[model_name]
        if target_dir is None:
            save_dir = Path.home() / ".cache" / "doraneural" / "models" / model_name
        else:
            save_dir = Path(target_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / meta["model_file"]
        tok_path = save_dir / meta["tok_file"]
        _fetch_file(meta["model_url"], model_path)
        _fetch_file(meta["tok_url"], tok_path)
        return model_path, tok_path

    # Arbitrary HuggingFace repo: "owner/model"
    if "/" not in model_name:
        available = list(HF_REPOSITORIES.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Built-in: {available}. "
            f"For HuggingFace repos, use 'owner/model' format (e.g. 'arnir0/Tiny-LLM')."
        )

    repo_id = model_name
    safe_name = repo_id.replace("/", "_")
    if target_dir is None:
        save_dir = Path.home() / ".cache" / "doraneural" / "models" / safe_name
    else:
        save_dir = Path(target_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    # Always download config.json
    config_path = save_dir / "config.json"
    _fetch_file(f"{base_url}/config.json", config_path)

    # Detect model file format. Prefer a single safetensors file, then fall
    # back to HuggingFace's sharded index and download every referenced shard.
    model_path = save_dir / "model.safetensors"
    try:
        _fetch_file(f"{base_url}/model.safetensors", model_path)
    except Exception:
        model_path = save_dir / "model.safetensors.index.json"
        _fetch_file(f"{base_url}/model.safetensors.index.json", model_path)
        with model_path.open("r", encoding="utf-8") as handle:
            shard_names = sorted(set(json.load(handle).get("weight_map", {}).values()))
        for shard_name in shard_names:
            _fetch_file(f"{base_url}/{shard_name}", save_dir / shard_name)

    # Try tokenizer.json (HuggingFace standard)
    tok_path = save_dir / "tokenizer.json"
    if not tok_path.exists():
        try:
            _fetch_file(f"{base_url}/tokenizer.json", tok_path)
        except Exception:
            # Fallback to tokenizer.model (SentencePiece binary)
            tok_path = save_dir / "tokenizer.model"
            _fetch_file(f"{base_url}/tokenizer.model", tok_path)

    return model_path, tok_path


def load_pretrained_llm(
    model_name: str = "stories260K",
    cache_dir: Optional[Union[str, Path]] = None,
    backend: str = "auto",
) -> LlamaLLM:
    """Load a ready-to-run pretrained LLaMA model from Hugging Face.

    Args:
        model_name: Built-in name ("stories260K", "stories15M") or
                     HuggingFace "owner/model" (e.g. "arnir0/Tiny-LLM").
        cache_dir: Optional directory for model files.
        backend: Execution engine ("auto", "cpp", or "numpy").

    Returns:
        LlamaLLM: Initialized model instance ready for .generate().
    """
    model_path, tok_path = download_hf_model(model_name=model_name, target_dir=cache_dir)
    return LlamaLLM(model_path=model_path, tokenizer_path=tok_path, backend=backend)
