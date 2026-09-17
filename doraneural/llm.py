"""Pure NumPy Pretrained LLaMA Language Model Engine.

Implements full autoregressive Transformer decoder inference in pure NumPy with:
- RoPE (Rotary Position Embeddings)
- RMSNorm
- Multi-Head Attention with KV-Caching
- SwiGLU Feed-Forward Network
- SentencePiece Byte-Fallback BPE Tokenizer
- Nucleus (Top-p) & Temperature Sampling
- Streaming Generation Generator
- Automatic download from Hugging Face Hub (karpathy/tinyllamas)
"""

import os
import struct
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

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

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return (self.dim * self.n_kv_heads) // self.n_heads


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

    def encode(self, text: str, bos: bool = True) -> List[int]:
        """Encode a string into a list of token IDs with BPE merge loop."""
        tokens: List[int] = []
        if bos:
            tokens.append(1)  # BOS token

        if text:
            # Add SentencePiece leading space prefix
            dummy = self.str_to_id.get(" ", self.str_to_id.get(" ", None))
            if dummy is not None:
                tokens.append(dummy)

        for ch in text:
            if ch in self.str_to_id:
                tokens.append(self.str_to_id[ch])
            else:
                for b in ch.encode("utf-8"):
                    tokens.append(b + 3)

        # Iteratively merge highest-scoring adjacent pairs
        while len(tokens) >= 2:
            best_score = -1e10
            best_id = -1
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair_str = self.vocab[tokens[i]] + self.vocab[tokens[i + 1]]
                if pair_str in self.str_to_id:
                    candidate_id = self.str_to_id[pair_str]
                    candidate_score = self.scores[candidate_id]
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_id = candidate_id
                        best_idx = i

            if best_idx == -1:
                break

            tokens[best_idx] = best_id
            tokens.pop(best_idx + 1)

        return tokens

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


class LlamaLLM:
    """Pure NumPy LLaMA transformer model for autoregressive text generation."""

    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        backend: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # 1. Read config header
        with open(self.model_path, "rb") as f:
            header = f.read(28)
            dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack("<7i", header)
            # Handle shared weights convention (negative vocab_size)
            shared_weights = vocab_size > 0
            vocab_size = abs(vocab_size)

            self.config = LlamaConfig(
                dim=dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                vocab_size=vocab_size,
                seq_len=seq_len,
            )

            # Read all remaining float32 parameters
            raw = np.fromfile(f, dtype=np.float32)

        # 2. Slice weights according to llama2.c binary specification
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

        # Skip legacy frequency table floats
        offset += p.seq_len * p.head_size

        # Output classifier weights
        if shared_weights or offset >= len(raw):
            self.wcls = self.tok_emb
        else:
            self.wcls = take((p.vocab_size, p.dim))

        # 3. Load Tokenizer
        self.tokenizer = LlamaTokenizer(self.tokenizer_path, vocab_size=p.vocab_size)

        # 4. Allocate Key-Value Cache
        self.key_cache = np.zeros((p.n_layers, p.seq_len, p.kv_dim), dtype=np.float32)
        self.val_cache = np.zeros((p.n_layers, p.seq_len, p.kv_dim), dtype=np.float32)

        # 5. C++ Backend Integration
        self.shared_weights = bool(shared_weights)
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
            "shared_classifier": int(shared_weights),
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
            for i in range(0, p.dim, 2):
                h_dim = i % p.head_size
                freq = 1.0 / (10000.0 ** (h_dim / p.head_size))
                val = pos * freq
                fcr, fci = np.cos(val), np.sin(val)

                q0, q1 = q[i], q[i + 1]
                q[i] = q0 * fcr - q1 * fci
                q[i + 1] = q0 * fci + q1 * fcr

                if i < p.kv_dim:
                    k0, k1 = k[i], k[i + 1]
                    k[i] = k0 * fcr - k1 * fci
                    k[i + 1] = k0 * fci + k1 * fcr

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
        return self.wcls @ x

    def sample(self, logits: np.ndarray, temperature: float = 0.7, top_p: float = 0.9) -> int:
        """Sample next token from logits using nucleus (top-p) and temperature."""
        if temperature <= 0.0:
            return int(np.argmax(logits))

        logits = logits / max(1e-4, temperature)
        probs = self._softmax(logits)

        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum = np.cumsum(sorted_probs)

            # Mask out probabilities beyond top_p cutoff
            cutoff_mask = cumsum > top_p
            cutoff_mask[0] = False  # Keep at least the top-1 choice
            sorted_probs[cutoff_mask] = 0.0
            sorted_probs /= np.sum(sorted_probs)

            selected_idx = np.random.choice(len(sorted_probs), p=sorted_probs)
            return int(sorted_indices[selected_idx])

        return int(np.random.choice(len(probs), p=probs))

    def generate(
        self,
        prompt: str = "Once upon a time",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Autoregressively generate text from a prompt.

        Args:
            prompt: Seed text string.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (>0.0 for creativity, 0.0 for deterministic greedy).
            top_p: Nucleus sampling probability cutoff (0.0 to 1.0).
            stream: If True, returns a generator yielding pieces as they are produced.

        Returns:
            Generated text string, or Generator of token strings.
        """
        self.reset_cache()
        prompt_tokens = self.tokenizer.encode(prompt, bos=True)

        if not prompt_tokens:
            prompt_tokens = [1]

        def _generator() -> Generator[str, None, None]:
            pos = 0
            logits = None

            # Pre-fill prompt
            for tok in prompt_tokens:
                logits = self.forward(tok, pos)
                pos += 1

            for _ in range(max_tokens):
                if pos >= self.config.seq_len - 1:
                    break

                next_token = self.sample(logits, temperature=temperature, top_p=top_p)
                if next_token == 2:  # EOS
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
    ) -> float:
        """Pure NumPy fallback training step."""
        seq_len = len(input_tokens)
        self.reset_cache()
        total_loss = 0.0

        for pos in range(seq_len):
            in_tok = input_tokens[pos]
            target_tok = target_tokens[pos]
            logits = self.forward(in_tok, pos)

            e = np.exp(logits - np.max(logits))
            probs = e / np.sum(e)

            target_prob = max(1e-12, float(probs[target_tok]))
            total_loss += -np.log(target_prob)

            dlogits = probs.copy()
            dlogits[target_tok] -= 1.0
            dlogits /= seq_len

            cls_w = self.wcls if not self.shared_weights else self.tok_emb
            g_emb = cls_w.T @ dlogits
            self.tok_emb[in_tok] -= lr * (g_emb + weight_decay * self.tok_emb[in_tok])

        return total_loss / seq_len

    def train(
        self,
        text: str,
        epochs: int = 3,
        lr: float = 1e-4,
        seq_len: int = 32,
        weight_decay: float = 0.01,
        verbose: int = 1,
    ) -> dict:
        """Fine-tune the model on custom text with cross-entropy loss and in-place updates.

        Args:
            text: Training text corpus.
            epochs: Number of complete passes over the text.
            lr: Learning rate for parameter updates.
            seq_len: Chunk length for training sequences.
            weight_decay: L2 regularization factor.
            verbose: 1 to print epoch progress, 0 to silence.

        Returns:
            Dictionary containing 'loss' history list.
        """
        tokens = self.tokenizer.encode(text, bos=False)
        if len(tokens) < seq_len + 1:
            tokens = tokens * ((seq_len + 2) // max(1, len(tokens)) + 1)

        history = {"loss": []}

        for ep in range(1, epochs + 1):
            ep_loss = 0.0
            steps = 0
            for i in range(0, len(tokens) - seq_len, seq_len):
                in_seq = tokens[i : i + seq_len]
                target_seq = tokens[i + 1 : i + seq_len + 1]
                loss = self.train_step(in_seq, target_seq, lr=lr, weight_decay=weight_decay)
                ep_loss += loss
                steps += 1

            avg_loss = ep_loss / max(1, steps)
            history["loss"].append(avg_loss)
            if verbose:
                print(f"Epoch {ep:2d}/{epochs} | Loss: {avg_loss:.4f} (Engine: {self.backend.upper()})")

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


def download_hf_model(model_name: str = "stories260K", target_dir: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
    """Download pretrained weights and tokenizer from Hugging Face Hub.

    Args:
        model_name: "stories260K" (1MB) or "stories15M" (60MB).
        target_dir: Directory to save the checkpoint and tokenizer. Defaults to ~/.cache/doraneural/models/<model_name>.

    Returns:
        Tuple of (model_path, tokenizer_path).
    """
    if model_name not in HF_REPOSITORIES:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(HF_REPOSITORIES.keys())}")

    meta = HF_REPOSITORIES[model_name]
    if target_dir is None:
        save_dir = Path.home() / ".cache" / "doraneural" / "models" / model_name
    else:
        save_dir = Path(target_dir)

    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / meta["model_file"]
    tok_path = save_dir / meta["tok_file"]

    def _fetch(url: str, dest: Path) -> None:
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

    _fetch(meta["model_url"], model_path)
    _fetch(meta["tok_url"], tok_path)

    return model_path, tok_path


def load_pretrained_llm(
    model_name: str = "stories260K",
    cache_dir: Optional[Union[str, Path]] = None,
    backend: str = "auto",
) -> LlamaLLM:
    """Load a ready-to-run pretrained LLaMA model from Hugging Face.

    Args:
        model_name: Pretrained model name ("stories260K" or "stories15M").
        cache_dir: Optional directory for model files.
        backend: Execution engine ("auto", "cpp", or "numpy").

    Returns:
        LlamaLLM: Initialized model instance ready for .generate().
    """
    model_path, tok_path = download_hf_model(model_name=model_name, target_dir=cache_dir)
    return LlamaLLM(model_path=model_path, tokenizer_path=tok_path, backend=backend)
