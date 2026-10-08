"""Zexo AI: Architectural Configurations & Scalable Tiers.

Defines the hyperparameter specifications for the Zexo conversational AI,
spanning ultra-compact prototyping tiers to multi-layer conversational models.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Dict, Optional, Union
import sys

# Ensure doraneural can be imported from parent directory
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doraneural.llm import LlamaConfig

# Vocabulary-free byte tiers use raw UTF-8 byte ids (0-255) plus BOS and EOS.
# See research/10_recurrent_trace_units_latent.md for the derivation.
BYTE_VOCAB_SIZE = 258

@dataclass
class ZexoConfig:
    """Configuration hyperparameter blueprint for Zexo AI models."""

    name: str = "zexo"
    tier: str = "chat"
    dim: int = 384
    hidden_dim: int = 1024
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4  # Grouped Query Attention (2:1 ratio)
    vocab_size: int = 32000
    seq_len: int = 1024
    rope_type: str = "interleaved"
    novel_neurons: bool = False
    tokenizer: str = "bpe"  # "bpe" (SentencePiece) or "byte" (vocabulary-free)
    system_prompt: str = (
        "You are Zexo, an intelligent, thoughtful, and creative conversational AI assistant. "
        "You communicate with clarity, warmth, and precision. You love helping users explore ideas, "
        "solve challenging problems, explain complex topics simply, and hold engaging, meaningful conversations."
    )
    version: str = "1.0.0"

    @property
    def is_byte_level(self) -> bool:
        """True when this tier models raw UTF-8 bytes instead of BPE tokens."""
        return self.tokenizer == "byte"

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return (self.dim * self.n_kv_heads) // self.n_heads

    @property
    def parameter_count(self) -> int:
        """Exact parameter count assuming tied input/output embeddings."""
        emb = self.vocab_size * self.dim
        layer_attn = self.dim + (self.dim * self.dim) + 2 * (self.kv_dim * self.dim) + (self.dim * self.dim)
        layer_ffn = self.dim + 3 * (self.hidden_dim * self.dim)
        per_layer = layer_attn + layer_ffn
        if self.novel_neurons:
            novel_per_layer = (self.dim * self.hidden_dim) + (3 * self.hidden_dim) + (self.dim * self.dim)
            per_layer += novel_per_layer
        final_norm = self.dim
        return emb + (per_layer * self.n_layers) + final_norm

    @property
    def llama_config(self) -> LlamaConfig:
        """Convert to doraneural's native LlamaConfig for engine execution."""
        cfg = LlamaConfig(
            dim=self.dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            rope_type=self.rope_type,
        )
        cfg.novel_neurons = self.novel_neurons
        if self.is_byte_level:
            # Byte tiers reserve ids 256/257 for BOS/EOS; the LLaMA default of 2
            # would otherwise treat a raw control byte as end-of-text.
            cfg.eos_token_id = 257
        return cfg

    # ---------------- Predefined Architectural Tiers ----------------

    @classmethod
    def micro(cls) -> "ZexoConfig":
        """Zexo-Micro (260K parameters): Ultra-lightweight CPU prototyping model."""
        return cls(
            tier="micro",
            dim=64,
            hidden_dim=172,
            n_layers=5,
            n_heads=4,
            n_kv_heads=4,
            vocab_size=512,
            seq_len=256,
        )

    @classmethod
    def mini(cls) -> "ZexoConfig":
        """Zexo-Mini (6.8M parameters): High-efficiency 32K-vocab model for fast CPU chat."""
        return cls(
            tier="mini",
            dim=192,
            hidden_dim=1024,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            vocab_size=32000,
            seq_len=1024,
        )

    @classmethod
    def chat(cls) -> "ZexoConfig":
        """Zexo-Chat (25.3M parameters): Balanced conversational model with GQA."""
        return cls(
            tier="chat",
            dim=384,
            hidden_dim=1024,
            n_layers=8,
            n_heads=8,
            n_kv_heads=4,
            vocab_size=32000,
            seq_len=1024,
        )

    @classmethod
    def dora(cls) -> "ZexoConfig":
        """Zexo-Dora (54M parameters): 12-layer Bio-Reflective KAN Novel-Neuron Transformer."""
        return cls(
            tier="dora",
            dim=512,
            hidden_dim=1536,
            n_layers=12,
            n_heads=8,
            n_kv_heads=4,
            vocab_size=32000,
            seq_len=1024,
            novel_neurons=True,
        )

    @classmethod
    def base(cls) -> "ZexoConfig":
        """Zexo-Base (109.5M parameters): Standard conversational base capable of deep reasoning."""
        return cls(
            tier="base",
            dim=768,
            hidden_dim=2048,
            n_layers=12,
            n_heads=12,
            n_kv_heads=12,
            vocab_size=32000,
            seq_len=2048,
        )

    @classmethod
    def large(cls) -> "ZexoConfig":
        """Zexo-Large (221.5M parameters): Scaled architecture with 4K context and GQA."""
        return cls(
            tier="large",
            dim=1024,
            hidden_dim=2816,
            n_layers=16,
            n_heads=16,
            n_kv_heads=8,
            vocab_size=32000,
            seq_len=4096,
        )

    @classmethod
    def micro_byte(cls) -> "ZexoConfig":
        """Byte-level Zexo-Micro: same transformer body as `micro`, 258-token vocab."""
        return cls(
            tier="micro-byte",
            dim=64,
            hidden_dim=172,
            n_layers=5,
            n_heads=4,
            n_kv_heads=4,
            vocab_size=BYTE_VOCAB_SIZE,
            seq_len=256,
            tokenizer="byte",
        )

    @classmethod
    def mini_byte(cls) -> "ZexoConfig":
        """Byte-level Zexo-Mini: identical body to `mini` with a 258-token vocab.

        This is the single largest parameter saving in the ladder: the BPE tier
        spends 89.8% of its 6.8M parameters on the embedding matrix, while the
        byte tier holds the same 700,992 transformer parameters in 750K total.
        """
        return cls(
            tier="mini-byte",
            dim=192,
            hidden_dim=1024,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            vocab_size=BYTE_VOCAB_SIZE,
            seq_len=512,
            tokenizer="byte",
        )

    @classmethod
    def chat_byte(cls) -> "ZexoConfig":
        """Byte-level Zexo-Chat: identical body to `chat` with a 258-token vocab.

        Halves the parameter count (25.3M -> 13.1M) without touching any
        transformer capacity. Freed budget can be spent on depth if desired.
        """
        return cls(
            tier="chat-byte",
            dim=384,
            hidden_dim=1024,
            n_layers=8,
            n_heads=8,
            n_kv_heads=4,
            vocab_size=BYTE_VOCAB_SIZE,
            seq_len=1024,
            tokenizer="byte",
        )

    @classmethod
    def base_byte(cls) -> "ZexoConfig":
        """Byte-level Zexo-Base: identical body to `base` with a 258-token vocab."""
        return cls(
            tier="base-byte",
            dim=768,
            hidden_dim=2048,
            n_layers=12,
            n_heads=12,
            n_kv_heads=12,
            vocab_size=BYTE_VOCAB_SIZE,
            seq_len=2048,
            tokenizer="byte",
        )

    @classmethod
    def from_tier(cls, tier: str) -> "ZexoConfig":
        """Factory constructor by tier name string."""
        t = tier.lower().strip()
        byte_tiers = {
            "micro-byte": cls.micro_byte,
            "microbyte": cls.micro_byte,
            "mini-byte": cls.mini_byte,
            "minibyte": cls.mini_byte,
            "chat-byte": cls.chat_byte,
            "chatbyte": cls.chat_byte,
            "small-byte": cls.chat_byte,
            "base-byte": cls.base_byte,
            "basebyte": cls.base_byte,
        }
        if t in byte_tiers:
            return byte_tiers[t]()
        if t == "micro":
            return cls.micro()
        elif t == "mini":
            return cls.mini()
        elif t in ("chat", "small"):
            return cls.chat()
        elif t in ("dora", "neuro", "dora_12"):
            return cls.dora()
        elif t in ("base", "medium"):
            return cls.base()
        elif t == "large":
            return cls.large()
        else:
            raise ValueError(
                f"Unknown Zexo tier '{tier}'. Available: micro, mini, chat, dora, base, large, "
                "or the vocabulary-free byte tiers micro-byte, mini-byte, chat-byte, base-byte."
            )

    def to_dict(self) -> Dict[str, any]:
        d = asdict(self)
        d["parameters"] = self.parameter_count
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "ZexoConfig":
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, filepath: Union[str, Path]) -> Path:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ZexoConfig":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
