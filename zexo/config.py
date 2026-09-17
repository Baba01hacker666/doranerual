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
    system_prompt: str = "You are Zexo, an intelligent, thoughtful, and creative conversational AI assistant."
    version: str = "1.0.0"

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
        final_norm = self.dim
        return emb + (per_layer * self.n_layers) + final_norm

    @property
    def llama_config(self) -> LlamaConfig:
        """Convert to doraneural's native LlamaConfig for engine execution."""
        return LlamaConfig(
            dim=self.dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            rope_type=self.rope_type,
        )

    # ---------------- Predefined Architectural Tiers ----------------

    @classmethod
    def micro(cls) -> "ZexoConfig":
        """Zexo-Micro (280K parameters): Fast CPU testing, prototyping, and CI runners."""
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
    def from_tier(cls, tier: str) -> "ZexoConfig":
        """Factory constructor by tier name string."""
        t = tier.lower().strip()
        if t == "micro":
            return cls.micro()
        elif t == "mini":
            return cls.mini()
        elif t in ("chat", "small"):
            return cls.chat()
        elif t in ("base", "medium"):
            return cls.base()
        elif t == "large":
            return cls.large()
        else:
            raise ValueError(f"Unknown Zexo tier '{tier}'. Available: micro, mini, chat, base, large.")

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
