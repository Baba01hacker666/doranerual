"""Zexo AI: Conversational Model & Engine Wrapper.

Wraps the underlying high-performance LLaMA execution engine with Zexo's
conversational persona, multi-turn sliding context window, and generation routines.
"""

from pathlib import Path
from typing import Generator, List, Optional, Union
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doraneural.llm import LlamaLLM, HFTokenizer, LlamaTokenizer
from doraneural.chat import ChatSession
from zexo.config import ZexoConfig
from zexo.persona import ZEXO_SYSTEM_PROMPT, ZEXO_STOP_SEQUENCES
from zexo.init_weights import initialize_zexo_checkpoint


class ZexoModel:
    """The Zexo Autonomous Conversational AI Model."""

    def __init__(
        self,
        llm: LlamaLLM,
        config: Optional[ZexoConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.config = config or ZexoConfig.from_tier("chat")
        self.system_prompt = system_prompt or self.config.system_prompt or ZEXO_SYSTEM_PROMPT

        self.chat_session = ChatSession(
            llm=self.llm,
            system_prompt=self.system_prompt,
            stop_sequences=ZEXO_STOP_SEQUENCES,
        )

    def chat(
        self,
        user_input: str,
        temperature: float = 0.7,
        max_tokens: int = 80,
        stream: bool = False,
    ) -> Union[str, any]:
        """Conduct a single conversational turn with Zexo, retaining multi-turn history."""
        self.chat_session.temperature = float(temperature)
        self.chat_session.max_new_tokens = int(max_tokens)
        if stream:
            return self.chat_session.stream_chat(user_input)
        return self.chat_session.chat(user_input)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Run raw autoregressive token completion."""
        return self.llm.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature, stream=stream)

    def train(
        self,
        text: str,
        epochs: int = 3,
        lr: float = 5e-4,
        seq_len: int = 16,
        verbose: int = 1,
    ) -> dict:
        """Fine-tune Zexo on custom text or conversational dialogue."""
        return self.llm.train(text, epochs=epochs, lr=lr, seq_len=seq_len, verbose=verbose)

    def reset(self) -> None:
        """Clear active conversation context and model KV cache."""
        self.chat_session.clear()

    def save(self, filepath: Union[str, Path]) -> Path:
        """Save Zexo checkpoint weights (.bin) and configuration (.json)."""
        out_bin = Path(filepath)
        out_bin.parent.mkdir(parents=True, exist_ok=True)
        self.llm.save(out_bin)
        cfg_path = out_bin.with_suffix(".json")
        self.config.save(cfg_path)
        return out_bin


def load_zexo(
    checkpoint_path: Optional[Union[str, Path]] = None,
    tokenizer_path: Optional[Union[str, Path]] = None,
    tier: str = "micro",
    backend: str = "auto",
    from_scratch: bool = False,
) -> ZexoModel:
    """Load or initialize a ready-to-run Zexo model instance."""
    repo_root = REPO_ROOT

    # 1. Resolve Tokenizer
    tok_p = Path(tokenizer_path) if tokenizer_path else None
    if not tok_p or not tok_p.exists():
        if tier == "micro":
            tok_p = repo_root / "models" / "stories260K" / "tok512.bin"
        else:
            tok_p = repo_root / "models" / "arnir0_Tiny-LLM" / "tokenizer.json"
            if not tok_p.exists():
                tok_p = repo_root / "models" / "stories260K" / "tok512.bin"

    # 2. Check explicit checkpoint path
    ckpt_p = Path(checkpoint_path) if checkpoint_path else None
    if ckpt_p and ckpt_p.exists() and not from_scratch:
        llm = LlamaLLM(model_path=ckpt_p, tokenizer_path=tok_p, backend=backend)
        cfg_file = ckpt_p.with_suffix(".json")
        cfg = ZexoConfig.load(cfg_file) if cfg_file.exists() else ZexoConfig.from_tier(tier)
        return ZexoModel(llm=llm, config=cfg)

    # 3. Check for existing zexo checkpoint in zexo/checkpoints/ (if not from_scratch)
    latest_zexo = repo_root / "zexo" / "checkpoints" / "latest.bin"
    if latest_zexo.exists() and not from_scratch:
        cfg_file = latest_zexo.with_suffix(".json")
        loaded_cfg = ZexoConfig.load(cfg_file) if cfg_file.exists() else None
        if loaded_cfg is None or loaded_cfg.tier == tier:
            cfg = loaded_cfg or ZexoConfig.from_tier(tier)
            llm = LlamaLLM(model_path=latest_zexo, tokenizer_path=tok_p, backend=backend)
            return ZexoModel(llm=llm, config=cfg)

    # 4. Check or initialize base weights for the requested tier (no story dependency)
    cfg = ZexoConfig.from_tier(tier)
    fresh_bin = repo_root / "zexo" / "checkpoints" / f"zexo_{tier}_base.bin"
    if not fresh_bin.exists():
        initialize_zexo_checkpoint(cfg, fresh_bin)
    llm = LlamaLLM(model_path=fresh_bin, tokenizer_path=tok_p, backend=backend)
    return ZexoModel(llm=llm, config=cfg)
