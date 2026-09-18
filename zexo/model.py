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
        weight_decay: float = 0.01,
        seq_len: int = 64,
        verbose: int = 1,
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
        lora_targets: Optional[List[str]] = None,
        adapter_path: Optional[Union[str, Path]] = None,
    ) -> dict:
        """Fine-tune Zexo, optionally differentiating the complete transformer."""
        return self.llm.train(
            text,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seq_len=seq_len,
            verbose=verbose,
            eval_text=eval_text,
            validation_split=validation_split,
            stride=stride,
            shuffle=shuffle,
            seed=seed,
            max_eval_steps=max_eval_steps,
            full_backprop=full_backprop,
            native_full=native_full,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
            adapter_path=adapter_path,
        )

    def full_backprop_model(self):
        """Expose the CPU autograd model for advanced full-layer training."""
        return self.llm.full_backprop_model()

    def load_lora(self, adapter_path: Union[str, Path]):
        """Load a saved LoRA adapter on top of the current pretrained checkpoint."""
        return self.llm.load_lora(adapter_path)

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
            candidates = [
                repo_root / "zexo" / "tokenizer" / "tok512.bin",
                repo_root / "models" / "stories260K" / "tok512.bin",
            ]
        else:
            candidates = [
                repo_root / "zexo" / "tokenizer" / "tokenizer.json",
                repo_root / "models" / "arnir0_Tiny-LLM" / "tokenizer.json",
                repo_root / "zexo" / "tokenizer" / "tok512.bin",
                repo_root / "models" / "stories260K" / "tok512.bin",
            ]
        for c in candidates:
            if c.exists():
                tok_p = c
                break

    if not tok_p or not tok_p.exists():
        import urllib.request
        tok_dir = repo_root / "zexo" / "tokenizer"
        tok_dir.mkdir(parents=True, exist_ok=True)
        if tier == "micro":
            tok_p = tok_dir / "tok512.bin"
            url = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin"
        else:
            tok_p = tok_dir / "tokenizer.json"
            url = "https://huggingface.co/arnir0/Tiny-LLM/resolve/main/tokenizer.json"
        try:
            print(f"📥 Auto-downloading tokenizer: {tok_p.name}...")
            urllib.request.urlretrieve(url, tok_p)
        except Exception as err:
            print(f"⚠️ Tokenizer download failed: {err}")

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
