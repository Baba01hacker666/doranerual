#!/usr/bin/env python3
"""Refine a trained Zexo checkpoint from explicit preference pairs with DPO.

Example::

    python zexo/refine.py \
        --checkpoint zexo/checkpoints/sft.bin \
        --preferences preferences/train.jsonl \
        --eval-preferences preferences/eval.jsonl \
        --output zexo/checkpoints/refined.bin

The reference policy is copied from the input checkpoint and remains frozen.
Use reviewed chosen/rejected pairs; this command does not invent its own reward
signal or recursively promote unverified model-generated answers.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import math
import json
from pathlib import Path
import sys
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doraneural.transformer import TransformerDecoderLM
from zexo.config import ZexoConfig
from zexo.model import load_zexo
from zexo.preferences import PreferenceTrainer, read_preferences


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DPO-align a trained Zexo checkpoint from chosen/rejected JSONL pairs."
    )
    parser.add_argument("--checkpoint", "-c", required=True, help="Existing trained base/SFT checkpoint (.bin).")
    parser.add_argument("--preferences", required=True, help="Training JSONL with prompt/chosen/rejected fields.")
    parser.add_argument("--eval-preferences", default=None, help="Optional held-out preference JSONL for best-epoch selection.")
    parser.add_argument("--output", "-o", default="zexo/checkpoints/refined.bin", help="Output merged checkpoint path.")
    parser.add_argument("--tier", default="chat", help="Fallback architecture tier if checkpoint metadata is absent.")
    parser.add_argument("--backend", choices=("auto", "cpp", "numpy"), default="numpy", help="Checkpoint loader backend.")
    parser.add_argument("--epochs", type=int, default=1, help="Preference-training epochs (default: 1).")
    parser.add_argument("--lr", type=float, default=5e-6, help="DPO learning rate (default: 5e-6).")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature/regularization coefficient (default: 0.1).")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (default: 0).")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Global gradient clipping norm (0 disables).")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Maximum prompt plus response tokens (defaults to model context).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42).")
    parser.add_argument("--lora-rank", type=int, default=8, help="Train LoRA adapters at this rank; use --full-model for all weights.")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha (default: 16).")
    parser.add_argument("--lora-targets", default="q,v", help="Comma-separated LoRA targets (default: q,v).")
    parser.add_argument("--full-model", action="store_true", help="DPO-train every model weight instead of LoRA adapters.")
    parser.add_argument("--adapter-output", default=None, help="Optional path for saving LoRA adapter-only weights before merging.")
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint_path}")
    preference_path = Path(args.preferences)
    eval_path = Path(args.eval_preferences) if args.eval_preferences else None
    if not preference_path.is_file():
        parser.error(f"preference data does not exist: {preference_path}")
    if eval_path is not None and not eval_path.is_file():
        parser.error(f"evaluation preference data does not exist: {eval_path}")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if not math.isfinite(args.lr) or not math.isfinite(args.beta) or args.lr <= 0.0 or args.beta <= 0.0:
        parser.error("--lr and --beta must be finite and positive")
    if (not math.isfinite(args.weight_decay) or not math.isfinite(args.grad_clip)
            or args.weight_decay < 0.0 or args.grad_clip < 0.0):
        parser.error("--weight-decay and --grad-clip must be finite and non-negative")
    if args.max_seq_len is not None and args.max_seq_len < 2:
        parser.error("--max-seq-len must be at least 2")
    if not args.full_model and args.lora_rank <= 0:
        parser.error("--lora-rank must be positive (or pass --full-model)")
    if not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0.0:
        parser.error("--lora-alpha must be finite and positive")
    if args.full_model and args.adapter_output:
        parser.error("--adapter-output is only used with LoRA; remove it or omit --full-model")

    parent_sha256 = _sha256(checkpoint_path)
    preference_sha256 = _sha256(preference_path)
    eval_preference_sha256 = _sha256(eval_path) if eval_path is not None else None
    if eval_preference_sha256 is not None and eval_preference_sha256 == preference_sha256:
        parser.error("--eval-preferences must contain held-out data, not the training file contents")
    train_examples = read_preferences(preference_path)
    eval_examples = read_preferences(eval_path) if eval_path is not None else None

    config_path = checkpoint_path.with_suffix(".json")
    tier = args.tier
    if config_path.exists():
        try:
            tier = ZexoConfig.load(config_path).tier
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    print("=" * 72)
    print("Zexo preference refinement (offline DPO)")
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Training pairs:  {len(train_examples):,}")
    print(f"Eval pairs:      {len(eval_examples):,}" if eval_examples is not None else "Eval pairs:      none (no best-epoch selection)")
    print(f"Tier:            {tier}")
    print(f"Training mode:   {'full model' if args.full_model else f'LoRA rank {args.lora_rank}'}")
    print("=" * 72)

    zexo = load_zexo(checkpoint_path=checkpoint_path, tier=tier, backend=args.backend)
    policy = zexo.full_backprop_model()
    # Create an independent reference copy from the untouched loaded checkpoint.
    reference = TransformerDecoderLM.from_llama(zexo.llm)
    for parameter in reference.parameters():
        parameter.requires_grad = False
    if not args.full_model:
        targets = [target.strip() for target in args.lora_targets.split(",") if target.strip()]
        policy.enable_lora(rank=args.lora_rank, alpha=args.lora_alpha, target_modules=targets)

    trainer = PreferenceTrainer(
        policy_model=policy,
        reference_model=reference,
        tokenizer=zexo.llm.tokenizer,
        system_prompt=zexo.system_prompt,
        beta=args.beta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_seq_len=args.max_seq_len,
    )
    history = trainer.fit(
        train_examples,
        eval_examples=eval_examples,
        epochs=args.epochs,
        seed=args.seed,
        verbose=1,
    )

    output_path = Path(args.output)
    protected_paths = {preference_path.resolve()}
    if eval_path is not None:
        protected_paths.add(eval_path.resolve())
    if output_path.resolve() in protected_paths:
        parser.error("--output must not overwrite preference data")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path = None
    if not args.full_model:
        if args.adapter_output:
            adapter_path = Path(args.adapter_output)
        else:
            adapter_path = output_path.with_name(output_path.stem + ".adapter.npz")
        adapter_path = policy.save_lora(adapter_path)
        policy.merge_lora()
    policy.copy_to_llama(zexo.llm)
    zexo.save(output_path)

    metadata = {
        "format_version": 1,
        "method": "DPO",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": str(checkpoint_path),
        "parent_sha256": parent_sha256,
        "preference_data": str(preference_path),
        "preference_data_sha256": preference_sha256,
        "eval_preference_data": str(eval_path) if eval_path is not None else None,
        "eval_preference_data_sha256": eval_preference_sha256,
        "tier": zexo.config.tier,
        "parameters": zexo.config.parameter_count,
        "beta": args.beta,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "training_mode": "full_model" if args.full_model else "lora",
        "lora_rank": None if args.full_model else args.lora_rank,
        "lora_alpha": None if args.full_model else args.lora_alpha,
        "lora_targets": None if args.full_model else [item.strip() for item in args.lora_targets.split(",") if item.strip()],
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "history": history,
        "output_checkpoint": str(output_path),
        "output_sha256": _sha256(output_path),
    }
    metadata_path = output_path.with_suffix(".refine.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved refined checkpoint: {output_path} ({output_path.stat().st_size:,} bytes)")
    if adapter_path is not None:
        print(f"Saved adapter:           {adapter_path}")
    print(f"Saved run metadata:      {metadata_path}")
    if "best_epoch" in history:
        print(f"Restored best eval epoch: {history['best_epoch']} (loss={history['best_eval_loss']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
