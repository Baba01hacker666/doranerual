#!/usr/bin/env python3
"""Offline DPO refinement for a FlashLite .npz checkpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doraneural.llm import ByteTokenizer
from zexo.flash_lite import DeepSeekFlashLiteLM
from zexo.preferences import PreferenceTrainer, read_preferences


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Full-model offline DPO for FlashLite NPZ checkpoints.")
    parser.add_argument("--checkpoint", required=True, help="Input FlashLite .npz checkpoint.")
    parser.add_argument("--preferences", required=True, help="Reviewed chosen/rejected training JSONL.")
    parser.add_argument("--eval-preferences", default=None, help="Optional held-out JSONL for best-epoch selection.")
    parser.add_argument("--output", default="zexo/checkpoints/flash_lite_refined.npz")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    train_path = Path(args.preferences)
    eval_path = Path(args.eval_preferences) if args.eval_preferences else None
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    if not train_path.is_file():
        parser.error(f"preference data does not exist: {train_path}")
    if eval_path is not None and not eval_path.is_file():
        parser.error(f"evaluation preference data does not exist: {eval_path}")
    if eval_path is not None and _sha256(train_path) == _sha256(eval_path):
        parser.error("--eval-preferences must contain held-out data, not the training file contents")
    if (args.epochs <= 0 or not math.isfinite(args.lr) or not math.isfinite(args.beta)
            or args.lr <= 0.0 or args.beta <= 0.0):
        parser.error("--epochs must be positive; --lr and --beta must be finite and positive")
    if args.max_seq_len is not None and args.max_seq_len < 2:
        parser.error("--max-seq-len must be at least 2")

    parent_hash = _sha256(checkpoint)
    train_examples = read_preferences(train_path)
    eval_examples = read_preferences(eval_path) if eval_path is not None else None
    policy = DeepSeekFlashLiteLM.load(checkpoint)
    reference = deepcopy(policy)
    trainer = PreferenceTrainer(
        policy_model=policy,
        reference_model=reference,
        tokenizer=ByteTokenizer(),
        system_prompt="",
        beta=args.beta,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
    )
    history = trainer.fit(
        train_examples,
        eval_examples=eval_examples,
        epochs=args.epochs,
        seed=args.seed,
        verbose=1,
    )

    output = Path(args.output)
    if output.suffix != ".npz":
        output = output.with_suffix(".npz")
    protected_paths = {train_path.resolve()}
    if eval_path is not None:
        protected_paths.add(eval_path.resolve())
    if output.resolve() in protected_paths:
        parser.error("--output must not overwrite preference data")
    output.parent.mkdir(parents=True, exist_ok=True)
    policy.save(output)
    metadata = {
        "format_version": 1,
        "method": "DPO",
        "architecture": "deepseek-v4.1-flash-inspired-flashlite",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": str(checkpoint),
        "parent_sha256": parent_hash,
        "preference_data": str(train_path),
        "preference_data_sha256": _sha256(train_path),
        "eval_preference_data": str(eval_path) if eval_path else None,
        "eval_preference_data_sha256": _sha256(eval_path) if eval_path else None,
        "config": policy.config.to_dict(),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "beta": args.beta,
        "history": history,
        "output_checkpoint": str(output),
        "output_sha256": _sha256(output),
    }
    metadata_path = output.with_suffix(".refine.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved refined FlashLite checkpoint: {output}")
    print(f"Saved DPO metadata:                {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
