#!/usr/bin/env python3
"""Pretrain the compact DeepSeek-Flash-inspired reference model on UTF-8 text.

This experimental runner writes a separate NPZ checkpoint and does not alter
Zexo's llama2.c-compatible .bin checkpoint format.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from zexo.flash_lite import DeepSeekFlashLiteLM, FlashLiteConfig
from doraneural.llm import ByteTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-backprop pretraining for the small FlashLite research architecture."
    )
    parser.add_argument("--data", required=True, help="UTF-8 plain-text training corpus.")
    parser.add_argument("--eval-data", default=None, help="Optional separate held-out UTF-8 text file.")
    parser.add_argument("--checkpoint", default=None, help="Optional FlashLite .npz checkpoint to continue from.")
    parser.add_argument("--output", default="zexo/checkpoints/flash_lite.npz", help="Output model checkpoint (.npz).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seq-len", type=int, default=128, help="Training window length; must be shorter than context.")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum optimizer updates per epoch.")
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--ced-split", type=int, default=None)
    parser.add_argument("--local-only-encoder-layers", type=int, default=2)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--global-top-k", type=int, default=32)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--experts-per-token", type=int, default=2)
    parser.add_argument("--router-aux-loss-coef", type=float, default=0.01)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    train_path = Path(args.data)
    eval_path = Path(args.eval_data) if args.eval_data else None
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if not train_path.is_file():
        parser.error(f"training data does not exist: {train_path}")
    if eval_path is not None and not eval_path.is_file():
        parser.error(f"evaluation data does not exist: {eval_path}")
    if eval_path is not None and (
        eval_path.resolve() == train_path.resolve() or _sha256(eval_path) == _sha256(train_path)
    ):
        parser.error("--eval-data must contain held-out data, not the training file contents")
    if checkpoint_path is not None and not checkpoint_path.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint_path}")
    if (args.epochs <= 0 or not np.isfinite(args.lr) or not np.isfinite(args.weight_decay)
            or args.lr <= 0.0 or args.weight_decay < 0.0):
        parser.error("--epochs and --lr must be positive; --weight-decay must be non-negative")
    if args.seq_len <= 0:
        parser.error("--seq-len must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.max_eval_steps is not None and args.max_eval_steps <= 0:
        parser.error("--max-eval-steps must be positive")
    if not np.isfinite(args.grad_clip) or args.grad_clip < 0.0:
        parser.error("--grad-clip must be finite and non-negative")

    np.random.seed(args.seed)
    if checkpoint_path is not None:
        model = DeepSeekFlashLiteLM.load(checkpoint_path)
    else:
        try:
            config = FlashLiteConfig(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                n_layers=args.layers,
                n_heads=args.heads,
                n_kv_heads=args.kv_heads,
                seq_len=args.context,
                ced_split=args.ced_split,
                local_only_encoder_layers=args.local_only_encoder_layers,
                sliding_window=args.window,
                global_top_k=args.global_top_k,
                n_experts=args.experts,
                experts_per_token=args.experts_per_token,
                router_aux_loss_coef=args.router_aux_loss_coef,
            )
        except ValueError as exc:
            parser.error(str(exc))
        model = DeepSeekFlashLiteLM(config)

    if args.seq_len >= model.config.seq_len:
        parser.error(f"--seq-len must be less than model context ({model.config.seq_len})")
    tokenizer = ByteTokenizer()
    training_text = train_path.read_text(encoding="utf-8", errors="strict")
    eval_text = eval_path.read_text(encoding="utf-8", errors="strict") if eval_path else None
    output_path = Path(args.output)
    if output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    protected_paths = [train_path.resolve()]
    if eval_path is not None:
        protected_paths.append(eval_path.resolve())
    if output_path.resolve() in protected_paths:
        parser.error("--output must not overwrite the training or evaluation data file")
    parent_sha256 = _sha256(checkpoint_path) if checkpoint_path is not None else None

    print("=" * 72)
    print("DeepSeek-Flash-inspired FlashLite (small CPU reference; not DeepSeek weights)")
    print(f"Architecture: {model.config.n_layers} layers | {model.config.dim} dim | {model.parameter_count:,} params")
    print(f"Attention:    CED split={model.config.encoder_layers}; SWA window={model.config.sliding_window}; global top-k={model.config.global_top_k}")
    print(f"CSA2-lite:    {', '.join(model.attention_modes)}")
    print(f"MoE:          {model.config.n_experts} routed experts, top-{model.config.experts_per_token} + shared expert")
    print(f"Data:         {train_path} ({len(training_text):,} chars, UTF-8 byte tokens)")
    print("=" * 72)

    history = model.fit_text(
        training_text,
        tokenizer=tokenizer,
        eval_text=eval_text,
        epochs=args.epochs,
        seq_len=args.seq_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        stride=args.stride,
        max_steps=args.max_steps,
        max_eval_steps=args.max_eval_steps,
        grad_clip=args.grad_clip,
        seed=args.seed,
        verbose=0 if args.quiet else 1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    metadata = {
        "format_version": 1,
        "architecture": "deepseek-v4.1-flash-inspired-flashlite",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "parent_sha256": parent_sha256,
        "training_data": str(train_path),
        "training_data_sha256": _sha256(train_path),
        "eval_data": str(eval_path) if eval_path else None,
        "eval_data_sha256": _sha256(eval_path) if eval_path else None,
        "config": model.config.to_dict(),
        "epochs": args.epochs,
        "seq_len": args.seq_len,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "history": history,
        "output_checkpoint": str(output_path),
        "output_sha256": _sha256(output_path),
    }
    metadata_path = output_path.with_suffix(".train.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved FlashLite checkpoint: {output_path} ({output_path.stat().st_size:,} bytes)")
    print(f"Saved run metadata:          {metadata_path}")
    if "best_epoch" in history:
        print(f"Restored best held-out epoch: {history['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
