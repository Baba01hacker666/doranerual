#!/usr/bin/env python3
"""Minimal interactive front end for a FlashLite .npz checkpoint.

FlashLite checkpoints from the example runner are next-token pretrained, not
instruction-tuned; this loop is a smoke test rather than a quality guarantee.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doraneural.llm import ByteTokenizer
from zexo.flash_lite import DeepSeekFlashLiteLM


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Chat-smoke-test a FlashLite NPZ checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="FlashLite .npz checkpoint from train_flash.py")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    if args.max_tokens < 0 or args.temperature < 0.0 or args.top_k < 0:
        parser.error("generation limits must be non-negative")

    model = DeepSeekFlashLiteLM.load(checkpoint)
    tokenizer = ByteTokenizer()
    history = []
    print("FlashLite reference chat (byte-level base model; not guaranteed to follow instructions).")
    print("Type /clear to reset context or /quit to exit.\n")
    while True:
        try:
            user_text = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user_text:
            continue
        if user_text.lower() in {"/quit", "/exit", "quit", "exit"}:
            return 0
        if user_text.lower() == "/clear":
            history.clear()
            print("Context cleared.\n")
            continue
        history.append(f"User: {user_text}\nAssistant:")
        prompt = "\n".join(history)
        reply = model.generate(
            prompt,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
        print(f"FlashLite> {reply}\n")
        history[-1] += f" {reply}"


if __name__ == "__main__":
    raise SystemExit(main())
