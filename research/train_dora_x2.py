"""Training Engine for Dora-X2 16-Layer MH-RTU Language Model.

Executes chunk-based sequence training on curated conversational datasets
with AdamW updates, O(1) recurrent trace state management, and real-time
evaluation benchmarking.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional
import numpy as np

# Force line buffering for CI/CD live output
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from doraneural.dora_x2 import DoraX2Config, DoraX2LM, DoraX2ChatSession


def load_text_data(data_path: Path, max_bytes: int = -1) -> bytes:
    """Load training text as raw bytes."""
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")
    raw = data_path.read_bytes()
    if max_bytes > 0 and len(raw) > max_bytes:
        raw = raw[:max_bytes]
    return raw


def train_dora_x2_engine(
    data_path: Path,
    output_dir: Path,
    dim: int = 768,
    hidden_dim: int = 2048,
    n_layers: int = 16,
    n_heads: int = 12,
    epochs: int = 3,
    chunk_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    max_bytes: int = 200000,
    tag: str = "dora_x2_latest",
    test_prompt: str = "User: Hello Dora-X2! What is your architecture?\nAssistant: ",
) -> Dict[str, any]:
    print("=" * 80)
    print(" ⚡ DORA-X2 16-LAYER MH-RTU CONVERSATIONAL TRAINING")
    print("=" * 80)
    print(f"Run Tag:            {tag}")
    print(f"Architecture:       {n_layers} Layers | {dim} Dim | {n_heads} Heads (head_dim={dim//n_heads})")
    print(f"FFN Hidden Dim:     {hidden_dim} (SwiGLU + Bio-Reflective KANs)")
    print(f"Corpus Path:        {data_path} (max_bytes={max_bytes})")
    print(f"Hyperparameters:    epochs={epochs}, chunk_size={chunk_size}, lr={lr}")
    print(f"Output Directory:   {output_dir}")
    print("-" * 80, flush=True)

    # 1. Load Data
    raw_bytes = load_text_data(data_path, max_bytes=max_bytes)
    print(f"Loaded {len(raw_bytes):,} raw bytes of training data.", flush=True)

    # 2. Build Model
    config = DoraX2Config(
        name="dora-x2",
        dim=dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=256,
    )
    model = DoraX2LM(config)
    print(f"Model instantiated with {config.parameter_count:,} parameters ({config.parameter_count/1e6:.1f}M).", flush=True)
    print("-" * 80, flush=True)

    # 3. Pre-Training Generation Probe
    print(f"🔍 Pre-Training Generation Probe (Prompt: '{test_prompt[:40]}...'):")
    pre_gen = model.generate(test_prompt, max_tokens=35, temperature=0.7)
    print(f"  ➜ Generated: \"{pre_gen.strip()}\"", flush=True)
    print("-" * 80, flush=True)

    # 4. Training Loop
    t_start = time.perf_counter()
    n_chunks = max(1, (len(raw_bytes) - 1) // chunk_size)
    print(f"🚀 Commencing training across {epochs} epochs ({n_chunks} chunks of {chunk_size} bytes per epoch)...", flush=True)

    loss_history = []
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        ep_tokens = 0
        ep_start = time.perf_counter()

        for c_idx in range(n_chunks):
            start = c_idx * chunk_size
            end = min(len(raw_bytes), start + chunk_size + 1)
            chunk = list(raw_bytes[start:end])
            if len(chunk) < 2:
                continue

            res = model.train_sequence(chunk, lr=lr, weight_decay=weight_decay, reset_state=(c_idx % 4 == 0))
            ep_loss += res["loss"] * res["tokens"]
            ep_tokens += res["tokens"]

            if (c_idx + 1) % 25 == 0 or (c_idx + 1) == n_chunks:
                elapsed = time.perf_counter() - ep_start
                cur_loss = ep_loss / max(1, ep_tokens)
                tps = ep_tokens / elapsed if elapsed > 0 else 0.0
                print(
                    f"   [Epoch {ep:02d}/{epochs:02d}] Step {c_idx+1:04d}/{n_chunks:04d} "
                    f"| Loss: {cur_loss:.4f} | Speed: {tps:.1f} tok/s",
                    flush=True,
                )

        avg_loss = ep_loss / max(1, ep_tokens)
        loss_history.append(avg_loss)
        ep_time = time.perf_counter() - ep_start
        print(f"✨ Epoch {ep:02d}/{epochs:02d} Complete -> Avg Loss: {avg_loss:.4f} in {ep_time:.1f}s", flush=True)

    total_time = time.perf_counter() - t_start
    print("-" * 80, flush=True)
    print(f"✓ Dora-X2 Training finished in {total_time:.2f}s ({ep_tokens * epochs / total_time:.1f} tok/s)", flush=True)
    print("-" * 80, flush=True)

    # 5. Post-Training Generation Probe
    print(f"✨ Post-Training Generation Probe (Prompt: '{test_prompt[:40]}...'):")
    post_gen = model.generate(test_prompt, max_tokens=40, temperature=0.7)
    print(f"  ➜ Generated: \"{post_gen.strip()}\"", flush=True)
    print("-" * 80, flush=True)

    # 6. Save Checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest"
    tagged_path = output_dir / tag
    model.save(latest_path)
    model.save(tagged_path)
    print("💾 Dora-X2 Checkpoint Saved:")
    print(f"  • Latest: {latest_path.with_suffix('.npz')}")
    print(f"  • Tagged: {tagged_path.with_suffix('.npz')}")
    print(f"  • Schema: {latest_path.with_suffix('.json')}")
    print("=" * 80, flush=True)

    return {
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "duration_s": total_time,
        "tokens_processed": ep_tokens * epochs,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Dora-X2 16-Layer MH-RTU Language Model.")
    default_data = REPO_ROOT / "zexo" / "data" / "zexo_quality_v1_train.txt"
    if not default_data.exists():
        default_data = REPO_ROOT / "zexo" / "data" / "step1_conversational_expanded.txt"

    parser.add_argument("--data", type=Path, default=default_data, help="Path to training text file.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "checkpoints" / "dora_x2", help="Directory to save checkpoints.")
    parser.add_argument("--dim", type=int, default=768, help="Model representation dimension.")
    parser.add_argument("--layers", type=int, default=16, help="Number of MH-RTU layers.")
    parser.add_argument("--heads", type=int, default=12, help="Number of attention / recurrent heads.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Recurrent chunk length.")
    parser.add_argument("--lr", type=float, default=0.001, help="AdamW learning rate.")
    parser.add_argument("--max-bytes", type=int, default=100000, help="Max corpus bytes to process (-1 for full).")
    parser.add_argument("--tag", type=str, default="dora_x2_run_1", help="Checkpoint tag.")
    parser.add_argument(
        "--test-prompt",
        type=str,
        default="User: Hello Dora-X2! What is your architecture?\nAssistant: ",
        help="Evaluation prompt.",
    )
    args = parser.parse_args()

    train_dora_x2_engine(
        data_path=args.data,
        output_dir=args.output_dir,
        dim=args.dim,
        hidden_dim=args.dim * 8 // 3,
        n_layers=args.layers,
        n_heads=args.heads,
        epochs=args.epochs,
        chunk_size=args.chunk_size,
        lr=args.lr,
        max_bytes=args.max_bytes,
        tag=args.tag,
        test_prompt=args.test_prompt,
    )


if __name__ == "__main__":
    main()
