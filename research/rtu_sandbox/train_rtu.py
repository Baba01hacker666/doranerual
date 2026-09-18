"""Continual Learning Trainer for RTU Recurrent Latent Language Model.

Supports:
1. Automatic Simple Wikipedia dataset download & preparation.
2. Continual learning: automatically resumes from `latest.npz` if present.
3. Multi-objective training: Cross-Entropy + Latent MSE + VICReg variance penalty.
4. Pre- and post-training generation benchmarks.
5. Checkpoint exports compatible with GitHub Actions caching & artifacts.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rtu import RTUConfig, RTULanguageModel
from prepare_wikipedia import prepare_simplewiki


def train_rtu_cli():
    parser = argparse.ArgumentParser(description="Train RTU Model with Continual Learning on Wikipedia")
    parser.add_argument("--data", default="research/rtu_sandbox/data/simplewiki_clean.txt", help="Path to text corpus")
    parser.add_argument("--download-wiki", action="store_true", default=True, help="Auto-download Wikipedia if data missing")
    parser.add_argument("--wiki-max-mb", type=float, default=200.0, help="Max MB of Simple Wikipedia to download/extract")
    parser.add_argument("--dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=4, help="Number of RTU layers")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--chunk-size", type=int, default=256, help="Recurrent chunk sequence size")
    parser.add_argument("--max-bytes", type=int, default=-1, help="Max bytes to train on per run (-1 = all)")
    parser.add_argument("--output-dir", default="research/rtu_sandbox/checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--from-scratch", action="store_true", help="Ignore existing checkpoint and train from scratch")
    parser.add_argument("--test-prompt", default="The Solar System is ", help="Prompt to evaluate before and after")
    parser.add_argument("--tag", default="", help="Run identification tag")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = out_dir / "latest.npz"
    latest_meta = out_dir / "latest.json"

    tag = args.tag or f"rtu_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    print("=" * 72)
    print(" 🤖 RTU LATENT ENGINE CONTINUAL TRAINING")
    print("=" * 72)
    print(f"Run ID:         {tag}")
    print(f"Dimensions:     dim={args.dim}, layers={args.layers}")
    print(f"Learning Rate:  {args.lr}")
    print(f"Output Dir:     {out_dir}")
    print(f"From Scratch:   {args.from_scratch}")
    print("-" * 72)

    # 1. Dataset Preparation
    data_path = Path(args.data)
    if not data_path.exists() and args.download_wiki:
        print(f"📖 Corpus {data_path} not found. Preparing Simple Wikipedia dataset (target: {args.wiki_max_mb} MB)...")
        prepare_simplewiki(output_path=str(data_path), max_mb=args.wiki_max_mb)

    if not data_path.exists():
        print(f"❌ Error: Dataset {data_path} does not exist!")
        sys.exit(1)

    file_size = data_path.stat().st_size
    print(f"✓ Found corpus: {data_path} ({file_size / (1024*1024):.2f} MB)")

    # 2. Model Initialization / Continual Checkpoint Loading
    cfg = RTUConfig(
        dim=args.dim,
        n_layers=args.layers,
        vocab_size=256,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    model = RTULanguageModel(cfg)

    meta_info: Dict[str, Any] = {
        "cumulative_bytes_trained": 0,
        "cumulative_runs": 0,
        "run_history": [],
    }

    if latest_checkpoint.exists() and not args.from_scratch:
        print(f"🔄 Continual Learning: Resuming from previous checkpoint: {latest_checkpoint}")
        try:
            model.load(latest_checkpoint)
            if latest_meta.exists():
                with open(latest_meta, "r", encoding="utf-8") as mf:
                    meta_info = json.load(mf)
            print(
                f"   Prior Stats: {meta_info.get('cumulative_runs', 0)} runs completed, "
                f"{meta_info.get('cumulative_bytes_trained', 0):,} cumulative bytes trained."
            )
        except Exception as err:
            print(f"⚠️ Warning: Could not restore checkpoint ({err}). Initializing fresh weights.")
    else:
        print("🌱 Initializing fresh RTU model weights from scratch.")

    print(f"Model Total Parameters: {cfg.parameter_count:,} params (Embedding: {cfg.vocab_size * cfg.dim * 2:,})")

    # Read bytes up to max_bytes with streaming offset continuation
    read_limit = args.max_bytes if args.max_bytes > 0 else file_size
    start_offset = (meta_info.get("cumulative_bytes_trained", 0) % file_size) if not args.from_scratch else 0
    with open(data_path, "rb") as f:
        f.seek(start_offset)
        raw_bytes = f.read(read_limit)
        if len(raw_bytes) < read_limit and start_offset > 0:
            f.seek(0)
            raw_bytes += f.read(read_limit - len(raw_bytes))

    print(
        f"✓ Loaded {len(raw_bytes):,} raw UTF-8 bytes for training window "
        f"(streaming offset: {start_offset:,}/{file_size:,} bytes)."
    )

    # 3. Pre-Training Prompt Evaluation
    print(f"\n🔍 Pre-Training Generation (Prompt: '{args.test_prompt}'):")
    pre_gen = model.generate(args.test_prompt, max_bytes=80, temperature=0.3, reset_state=True)
    print(f"   [Output]: {args.test_prompt}{pre_gen}\n")

    # 4. Training Loop
    print("🚀 Commencing Continual Training...")
    t0 = time.perf_counter()
    n_bytes = len(raw_bytes)
    total_chunks = max(1, (n_bytes + args.chunk_size - 1) // args.chunk_size)

    run_loss_history = []
    run_ce_history = []

    for ep in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        total_loss = 0.0
        total_ce = 0.0
        total_lat = 0.0
        total_var = 0.0
        processed_chunks = 0

        for c_idx in range(0, n_bytes - 1, args.chunk_size):
            chunk = raw_bytes[c_idx : c_idx + args.chunk_size + 1]
            stats = model.train_sequence(chunk, lr=args.lr, reset_state=(c_idx == 0))
            total_loss += stats["loss"]
            total_ce += stats["ce_loss"]
            total_lat += stats["latent_loss"]
            total_var += stats["variance_loss"]
            processed_chunks += 1

            if processed_chunks % 500 == 0 or processed_chunks == total_chunks:
                now_elapsed = max(1e-4, time.perf_counter() - ep_t0)
                bytes_done = min(n_bytes, c_idx + args.chunk_size)
                speed = bytes_done / now_elapsed
                print(
                    f"\r  Epoch {ep}/{args.epochs} | "
                    f"Chunk {processed_chunks:,}/{total_chunks:,} ({bytes_done/(1024*1024):.1f} MB) | "
                    f"Loss: {total_loss / processed_chunks:.4f} | "
                    f"CE: {total_ce / processed_chunks:.4f} | "
                    f"Latent: {total_lat / processed_chunks:.4f} | "
                    f"Var: {total_var / processed_chunks:.4f} | "
                    f"Speed: {speed:,.0f} B/s",
                    end="",
                    flush=True,
                )

        ep_elapsed = time.perf_counter() - ep_t0
        avg_loss = total_loss / max(1, processed_chunks)
        avg_ce = total_ce / max(1, processed_chunks)
        run_loss_history.append(avg_loss)
        run_ce_history.append(avg_ce)
        print(f"\n✓ Epoch {ep} complete in {ep_elapsed:.2f}s | Avg Loss: {avg_loss:.4f} | CE: {avg_ce:.4f}")

    total_time = time.perf_counter() - t0
    total_bytes_trained_this_run = n_bytes * args.epochs
    overall_speed = total_bytes_trained_this_run / max(1e-4, total_time)
    print(f"\n🎉 Training Phase Complete: {total_bytes_trained_this_run:,} bytes processed in {total_time:.2f}s ({overall_speed:,.0f} B/s)")

    # 5. Post-Training Prompt Evaluation
    print(f"\n✨ Post-Training Generation (Prompt: '{args.test_prompt}'):")
    post_gen = model.generate(args.test_prompt, max_bytes=80, temperature=0.3, reset_state=True)
    print(f"   [Output]: {args.test_prompt}{post_gen}\n")

    # 6. Save Updated Continual Checkpoints
    tagged_checkpoint = out_dir / f"{tag}.npz"
    model.save(latest_checkpoint)
    model.save(tagged_checkpoint)

    meta_info["cumulative_bytes_trained"] += total_bytes_trained_this_run
    meta_info["cumulative_runs"] += 1
    meta_info["last_updated"] = datetime.now(timezone.utc).isoformat()
    meta_info["dim"] = args.dim
    meta_info["layers"] = args.layers
    meta_info["parameter_count"] = cfg.parameter_count
    meta_info["run_history"].append({
        "tag": tag,
        "epochs": args.epochs,
        "bytes_trained": total_bytes_trained_this_run,
        "final_loss": run_loss_history[-1] if run_loss_history else 0.0,
        "final_ce": run_ce_history[-1] if run_ce_history else 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    with open(latest_meta, "w", encoding="utf-8") as mf:
        json.dump(meta_info, mf, indent=2)

    print("💾 Checkpoint Saved:")
    print(f"   - Continual Active State: {latest_checkpoint}")
    print(f"   - Tagged Snapshot:        {tagged_checkpoint}")
    print(f"   - Meta Manifest:          {latest_meta}")
    print("=" * 72)


if __name__ == "__main__":
    train_rtu_cli()
