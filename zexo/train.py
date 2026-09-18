#!/usr/bin/env python3
"""Zexo AI: Dedicated Conversational Training & Checkpointing Runner.

Fine-tunes Zexo AI on conversational dialogue corpora or Hugging Face datasets,
evaluates conversational answers before and after training, and maintains
continuous checkpoint lineage in zexo/checkpoints/.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import doraneural as dn
from zexo.config import ZexoConfig
from zexo.data.dataset_tools import load_corpus
from zexo.model import load_zexo


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Train and fine-tune Zexo AI on conversational data.")
    parser.add_argument(
        "--data", "-d",
        default=str(REPO_ROOT / "zexo" / "data" / "zexo_quality_v1_train.txt"),
        help="Path to a plain-text or validated JSONL training corpus (default: zexo/data/zexo_quality_v1_train.txt).",
    )
    parser.add_argument(
        "--eval-data",
        default=None,
        help="Optional held-out plain-text or JSONL evaluation corpus. Auto-detected for *_train.txt files.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.0,
        help="Reserve this fraction for validation when --eval-data is not provided (default: 0).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Training window stride. Defaults to seq-len; smaller values add overlapping examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to shuffle training windows (default: 42).",
    )
    parser.add_argument(
        "--max-eval-steps",
        type=int,
        default=None,
        help="Optional cap on validation windows per epoch.",
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Hugging Face dataset identifier (e.g. 'roneneldan/TinyStories') to auto-download.",
    )
    parser.add_argument(
        "--hf-samples",
        type=int,
        default=50,
        help="Max samples to download if using Hugging Face (default: 50).",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="Path to prior Zexo checkpoint to resume training from.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Initialize fresh base weights rather than resuming from latest checkpoint.",
    )
    parser.add_argument(
        "--tier", "-t",
        choices=["micro", "mini", "chat", "base", "large"],
        default="mini",
        help="Zexo tier architecture (default: mini).",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (default: 0.0005).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence chunk length (default: 64).",
    )
    parser.add_argument(
        "--test-prompt", "-p",
        default="Who are you?",
        help="Test prompt to evaluate conversational response before and after training.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=str(REPO_ROOT / "zexo" / "checkpoints"),
        help="Directory to save updated checkpoints (default: zexo/checkpoints).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode: uses all available CPU threads, optimized sequence length, and high-throughput execution.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Run identifier tag.",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fast mode optimizations
    if args.fast:
        n_cpus = os.cpu_count() or 4
        os.environ["OMP_NUM_THREADS"] = str(n_cpus)
        os.environ["PYTHONUNBUFFERED"] = "1"
        if args.seq_len == 64:
            args.seq_len = 128

    run_id = args.tag or datetime.now(timezone.utc).strftime("zexo_%Y%m%d_%H%M%S")
    timestamp = datetime.now(timezone.utc).isoformat()

    print("═" * 72)
    print(" 🤖 Zexo AI Conversational Training Engine")
    print("═" * 72)
    print(f"Run ID:         {run_id}")
    print(f"Target Tier:    {args.tier.upper()}")
    print(f"From Scratch:   {args.from_scratch}")
    print(f"Fast Mode:      {args.fast} ({os.environ.get('OMP_NUM_THREADS', 'auto')} OpenMP threads)")
    print(f"Output Dir:     {out_dir}")
    print("─" * 72)

    # 1. Resolve Data
    corpus_source = args.hf_dataset or args.data
    if args.hf_dataset or dn.is_hf_dataset_identifier(args.data):
        target_hf = args.hf_dataset or args.data
        if "," in target_hf:
            ds_list = [d.strip() for d in target_hf.split(",") if d.strip()]
            print(f"🤗 Downloading and merging {len(ds_list)} Hugging Face datasets: {ds_list}...")
            corpus_path = dn.download_and_merge_hf_datasets(ds_list, max_samples_each=args.hf_samples)
        else:
            print(f"🤗 Downloading training dataset from Hugging Face: '{target_hf}'...")
            corpus_path = dn.download_hf_dataset(target_hf, max_samples=args.hf_samples)
    else:
        corpus_path = Path(args.data)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Training data not found: {corpus_path}")

    text = load_corpus(corpus_path)
    eval_text = None
    eval_path = Path(args.eval_data) if args.eval_data else None
    if eval_path is None and corpus_path.name.endswith("_train.txt"):
        candidate = corpus_path.with_name(corpus_path.name.replace("_train.txt", "_eval.txt"))
        if candidate.exists():
            eval_path = candidate
    if eval_path is not None:
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation data not found: {eval_path}")
        eval_text = load_corpus(eval_path)
    print(f"📄 Dataset Loaded: {corpus_path.name} ({len(text):,} chars, {len(text.split()):,} words)")
    if eval_path is not None:
        print(f"🧪 Held-out Evaluation: {eval_path.name} ({len(eval_text):,} chars, {len(eval_text.split()):,} words)")

    # 2. Load or Initialize Zexo
    print("\n[1/4] Loading Zexo Model...")
    zexo = load_zexo(checkpoint_path=args.checkpoint, tier=args.tier, from_scratch=args.from_scratch)
    print(f"  Architecture: {zexo.config.dim} dim, {zexo.config.n_layers} layers, {zexo.config.parameter_count:,} params")
    print(f"  Backend:      {zexo.llm.backend.upper()}")

    # 3. Test Baseline Answer
    print(f"\n[2/4] Baseline Conversational Reply (Before Training):")
    print("─" * 72)
    print(f"User: {args.test_prompt}")
    ans_before = zexo.chat(args.test_prompt, max_tokens=50)
    print(f"Zexo: {ans_before}")
    print("─" * 72)

    # 4. Train Model
    print(f"\n[3/4] Fine-tuning Zexo for {args.epochs} epochs (lr={args.lr}, seq_len={args.seq_len})...")
    t0 = time.perf_counter()
    hist = zexo.train(
        text,
        epochs=args.epochs,
        lr=args.lr,
        seq_len=args.seq_len,
        verbose=1,
        eval_text=eval_text,
        validation_split=args.validation_split if eval_text is None else 0.0,
        stride=args.stride,
        seed=args.seed,
        max_eval_steps=args.max_eval_steps,
    )
    duration = time.perf_counter() - t0
    final_report = f"Final loss: {hist['loss'][-1]:.4f}"
    if "val_loss" in hist:
        final_report += f" | Final validation loss: {hist['val_loss'][-1]:.4f}"
    print(f"  Training finished in {duration:.2f}s! {final_report}")

    # 5. Test Fine-Tuned Answer
    zexo.reset()
    print(f"\n[4/4] Fine-Tuned Conversational Reply (After Training):")
    print("─" * 72)
    print(f"User: {args.test_prompt}")
    ans_after = zexo.chat(args.test_prompt, max_tokens=50)
    print(f"Zexo: {ans_after}")
    print("─" * 72)

    # 6. Save Checkpoint
    latest_bin = out_dir / "latest.bin"
    snapshot_bin = out_dir / f"{run_id}.bin"
    zexo.save(latest_bin)
    zexo.save(snapshot_bin)
    print(f"\n💾 Saved updated Zexo weights:")
    print(f"   -> Head:     {latest_bin.name} ({latest_bin.stat().st_size:,} bytes)")
    print(f"   -> Snapshot: {snapshot_bin.name}")

    # 7. Metadata
    meta_file = out_dir / "zexo_meta.json"
    meta_entry = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "tier": zexo.config.tier,
        "parameters": zexo.config.parameter_count,
        "dataset": corpus_path.name,
        "evaluation_dataset": eval_path.name if eval_path is not None else None,
        "chars": len(text),
        "words": len(text.split()),
        "epochs": args.epochs,
        "lr": args.lr,
        "seq_len": args.seq_len,
        "stride": args.stride or args.seq_len,
        "seed": args.seed,
        "loss_history": hist["loss"],
        "val_loss_history": hist.get("val_loss"),
        "test_prompt": args.test_prompt,
        "reply_before": ans_before,
        "reply_after": ans_after,
        "sha256": compute_sha256(latest_bin),
    }

    history = []
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                prev = json.load(f)
                history = prev.get("runs", [])
        except Exception:
            history = []

    history.append(meta_entry)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({"latest": meta_entry, "total_runs": len(history), "runs": history}, f, indent=2)

    print(f"   -> Metadata: {meta_file.name}")
    print("\n" + "═" * 72)
    print("✨ Zexo conversational training complete! Ready to chat: python zexo/chat.py")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
