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
from zexo.model import load_zexo


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    expanded_path = REPO_ROOT / "zexo" / "data" / "step1_conversational_expanded.txt"
    base_path = REPO_ROOT / "zexo" / "data" / "step1_conversational_base.txt"
    default_dataset = str(expanded_path if expanded_path.exists() else base_path)

    parser = argparse.ArgumentParser(description="Train and fine-tune Zexo AI on conversational data.")
    parser.add_argument(
        "--data", "-d",
        default=default_dataset,
        help=f"Path to training dialogue file or Hugging Face dataset ID (default: {Path(default_dataset).name}).",
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Hugging Face dataset identifier (e.g. 'yahma/alpaca-cleaned' or 'roneneldan/TinyStories') to auto-download.",
    )
    parser.add_argument(
        "--hf-samples",
        type=int,
        default=50,
        help="Max samples to download if using Hugging Face (-1 = all, default: 50).",
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
        default=3e-4,
        help="Learning rate for parameter updates (default: 0.0003).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence chunk length (default: 128).",
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=0.15,
        help="Continual learning: fraction of core persona & reasoning anchor dialogues to interleave (default: 0.15).",
    )
    parser.add_argument(
        "--no-mask-prompts",
        action="store_true",
        help="Disable SFT instruction loss masking on user prompt tokens.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps/batches per epoch (useful for large datasets on CPU).",
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
        help="Enable fast mode: uses all available CPU threads and unbuffered output.",
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
        if args.seq_len == 16:
            args.seq_len = 64

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

    text = corpus_path.read_text(encoding="utf-8", errors="replace")
    print(f"📄 Dataset Loaded: {corpus_path.name} ({len(text):,} chars, {len(text.split()):,} words)")

    # Continual Learning Anchor Replay: Interleave core persona/reasoning dialogues to prevent catastrophic forgetting
    if args.replay_ratio > 0 and corpus_path.name != "step1_conversational_expanded.txt" and expanded_path.exists():
        anchor_text = expanded_path.read_text(encoding="utf-8", errors="replace")
        anchor_blocks = [b.strip() for b in anchor_text.split("\n\n") if b.strip()]
        current_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        if current_blocks and anchor_blocks:
            import random
            rng = random.Random(42)
            n_replay = max(1, int(len(current_blocks) * args.replay_ratio))
            sampled_anchors = rng.sample(anchor_blocks, min(n_replay, len(anchor_blocks)))
            combined = []
            stride = max(1, len(current_blocks) // len(sampled_anchors))
            s_idx = 0
            for c_idx, cb in enumerate(current_blocks):
                combined.append(cb)
                if c_idx % stride == 0 and s_idx < len(sampled_anchors):
                    combined.append(sampled_anchors[s_idx])
                    s_idx += 1
            text = "\n\n".join(combined) + "\n"
            print(f"🔄 Continual Learning Anchor Replay: Interleaved {len(sampled_anchors)} core persona dialogues ({args.replay_ratio * 100:.0f}% buffer) to prevent catastrophic forgetting.")

    # 2. Load or Initialize Zexo
    print("\n[1/4] Loading Zexo Model...")
    zexo = load_zexo(checkpoint_path=args.checkpoint, tier=args.tier, from_scratch=args.from_scratch)
    print(f"  Architecture: {zexo.config.dim} dim, {zexo.config.n_layers} layers, {zexo.config.parameter_count:,} params")
    print(f"  Backend:      {zexo.llm.backend.upper()}")

    # 3. Test Baseline Answer (Continual Retention Probe + Task Probe)
    print(f"\n[2/4] Baseline Conversational Probes (Before Training):")
    print("─" * 72)
    retention_prompt = "Who are you?"
    ans_retention_before = zexo.chat(retention_prompt, max_tokens=50)
    zexo.reset()
    print(f"  [Retention Probe] User: {retention_prompt}")
    print(f"                    Zexo: {ans_retention_before}")
    ans_task_before = zexo.chat(args.test_prompt, max_tokens=50)
    zexo.reset()
    print(f"  [Task Probe]      User: {args.test_prompt}")
    print(f"                    Zexo: {ans_task_before}")
    print("─" * 72)

    # 4. Train Model with SFT instruction masking and Cosine Annealing LR
    steps_str = f", max_steps={args.max_steps}" if args.max_steps else ""
    print(f"\n[3/4] Fine-tuning Zexo for {args.epochs} epochs (lr={args.lr}, seq_len={args.seq_len}{steps_str}, mask_prompts={not args.no_mask_prompts})...")
    t0 = time.perf_counter()
    hist = zexo.train(
        text,
        epochs=args.epochs,
        lr=args.lr,
        seq_len=args.seq_len,
        verbose=1,
        mask_prompts=not args.no_mask_prompts,
        max_batches=args.max_steps,
    )
    duration = time.perf_counter() - t0
    print(f"  Training finished in {duration:.2f}s! Final loss: {hist['loss'][-1]:.4f}")

    # 5. Test Fine-Tuned Answer (Continual Retention Probe + Task Probe)
    zexo.reset()
    print(f"\n[4/4] Fine-Tuned Conversational Probes (After Training):")
    print("─" * 72)
    ans_retention_after = zexo.chat(retention_prompt, max_tokens=50)
    zexo.reset()
    print(f"  [Retention Probe] User: {retention_prompt}")
    print(f"                    Zexo: {ans_retention_after}")
    ans_task_after = zexo.chat(args.test_prompt, max_tokens=50)
    zexo.reset()
    print(f"  [Task Probe]      User: {args.test_prompt}")
    print(f"                    Zexo: {ans_task_after}")
    print("─" * 72)

    # 6. Save Checkpoint
    latest_bin = out_dir / "latest.bin"
    snapshot_bin = out_dir / f"{run_id}.bin"
    zexo.save(latest_bin)
    zexo.save(snapshot_bin)
    print(f"\n💾 Saved updated Zexo weights:")
    print(f"   -> Head:     {latest_bin.name} ({latest_bin.stat().st_size:,} bytes)")
    print(f"   -> Snapshot: {snapshot_bin.name}")

    # 7. Metadata with continual learning lineage
    meta_file = out_dir / "zexo_meta.json"
    meta_entry = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "tier": zexo.config.tier,
        "parameters": zexo.config.parameter_count,
        "dataset": corpus_path.name,
        "chars": len(text),
        "words": len(text.split()),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "loss_history": hist["loss"],
        "mask_prompts": not args.no_mask_prompts,
        "replay_ratio": args.replay_ratio,
        "retention_prompt": retention_prompt,
        "retention_before": ans_retention_before,
        "retention_after": ans_retention_after,
        "task_prompt": args.test_prompt,
        "reply_before": ans_task_before,
        "reply_after": ans_task_after,
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
