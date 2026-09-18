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
    expanded_path = REPO_ROOT / "zexo" / "data" / "step1_conversational_expanded.txt"
    quality_train_path = REPO_ROOT / "zexo" / "data" / "zexo_quality_v1_train.txt"
    default_dataset = str(quality_train_path if quality_train_path.exists() else (expanded_path if expanded_path.exists() else base_path))

    parser = argparse.ArgumentParser(description="Train and fine-tune Zexo AI on conversational data.")
    parser.add_argument(
        "--data", "-d",
        default=default_dataset,
        help=f"Path to training dialogue file or Hugging Face dataset ID (default: {Path(default_dataset).name}).",
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
        help="Hugging Face dataset identifier (e.g. 'yahma/alpaca-cleaned' or 'roneneldan/TinyStories') to auto-download.",
    )
    parser.add_argument(
        "--hf-config",
        default=None,
        help="Hugging Face dataset subset/configuration (e.g. '20231101.simple' or 'wikitext-2-raw-v1').",
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
        choices=["micro", "mini", "chat", "base", "large", "dora"],
        default="mini",
        help="Zexo tier architecture (default: mini).",
    )
    parser.add_argument(
        "--novel-neurons",
        action="store_true",
        help="Enable novel Bio-Reflective KAN neurons (dendritic gating, Chebyshev KAN, reflection).",
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
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW/L2 weight decay (default: 0.01).",
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
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm (default: 1.0, 0 to disable).",
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
        "--full-backprop",
        action="store_true",
        help="Train all transformer weights with native C++ (NumPy reference remains available via --numpy-full).",
    )
    parser.add_argument(
        "--numpy-full",
        action="store_true",
        help="Force the slower NumPy full-backprop reference path instead of native C++.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="Train a LoRA adapter instead of changing the pretrained base (for example: 8).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="LoRA scaling alpha (default: 16).",
    )
    parser.add_argument(
        "--lora-targets",
        default="q,v",
        help="Comma-separated LoRA targets: q,k,v,o,w1,w2,w3,lm_head (default: q,v).",
    )
    parser.add_argument(
        "--adapter-output",
        default=None,
        help="Optional .npz path for saving the trained LoRA adapter.",
    )
    parser.add_argument(
        "--adapter-input",
        default=None,
        help="Optional existing .npz LoRA adapter to load on the checkpoint before training/evaluation.",
    )
    parser.add_argument(
        "--threads", "--num-threads",
        dest="threads",
        type=int,
        default=None,
        help="Native OpenMP thread count. Any positive value is accepted (for example 200 or 3000); default keeps normal runtime auto-selection.",
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
    if args.full_backprop and args.lora_rank is not None:
        parser.error("--full-backprop and --lora-rank are mutually exclusive")
    if args.lora_rank is not None and args.lora_rank <= 0:
        parser.error("--lora-rank must be positive")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be a positive integer")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Thread selection is deliberately not capped to os.cpu_count(): OpenMP
    # accepts larger pools for machines/containers that expose CPUs later.
    # The default remains the normal runtime-selected CPU count.
    requested_threads = args.threads
    if requested_threads is None and args.fast:
        requested_threads = os.cpu_count() or 4
    if requested_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(requested_threads)
        os.environ["DORANEURAL_NUM_THREADS"] = str(requested_threads)
        available_cpus = os.cpu_count() or 1
        if requested_threads > available_cpus:
            print(
                f"⚠️ Requested {requested_threads:,} threads but this process currently exposes "
                f"{available_cpus:,} CPU(s); this may oversubscribe and run slower."
            )
    if args.fast:
        os.environ["PYTHONUNBUFFERED"] = "1"
        if args.seq_len == 64:
            args.seq_len = 128

    run_id = args.tag or datetime.now(timezone.utc).strftime("zexo_%Y%m%d_%H%M%S")
    timestamp = datetime.now(timezone.utc).isoformat()

    print("═" * 72)
    print(" 🤖 Zexo AI Conversational Training Engine")
    print("═" * 72)
    print(f"Run ID:         {run_id}")
    is_novel = args.novel_neurons or args.tier.lower() == "dora"
    print(f"Target Tier:    {args.tier.upper()}")
    print(f"Novel Neurons:  {is_novel} (Bio-Reflective KAN)")
    print(f"From Scratch:   {args.from_scratch}")
    mode = "FULL TRANSFORMER BACKPROP" if args.full_backprop else (
        f"LORA ADAPTER (rank {args.lora_rank})" if args.lora_rank is not None else "HEAD-ONLY FAST PATH"
    )
    print(f"Training Mode:  {mode}")
    print(f"Weight Decay:   {args.weight_decay}")
    print(f"Thread Request: {f'{requested_threads:,}' if requested_threads is not None else 'auto (normal CPU runtime)'}")
    print(f"Fast Mode:      {args.fast}")
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
            print(f"🤗 Downloading training dataset from Hugging Face: '{target_hf}' (config={args.hf_config})...")
            corpus_path = dn.download_hf_dataset(target_hf, config=args.hf_config, max_samples=args.hf_samples)
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
    if is_novel:
        zexo.config.novel_neurons = True
        zexo.llm.config.novel_neurons = True
    if args.threads is not None:
        # Apply explicitly after engine construction too; this covers a
        # previously loaded shared library and makes the effective setting visible.
        zexo.llm.set_num_threads(args.threads)
    if args.adapter_input:
        zexo.load_lora(args.adapter_input)
        print(f"  Loaded LoRA adapter: {args.adapter_input}")
    print(f"  Architecture: {zexo.config.dim} dim, {zexo.config.n_layers} layers, {zexo.config.parameter_count:,} params")
    print(f"  Backend:      {zexo.llm.backend.upper()}")
    print(f"  Native threads: {zexo.llm.num_threads:,} (effective; NumPy fallback reports 1)")

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
        weight_decay=args.weight_decay,
        seq_len=args.seq_len,
        verbose=1,
        mask_prompts=not args.no_mask_prompts,
        max_batches=args.max_steps,
        eval_text=eval_text,
        validation_split=args.validation_split if eval_text is None else 0.0,
        stride=args.stride,
        seed=args.seed,
        max_eval_steps=args.max_eval_steps,
        full_backprop=args.full_backprop,
        native_full=not args.numpy_full,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=[item.strip() for item in args.lora_targets.split(",") if item.strip()],
        adapter_path=args.adapter_output,
        grad_clip=args.grad_clip,
    )
    duration = time.perf_counter() - t0
    final_report = f"Final loss: {hist['loss'][-1]:.4f}"
    if "val_loss" in hist:
        final_report += f" | Final validation loss: {hist['val_loss'][-1]:.4f}"
    print(f"  Training finished in {duration:.2f}s! {final_report}")

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
        "evaluation_dataset": eval_path.name if eval_path is not None else None,
        "chars": len(text),
        "words": len(text.split()),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seq_len": args.seq_len,
        "stride": args.stride or args.seq_len,
        "seed": args.seed,
        "training_mode": "full_backprop" if args.full_backprop else (
            "lora" if args.lora_rank is not None else "head_only"
        ),
        "full_backprop_backend": "numpy" if args.numpy_full else "native_cpp",
        "threads_requested": requested_threads,
        "threads_effective": zexo.llm.num_threads,
        "available_cpus": os.cpu_count(),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha if args.lora_rank is not None else None,
        "lora_targets": [item.strip() for item in args.lora_targets.split(",") if item.strip()] if args.lora_rank is not None else None,
        "adapter_path": hist.get("adapter_path"),
        "adapter_input": args.adapter_input,
        "loss_history": hist["loss"],
        "val_loss_history": hist.get("val_loss"),
        "training_tokens": hist.get("tokens"),
        "windows_per_epoch": hist.get("windows_per_epoch"),
        "total_training_steps": hist.get("total_steps"),
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
