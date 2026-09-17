#!/usr/bin/env python3
"""Unified LLM Training & Continuous Checkpointing Runner.

Can be run locally or inside GitHub Actions workflows. Supports resuming
from prior checkpoint weights, training on a new text dataset, and saving
the updated model weights as `.bin` checkpoints with metadata for continuous
iterative training.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

# Add repository root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import doraneural as dn


def calculate_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def resolve_base_model(
    checkpoint_arg: Optional[str],
    tokenizer_arg: Optional[str],
    output_dir: Path,
) -> Tuple[dn.LlamaLLM, str, Path]:
    """Resolve which checkpoint and tokenizer to start training from.

    Priority:
    1. Explicit --checkpoint argument if provided and exists.
    2. checkpoints/latest.bin if it exists in output_dir (continual training loop).
    3. Local base model in models/stories260K/stories260K.bin.
    4. Auto-download stories260K from Hugging Face.
    """
    # 1. Check explicit checkpoint
    if checkpoint_arg:
        ckpt_path = Path(checkpoint_arg)
        if ckpt_path.exists():
            print(f"🎯 Using explicitly specified checkpoint: {ckpt_path}")
            tok_path = Path(tokenizer_arg) if tokenizer_arg and Path(tokenizer_arg).exists() else (
                REPO_ROOT / "models" / "stories260K" / "tok512.bin"
            )
            llm = dn.LlamaLLM(model_path=ckpt_path, tokenizer_path=tok_path, backend="auto")
            return llm, f"Specified checkpoint: {ckpt_path.name}", ckpt_path
        else:
            print(f"⚠️ Warning: Specified checkpoint '{checkpoint_arg}' not found.")

    # 2. Check for latest.bin in output_dir (for automated GitHub Actions iterative training)
    latest_path = output_dir / "latest.bin"
    if latest_path.exists() and latest_path.stat().st_size > 0:
        print(f"🔄 CONTINUAL TRAINING: Resuming from previous run checkpoint: {latest_path}")
        tok_path = Path(tokenizer_arg) if tokenizer_arg and Path(tokenizer_arg).exists() else (
            REPO_ROOT / "models" / "stories260K" / "tok512.bin"
        )
        llm = dn.LlamaLLM(model_path=latest_path, tokenizer_path=tok_path, backend="auto")
        return llm, f"Continued from: {latest_path.name}", latest_path

    # 3. Check local base model in models/stories260K/
    local_base = REPO_ROOT / "models" / "stories260K" / "stories260K.bin"
    local_tok = REPO_ROOT / "models" / "stories260K" / "tok512.bin"
    if local_base.exists() and local_tok.exists():
        print(f"📦 Using local base model: {local_base}")
        llm = dn.LlamaLLM(model_path=local_base, tokenizer_path=local_tok, backend="auto")
        return llm, "Base model: stories260K (local)", local_base

    # 4. Download pretrained base model
    print("🌐 Downloading pretrained base model (stories260K) from Hugging Face...")
    llm = dn.load_pretrained_llm("stories260K", backend="auto")
    return llm, "Base model: stories260K (downloaded)", Path(llm.model_path)


def load_dataset(
    data_arg: str,
    hf_dataset: Optional[str] = None,
    hf_split: str = "train",
    hf_column: Optional[str] = None,
    hf_samples: int = 100,
) -> Tuple[str, str, int]:
    """Read training corpus from a file, directory, or Hugging Face dataset."""
    target_hf = hf_dataset
    if not target_hf and dn.is_hf_dataset_identifier(data_arg):
        target_hf = data_arg

    if target_hf:
        print(f"🤗 Auto-resolving Hugging Face dataset: '{target_hf}' (split={hf_split}, max_samples={hf_samples})...")
        local_path = dn.download_hf_dataset(
            target_hf,
            split=hf_split,
            text_column=hf_column,
            max_samples=hf_samples,
        )
        data_arg = str(local_path)

    data_path = Path(data_arg)
    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset path not found: {data_path}")

    if data_path.is_file():
        dataset_name = data_path.name
        with open(data_path, "r", encoding="utf-8", errors="replace") as f:
            corpus = f.read()
    else:
        # Directory of .txt files
        dataset_name = f"Directory: {data_path.name}"
        txt_files = sorted(data_path.glob("*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt files found in dataset directory: {data_path}")
        corpus_parts = []
        for p in txt_files:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                corpus_parts.append(f.read())
        corpus = "\n\n".join(corpus_parts)

    word_count = len(corpus.split())
    return corpus, dataset_name, word_count


def write_summary(
    summary_path: Path,
    metadata: dict,
    loss_history: list,
    gen_before: str,
    gen_after: str,
    prompt: str,
) -> None:
    """Generate Markdown summary for GitHub Actions or local reporting."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    losses = loss_history
    initial_loss = losses[0] if losses else 0.0
    final_loss = losses[-1] if losses else 0.0
    loss_delta = initial_loss - final_loss
    loss_pct = (loss_delta / max(1e-8, initial_loss)) * 100.0 if initial_loss > 0 else 0.0

    lines = [
        "# 🚀 doraneural: Continuous LLM Training Report",
        "",
        f"**Run ID:** `{metadata['run_id']}` | **Timestamp:** `{metadata['timestamp_utc']}`",
        "",
        "## 📊 Training Overview",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Engine Backend** | `{metadata['backend'].upper()}` |",
        f"| **Dataset** | `{metadata['dataset_name']}` ({metadata['word_count']} words, ~{metadata['estimated_tokens']} tokens) |",
        f"| **Prior Checkpoint** | `{metadata['parent_checkpoint']}` |",
        f"| **Output Checkpoint** | `{metadata['output_checkpoint']}` ({metadata['checkpoint_bytes']:,} bytes) |",
        f"| **Epochs** | `{metadata['epochs']}` |",
        f"| **Learning Rate** | `{metadata['lr']}` |",
        f"| **Sequence Length** | `{metadata['seq_len']}` |",
        f"| **Training Duration** | `{metadata['duration_seconds']:.2f}s` |",
        f"| **Initial Loss** | `{initial_loss:.4f}` |",
        f"| **Final Loss** | `{final_loss:.4f}` ({loss_pct:+.1f}%) |",
        "",
        "## 📉 Loss Progression",
        "",
        "| Epoch | Cross-Entropy Loss |",
        "| :---: | :---: |",
    ]

    for ep, l in enumerate(losses, 1):
        lines.append(f"| {ep} | {l:.4f} |")

    lines.extend([
        "",
        "## 📝 Text Generation Comparison",
        "",
        f"**Test Prompt:** *`\"{prompt}\"`*",
        "",
        "### ⏳ Before Training (Prior Weights):",
        "> " + gen_before.replace("\n", "\n> "),
        "",
        "### ✨ After Training (Updated Weights):",
        "> " + gen_after.replace("\n", "\n> "),
        "",
        "## 🔁 Continuous Training Instructions",
        "To continue training this model on a **new dataset** in subsequent workflow runs:",
        "1. **Automatic:** The workflow uploads `latest.bin` to the GitHub Actions cache and artifacts.",
        "2. **Manual UI:** When triggering the workflow via `workflow_dispatch`, keep `resume_from_cache: true` or pass the artifact path.",
        "3. **Local CLI:** Run `python scripts/train_llm.py --checkpoint checkpoints/latest.bin --data path/to/new_dataset.txt`",
    ])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"📋 Step summary written to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="doraneural: Train / Fine-tune LLaMA models with continuous checkpointing."
    )
    parser.add_argument(
        "--data", "-d",
        default="data/train.txt",
        help="Path to training text file, directory of .txt files, or Hugging Face dataset identifier/URL (default: data/train.txt).",
    )
    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Hugging Face dataset identifier (e.g. 'roneneldan/TinyStories') or direct URL.",
    )
    parser.add_argument(
        "--hf-split",
        default="train",
        help="Hugging Face dataset split to download (default: 'train').",
    )
    parser.add_argument(
        "--hf-column",
        default=None,
        help="Column name to extract text from (auto-detected if omitted).",
    )
    parser.add_argument(
        "--hf-samples",
        type=int,
        default=100,
        help="Maximum number of dataset rows/samples to download (default: 100).",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="Path to base/prior checkpoint .bin file to resume from. If omitted, checks for latest.bin or base model.",
    )
    parser.add_argument(
        "--tokenizer", "-t",
        default=None,
        help="Path to tokenizer file (default: models/stories260K/tok512.bin).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="checkpoints",
        help="Directory to store updated checkpoints and metadata (default: checkpoints).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional identifier tag for this training run.",
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
        help="Learning rate for AdamW updates (default: 0.0005).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
        help="Sequence chunk length for training (default: 16).",
    )
    parser.add_argument(
        "--prompt", "-p",
        default="Once upon a time, there was a little robot",
        help="Prompt to evaluate text generation before and after training.",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Optional markdown file path to write training summary report (defaults to GITHUB_STEP_SUMMARY if set).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "cpp", "numpy"],
        default="auto",
        help="Execution engine (default: auto).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    print("=" * 72)
    print("🤖 doraneural: Continuous LLM Training & Checkpoint Engine")
    print("=" * 72)
    print(f"Run ID:         {run_id}")
    print(f"Timestamp UTC:  {timestamp_utc}")
    print(f"Output Dir:     {output_dir.resolve()}")
    print("-" * 72)

    # 1. Load Dataset
    print(f"\n[1/5] Loading training corpus from: {args.hf_dataset or args.data}")
    corpus, dataset_name, word_count = load_dataset(
        data_arg=args.data,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_column=args.hf_column,
        hf_samples=args.hf_samples,
    )
    print(f"  Dataset Name: {dataset_name}")
    print(f"  Words:        {word_count:,}")
    print(f"  Characters:   {len(corpus):,}")

    # 2. Resolve Base Model Checkpoint
    print(f"\n[2/5] Resolving model checkpoint...")
    llm, parent_desc, parent_path = resolve_base_model(args.checkpoint, args.tokenizer, output_dir)
    print(f"  Engine:       {llm.backend.upper()}")
    print(f"  Dimensions:   dim={llm.config.dim}, layers={llm.config.n_layers}, vocab={llm.config.vocab_size}")
    tokens_encoded = llm.tokenizer.encode(corpus, bos=False)
    print(f"  Token count:  {len(tokens_encoded):,} tokens")

    # 3. Test Baseline Generation (Before Training)
    print(f"\n[3/5] Baseline Generation (Before Training):")
    print("-" * 72)
    print(f"Prompt: \"{args.prompt}\"")
    t0_gen = time.perf_counter()
    gen_before = llm.generate(prompt=args.prompt, max_tokens=40, temperature=0.7)
    print(f"Output: {gen_before}")
    print("-" * 72)

    # 4. Train Model
    print(f"\n[4/5] Training LLM for {args.epochs} epochs (lr={args.lr}, seq_len={args.seq_len})...")
    t0_train = time.perf_counter()
    history = llm.train(
        text=corpus,
        epochs=args.epochs,
        lr=args.lr,
        seq_len=args.seq_len,
        verbose=1,
    )
    duration_train = time.perf_counter() - t0_train
    print(f"  Training finished in {duration_train:.2f}s!")

    # 5. Test Fine-Tuned Generation (After Training)
    print(f"\n[5/5] Fine-Tuned Generation (After Training):")
    print("-" * 72)
    print(f"Prompt: \"{args.prompt}\"")
    gen_after = llm.generate(prompt=args.prompt, max_tokens=40, temperature=0.7)
    print(f"Output: {gen_after}")
    print("-" * 72)

    # 6. Save Updated Checkpoints
    latest_ckpt_path = output_dir / "latest.bin"
    history_ckpt_path = output_dir / f"model_{run_id}.bin"

    print(f"\n💾 Saving updated model checkpoints...")
    llm.save(latest_ckpt_path)
    llm.save(history_ckpt_path)
    print(f"  -> Saved active head: {latest_ckpt_path} ({latest_ckpt_path.stat().st_size:,} bytes)")
    print(f"  -> Saved snapshot:    {history_ckpt_path}")

    # 7. Write Checkpoint Metadata JSON
    meta_path = output_dir / "checkpoint_meta.json"
    metadata = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "backend": llm.backend,
        "dataset_name": dataset_name,
        "dataset_path": str(Path(args.data).resolve()),
        "word_count": word_count,
        "estimated_tokens": len(tokens_encoded),
        "parent_checkpoint": parent_desc,
        "parent_checkpoint_path": str(parent_path.resolve()) if parent_path else None,
        "output_checkpoint": str(latest_ckpt_path.name),
        "output_snapshot": str(history_ckpt_path.name),
        "checkpoint_bytes": latest_ckpt_path.stat().st_size,
        "checkpoint_sha256": calculate_sha256(latest_ckpt_path),
        "epochs": args.epochs,
        "lr": args.lr,
        "seq_len": args.seq_len,
        "duration_seconds": duration_train,
        "loss_history": history["loss"],
        "prompt": args.prompt,
        "gen_before": gen_before,
        "gen_after": gen_after,
    }

    # If previous metadata exists, append to history
    meta_history = []
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                if isinstance(prev_data, dict) and "runs" in prev_data:
                    meta_history = prev_data["runs"]
                elif isinstance(prev_data, dict):
                    meta_history = [prev_data]
        except Exception:
            meta_history = []

    meta_history.append(metadata)
    full_meta = {
        "latest": metadata,
        "total_runs": len(meta_history),
        "runs": meta_history,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(full_meta, f, indent=2)
    print(f"  -> Metadata saved:     {meta_path}")

    # 8. Output Summary Report
    summary_file = args.summary_file or os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        write_summary(
            Path(summary_file),
            metadata,
            history["loss"],
            gen_before,
            gen_after,
            args.prompt,
        )

    print("\n" + "=" * 72)
    print("✅ Continuous training cycle completed successfully!")
    print(f"   Model is ready to be loaded for inference or next training run.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
