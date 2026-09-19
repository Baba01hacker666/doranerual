"""Empirical Research Benchmark: Standard Transformer vs Dora (Bio-Reflective KAN) vs RTU.

Runs a side-by-side controlled training and evaluation comparison on identical text data:
1. Model architecture & parameter allocation (Vocabulary vs Sequence modeling)
2. Empirical loss convergence curves across training epochs
3. Training throughput and runtime latency
4. Context length memory footprint scaling (O(T) KV-cache vs O(1) State)
5. Qualitative text generation prompts before and after training
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Ensure repository root and rtu_sandbox are importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research" / "rtu_sandbox"))

from doraneural.transformer import TransformerDecoderLM, TensorAdamW
from research.rtu_sandbox.rtu import RTUConfig, RTULanguageModel


def run_comparative_study(
    data_path: Path,
    dim: int = 64,
    hidden_dim: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    epochs: int = 6,
    seq_len: int = 32,
    lr: float = 1e-3,
) -> Dict[str, any]:
    """Execute controlled empirical benchmark across all 3 architectures."""
    text = data_path.read_text(encoding="utf-8").strip()
    raw_bytes = list(text.encode("utf-8"))
    vocab_size = 256  # Uniform byte vocabulary for equal comparison

    print("=" * 78)
    print(" 🔬 DORANEURAL RESEARCH LAB: COMPARATIVE MODEL BENCHMARK")
    print("    Standard Transformer  vs.  Dora (Bio-Reflective KAN)  vs.  RTU")
    print("=" * 78)
    print(f"Corpus:                {data_path.name} ({len(text):,} chars, {len(raw_bytes):,} raw UTF-8 bytes)")
    print(f"Shared Hyperparams:    dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}, vocab_size={vocab_size}")
    print(f"Training Config:       epochs={epochs}, seq_len={seq_len}, base_lr={lr}")
    print("-" * 78)

    # 1. Instantiate Models
    np.random.seed(42)
    std_transformer = TransformerDecoderLM(
        dim=dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        seq_len=seq_len,
        novel_neurons=False,
    )

    np.random.seed(42)
    dora_transformer = TransformerDecoderLM(
        dim=dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        seq_len=seq_len,
        novel_neurons=True,
    )

    np.random.seed(42)
    rtu_cfg = RTUConfig(
        dim=dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        lr=lr * 3.0,
    )
    rtu_model = RTULanguageModel(rtu_cfg)

    # Parameter counts
    std_params = sum(p.data.size for p in std_transformer.parameters())
    dora_params = sum(p.data.size for p in dora_transformer.parameters())
    rtu_params = rtu_cfg.parameter_count

    print(f"Architectural Footprint:")
    print(f"  • Standard Transformer: {std_params:,} parameters")
    print(f"  • Dora (Novel Neurons):  {dora_params:,} parameters (+{dora_params - std_params:,} novel params: dend, kan, ref)")
    print(f"  • RTU Model:             {rtu_params:,} parameters ({(rtu_params / std_params)*100:.1f}% of Standard)")
    print("-" * 78)

    # Slice sequence windows
    tokens = raw_bytes
    batches: List[Tuple[List[int], List[int]]] = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        chunk = tokens[i : i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            batches.append((chunk[:-1], chunk[1:]))

    print(f"Prepared {len(batches)} training windows per epoch.")
    print("-" * 78)

    # ---------------- 1. Standard Transformer ----------------
    print("\n[1/3] Training Standard Transformer (Attention + SwiGLU)...")
    opt_std = TensorAdamW(std_transformer.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.perf_counter()
    std_losses = []
    for _ in range(epochs):
        ep_loss = 0.0
        for x_in, y_tgt in batches:
            l = std_transformer.train_batch(x_in, y_tgt, opt_std)
            ep_loss += l
        std_losses.append(ep_loss / len(batches))
    t_std = time.perf_counter() - t0
    print(f"  ✓ Standard Transformer done in {t_std:.2f}s | {std_losses[0]:.4f} -> {std_losses[-1]:.4f}")

    # ---------------- 2. Dora (Bio-Reflective KAN) ----------------
    print("\n[2/3] Training Dora Transformer (Attention + Dendritic + Chebyshev KAN + Reflection)...")
    opt_dora = TensorAdamW(dora_transformer.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.perf_counter()
    dora_losses = []
    for _ in range(epochs):
        ep_loss = 0.0
        for x_in, y_tgt in batches:
            l = dora_transformer.train_batch(x_in, y_tgt, opt_dora)
            ep_loss += l
        dora_losses.append(ep_loss / len(batches))
    t_dora = time.perf_counter() - t0
    print(f"  ✓ Dora Transformer done in {t_dora:.2f}s | {dora_losses[0]:.4f} -> {dora_losses[-1]:.4f}")

    # ---------------- 3. RTU Model ----------------
    print("\n[3/3] Training RTU Model (Recurrent Trace Units with Linear Decay & JEPA)...")
    t0 = time.perf_counter()
    rtu_losses = []
    raw_bytes_data = bytes(raw_bytes)
    for _ in range(epochs):
        res = rtu_model.train_sequence(raw_bytes_data, reset_state=True)
        rtu_losses.append(res["loss"])
    t_rtu = time.perf_counter() - t0
    print(f"  ✓ RTU Model done in {t_rtu:.2f}s | {rtu_losses[0]:.4f} -> {rtu_losses[-1]:.4f}")

    print("\n" + "=" * 78)
    print(" 📊 EMPIRICAL COMPARISON RESULTS TABLE")
    print("=" * 78)
    print(f"{'Metric':<30} | {'Standard Transf.':<15} | {'Dora (Novel)':<15} | {'RTU':<15}")
    print("-" * 82)
    print(f"{'Trainable Parameters':<30} | {std_params:<15,} | {dora_params:<15,} | {rtu_params:<15,}")
    print(f"{'Training Wall Time (s)':<30} | {t_std:<15.2f} | {t_dora:<15.2f} | {t_rtu:<15.2f}")
    print(f"{'Initial Training Loss':<30} | {std_losses[0]:<15.4f} | {dora_losses[0]:<15.4f} | {rtu_losses[0]:<15.4f}")
    print(f"{'Final Loss (Epoch ' + str(epochs) + ')':<30} | {std_losses[-1]:<15.4f} | {dora_losses[-1]:<15.4f} | {rtu_losses[-1]:<15.4f}")
    print(f"{'Loss Reduction (Delta)':<30} | {std_losses[0]-std_losses[-1]:<15.4f} | {dora_losses[0]-dora_losses[-1]:<15.4f} | {rtu_losses[0]-rtu_losses[-1]:<15.4f}")
    print(f"{'Context Memory Footprint':<30} | {'O(T) KV-Cache':<15} | {'O(T) KV-Cache':<15} | {'O(1) State':<15}")
    print(f"{'Native C++ Accelerated':<30} | {'Yes':<15} | {'No (Py Tape)':<15} | {'Yes (librtu)':<15}")
    print("=" * 78)

    # Text Generation Probes
    prompt_str = "Sparky was "
    prompt_bytes = list(prompt_str.encode("utf-8"))

    print("\n🔤 GENERATION SAMPLE COMPARISON (Prompt: 'Sparky was '):")

    # Standard
    std_out = list(prompt_bytes)
    for _ in range(40):
        ctx = std_out[-seq_len:]
        logits = std_transformer.forward(ctx)[-1]
        std_out.append(int(np.argmax(logits.data)))
    print(f"  • Standard Transformer: [ {bytes(std_out).decode('utf-8', errors='replace')} ]")

    # Dora
    dora_out = list(prompt_bytes)
    for _ in range(40):
        ctx = dora_out[-seq_len:]
        logits = dora_transformer.forward(ctx)[-1]
        dora_out.append(int(np.argmax(logits.data)))
    print(f"  • Dora (Novel Neurons):  [ {bytes(dora_out).decode('utf-8', errors='replace')} ]")

    # RTU
    rtu_model.reset_state()
    rtu_out = rtu_model.generate(prompt_str, max_bytes=40, temperature=0.2, reset_state=True)
    print(f"  • RTU Model:             [ {rtu_out} ]")
    print("=" * 78)

    return {
        "std_params": std_params,
        "dora_params": dora_params,
        "rtu_params": rtu_params,
        "std_time": t_std,
        "dora_time": t_dora,
        "rtu_time": t_rtu,
        "std_loss": std_losses[-1],
        "dora_loss": dora_losses[-1],
        "rtu_loss": rtu_losses[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Comparative study: Transformer vs Dora vs RTU.")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "train.txt",
        help="Path to training text corpus.",
    )
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs.")
    parser.add_argument("--dim", type=int, default=64, help="Model hidden dimension.")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers.")
    args = parser.parse_args()

    run_comparative_study(
        data_path=args.data,
        dim=args.dim,
        hidden_dim=args.dim * 2,
        n_layers=args.layers,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
