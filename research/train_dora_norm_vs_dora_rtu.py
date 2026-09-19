"""Head-to-head Conversational Training: Dora-Norm vs Dora-RTU.

Demonstrates the stabilization of the novel Dora architecture:
1. Dora-Norm: Stabilized Transformer with bounded dendritic gating, normalized Chebyshev KAN,
   and additive cortical reflection with residual highway preservation.
2. Dora-RTU: Byte-level O(1) state memory with Bio-Reflective KAN transformations
   and JEPA multi-objective latent prediction.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "research" / "rtu_sandbox"))

from doraneural.dora_norm import DoraNormDecoderLM
from doraneural.transformer import TensorAdamW
from research.rtu_sandbox.dora_rtu import DoraRTUConfig, DoraRTULanguageModel


def train_dora_norm_vs_rtu(
    data_path: Path,
    epochs: int = 5,
    dim: int = 96,
    hidden_dim: int = 192,
    n_layers: int = 3,
    seq_len: int = 48,
    lr: float = 1e-3,
) -> Dict[str, any]:
    text = data_path.read_text(encoding="utf-8").strip()
    raw_bytes = list(text.encode("utf-8"))
    vocab_size = 256  # Byte-level vocabulary for fair comparison

    print("=" * 80)
    print(" 🚀 CONVERSATIONAL TRAINING RUN: DORA-NORM  vs.  DORA-RTU")
    print("=" * 80)
    print(f"Corpus:             {data_path.name} ({len(text):,} chars, {len(raw_bytes):,} UTF-8 bytes)")
    print(f"Model Configuration: dim={dim}, hidden_dim={hidden_dim}, layers={n_layers}, vocab_size={vocab_size}")
    print(f"Training Settings:   epochs={epochs}, seq_len={seq_len}, lr={lr}")
    print("-" * 80)

    # 1. Instantiate Models
    np.random.seed(42)
    dora_norm = DoraNormDecoderLM(
        dim=dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )

    np.random.seed(42)
    dora_rtu_cfg = DoraRTUConfig(
        dim=dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        lr=lr * 2.0,
    )
    dora_rtu = DoraRTULanguageModel(dora_rtu_cfg)

    p_norm = sum(p.data.size for p in dora_norm.parameters())
    p_rtu = dora_rtu_cfg.parameter_count

    print(f"Model Parameters:")
    print(f"  • Dora-Norm (Stabilized Transformer): {p_norm:,} parameters")
    print(f"  • Dora-RTU  (Recurrent Trace Unit):   {p_rtu:,} parameters ({(p_rtu / p_norm)*100:.1f}% of Dora-Norm)")
    print("-" * 80)

    # 2. Slice sequence windows for Dora-Norm
    batches: List[Tuple[List[int], List[int]]] = []
    for i in range(0, len(raw_bytes) - seq_len, seq_len):
        chunk = raw_bytes[i : i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            batches.append((chunk[:-1], chunk[1:]))

    print(f"Prepared {len(batches)} sequence windows per epoch.")
    print("-" * 80)

    # 3. Pre-Training Generation Probes
    probe_prompt = "User: Who are you?\nZexo: "
    print(f"🔍 Baseline Probes (Before Training, Prompt: '{probe_prompt.strip()}'):")

    # Dora-Norm probe
    p_bytes = list(probe_prompt.encode("utf-8"))
    out_norm = list(p_bytes)
    for _ in range(35):
        ctx = out_norm[-seq_len:]
        logits = dora_norm.forward(ctx)[-1]
        out_norm.append(int(np.argmax(logits.data)))
    print(f"  [Dora-Norm (Untrained)]: {bytes(out_norm).decode('utf-8', errors='replace')}")

    # Dora-RTU probe
    out_rtu = dora_rtu.generate(probe_prompt, max_bytes=35, temperature=0.7, reset_state=True)
    print(f"  [Dora-RTU  (Untrained)]: {out_rtu}")
    print("-" * 80)

    # 4. Train Dora-Norm
    print("\n⚡ [1/2] Training Dora-Norm (Stabilized Attention + Normalized KAN + Highway Reflection)...")
    opt_norm = TensorAdamW(dora_norm.parameters(), lr=lr, weight_decay=0.01)
    norm_losses = []
    t_start_norm = time.perf_counter()

    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for x_in, y_tgt in batches:
            loss_val = dora_norm.train_batch(x_in, y_tgt, opt_norm)
            ep_loss += loss_val
        avg_loss = ep_loss / len(batches)
        norm_losses.append(avg_loss)
        print(f"   Epoch {ep}/{epochs} -> Loss: {avg_loss:.4f}")
    t_norm = time.perf_counter() - t_start_norm
    print(f"  ✓ Dora-Norm complete in {t_norm:.2f}s ({len(batches)*epochs/t_norm:.2f} windows/s) | Initial: {norm_losses[0]:.4f} -> Final: {norm_losses[-1]:.4f}")

    # 5. Train Dora-RTU
    print("\n🌀 [2/2] Training Dora-RTU (Linear Decay Recurrence + Bio-Reflective KAN + JEPA)...")
    rtu_losses = []
    t_start_rtu = time.perf_counter()
    raw_bytes_data = bytes(raw_bytes)

    # Train in chunks to allow progressive state tracking
    chunk_sz = 1024
    chunks = [raw_bytes_data[i : i + chunk_sz] for i in range(0, len(raw_bytes_data), chunk_sz)]

    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        dora_rtu.reset_state()
        for ch in chunks:
            res = dora_rtu.train_sequence(ch, reset_state=False)
            ep_loss += res["loss"]
        avg_loss = ep_loss / len(chunks)
        rtu_losses.append(avg_loss)
        print(f"   Epoch {ep}/{epochs} -> Loss: {avg_loss:.4f} (CE: {res['ce_loss']:.4f}, Latent: {res['latent_loss']:.4f})")
    t_rtu = time.perf_counter() - t_start_rtu
    print(f"  ✓ Dora-RTU complete in {t_rtu:.2f}s ({len(raw_bytes_data)*epochs/t_rtu:.1f} bytes/s) | Initial: {rtu_losses[0]:.4f} -> Final: {rtu_losses[-1]:.4f}")

    # 6. Post-Training Generation Probes
    print("\n" + "=" * 80)
    print(" ✨ POST-TRAINING GENERATION PROBES:")
    print("=" * 80)

    # Dora-Norm probe
    out_norm_post = list(p_bytes)
    for _ in range(45):
        ctx = out_norm_post[-seq_len:]
        logits = dora_norm.forward(ctx)[-1]
        out_norm_post.append(int(np.argmax(logits.data)))
    gen_norm = bytes(out_norm_post).decode("utf-8", errors="replace")
    print(f"  • Dora-Norm Output:\n    {gen_norm.replace(chr(10), ' ')}")

    # Dora-RTU probe
    dora_rtu.reset_state()
    gen_rtu = dora_rtu.generate(probe_prompt, max_bytes=45, temperature=0.3, reset_state=True)
    print(f"  • Dora-RTU Output:\n    {gen_rtu.replace(chr(10), ' ')}")

    # 7. Summary Comparison
    print("\n" + "=" * 80)
    print(" 📊 HEAD-TO-HEAD COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<32} | {'Dora-Norm (Stabilized)':<22} | {'Dora-RTU (Recurrent)':<22}")
    print("-" * 84)
    print(f"{'Model Parameters':<32} | {p_norm:<22,} | {p_rtu:<22,}")
    print(f"{'Training Duration (s)':<32} | {t_norm:<22.2f} | {t_rtu:<22.2f}")
    print(f"{'Initial Loss':<32} | {norm_losses[0]:<22.4f} | {rtu_losses[0]:<22.4f}")
    print(f"{'Final Loss (Epoch ' + str(epochs) + ')':<32} | {norm_losses[-1]:<22.4f} | {rtu_losses[-1]:<22.4f}")
    print(f"{'Total Loss Reduction':<32} | {norm_losses[0]-norm_losses[-1]:<22.4f} | {rtu_losses[0]-rtu_losses[-1]:<22.4f}")
    print(f"{'Inference Context Memory':<32} | {'O(T) KV-Cache':<22} | {'O(1) State Memory':<22}")
    print(f"{'Sequence Scaling Compute':<32} | {'O(T^2) Attention':<22} | {'O(T) Linear Recurrence':<22}")
    print(f"{'Objective Formulation':<32} | {'Next-Byte CrossEntropy':<22} | {'CE + JEPA + VICReg':<22}")
    print("=" * 80)

    return {
        "p_norm": p_norm,
        "p_rtu": p_rtu,
        "t_norm": t_norm,
        "t_rtu": t_rtu,
        "loss_norm": norm_losses[-1],
        "loss_rtu": rtu_losses[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Train Dora-Norm vs Dora-RTU on conversational data.")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "zexo" / "data" / "step1_conversational_base.txt",
        help="Path to conversational corpus.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--dim", type=int, default=96, help="Model hidden dimension.")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers.")
    args = parser.parse_args()

    train_dora_norm_vs_rtu(
        data_path=args.data,
        epochs=args.epochs,
        dim=args.dim,
        hidden_dim=args.dim * 2,
        n_layers=args.layers,
    )


if __name__ == "__main__":
    main()
