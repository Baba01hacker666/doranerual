#!/usr/bin/env python3
"""Zexo AI: Base Checkpoint Weight Initializer.

Initializes mathematically calibrated binary weights (.bin) for any Zexo
architectural tier (micro, mini, chat, base, large) so training can begin
from a clean, grounded base state.
"""

import argparse
from pathlib import Path
import struct
import sys
import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from zexo.config import ZexoConfig


def initialize_zexo_checkpoint(
    config: ZexoConfig,
    output_path: Path,
    seed: int = 42,
) -> Path:
    """Generate a clean binary model checkpoint for Zexo."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    p = config

    vocab_sign = p.vocab_size  # Tied embeddings

    header = struct.pack(
        "<7i",
        p.dim,
        p.hidden_dim,
        p.n_layers,
        p.n_heads,
        p.n_kv_heads,
        vocab_sign,
        p.seq_len,
    )

    # Variance scaling for deep transformer stability
    emb_std = 1.0 / np.sqrt(p.dim)
    proj_std = 0.02 / np.sqrt(2.0 * p.n_layers)
    # Output projections must NOT be exact zeros: a zero wo (attention out)
    # or w2 (FFN down) makes those residual branches contribute nothing, so
    # with a frozen base (LoRA) the adapters there receive zero gradient
    # forever. Small nonzero init keeps every gradient path alive.

    print(f"🔧 Initializing Zexo-{config.tier.capitalize()} Base Weights...")
    print(f"   Architecture: {p.dim} dim, {p.hidden_dim} hidden, {p.n_layers} layers, {p.n_heads} heads, {p.n_kv_heads} kv-heads")
    print(f"   Parameters:   {config.parameter_count:,}")
    print(f"   Destination:  {output_path}")

    with open(output_path, "wb") as f:
        f.write(header)
        # 1. Token embeddings
        rng.normal(0.0, emb_std, (p.vocab_size, p.dim)).astype(np.float32).tofile(f)
        # 2. Attention RMSNorm weights
        np.ones((p.n_layers, p.dim), dtype=np.float32).tofile(f)
        # 3. Query projection
        rng.normal(0.0, proj_std, (p.n_layers, p.dim, p.dim)).astype(np.float32).tofile(f)
        # 4. Key projection
        rng.normal(0.0, proj_std, (p.n_layers, p.kv_dim, p.dim)).astype(np.float32).tofile(f)
        # 5. Value projection
        rng.normal(0.0, proj_std, (p.n_layers, p.kv_dim, p.dim)).astype(np.float32).tofile(f)
        # 6. Output projection (nonzero: see note above)
        rng.normal(0.0, proj_std, (p.n_layers, p.dim, p.dim)).astype(np.float32).tofile(f)
        # 7. FFN RMSNorm weights
        np.ones((p.n_layers, p.dim), dtype=np.float32).tofile(f)
        # 8. Gate projection (w1)
        rng.normal(0.0, proj_std, (p.n_layers, p.hidden_dim, p.dim)).astype(np.float32).tofile(f)
        # 9. Down projection (w2, nonzero: see note above)
        rng.normal(0.0, proj_std, (p.n_layers, p.dim, p.hidden_dim)).astype(np.float32).tofile(f)
        # 10. Up projection (w3)
        rng.normal(0.0, proj_std, (p.n_layers, p.hidden_dim, p.dim)).astype(np.float32).tofile(f)
        # 11. Final RMSNorm
        np.ones(p.dim, dtype=np.float32).tofile(f)
        # 12. Legacy frequency table padding
        np.zeros(p.seq_len * p.head_size, dtype=np.float32).tofile(f)
        # 13. Novel neuron weights if enabled
        if config.novel_neurons:
            # Dendritic gating weights (n_layers, hidden_dim, dim)
            rng.normal(0.0, proj_std * 0.1, (p.n_layers, p.hidden_dim, p.dim)).astype(np.float32).tofile(f)
            # Chebyshev KAN polynomial coefficients (n_layers, 3, hidden_dim)
            kan_std = 0.05 / (np.sqrt(p.hidden_dim) * 4.0)
            rng.normal(0.0, kan_std, (p.n_layers, 3, p.hidden_dim)).astype(np.float32).tofile(f)
            # Cortical reflection projection (n_layers, dim, dim)
            rng.normal(0.0, proj_std * 0.1, (p.n_layers, p.dim, p.dim)).astype(np.float32).tofile(f)

    # Save JSON configuration alongside binary
    cfg_path = output_path.with_suffix(".json")
    config.save(cfg_path)
    print(f"✅ Checkpoint successfully created: {output_path.name} ({output_path.stat().st_size:,} bytes)")
    print(f"   Config saved: {cfg_path.name}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Initialize base weights for Zexo AI models.")
    parser.add_argument(
        "--tier", "-t",
        choices=["micro", "mini", "chat", "base", "large", "dora"],
        default="micro",
        help="Zexo tier to initialize (default: micro).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output filepath (defaults to zexo/checkpoints/zexo_<tier>_base.bin).",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for weight initialization.",
    )
    args = parser.parse_args()

    config = ZexoConfig.from_tier(args.tier)
    out_dir = REPO_ROOT / "zexo" / "checkpoints"
    out_file = Path(args.output) if args.output else (out_dir / f"zexo_{args.tier}_base.bin")

    initialize_zexo_checkpoint(config, out_file, seed=args.seed)


if __name__ == "__main__":
    main()
