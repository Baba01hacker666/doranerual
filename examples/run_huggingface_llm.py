"""Pretrained LLaMA Language Model Demo in Pure NumPy.

Demonstrates:
1. Downloading and loading a pretrained LLaMA model checkpoint from Hugging Face Hub.
2. Running autoregressive transformer decoder inference in pure NumPy with:
   - RoPE (Rotary Positional Embeddings)
   - RMSNorm
   - Key-Value Cache
   - Multi-Head Attention
   - SwiGLU Feed-Forward Network
   - SentencePiece Byte-Fallback BPE Tokenizer
3. Streaming generation with top-p nucleus and temperature sampling.
4. Achieving high token generation throughput on pure CPU with 0 heavy dependencies.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import doraneural as dn


def main():
    print("=" * 70)
    print("📖 doraneural: Pretrained Hugging Face LLM in Pure NumPy")
    print("=" * 70)
    print("Architecture: LLaMA (RoPE + RMSNorm + KV-Cache + SwiGLU FFN)")
    print("Weight Source: Hugging Face (karpathy/tinyllamas - stories260K)")
    print("Dependencies: Pure NumPy + Python standard library (0 PyTorch/Transformers)")
    print("=" * 70)

    # 1. Download and load model
    print("\n[1/3] Loading model and SentencePiece tokenizer...")
    t0 = time.perf_counter()
    llm = dn.load_pretrained_llm("stories260K")
    t_load = time.perf_counter() - t0

    cfg = llm.config
    print(f"Loaded in {t_load:.2f}s!")
    print(f"  • Hidden Dimension:   {cfg.dim}")
    print(f"  • FFN Hidden Dim:     {cfg.hidden_dim}")
    print(f"  • Transformer Layers: {cfg.n_layers}")
    print(f"  • Attention Heads:    {cfg.n_heads} (Query) / {cfg.n_kv_heads} (Key/Value)")
    print(f"  • Head Dimension:     {cfg.head_size}")
    print(f"  • Vocabulary Size:    {cfg.vocab_size} tokens")
    print(f"  • Max Sequence:       {cfg.seq_len} tokens")

    # 2. Test generation prompts
    prompts = [
        "Once upon a time, Lily found a tiny kitten in the garden.",
        "One sunny morning, Tim had a big red toy car.",
    ]

    print("\n[2/3] Streaming Autoregressive Story Generation:")
    print("=" * 70)

    for idx, prompt in enumerate(prompts, 1):
        print(f"\n--- Story {idx} (Temperature: 0.7, Top-p: 0.85) ---")
        sys.stdout.write(f"\033[1m{prompt}\033[0m")
        sys.stdout.flush()

        t_gen_start = time.perf_counter()
        token_count = 0

        # Stream tokens one by one
        for piece in llm.generate(
            prompt=prompt,
            max_tokens=75,
            temperature=0.7,
            top_p=0.85,
            stream=True,
        ):
            sys.stdout.write(piece)
            sys.stdout.flush()
            token_count += 1

        t_gen = time.perf_counter() - t_gen_start
        tok_per_sec = token_count / max(1e-4, t_gen)
        print(f"\n\033[90m[{token_count} tokens in {t_gen:.2f}s | {tok_per_sec:.1f} tok/s]\033[0m\n")

    # 3. Interactive prompt
    print("=" * 70)
    print("[3/3] Interactive Mode (Try your own prompt!):")
    print("Type a prompt and press Enter (or press Enter for default):")
    try:
        user_prompt = input("Prompt > ").strip()
    except (EOFError, KeyboardInterrupt):
        user_prompt = ""

    if not user_prompt:
        user_prompt = "There once was a brave little puppy who"

    print(f"\nGenerating from prompt: \"{user_prompt}\"...\n")
    sys.stdout.write(f"\033[1m{user_prompt}\033[0m")
    sys.stdout.flush()

    for piece in llm.generate(
        prompt=user_prompt,
        max_tokens=90,
        temperature=0.7,
        top_p=0.9,
        stream=True,
    ):
        sys.stdout.write(piece)
        sys.stdout.flush()

    print("\n\n" + "=" * 70)
    print("Story generation complete! Pure NumPy LLaMA inference verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
