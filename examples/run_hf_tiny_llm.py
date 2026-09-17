#!/usr/bin/env python3
"""doraneural: Load any HuggingFace LLM from safetensors (zero PyTorch dependency).

This example demonstrates loading arnir0/Tiny-LLM (10M parameters, 1 layer, 32K vocab)
directly from HuggingFace Hub using doraneural's pure NumPy + C++ engine.

No PyTorch. No transformers library. Just NumPy + our own safetensors parser.

Usage:
    python examples/run_hf_tiny_llm.py
"""

import time
from doraneural import load_pretrained_llm

PROMPTS = [
    "The cat sat on the",
    "Once upon a time",
    "According to all known laws of aviation",
    "In a galaxy far far away",
]


def main():
    print("=" * 70)
    print("doraneural: HuggingFace SafeTensors LLM (arnir0/Tiny-LLM)")
    print("=" * 70)
    print()

    # One-liner: auto-downloads model.safetensors + tokenizer.json from HF Hub
    t0 = time.time()
    llm = load_pretrained_llm("arnir0/Tiny-LLM")
    load_time = time.time() - t0

    p = llm.config
    print(f"Model loaded in {load_time:.2f}s")
    print(f"  Architecture: LlamaForCausalLM")
    print(f"  Parameters:   ~10M (dim={p.dim}, layers={p.n_layers}, heads={p.n_heads}, kv_heads={p.n_kv_heads})")
    print(f"  Vocab:        {p.vocab_size:,} tokens (SentencePiece BPE)")
    print(f"  Context:      {p.seq_len} tokens")
    print(f"  RoPE:         {p.rope_type} (HuggingFace split-half)")
    print(f"  Weights:      SafeTensors F16 → float32")
    print(f"  Backend:      {llm.backend.upper()}")
    print(f"  Tokenizer:    {type(llm.tokenizer).__name__}")
    print()

    # Tokenizer demo
    text = "Hello, world! This is doraneural."
    tokens = llm.tokenizer.encode(text)
    decoded = llm.tokenizer.decode(tokens)
    print(f"Tokenizer test: {repr(text)}")
    print(f"  Encoded: {tokens}")
    print(f"  Decoded: {repr(decoded)}")
    print()

    # Generation demo
    print("-" * 70)
    total_tokens = 0
    total_time = 0.0

    for prompt in PROMPTS:
        t0 = time.time()
        output = llm.generate(prompt=prompt, max_tokens=50, temperature=0.7, top_p=0.9)
        dur = time.time() - t0
        gen_tokens = len(llm.tokenizer.encode(output, bos=False)) - len(llm.tokenizer.encode(prompt, bos=False))
        total_tokens += gen_tokens
        total_time += dur

        print(f"\n[Prompt] {prompt}")
        print(f"[Output] {output}")
        print(f"  ({gen_tokens} tokens in {dur:.2f}s = {gen_tokens/dur:.0f} tok/s)")

    print()
    print("-" * 70)
    print(f"Total: {total_tokens} tokens in {total_time:.2f}s ({total_tokens/total_time:.0f} tok/s)")
    print()
    print("Zero external dependencies used. Pure NumPy + C++ engine.")
    print("=" * 70)


if __name__ == "__main__":
    main()
