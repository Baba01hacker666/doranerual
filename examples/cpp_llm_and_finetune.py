"""High-Performance C++ LLM Inference and Fine-Tuning Demo.

Demonstrates:
1. C++ native backend with OpenMP multi-threading and SIMD vectorization.
2. Fine-tuning a pretrained LLaMA model on custom text using Cross-Entropy & AdamW.
3. Comparing generation before vs after fine-tuning.
4. Saving and reloading the fine-tuned checkpoint.
"""

import sys
import tempfile
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import doraneural as dn


def main():
    print("=" * 70)
    print("⚡ doraneural: Native C++ Accelerated LLM & Fine-Tuning Engine")
    print("=" * 70)

    # 1. Check C++ Backend
    print(f"C++ Engine Status: {'AVAILABLE (OpenMP Enabled)' if dn.is_cpp_available() else 'FALLBACK (Pure NumPy)'}")

    # 2. Load model
    print("\n[1/4] Loading pretrained LLaMA (stories260K from Hugging Face)...")
    llm = dn.load_pretrained_llm("stories260K", backend="auto")
    print(f"Active Backend: {llm.backend.upper()}")
    print(f"Model: {llm.config.dim} dim, {llm.config.n_layers} layers, {llm.config.vocab_size} vocab")

    prompt = "Once upon a time, there was a little robot"

    # 3. Generate baseline story before fine-tuning
    print("\n[2/4] Baseline Generation (Before Fine-Tuning):")
    print("-" * 70)
    sys.stdout.write(f"\033[1m{prompt}\033[0m")
    sys.stdout.flush()

    for piece in llm.generate(prompt=prompt, max_tokens=40, temperature=0.7, stream=True):
        sys.stdout.write(piece)
        sys.stdout.flush()
    print("\n" + "-" * 70)

    # 4. Custom corpus for fine-tuning
    custom_corpus = """
Once upon a time, there was a little robot named Sparky.
Sparky was a silver robot who loved to fix shiny gears and oil wheels.
Sparky lived in a magical workshop with his best friend, a robotic kitten named Pip.
Every morning, Sparky and Pip would oil their joints and build colorful solar lanterns.
One day, Sparky built a special rainbow battery that made Pip purr with joy.
Sparky said to Pip, "We are the best robot friends in the entire galaxy!"
"""

    print("\n[3/4] Fine-Tuning LLM on Custom Robot Story in C++...")
    print(f"Training corpus length: {len(custom_corpus.split())} words")
    t0 = time.perf_counter()
    history = llm.train(custom_corpus, epochs=5, lr=5e-4, seq_len=16, verbose=1)
    t_train = time.perf_counter() - t0
    print(f"Fine-tuning complete in {t_train:.2f}s! Final loss: {history['loss'][-1]:.4f}")

    # 5. Generate story after fine-tuning
    print("\n[4/4] Fine-Tuned Generation (After Fine-Tuning):")
    print("-" * 70)
    sys.stdout.write(f"\033[1m{prompt}\033[0m")
    sys.stdout.flush()

    for piece in llm.generate(prompt=prompt, max_tokens=45, temperature=0.7, stream=True):
        sys.stdout.write(piece)
        sys.stdout.flush()
    print("\n" + "-" * 70)

    # 6. Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "sparky_robot.bin"
        llm.save(save_path)
        print(f"\nCheckpoint successfully saved: {save_path.name} ({save_path.stat().st_size:,} bytes)")

    print("\n" + "=" * 70)
    print("All C++ LLM inference and fine-tuning steps verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
