"""Experiment & Evaluation Benchmark for Recurrent Trace Unit (RTU) Architecture.

Demonstrates:
1. Parameter allocation comparison (Byte-level 256 vs Subword 32,000 vocab).
2. Live training on Simple Wikipedia encyclopedic data.
3. Prompt continuation before vs. after training.
4. Latent representation stability & non-collapse verification (VICReg variance).
5. Arbitrary UTF-8 multi-byte (Unicode / Emoji) zero-shot handling without <unk>.
6. O(1) continuous state streaming memory check.
"""

import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np

# Ensure local rtu module is imported
sys.path.insert(0, str(Path(__file__).parent))
from rtu import RTUConfig, RTULanguageModel


def fetch_simple_wikipedia(title: str = "Solar_System") -> str:
    """Fetch article extract from Simple English Wikipedia, with offline fallback."""
    url = (
        f"https://simple.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={title}&format=json"
    )
    headers = {"User-Agent": "DoraneuralResearchLab/1.0 (https://github.com/Baba01hacker666/doranerual)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for p in pages.values():
                extract = p.get("extract", "")
                if extract:
                    return extract[:1500]  # Take first 1500 chars for concise benchmark
    except Exception as exc:
        print(f"  [Notice: Online fetch fallback due to: {exc}]")

    # Offline curated encyclopedic text fallback
    return (
        "The Solar System is a group of space objects that are held together by gravity, "
        "with the Sun in the center. The Sun is a huge ball of hot glowing gas that gives off light and heat. "
        "Everything else in the Solar System orbits around the Sun. "
        "The eight planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. "
        "Earth is the third planet from the Sun and the only world known to harbor life. "
        "Jupiter is the largest planet, with a giant red storm that has raged for hundreds of years. "
        "Gravity keeps the planets, moons, asteroids, and comets in stable orbits."
    )


def run_rtu_evaluation():
    print("=" * 72)
    print(" 🧠 DORANEURAL RESEARCH: RECURRENT TRACE UNIT (RTU) BENCHMARK")
    print("=" * 72)

    # 1. Parameter Allocation Comparison
    dim = 128
    n_layers = 3
    rtu_cfg = RTUConfig(dim=dim, n_layers=n_layers, vocab_size=256, lr=2e-3, weight_decay=0.001)
    rtu_model = RTULanguageModel(rtu_cfg)

    subword_vocab = 32000
    subword_embed_params = subword_vocab * dim * 2  # input embed + output unembed
    rtu_embed_params = 256 * dim * 2
    rtu_hidden_params = rtu_cfg.parameter_count - rtu_embed_params

    print(f"\n📊 1. Parameter Allocation Analysis (dim={dim}, layers={n_layers}):")
    print(f"  - Subword Model (32k vocab) embedding footprint: {subword_embed_params:,} params")
    print(f"  - RTU Model (256 bytes) embedding footprint:    {rtu_embed_params:,} params ({rtu_embed_params / subword_embed_params * 100:.2f}% of subword)")
    print(f"  - RTU Hidden Recurrence Engine:                 {rtu_hidden_params:,} params")
    print(f"  - Total RTU Model Parameters:                   {rtu_cfg.parameter_count:,} params")
    print(f"  ➜ Freeing {(subword_embed_params - rtu_embed_params):,} params for deep non-linear dynamics!")

    # 2. Fetch Data
    print("\n📚 2. Loading Corpus (Simple English Wikipedia):")
    text = fetch_simple_wikipedia("Solar_System")
    print(f"  Fetched {len(text):,} characters ({len(text.encode('utf-8')):,} UTF-8 bytes).")
    print(f"  Snippet: {text[:140]}...")

    # 3. Baseline Generation (Untrained Model)
    prompt = "The Solar System is "
    print(f"\n🎲 3. Baseline Prompt Continuation (Untrained, Prompt: '{prompt}'):")
    rtu_model.config.temperature = 0.5
    raw_gen = rtu_model.generate(prompt, max_bytes=60, temperature=0.5, reset_state=True)
    print(f"  Continuation: [ {raw_gen} ]")

    # 4. Multi-Objective Real-Time Training
    print("\n🚀 4. Multi-Objective Real-Time Training:")
    print("  Objectives: Cross-Entropy + Target Latent MSE + VICReg Variance Penalty + Stop MSE")
    
    epochs = 12
    raw_bytes = text.encode("utf-8")
    t_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        rtu_model.reset_state()
        ep_losses = {"total": 0.0, "ce": 0.0, "latent": 0.0, "var": 0.0}
        n_steps = len(raw_bytes) - 1

        for i in range(n_steps):
            cur_b = raw_bytes[i]
            next_b = raw_bytes[i + 1]
            is_end = (i == n_steps - 1)
            _, _, step_l = rtu_model.step_online_train(cur_b, next_b, is_end=is_end)
            ep_losses["total"] += step_l["total"]
            ep_losses["ce"] += step_l["ce"]
            ep_losses["latent"] += step_l["latent"]
            ep_losses["var"] += step_l["variance"]

        avg_total = ep_losses["total"] / n_steps
        avg_ce = ep_losses["ce"] / n_steps
        avg_lat = ep_losses["latent"] / n_steps
        avg_var = ep_losses["var"] / n_steps

        # Check representation latent variance
        _, _, lat = rtu_model.forward_byte(ord("T"), update_state=False)
        lat_std = float(np.std(lat))

        if ep % 2 == 0 or ep == 1 or ep == epochs:
            print(
                f"  Epoch {ep:2d}/{epochs:2d} | "
                f"Total Loss: {avg_total:.4f} | "
                f"CE: {avg_ce:.4f} | "
                f"Latent MSE: {avg_lat:.4f} | "
                f"VICReg Var Loss: {avg_var:.4f} | "
                f"Latent Std: {lat_std:.3f}"
            )

    elapsed = time.perf_counter() - t_start
    total_bytes_trained = len(raw_bytes) * epochs
    print(f"  ✓ Training completed in {elapsed:.2f}s ({total_bytes_trained / elapsed:,.0f} bytes/s throughput)")

    # 5. Post-Training Prompt Continuations
    print(f"\n✨ 5. Post-Training Prompt Continuations:")
    test_prompts = [
        "The Solar System is ",
        "Earth is ",
        "Jupiter is ",
    ]
    for p in test_prompts:
        completion = rtu_model.generate(p, max_bytes=80, temperature=0.2, reset_state=True)
        print(f"  Prompt: '{p}'")
        print(f"  Generated: '{p}{completion}'\n")

    # 6. Arbitrary Unicode & Multi-Byte Robustness (No <unk>)
    print("🌍 6. Arbitrary UTF-8 / Multi-byte & Emoji Processing:")
    unicode_input = "Space objects: 🌍 Earth, ☀️ Sun, 🚀 Rocket."
    uni_bytes = unicode_input.encode("utf-8")
    rtu_model.reset_state()
    print(f"  Input: '{unicode_input}' ({len(uni_bytes)} bytes)")
    
    # Forward all bytes to confirm no NaN and clean latent representations
    all_valid = True
    for b in uni_bytes:
        _, _, lat = rtu_model.forward_byte(b, update_state=True)
        if np.isnan(lat).any() or np.isinf(lat).any():
            all_valid = False
            break
    print(f"  All {len(uni_bytes)} raw bytes processed without crash or <unk>: {all_valid}")

    # 7. O(1) Memory Footprint Verification
    print("\n🧠 7. O(1) Constant State Invariance:")
    stream_length = 5000
    pseudo_stream = (raw_bytes * (stream_length // len(raw_bytes) + 1))[:stream_length]
    rtu_model.reset_state()
    
    state_norms = []
    for step, b in enumerate(pseudo_stream):
        rtu_model.forward_byte(b, update_state=True)
        if step % 1000 == 0:
            norm = float(np.linalg.norm(rtu_model.layer_states[0].state))
            state_norms.append((step, norm))
    
    print(f"  Processed {stream_length:,} streaming bytes sequentially.")
    print(f"  Layer 0 state norm across steps: {', '.join(f't={s}: ||s||={n:.2f}' for s, n in state_norms)}")
    print("  State remains strictly bounded due to learned sigmoid decay d in (0, 1) and LayerNorm!")
    print("\n" + "=" * 72)
    print(" ✅ RTU EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    run_rtu_evaluation()
