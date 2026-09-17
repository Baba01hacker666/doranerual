#!/usr/bin/env python3
"""Train a Unified Bio-Reflective KAN Hybrid Network in doraneural.

Unifies four novel neuron architectures into a single synergistic model:
1. Stage 1 (Cortical Reflection): ReflectiveDense (2 reflection passes, iterative input denoising)
2. Stage 2 (Dynamic Routing):    BifurcatedDense (Threshold-gated dual-pathway signal bifurcation)
3. Stage 3 (Dendritic Gating):   DendriticDense (Multi-compartment pyramidal neuron gating)
4. Stage 4 (Synaptic KAN):       ChebyshevKAN (Continuous polynomial basis expansions on synapses)
5. Readout:                      Softmax (Multi-class probability distribution)

Task:
    8x8 Handwritten Digit Classification under high noise (0.18 sigma),
    comparing the Bio-Hybrid Architecture against a Classical MLP.

Usage:
    python examples/train_bio_hybrid_model.py
"""

import sys
import time
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import doraneural as dn


def count_params(model: dn.Sequential) -> int:
    """Return total number of learnable parameters in the model."""
    total = 0
    for layer in model.layers:
        params = layer.get_params()
        total += sum(p.size for p in params.values() if p is not None)
    return total


def render_ascii_digit(image_flat: np.ndarray, threshold: float = 0.35) -> str:
    """Render an 8x8 flattened grayscale image as an ASCII character grid."""
    img = image_flat.reshape(8, 8)
    lines = []
    for row in img:
        chars = []
        for val in row:
            if val > 0.65:
                chars.append("██")
            elif val > threshold:
                chars.append("▒▒")
            else:
                chars.append("  ")
        lines.append("    " + "".join(chars))
    return "\n".join(lines)


def main():
    print("=" * 82)
    print("🧠 BIO-REFLECTIVE KAN: UNIFIED NOVEL NEURON ARCHITECTURE")
    print("=" * 82)
    print("Synthesizing 4 novel neuron paradigms into a single neural network:")
    print("  1. ReflectiveDense: Top-down cortical reflection to iteratively denoise input.")
    print("  2. BifurcatedDense: Threshold-gated routing (sub- vs supra-threshold stroke pathways).")
    print("  3. DendriticDense:  Multi-compartment pyramidal multiplicative conjunctions.")
    print("  4. ChebyshevKAN:    Continuous non-linear polynomial synaptic functions.")
    print("=" * 82)

    # 1. Dataset Generation: Digits with realistic noise
    dn.set_seed(42)
    n_samples = 1000
    noise_level = 0.18
    print(f"\nGenerating {n_samples} synthetic 8x8 digits with noise level = {noise_level}...")
    X, y = dn.make_digits(n_samples=n_samples, noise=noise_level, flatten=True, seed=42)
    y_cat = dn.to_categorical(y, num_classes=10)

    X_train, X_test, y_train, y_test = dn.train_test_split(
        X, y_cat, test_size=0.25, seed=42
    )
    raw_test = np.argmax(y_test, axis=1)

    print(f"Train Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples\n")

    # 2. Model 1: Classical MLP Baseline
    classical_model = dn.Sequential([
        dn.Dense(64, 48),
        dn.SiLU(),
        dn.Dense(48, 24),
        dn.SiLU(),
        dn.Dense(24, 10),
        dn.Softmax(),
    ])
    opt_classic = dn.Adam(lr=0.012, weight_decay=1e-4)
    classical_model.compile(optimizer=opt_classic, loss=dn.CategoricalCrossEntropy(), metrics=["accuracy"])
    sched_classic = dn.CosineAnnealingLR(opt_classic, T_max=15, eta_min=0.002)

    # 3. Model 2: Bio-Reflective KAN Architecture
    bio_model = dn.Sequential([
        # Stage 1: Cortical reflection (2 passes) to resolve noisy pixels
        dn.ReflectiveDense(in_features=64, out_features=36, reflection_steps=2, alpha=0.35),
        # Stage 2: Threshold bifurcation (separate faint vs strong stroke intensities)
        dn.BifurcatedDense(in_features=36, out_features=24, temperature=0.8),
        dn.SiLU(),
        # Stage 3: Multi-compartment pyramidal integration (2 dendritic branches)
        dn.DendriticDense(in_features=24, out_features=16, num_branches=2),
        # Stage 4: High-order polynomial synaptic readout
        dn.ChebyshevKAN(in_features=16, out_features=10, degree=3),
        # Readout: Class distribution
        dn.Softmax(),
    ])
    opt_bio = dn.Adam(lr=0.012, weight_decay=1e-4)
    bio_model.compile(optimizer=opt_bio, loss=dn.CategoricalCrossEntropy(), metrics=["accuracy"])
    sched_bio = dn.CosineAnnealingLR(opt_bio, T_max=15, eta_min=0.002)

    epochs = 15
    batch_size = 32

    # Train Classical MLP
    n_params_classic = count_params(classical_model)
    print(f"--- Training Model 1: Classical MLP (Params: {n_params_classic:,d}) ---")
    t0 = time.perf_counter()
    hist_classic = classical_model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
        validation_data=(X_test, y_test), scheduler=sched_classic
    )
    time_classic = (time.perf_counter() - t0) * 1000

    # Train Bio-Reflective KAN
    n_params_bio = count_params(bio_model)
    print(f"\n--- Training Model 2: Bio-Reflective KAN (Params: {n_params_bio:,d}) ---")
    t0 = time.perf_counter()
    hist_bio = bio_model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
        validation_data=(X_test, y_test), scheduler=sched_bio
    )
    time_bio = (time.perf_counter() - t0) * 1000

    # Evaluation
    test_eval_classic = classical_model.evaluate(X_test, y_test)
    test_eval_bio = bio_model.evaluate(X_test, y_test)

    # 4. Results Summary
    print("\n" + "=" * 82)
    print("🏆 FINAL COMPARISON RESULTS")
    print("=" * 82)
    print(f"{'Model Architecture':<36} | {'Params':>7} | {'Val Loss':>9} | {'Val Acc':>8} | {'Time':>8}")
    print("-" * 82)
    print(f"{'Classical MLP (Dense + SiLU)':<36} | {n_params_classic:>7d} | {test_eval_classic.loss:>9.4f} | {test_eval_classic.accuracy * 100:>7.1f}% | {time_classic:>6.0f}ms")
    print(f"{'Bio-Reflective KAN (Unified Novel)':<36} | {n_params_bio:>7d} | {test_eval_bio.loss:>9.4f} | {test_eval_bio.accuracy * 100:>7.1f}% | {time_bio:>6.0f}ms")
    print("=" * 82)

    # 5. Visual Predictions Demo
    print("\n🔍 Visual Samples & Bio-Reflective KAN Predictions:")
    sample_indices = [0, 1, 2, 3, 4]
    probs = bio_model.predict_proba(X_test[sample_indices])
    preds = np.argmax(probs, axis=1)

    for i, idx in enumerate(sample_indices):
        true_label = int(raw_test[idx])
        pred_label = int(preds[i])
        confidence = probs[i, pred_label] * 100.0
        status = "✓ CORRECT" if pred_label == true_label else "✗ MISMATCH"

        print(f"\nSample #{i+1} [True: {true_label} | Pred: {pred_label} ({confidence:.1f}%) | {status}]:")
        print(render_ascii_digit(X_test[idx]))

    # 6. Save Model Demonstration
    save_path = "bio_reflective_kan_digit_classifier.json"
    print(f"\n💾 Serializing Bio-Reflective KAN model to '{save_path}'...")
    dn.save_model(bio_model, save_path)

    reloaded = dn.load_model(save_path)
    reloaded.compile(loss=dn.CategoricalCrossEntropy(), metrics=["accuracy"])
    reloaded_acc = reloaded.evaluate(X_test, y_test).accuracy * 100.0
    print(f"✓ Model successfully reloaded! Verified accuracy: {reloaded_acc:.1f}%")
    print("=" * 82)


if __name__ == "__main__":
    main()
