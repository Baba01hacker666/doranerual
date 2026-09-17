#!/usr/bin/env python3
"""Explore Reflective & Threshold-Gated Bifurcated Neurons in doraneural.

Demonstrates two architectural breakthroughs inspired by your ideas:

1. BifurcatedDense (Threshold-Gated Dual-Pathway Neurons):
   - "Apply gates if input is lower than set weights it will go to a different neuron"
   - Dynamically routes sub-threshold inputs to low-intensity weights (W_low)
     and supra-threshold inputs to high-intensity weights (W_high).
   - Solves discontinuous dual-regime physics / burst-firing dynamics.

2. ReflectiveDense (Iterative Cortical Feedback & Reflection):
   - "Reflect back the input back to previous layer again"
   - Implements biological top-down cortical feedback / predictive coding loops.
   - The neuron emits an initial hypothesis, projects it BACK to input space
     via feedback weights W_ref, refines the representation, and re-evaluates!

Usage:
    python examples/explore_reflection_and_gating.py
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


# ==============================================================================
# DEMO 1: Bifurcated Threshold-Gated Dynamic Routing
# Dual-regime problem: Sub-threshold laminar flow vs Supra-threshold turbulent flow
# ==============================================================================
def demo_bifurcated_gating():
    print("=" * 80)
    print("⚡ 1. BIFURCATED NEURONS: Dynamic Threshold-Gated Dual-Pathway Routing")
    print("   Concept: 'If input is lower than threshold, route to a different neuron!'")
    print("=" * 80)
    print("Task: Piecewise Dual-Regime Physical Transition:")
    print("  • Sub-threshold regime (|x| <= 0.5): Gentle linear damping [f(x) = 0.2*x1 + 0.1*x2]")
    print("  • Supra-threshold regime (|x| > 0.5): Non-linear turbulent burst [f(x) = 2*sin(3*pi*x1) + x2^2]")
    print()

    dn.set_seed(42)
    N_train, N_test = 500, 200

    def generate_regime_data(n: int):
        X = np.random.uniform(-1.2, 1.2, size=(n, 2)).astype(np.float32)
        radius = np.linalg.norm(X, axis=1, keepdims=True)
        # Regime mask: True for active burst regime
        is_burst = (radius > 0.5).astype(np.float32)

        # Low-energy regime: quiet linear
        y_low = 0.2 * X[:, 0:1] + 0.1 * X[:, 1:2]
        # High-energy regime: turbulent oscillation
        y_high = 2.0 * np.sin(3.0 * np.pi * X[:, 0:1]) + (X[:, 1:2] ** 2)

        y = (1.0 - is_burst) * y_low + is_burst * y_high
        return X, y.astype(np.float32)

    X_train, y_train = generate_regime_data(N_train)
    X_test, y_test = generate_regime_data(N_test)

    # 1. Classical MLP: standard feedforward
    mlp_model = dn.Sequential([
        dn.Dense(2, 16),
        dn.SiLU(),
        dn.Dense(16, 1),
    ])
    mlp_model.compile(optimizer=dn.Adam(lr=0.03), loss=dn.MSELoss())
    n_params_mlp = count_params(mlp_model)
    print(f"Training Classical MLP (Params: {n_params_mlp})...")
    hist_mlp = mlp_model.fit(X_train, y_train, epochs=80, batch_size=20, verbose=0)
    preds_mlp = mlp_model.forward(X_test)
    mse_mlp = float(np.mean((preds_mlp - y_test) ** 2))
    r2_mlp = max(0.0, float(1.0 - (mse_mlp / (np.var(y_test) + 1e-8)))) * 100.0
    print(f"  Classical MLP: Test MSE: {mse_mlp:.5f} | R²: {r2_mlp:.2f}%\n")

    # 2. Bifurcated Gated Model: dynamic threshold router
    bif_model = dn.Sequential([
        dn.BifurcatedDense(2, 16, temperature=0.8),
        dn.SiLU(),
        dn.Dense(16, 1),
    ])
    bif_model.compile(optimizer=dn.Adam(lr=0.03), loss=dn.MSELoss())
    n_params_bif = count_params(bif_model)
    print(f"Training Bifurcated Dual-Pathway Model (Params: {n_params_bif})...")
    hist_bif = bif_model.fit(X_train, y_train, epochs=80, batch_size=20, verbose=0)
    preds_bif = bif_model.forward(X_test)
    mse_bif = float(np.mean((preds_bif - y_test) ** 2))
    r2_bif = max(0.0, float(1.0 - (mse_bif / (np.var(y_test) + 1e-8)))) * 100.0
    print(f"  Bifurcated Model: Test MSE: {mse_bif:.5f} | R²: {r2_bif:.2f}%\n")

    print(f"⚡ Result: Bifurcated routing achieved MSE: {mse_bif:.5f} vs Classical: {mse_mlp:.5f}")
    if mse_bif < mse_mlp:
        print(f"  → Bifurcated neurons achieved a {mse_mlp / mse_bif:.2f}x reduction in MSE!")
    print()


# ==============================================================================
# DEMO 2: Reflective Cortical Feedback & Iterative Hypothesis Refinement
# Ambiguous / Noisy feature resolution via iterative top-down reflection
# ==============================================================================
def demo_reflective_feedback():
    print("=" * 80)
    print("🔄 2. REFLECTIVE NEURONS: Iterative Cortical Feedback & Reflection Loops")
    print("   Concept: 'Reflect the output back to the input/previous layer again!'")
    print("=" * 80)
    print("Task: Iterative Hypothesis Refinement on Coupled Non-Linear Ambiguity:")
    print("  The network receives noisy inputs and refines its state over K reflection steps.")
    print()

    dn.set_seed(42)
    N_train, N_test = 400, 150

    # True non-linear system
    def generate_ambiguous_data(n: int):
        clean_X = np.random.uniform(-1.0, 1.0, size=(n, 4)).astype(np.float32)
        # Target: coupled quadratic and trigonometric interaction
        y = (
            np.sin(np.pi * clean_X[:, 0:1] * clean_X[:, 1:2])
            + (clean_X[:, 2:3] ** 2 - clean_X[:, 3:4] ** 2)
        ).astype(np.float32)
        # Add realistic observation noise to inputs
        noisy_X = clean_X + np.random.normal(0, 0.15, size=clean_X.shape).astype(np.float32)
        return noisy_X, y

    X_train, y_train = generate_ambiguous_data(N_train)
    X_test, y_test = generate_ambiguous_data(N_test)

    # 1. Classical Single-Pass Dense Model (No Reflection: 1 pass)
    classic_model = dn.Sequential([
        dn.Dense(4, 16),
        dn.SiLU(),
        dn.Dense(16, 1),
    ])
    classic_model.compile(optimizer=dn.Adam(lr=0.03), loss=dn.MSELoss())
    print(f"Training Classical 1-Pass Model (Params: {count_params(classic_model)})...")
    hist_classic = classic_model.fit(X_train, y_train, epochs=80, batch_size=20, verbose=0)
    preds_classic = classic_model.forward(X_test)
    mse_classic = float(np.mean((preds_classic - y_test) ** 2))
    r2_classic = max(0.0, float(1.0 - (mse_classic / (np.var(y_test) + 1e-8)))) * 100.0
    print(f"  Classical Model: Test MSE: {mse_classic:.5f} | R²: {r2_classic:.2f}%\n")

    # 2. Reflective Model: 3 iterative reflection passes (K=3)
    reflective_model = dn.Sequential([
        dn.ReflectiveDense(4, 16, reflection_steps=3, alpha=0.4),
        dn.Dense(16, 1),
    ])
    reflective_model.compile(optimizer=dn.Adam(lr=0.03), loss=dn.MSELoss())
    print(f"Training Reflective Feedback Model (K=3 Reflection Steps, Params: {count_params(reflective_model)})...")
    hist_refl = reflective_model.fit(X_train, y_train, epochs=80, batch_size=20, verbose=0)
    preds_refl = reflective_model.forward(X_test)
    mse_refl = float(np.mean((preds_refl - y_test) ** 2))
    r2_refl = max(0.0, float(1.0 - (mse_refl / (np.var(y_test) + 1e-8)))) * 100.0
    print(f"  Reflective Model (K=3): Test MSE: {mse_refl:.5f} | R²: {r2_refl:.2f}%\n")

    print(f"🔄 Result: Reflective feedback achieved MSE: {mse_refl:.5f} vs 1-Pass: {mse_classic:.5f}")
    if mse_refl < mse_classic:
        print(f"  → Iterative reflection achieved a {mse_classic / mse_refl:.2f}x reduction in MSE!")
    print()


def main():
    demo_bifurcated_gating()
    demo_reflective_feedback()


if __name__ == "__main__":
    main()
