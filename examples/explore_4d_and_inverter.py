#!/usr/bin/env python3
"""Explore 4D Hyper-Tensor Models & Synaptic Inverters in doraneural.

Demonstrates two radical frontier concepts:
1. Tensor4DDense (4D Spacetime Hyper-Tensor Networks):
   - "Models exist in 3D vectors, so let's try 4D models!"
   - Processes 4-dimensional hypercube tensors (Batch, Time, Space, Channels)
     preserving 4D geometric spacetime manifold structure without early flattening.

2. InvertedDense (Synaptic Inverters with Learnable Duality):
   - "Let's try inverter like invert module weights because no other will!"
   - Each neuron possesses a continuous learnable inversion dial:
       g_inv = Sigmoid(inversion_logits)
       scale = 1 - 2 * g_inv in [+1, -1]
     allowing neurons to autonomously invert from excitatory (+W) to inhibitory (-W)
     to detect negative spaces, antagonistic contrast, and anti-correlations.

Usage:
    python examples/explore_4d_and_inverter.py
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
# DEMO 1: 4D Spatiotemporal Hyper-Tensor Model
# Shape: (Batch, Time, Space, Features)
# ==============================================================================
def demo_4d_hyper_tensor_model():
    print("=" * 80)
    print("🌌 1. 4D HYPER-TENSOR NETWORK (Spatiotemporal Manifold Modeling)")
    print("   Concept: 'Models exist in 3D vectors, so let's try 4D models!'")
    print("=" * 80)
    print("Data Geometry: 4D Tensors of shape (Batch, Time, Space, Channels)")
    print("  • Batch B = 120 samples")
    print("  • Time Dimension T = 6 frames")
    print("  • Spatial Grid S = 8 points")
    print("  • Channel Features C = 4 vector components (e.g. pressure, velocity_x, velocity_y, temp)")
    print()

    dn.set_seed(42)
    B_train, B_test = 120, 40
    T, S, C_in = 6, 8, 4

    # Generate synthetic 4D wave diffusion system
    def generate_4d_field(n_batches: int):
        X = np.random.randn(n_batches, T, S, C_in).astype(np.float32)
        # Target is non-linear spatiotemporal energy invariant
        # y = sum_{t, s} (X_{t,s,0}^2 - X_{t,s,1}^2 + sin(X_{t,s,2} * X_{t,s,3}))
        e1 = X[:, :, :, 0] ** 2 - X[:, :, :, 1] ** 2
        e2 = np.sin(X[:, :, :, 2] * X[:, :, :, 3])
        y = np.mean(e1 + e2, axis=(1, 2)).reshape(-1, 1).astype(np.float32)
        return X, y

    X_train, y_train = generate_4d_field(B_train)
    X_test, y_test = generate_4d_field(B_test)

    print(f"X_train 4D Tensor Shape: {X_train.shape}")
    print(f"y_train Target Shape:     {y_train.shape}\n")

    # Build 4D Hyper-Tensor Model
    model_4d = dn.Sequential([
        # 4D Hyper-Tensor Layer 1: 4 channels -> 12 channels in 4D space
        dn.Tensor4DDense(in_channels=4, out_channels=12),
        dn.SiLU(),
        # 4D Hyper-Tensor Layer 2: 12 channels -> 6 channels in 4D space
        dn.Tensor4DDense(in_channels=12, out_channels=6),
        dn.SiLU(),
        # Flatten 4D hypercube into readout
        dn.Flatten(),
        dn.Dense(in_features=T * S * 6, out_features=16),
        dn.SiLU(),
        dn.Dense(in_features=16, out_features=1),
    ])

    opt = dn.Adam(lr=0.015)
    model_4d.compile(optimizer=opt, loss=dn.MSELoss())

    print(f"Training 4D Hyper-Tensor Model (Params: {count_params(model_4d):,d})...")
    t0 = time.perf_counter()
    hist = model_4d.fit(X_train, y_train, epochs=30, batch_size=16, verbose=1)
    train_time = (time.perf_counter() - t0) * 1000

    preds = model_4d.forward(X_test)
    mse = float(np.mean((preds - y_test) ** 2))
    r2 = max(0.0, float(1.0 - (mse / (np.var(y_test) + 1e-8)))) * 100.0

    print(f"\n✓ 4D Hyper-Tensor Training Complete in {train_time:.1f}ms!")
    print(f"  Test MSE: {mse:.5f} | R² Variance Explained: {r2:.2f}%\n")


# ==============================================================================
# DEMO 2: Synaptic Inverters with Learnable Polarity
# Antagonistic push-pull / contrast-inversion learning
# ==============================================================================
def demo_synaptic_inverter():
    print("=" * 80)
    print("🔀 2. SYNAPTIC INVERTER (Learnable Weight Inversion & Dual Polarity)")
    print("   Concept: 'Invert module weights because no other will!'")
    print("=" * 80)
    print("Task: Antagonistic Dual-Phase Physics (Wave Cancellation & Contrast Inversion):")
    print("  • Inputs contain both positive excitations and destructive interference phases.")
    print("  • Standard neurons must balance positive/negative weights through gradient conflict.")
    print("  • Inverted neurons can dynamically flip their whole synaptic field (+W -> -W)!")
    print()

    dn.set_seed(42)
    N = 300
    X = np.random.uniform(-1.0, 1.0, size=(N, 6)).astype(np.float32)
    # Target: 3 positive phase features, 3 inverted anti-phase features
    y = (
        (X[:, 0:1] + X[:, 1:2] + X[:, 2:3])
        - 2.5 * (X[:, 3:4] + X[:, 4:5] + X[:, 5:6])
    ).astype(np.float32)

    X_train, X_test, y_train, y_test = dn.train_test_split(X, y, test_size=0.25, seed=42)

    # Build model with InvertedDense
    inv_model = dn.Sequential([
        dn.InvertedDense(in_features=6, out_features=12, init_inverted=False),
        dn.SiLU(),
        dn.InvertedDense(in_features=12, out_features=1, init_inverted=False),
    ])
    inv_model.compile(optimizer=dn.Adam(lr=0.03), loss=dn.MSELoss())

    print(f"Training Inverted Synaptic Model (Params: {count_params(inv_model)})...")
    hist = inv_model.fit(X_train, y_train, epochs=40, batch_size=16, verbose=0)

    preds = inv_model.forward(X_test)
    mse = float(np.mean((preds - y_test) ** 2))
    r2 = max(0.0, float(1.0 - (mse / (np.var(y_test) + 1e-8)))) * 100.0

    print(f"✓ Training Complete! Final Loss: {hist['loss'][-1]:.5f} | Test MSE: {mse:.5f} | R²: {r2:.2f}%\n")

    # Inspect the learned inversion dials!
    layer1 = inv_model.layers[0]
    logits = layer1.inversion_logits.ravel()
    g_inv = 1.0 / (1.0 + np.exp(-logits))
    scale = 1.0 - 2.0 * g_inv

    print("🔍 Synaptic Inversion State of Layer 1 Neurons:")
    print("  Neuron # | Logit  | Inversion Gate g | Scale Factor s | Polarity Mode")
    print("  " + "-" * 66)
    for i in range(len(scale)):
        if scale[i] > 0.3:
            mode = "🟢 Standard (+W Excitatory)"
        elif scale[i] < -0.3:
            mode = "🔴 Inverted (-W Inhibitory)"
        else:
            mode = "⚪ Neutral / Pruned"
        print(f"    #{i+1:02d}    | {logits[i]:+5.2f} |     {g_inv[i]:5.3f}        |     {scale[i]:+5.2f}      | {mode}")

    print("\n" + "=" * 80)
    print("Architectural Takeaways:")
    print("  ✓ Tensor4DDense natively processes 4D spacetime manifolds (B, T, S, C).")
    print("  ✓ InvertedDense differentiably inverts synaptic polarity (+W <-> -W).")
    print("  ✓ Standalone Inverter layer provides zero-parameter pure sign inversion.")
    print("=" * 80)


def main():
    demo_4d_hyper_tensor_model()
    demo_synaptic_inverter()


if __name__ == "__main__":
    main()
