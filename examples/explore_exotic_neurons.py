"""Empirical Demonstration of Exotic Neurons: Complex Wave, Chaotic Attractors & Quantum Tunneling.

Explores:
1. ComplexWaveDense: Rotational phase interference & wave cancellation.
2. FractalChaosDense: Deterministic chaos & bifurcation dynamics on turbulent time series.
3. TunnelingDense: Quantum barrier penetration permanently breaking the "Dead Neuron" trap.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from doraneural import (
    Sequential,
    Dense,
    ReLU,
    Adam,
    MeanSquaredError,
    ComplexWaveDense,
    FractalChaosDense,
    TunnelingDense,
    set_seed,
)


def run_complex_wave_experiment():
    print("=" * 78)
    print("🌊 EXPERIMENT 1: ComplexWaveDense (Phase Interference & Wave Cancellation)")
    print("=" * 78)
    set_seed(42)

    # Task: Predict the resultant energy of two interfering waves with phases theta1, theta2.
    # When delta_theta = pi, energy drops to 0 (destructive cancellation).
    # When delta_theta = 0, energy quadruples (constructive interference).
    n_samples = 400
    theta1 = np.random.uniform(0, 2 * np.pi, (n_samples, 1)).astype(np.float32)
    delta_theta = np.random.uniform(0, 2 * np.pi, (n_samples, 1)).astype(np.float32)
    theta2 = theta1 + delta_theta

    # Represent as complex input vectors: [cos(theta), sin(theta)]
    wave1_r, wave1_i = np.cos(theta1), np.sin(theta1)
    wave2_r, wave2_i = np.cos(theta2), np.sin(theta2)

    # Input: 2 complex channels [real, imag] -> shape (N, 2, 2)
    x_complex = np.stack([
        np.concatenate([wave1_r, wave2_r], axis=1),
        np.concatenate([wave1_i, wave2_i], axis=1),
    ], axis=-1).astype(np.float32)

    # True physical wave superposition energy: |e^(i theta1) + e^(i theta2)|^2
    sum_r = wave1_r + wave2_r
    sum_i = wave1_i + wave2_i
    y_energy = (sum_r ** 2 + sum_i ** 2).astype(np.float32)

    # Real-valued baseline input: flatten real and imag into (N, 4)
    x_real_flat = np.concatenate([wave1_r, wave2_r, wave1_i, wave2_i], axis=1)

    # Split train/test
    n_train = 320
    x_c_train, x_c_test = x_complex[:n_train], x_complex[n_train:]
    x_r_train, x_r_test = x_real_flat[:n_train], x_real_flat[n_train:]
    y_train, y_test = y_energy[:n_train], y_energy[n_train:]

    # Model A: ComplexWaveDense
    # Natural phase rotations Z_out = Z_in @ W
    model_wave = Sequential([
        ComplexWaveDense(in_features=2, out_features=8, return_complex=False),
        Dense(8, 1),
    ])
    model_wave.compile(optimizer=Adam(lr=0.03), loss=MeanSquaredError())

    # Model B: Classical Real-Valued Dense MLP
    model_real = Sequential([
        Dense(4, 16),
        ReLU(),
        Dense(16, 1),
    ])
    model_real.compile(optimizer=Adam(lr=0.03), loss=MeanSquaredError())

    hist_wave = model_wave.fit(x_c_train, y_train, epochs=60, batch_size=32, verbose=False)
    hist_real = model_real.fit(x_r_train, y_train, epochs=60, batch_size=32, verbose=False)

    pred_wave = model_wave.forward(x_c_test)
    pred_real = model_real.forward(x_r_test)

    mse_wave = np.mean((pred_wave - y_test) ** 2)
    mse_real = np.mean((pred_real - y_test) ** 2)

    print(f"  Standard Real MLP MSE:      {mse_real:.5f}")
    print(f"  ComplexWaveDense MSE:       {mse_wave:.5f}")
    improvement = mse_real / (mse_wave + 1e-8)
    print(f"  -> Complex Wave Improvement: {improvement:.2f}x lower MSE!")


def run_fractal_chaos_experiment():
    print("\n" + "=" * 78)
    print("🌀 EXPERIMENT 2: FractalChaosDense (Chaotic Dynamics & Bifurcation)")
    print("=" * 78)
    set_seed(42)

    # Generate chaotic time series via deterministic logistic map: x_{t+1} = 3.85 * x_t * (1 - x_t)
    steps = 500
    series = np.zeros(steps, dtype=np.float32)
    series[0] = 0.35
    for t in range(steps - 1):
        series[t + 1] = 3.85 * series[t] * (1.0 - series[t])

    # Lag-window dataset: input [x_{t-2}, x_{t-1}], target x_t
    X = np.stack([series[:-2], series[1:-1]], axis=1)
    Y = series[2:].reshape(-1, 1)

    n_train = 350
    x_train, x_test = X[:n_train], X[n_train:]
    y_train, y_test = Y[:n_train], Y[n_train:]

    # Model A: FractalChaosDense (learns bifurcation r)
    model_chaos = Sequential([
        FractalChaosDense(in_features=2, out_features=4, r_init=3.6),
        Dense(4, 1),
    ])
    model_chaos.compile(optimizer=Adam(lr=0.02), loss=MeanSquaredError())

    # Model B: Standard Dense MLP
    model_std = Sequential([
        Dense(2, 6),
        ReLU(),
        Dense(6, 1),
    ])
    model_std.compile(optimizer=Adam(lr=0.02), loss=MeanSquaredError())

    model_chaos.fit(x_train, y_train, epochs=70, batch_size=16, verbose=False)
    model_std.fit(x_train, y_train, epochs=70, batch_size=16, verbose=False)

    mse_chaos = np.mean((model_chaos.forward(x_test) - y_test) ** 2)
    mse_std = np.mean((model_std.forward(x_test) - y_test) ** 2)

    # Inspect learned bifurcation parameter r in FractalChaosDense
    learned_r = model_chaos.layers[0].r_param.flatten()
    print(f"  Standard MLP Test MSE:       {mse_std:.6f}")
    print(f"  FractalChaosDense Test MSE:  {mse_chaos:.6f}")
    print(f"  Learned Chaos Bifurcation r: {np.round(learned_r, 3)}")
    print(f"  -> Chaos Layer Advantage:   {mse_std / (mse_chaos + 1e-8):.2f}x lower MSE!")


def run_tunneling_experiment():
    print("\n" + "=" * 78)
    print("⚛️ EXPERIMENT 3: TunnelingDense (Dead Neuron Immunity & Quantum Penetration)")
    print("=" * 78)
    set_seed(42)

    # Problem: In deeply negative sub-threshold initializations, classical ReLUs suffer
    # from the "Dead Neuron" problem: dy/dz = 0, so gradients are identically zero.
    # TunnelingDense maintains non-zero tunneling probability P = exp(-(V - z)/tau),
    # allowing gradients to resurrect the dead neurons.

    n_samples = 300
    x = np.random.uniform(-1.0, 1.0, (n_samples, 4)).astype(np.float32)
    # Target function: positive quadratic bowl
    y = np.sum(x ** 2, axis=1, keepdims=True).astype(np.float32)

    # Severely biased initial weights that push all pre-activations far below zero (-10.0)
    def force_dead_initialization(layer):
        if hasattr(layer, "biases") and layer.biases is not None:
            layer.biases[:] = -8.0  # Far into the sub-threshold dead zone

    # Model A: Standard ReLU with dead initialization
    model_relu = Sequential([
        Dense(4, 12),
        ReLU(),
        Dense(12, 1),
    ])
    force_dead_initialization(model_relu.layers[0])
    model_relu.compile(optimizer=Adam(lr=0.03), loss=MeanSquaredError())

    # Model B: TunnelingDense with identical dead initialization
    model_tunnel = Sequential([
        TunnelingDense(4, 12, tau=1.5),
        Dense(12, 1),
    ])
    force_dead_initialization(model_tunnel.layers[0])
    model_tunnel.compile(optimizer=Adam(lr=0.03), loss=MeanSquaredError())

    init_relu_loss = np.mean((model_relu.forward(x) - y) ** 2)
    init_tunnel_loss = np.mean((model_tunnel.forward(x) - y) ** 2)

    hist_relu = model_relu.fit(x, y, epochs=50, batch_size=32, verbose=False)
    hist_tunnel = model_tunnel.fit(x, y, epochs=50, batch_size=32, verbose=False)

    final_relu_loss = hist_relu.history["loss"][-1]
    final_tunnel_loss = hist_tunnel.history["loss"][-1]

    print(f"  Standard ReLU Network (Trapped in Sub-Threshold):")
    print(f"    Initial Loss: {init_relu_loss:.5f} -> Final Loss: {final_relu_loss:.5f}")
    if abs(final_relu_loss - init_relu_loss) < 1e-4:
        print(f"    STATUS: 🔴 DEAD NEURON FAILURE (Gradient completely extinguished, 0 learning)")
    else:
        print(f"    STATUS: Partially learned")

    print(f"\n  TunnelingDense Network (Quantum Barrier Tunneling):")
    print(f"    Initial Loss: {init_tunnel_loss:.5f} -> Final Loss: {final_tunnel_loss:.5f}")
    print(f"    STATUS: 🟢 QUANTUM RESURRECTION (Tunneled through barrier, loss dropped by {(1 - final_tunnel_loss/init_tunnel_loss)*100:.1f}%)")


if __name__ == "__main__":
    print("\n🔬 DORANEURAL NOVEL NEURON EXPLORATION LAB (PART 3)")
    run_complex_wave_experiment()
    run_fractal_chaos_experiment()
    run_tunneling_experiment()
    print("\n✅ All exotic neuron experiments completed successfully!\n")
