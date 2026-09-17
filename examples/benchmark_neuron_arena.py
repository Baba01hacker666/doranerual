#!/usr/bin/env python3
"""Neuron Arena: Empirical Benchmark of Classical vs Novel Neuron Architectures.

Pits classical neurons (Dense + ReLU / SiLU) against novel architectures:
1. Multi-Compartment Pyramidal Neurons (DendriticDense)
2. Kolmogorov-Arnold Networks (ChebyshevKAN)

Under strictly controlled, parameter-matched budgets across three benchmark tasks:
- Task 1: Non-Linear Physics / Multiplicative Interaction Regression
- Task 2: Geometric Manifold / Two-Spirals Non-Convex Classification
- Task 3: High-Frequency Wave Fitting (Spectral Bias & F-Principle)

Usage:
    python examples/benchmark_neuron_arena.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
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
# Benchmark 1: Non-Linear Physics / Multiplicative Interaction Regression
# Formula: f(x) = (x1 * x2) / (1 + x3^2) + sin(pi * x4)
# ==============================================================================
def run_physics_regression() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("🥊 ARENA TASK 1: Non-Linear Physics Regression (Multiplicative & Rational)")
    print("   Target: f(x) = (x1 * x2) / (1 + x3^2) + sin(pi * x4)")
    print("=" * 78)

    dn.set_seed(42)
    N_train, N_test = 500, 200

    X_train = np.random.uniform(-1.5, 1.5, size=(N_train, 4)).astype(np.float32)
    y_train = (
        (X_train[:, 0:1] * X_train[:, 1:2]) / (1.0 + X_train[:, 2:3] ** 2)
        + np.sin(np.pi * X_train[:, 3:4])
    ).astype(np.float32)

    X_test = np.random.uniform(-1.5, 1.5, size=(N_test, 4)).astype(np.float32)
    y_test = (
        (X_test[:, 0:1] * X_test[:, 1:2]) / (1.0 + X_test[:, 2:3] ** 2)
        + np.sin(np.pi * X_test[:, 3:4])
    ).astype(np.float32)

    # Models under ~280-300 parameter budget
    models = {
        "Classical MLP (Dense + SiLU)": dn.Sequential([
            dn.Dense(4, 16),
            dn.SiLU(),
            dn.Dense(16, 12),
            dn.SiLU(),
            dn.Dense(12, 1),
        ]),
        "Dendritic Net (DendriticDense)": dn.Sequential([
            dn.DendriticDense(4, 7, num_branches=2),
            dn.DendriticDense(7, 1, num_branches=2),
        ]),
        "KAN Net (ChebyshevKAN)": dn.Sequential([
            dn.ChebyshevKAN(4, 9, degree=4),
            dn.ChebyshevKAN(9, 1, degree=4),
        ]),
    }

    results = {}
    epochs = 120
    batch_size = 32

    for name, model in models.items():
        dn.set_seed(42)
        n_params = count_params(model)
        optimizer = dn.Adam(lr=0.03, weight_decay=1e-5)
        model.compile(optimizer=optimizer, loss=dn.MSELoss())

        t0 = time.perf_counter()
        hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        train_time = (time.perf_counter() - t0) * 1000

        preds = model.forward(X_test)
        mse = float(np.mean((preds - y_test) ** 2))
        mae = float(np.mean(np.abs(preds - y_test)))
        var_y = float(np.var(y_test))
        r2 = max(0.0, float(1.0 - (mse / (var_y + 1e-8)))) * 100.0

        results[name] = {
            "params": n_params,
            "train_time_ms": train_time,
            "final_train_loss": hist["loss"][-1],
            "test_mse": mse,
            "test_mae": mae,
            "test_r2": r2,
        }
        print(f"  [{name:^30}] | Params: {n_params:3d} | Test MSE: {mse:.5f} | R²: {r2:5.2f}% | Time: {train_time:5.1f}ms")

    return results


# ==============================================================================
# Benchmark 2: Geometric Manifold / Two-Spirals Non-Convex Classification
# Lang & Witbrock benchmark: spirals winding around origin
# ==============================================================================
def run_spirals_classification() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("🥊 ARENA TASK 2: Non-Convex Decision Manifold (Two Interleaved Spirals)")
    print("   Target: Separate intertwined non-linear spiral branches in 2D")
    print("=" * 78)

    dn.set_seed(42)
    n_points = 180

    # Spiral 1 (class 0)
    theta1 = np.linspace(0.5, 3.5 * np.pi, n_points)
    r1 = theta1 / (3.5 * np.pi)
    x1 = r1 * np.cos(theta1) + np.random.normal(0, 0.02, n_points)
    y1 = r1 * np.sin(theta1) + np.random.normal(0, 0.02, n_points)
    class0 = np.stack([x1, y1], axis=1)

    # Spiral 2 (class 1, rotated by pi)
    theta2 = np.linspace(0.5, 3.5 * np.pi, n_points)
    r2 = theta2 / (3.5 * np.pi)
    x2 = -r2 * np.cos(theta2) + np.random.normal(0, 0.02, n_points)
    y2 = -r2 * np.sin(theta2) + np.random.normal(0, 0.02, n_points)
    class1 = np.stack([x2, y2], axis=1)

    X = np.vstack([class0, class1]).astype(np.float32)
    y = np.vstack([np.zeros((n_points, 1)), np.ones((n_points, 1))]).astype(np.float32)

    # Train / Test split (80/20 stratified)
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Parameter matched models (~350-385 parameters)
    models = {
        "Classical MLP (Dense + SiLU)": dn.Sequential([
            dn.Dense(2, 24),
            dn.SiLU(),
            dn.Dense(24, 12),
            dn.SiLU(),
            dn.Dense(12, 1),
            dn.Sigmoid(),
        ]),
        "Dendritic Net (DendriticDense)": dn.Sequential([
            dn.DendriticDense(2, 10, num_branches=2),
            dn.DendriticDense(10, 1, num_branches=2),
            dn.Sigmoid(),
        ]),
        "KAN Net (ChebyshevKAN)": dn.Sequential([
            dn.ChebyshevKAN(2, 16, degree=5),
            dn.ChebyshevKAN(16, 1, degree=5),
            dn.Sigmoid(),
        ]),
    }

    results = {}
    epochs = 200
    batch_size = 32

    for name, model in models.items():
        dn.set_seed(42)
        n_params = count_params(model)
        optimizer = dn.Adam(lr=0.04)
        model.compile(optimizer=optimizer, loss=dn.BinaryCrossEntropy(), metrics=["accuracy"])

        t0 = time.perf_counter()
        hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        train_time = (time.perf_counter() - t0) * 1000

        probs = model.predict_proba(X_test)
        preds = (probs > 0.5).astype(np.float32)
        acc = float(np.mean(preds == y_test) * 100.0)
        test_loss = float(dn.BinaryCrossEntropy().forward(probs, y_test))

        results[name] = {
            "params": n_params,
            "train_time_ms": train_time,
            "final_train_loss": hist["loss"][-1],
            "test_loss": test_loss,
            "test_accuracy": acc,
        }
        print(f"  [{name:^30}] | Params: {n_params:3d} | Test Acc: {acc:5.1f}% | Loss: {test_loss:.4f} | Time: {train_time:5.1f}ms")

    return results


# ==============================================================================
# Benchmark 3: High-Frequency Wave Fitting (Spectral Bias & F-Principle)
# Formula: f(x) = sin(3*pi*x) + 0.5*cos(9*pi*x)
# ==============================================================================
def run_fourier_wave() -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("🥊 ARENA TASK 3: High-Frequency Wave Approximation (Spectral Bias & F-Principle)")
    print("   Target: f(x) = sin(3π x) + 0.5 cos(9π x) on x in [-1, 1]")
    print("=" * 78)

    dn.set_seed(42)
    N_train = 250
    N_test = 150

    X_train = np.random.uniform(-1.0, 1.0, size=(N_train, 1)).astype(np.float32)
    y_train = (np.sin(3.0 * np.pi * X_train) + 0.5 * np.cos(9.0 * np.pi * X_train)).astype(np.float32)

    X_test = np.linspace(-1.0, 1.0, N_test).reshape(-1, 1).astype(np.float32)
    y_test = (np.sin(3.0 * np.pi * X_test) + 0.5 * np.cos(9.0 * np.pi * X_test)).astype(np.float32)

    # Models under ~170-205 parameter budget
    models = {
        "Classical MLP (Dense + SiLU)": dn.Sequential([
            dn.Dense(1, 24),
            dn.SiLU(),
            dn.Dense(24, 6),
            dn.SiLU(),
            dn.Dense(6, 1),
        ]),
        "Dendritic Net (DendriticDense)": dn.Sequential([
            dn.DendriticDense(1, 6, num_branches=2),
            dn.DendriticDense(6, 1, num_branches=2),
        ]),
        "KAN Net (ChebyshevKAN)": dn.Sequential([
            dn.ChebyshevKAN(1, 10, degree=6),
            dn.ChebyshevKAN(10, 1, degree=6),
        ]),
    }

    results = {}
    epochs = 150
    batch_size = 25

    for name, model in models.items():
        dn.set_seed(42)
        n_params = count_params(model)
        optimizer = dn.Adam(lr=0.03)
        model.compile(optimizer=optimizer, loss=dn.MSELoss())

        t0 = time.perf_counter()
        hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        train_time = (time.perf_counter() - t0) * 1000

        preds = model.forward(X_test)
        mse = float(np.mean((preds - y_test) ** 2))
        max_err = float(np.max(np.abs(preds - y_test)))
        var_y = float(np.var(y_test))
        r2 = max(0.0, float(1.0 - (mse / (var_y + 1e-8)))) * 100.0

        results[name] = {
            "params": n_params,
            "train_time_ms": train_time,
            "final_train_loss": hist["loss"][-1],
            "test_mse": mse,
            "max_error": max_err,
            "test_r2": r2,
        }
        print(f"  [{name:^30}] | Params: {n_params:3d} | Test MSE: {mse:.5f} | Max Err: {max_err:.4f} | R²: {r2:5.2f}%")

    return results


def print_grand_summary(
    res_physics: Dict[str, Any],
    res_spirals: Dict[str, Any],
    res_wave: Dict[str, Any],
) -> None:
    print("\n" + "=" * 88)
    print("🏆 GRAND NEURON ARENA SUMMARY: CLASSICAL VS NOVEL ARCHITECTURES")
    print("=" * 88)

    print("\n1. Task 1 - Non-Linear Physics Regression:")
    print(f"   {'Architecture':<32} | {'Params':>6} | {'Test MSE':>10} | {'Test MAE':>10} | {'R² (%)':>8}")
    print("   " + "-" * 76)
    for k, v in res_physics.items():
        print(f"   {k:<32} | {v['params']:>6d} | {v['test_mse']:>10.5f} | {v['test_mae']:>10.5f} | {v['test_r2']:>7.2f}%")

    print("\n2. Task 2 - Two Intertwined Spirals Classification:")
    print(f"   {'Architecture':<32} | {'Params':>6} | {'Test Loss':>10} | {'Accuracy':>10} | {'Train Time':>10}")
    print("   " + "-" * 76)
    for k, v in res_spirals.items():
        print(f"   {k:<32} | {v['params']:>6d} | {v['test_loss']:>10.4f} | {v['test_accuracy']:>9.1f}% | {v['train_time_ms']:>8.1f}ms")

    print("\n3. Task 3 - High-Frequency Wave Fitting (Spectral Bias):")
    print(f"   {'Architecture':<32} | {'Params':>6} | {'Test MSE':>10} | {'Max L_inf':>10} | {'R² (%)':>8}")
    print("   " + "-" * 76)
    for k, v in res_wave.items():
        print(f"   {k:<32} | {v['params']:>6d} | {v['test_mse']:>10.5f} | {v['max_error']:>10.4f} | {v['test_r2']:>7.2f}%")

    print("\n" + "=" * 88)


def main():
    print("=" * 78)
    print("🔬 DORANEURAL NOVEL NEURON ARENA")
    print("   Comparing Classical Dense vs DendriticDense vs ChebyshevKAN")
    print("=" * 78)

    res_physics = run_physics_regression()
    res_spirals = run_spirals_classification()
    res_wave = run_fourier_wave()

    print_grand_summary(res_physics, res_spirals, res_wave)


if __name__ == "__main__":
    main()
