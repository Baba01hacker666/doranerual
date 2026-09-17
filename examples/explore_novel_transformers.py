#!/usr/bin/env python3
"""Explore Novel Transformer Architectures: Standard vs KAN vs Dendritic Transformers.

Compares three Transformer architectures on non-linear sequence modeling:
1. Standard Transformer Block (Pre-LN Multi-Head Attention + Classical Dense MLP)
2. Dendritic Transformer Block (Multi-Head Attention + Multi-Branch Pyramidal Dendritic Gating)
3. KAN Transformer Block (Multi-Head Attention + Chebyshev Polynomial KAN)

Task:
    Algorithmic Sequence Regression on Non-Linear Coupled Dynamics:
    Given a sequence of vector states X in R^(T x D), predict the non-linear
    accumulated target value driven by cross-feature and cross-time polynomial couplings.

Usage:
    python examples/explore_novel_transformers.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import doraneural as dn


def count_params(model: dn.Sequential) -> int:
    """Return total number of learnable parameters in the model."""
    total = 0
    for layer in model.layers:
        params = layer.get_params()
        total += sum(p.size for p in params.values() if p is not None)
    return total


def generate_coupled_sequence_data(
    n_samples: int = 400,
    seq_len: int = 8,
    d_in: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sequence dataset with non-linear coupled dynamics across time steps."""
    np.random.seed(seed)
    X = np.random.uniform(-1.0, 1.0, size=(n_samples, seq_len, d_in)).astype(np.float32)

    # Compute target: coupled non-linear polynomial dynamics across sequence steps
    # y = sum_t [ sin(pi * x_{t,0}) + (x_{t,1} * x_{t,2}) / (1 + x_{t,3}^2) ]
    term1 = np.sin(np.pi * X[:, :, 0])
    term2 = (X[:, :, 1] * X[:, :, 2]) / (1.0 + X[:, :, 3] ** 2)
    y = np.sum(term1 + term2, axis=1, keepdims=True).astype(np.float32) / float(seq_len)

    # Train / Test split (75/25)
    split = int(0.75 * n_samples)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    return X_train, y_train, X_test, y_test


def build_models(d_in: int = 4, d_model: int = 16, seq_len: int = 8) -> Dict[str, dn.Sequential]:
    """Build parameter-matched sequence models with different Transformer blocks."""
    models = {}

    # 1. Standard Transformer
    models["Standard Transformer (MLP FFN)"] = dn.Sequential([
        dn.Dense(in_features=d_in, out_features=d_model),
        dn.PositionalEncoding(d_model=d_model, max_len=seq_len + 4),
        dn.TransformerBlock(d_model=d_model, num_heads=4, d_ff=32, dropout=0.0),
        dn.Flatten(),
        dn.Dense(in_features=seq_len * d_model, out_features=16),
        dn.SiLU(),
        dn.Dense(in_features=16, out_features=1),
    ])

    # 2. Dendritic Transformer
    models["Dendritic Transformer (Pyramidal FFN)"] = dn.Sequential([
        dn.Dense(in_features=d_in, out_features=d_model),
        dn.PositionalEncoding(d_model=d_model, max_len=seq_len + 4),
        dn.DendriticTransformerBlock(d_model=d_model, num_heads=4, d_ff=24, num_branches=2, dropout=0.0),
        dn.Flatten(),
        dn.Dense(in_features=seq_len * d_model, out_features=16),
        dn.SiLU(),
        dn.Dense(in_features=16, out_features=1),
    ])

    # 3. KAN Transformer
    models["KAN Transformer (Chebyshev FFN)"] = dn.Sequential([
        dn.Dense(in_features=d_in, out_features=d_model),
        dn.PositionalEncoding(d_model=d_model, max_len=seq_len + 4),
        dn.KANTransformerBlock(d_model=d_model, num_heads=4, d_ff=24, degree=3, dropout=0.0),
        dn.Flatten(),
        dn.Dense(in_features=seq_len * d_model, out_features=16),
        dn.SiLU(),
        dn.Dense(in_features=16, out_features=1),
    ])

    return models


def run_transformer_arena() -> None:
    print("=" * 80)
    print("🧠 EXPLORING NOVEL TRANSFORMER ARCHITECTURES: STANDARD vs DENDRITIC vs KAN")
    print("=" * 80)
    print("Task: Coupled Non-Linear Sequence Dynamics [y = (1/T) sum_t (sin(pi*x0) + x1*x2/(1+x3^2))]")
    print("Evaluating parameter efficiency, test MSE, and R² variance explained.\n")

    seq_len = 8
    d_in = 4
    d_model = 16
    epochs = 40
    batch_size = 20

    X_train, y_train, X_test, y_test = generate_coupled_sequence_data(
        n_samples=400, seq_len=seq_len, d_in=d_in, seed=42
    )

    print(f"Dataset: X_train={X_train.shape}, y_train={y_train.shape} | X_test={X_test.shape}, y_test={y_test.shape}\n")

    models = build_models(d_in=d_in, d_model=d_model, seq_len=seq_len)
    results = {}

    for name, model in models.items():
        dn.set_seed(42)
        n_params = count_params(model)
        optimizer = dn.Adam(lr=0.015, weight_decay=1e-5)
        model.compile(optimizer=optimizer, loss=dn.MSELoss())

        print(f"Training [{name}] (Params: {n_params})...")
        t0 = time.perf_counter()
        hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Evaluate on test set
        preds = model.forward(X_test)
        mse = float(np.mean((preds - y_test) ** 2))
        mae = float(np.mean(np.abs(preds - y_test)))
        var_y = float(np.var(y_test))
        r2 = max(0.0, float(1.0 - (mse / (var_y + 1e-8)))) * 100.0

        results[name] = {
            "params": n_params,
            "train_time_ms": elapsed_ms,
            "final_train_loss": hist["loss"][-1],
            "test_mse": mse,
            "test_mae": mae,
            "test_r2": r2,
            "history": hist,
        }
        print(f"  ✓ Finished in {elapsed_ms:.1f}ms | Final Loss: {hist['loss'][-1]:.5f} | Test MSE: {mse:.5f} | R²: {r2:.2f}%\n")

    # Grand Comparison Table
    print("=" * 86)
    print("📊 TRANSFORMER BENCHMARK SUMMARY")
    print("=" * 86)
    print(f"{'Architecture':<40} | {'Params':>7} | {'Test MSE':>10} | {'Test MAE':>10} | {'R² (%)':>8}")
    print("-" * 86)
    for name, res in results.items():
        print(f"{name:<40} | {res['params']:>7d} | {res['test_mse']:>10.5f} | {res['test_mae']:>10.5f} | {res['test_r2']:>7.2f}%")
    print("=" * 86)

    # Sample Predictions Comparison
    print("\n🔍 Sample Sequence Predictions vs Ground Truth:")
    sample_indices = [0, 5, 10, 15, 20]
    print(f"{'Sample #':<10} | {'True y':>10} | " + " | ".join(f"{name.split()[0]:>12}" for name in models.keys()))
    print("-" * 65)
    for idx in sample_indices:
        yt = y_test[idx, 0]
        preds_str = []
        for name, model in models.items():
            yp = model.forward(X_test[idx:idx+1])[0, 0]
            preds_str.append(f"{yp:>12.4f}")
        print(f"Sample {idx:<3d} | {yt:>10.4f} | " + " | ".join(preds_str))

    print("\n" + "=" * 80)
    print("Architectural Insights:")
    print("  • Standard Transformer uses static activation MLP (Dense + ReLU/SiLU).")
    print("  • Dendritic Transformer utilizes multi-compartment branch gating in token space.")
    print("  • KAN Transformer learns continuous orthogonal polynomial basis functions on channels.")
    print("  • Both KANTransformerBlock and DendriticTransformerBlock are 100% plug-and-play!")
    print("=" * 80)


if __name__ == "__main__":
    run_transformer_arena()
