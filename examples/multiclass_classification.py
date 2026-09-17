"""Multiclass Classification Example with neural_lib.

Demonstrates multi-class classification using Softmax activation and Categorical Cross-Entropy:
1. Synthetic 3-class dataset generation with pure NumPy (make_blobs).
2. One-hot encoding of target labels.
3. Train / test split.
4. Sequential network: Dense -> ReLU -> Dense -> Softmax.
5. Compile with CategoricalCrossEntropy and SGD + Momentum.
6. Training and validation monitoring.
7. Test evaluation.
8. Inference and prediction probability distributions.
9. Persistence verification (save & load).
"""

import os
import sys
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_lib import (
    Sequential,
    Dense,
    ReLU,
    Softmax,
    CategoricalCrossEntropy,
    SGD,
    set_seed,
    train_test_split,
    one_hot_encode,
    make_blobs,
    load_model,
)


def main() -> None:
    print("=" * 70)
    print(" neural_lib: Multiclass Classification Demo (3-Class Blobs) ")
    print("=" * 70)

    # 1. Set seed for deterministic reproducibility
    set_seed(42)
    print("[1/6] Random seed set to 42.")

    # 2. Generate 3-class isotropic Gaussian clusters
    n_samples = 1500
    n_classes = 3
    X, y_labels = make_blobs(n_samples=n_samples, n_features=2, centers=n_classes, cluster_std=0.8, seed=42)
    print(f"[2/6] Generated {n_samples} samples across {n_classes} clusters.")

    # One-hot encode targets for categorical cross-entropy
    y_one_hot = one_hot_encode(y_labels, num_classes=n_classes)
    print(f"      X shape: {X.shape}, y (labels) shape: {y_labels.shape}, y (one-hot) shape: {y_one_hot.shape}")

    # Train / test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, shuffle=True, seed=42
    )
    print(f"      Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 3. Build model architecture: 2 features -> 24 hidden -> 3 classes
    print("\n[3/6] Defining network architecture:")
    model = Sequential([
        Dense(in_features=2, out_features=24, weight_init="he"),
        ReLU(),
        Dense(in_features=24, out_features=n_classes, weight_init="xavier"),
        Softmax(),
    ])
    model.summary()

    # 4. Compile with Categorical Cross-Entropy and SGD with Momentum
    print("\n[4/6] Compiling model...")
    optimizer = SGD(lr=0.05, momentum=0.9)
    model.compile(
        loss=CategoricalCrossEntropy(),
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    # 5. Train model
    epochs = 40
    batch_size = 32
    print(f"\n[5/6] Training model for {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=True,
        validation_data=(X_test, y_test),
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\n" + "-" * 50)
    print(f"Final Test Evaluation: Loss = {test_loss:.4f}, Accuracy = {test_acc * 100:.2f}%")
    print("-" * 50)

    # Check assertions
    assert history["loss"][-1] < history["loss"][0], "Loss failed to decrease during training."
    assert test_acc > 0.90, f"Expected test accuracy > 90%, got {test_acc * 100:.2f}%"

    # 6. Inference on sample coordinates
    sample_queries = np.array([
        [3.5, 0.0],    # Near class 0 center
        [-1.7, 3.0],   # Near class 1 center
        [-1.7, -3.0],  # Near class 2 center
    ], dtype=np.float32)

    probas = model.predict_proba(sample_queries)
    preds = model.predict(sample_queries)

    print("\nInference on sample points:")
    for i, (point, prob, pred) in enumerate(zip(sample_queries, probas, preds)):
        prob_str = ", ".join([f"Class {c}: {p * 100:5.1f}%" for c, p in enumerate(prob)])
        print(f"  Point ({point[0]:5.1f}, {point[1]:5.1f}) -> [{prob_str}] -> Predicted Class: {pred[0]}")

    # 7. Model serialization roundtrip
    print("\n[6/6] Testing Save & Load persistence roundtrip...")
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "multiclass_blobs_model"

    model.save(save_path)
    print(f"  Model saved to: {save_path}.json and {save_path}.npz")

    loaded_model = load_model(save_path)
    print("  Model loaded back into memory.")

    original_preds = model.predict_proba(X_test)
    loaded_preds = loaded_model.predict_proba(X_test)

    max_diff = np.max(np.abs(original_preds - loaded_preds))
    print(f"  Max absolute prediction difference between original & loaded model: {max_diff:.8e}")
    np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-6, atol=1e-6)
    print("  SUCCESS: Original and loaded models produce 100% identical predictions!")

    print("\n" + "=" * 70)
    print(" Multiclass Classification Demo completed successfully! ")
    print("=" * 70)


if __name__ == "__main__":
    main()
