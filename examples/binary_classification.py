"""Binary Classification Example with neural_lib.

Demonstrates end-to-end training of a neural network on a synthetic 2D moons dataset:
1. Dataset generation using pure NumPy (make_moons).
2. Train / test split.
3. Sequential model building with Dense, ReLU, and Sigmoid layers.
4. Model compilation with BinaryCrossEntropy and SGD with Momentum.
5. Training with epoch-by-epoch loss & accuracy tracking.
6. Test set evaluation.
7. Inference on sample queries.
8. Model serialization (save/load roundtrip) and verification of identical predictions.
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
    Sigmoid,
    BinaryCrossEntropy,
    SGD,
    set_seed,
    train_test_split,
    make_moons,
    load_model,
)


def main() -> None:
    print("=" * 70)
    print(" neural_lib: Binary Classification Demo (CPU / Pure NumPy) ")
    print("=" * 70)

    # 1. Set seed for deterministic reproducibility
    set_seed(42)
    print("[1/6] Random seed set to 42 for reproducible results.")

    # 2. Generate synthetic non-linear 2D dataset (Two Moons)
    n_samples = 1200
    noise = 0.15
    X, y = make_moons(n_samples=n_samples, noise=noise, seed=42)
    print(f"[2/6] Generated {n_samples} samples of 'Two Moons' dataset (noise={noise}).")
    print(f"      Feature matrix X shape: {X.shape}, Target vector y shape: {y.shape}")

    # Train / test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, seed=42
    )
    print(f"      Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 3. Define Sequential model architecture
    print("\n[3/6] Defining network architecture:")
    model = Sequential([
        Dense(in_features=2, out_features=32, weight_init="he"),
        ReLU(),
        Dense(in_features=32, out_features=16, weight_init="he"),
        ReLU(),
        Dense(in_features=16, out_features=1, weight_init="xavier"),
        Sigmoid(),
    ])
    model.summary()

    # 4. Compile model with Binary Cross-Entropy loss and SGD with momentum
    print("\n[4/6] Compiling model...")
    optimizer = SGD(lr=0.1, momentum=0.9)
    model.compile(
        loss=BinaryCrossEntropy(),
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    # 5. Fit the model
    epochs = 50
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

    # Evaluate on unseen test data
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\n" + "-" * 50)
    print(f"Final Test Evaluation: Loss = {test_loss:.4f}, Accuracy = {test_acc * 100:.2f}%")
    print("-" * 50)

    # Verify performance criteria
    initial_loss = history["loss"][0]
    final_loss = history["loss"][-1]
    print(f"Training loss decreased from {initial_loss:.4f} to {final_loss:.4f}.")
    assert final_loss < initial_loss, "Training loss did not decrease!"
    assert test_acc > 0.85, f"Expected test accuracy > 85%, got {test_acc * 100:.2f}%"

    # 6. Inference demo
    sample_queries = np.array([
        [0.0, 0.5],   # Expected class 0
        [1.0, -0.2],  # Expected class 1
        [-0.5, 0.8],  # Expected class 0
        [1.8, 0.1],   # Expected class 1
    ], dtype=np.float32)

    probas = model.predict_proba(sample_queries)
    preds = model.predict(sample_queries)

    print("\nInference on sample coordinates:")
    for i, (coord, prob, pred) in enumerate(zip(sample_queries, probas, preds)):
        print(f"  Sample {i + 1}: Point ({coord[0]:5.2f}, {coord[1]:5.2f}) -> "
              f"P(class=1) = {prob[0]:.4f} -> Predicted Class = {pred[0]}")

    # 7. Model serialization roundtrip
    print("\n[6/6] Testing Save & Load persistence roundtrip...")
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "binary_moons_model"

    model.save(save_path)
    print(f"  Model saved to: {save_path}.json and {save_path}.npz")

    # Re-load model from disk
    loaded_model = load_model(save_path)
    print("  Model successfully loaded back into memory.")

    # Compare predictions
    original_preds = model.predict_proba(X_test)
    loaded_preds = loaded_model.predict_proba(X_test)

    max_diff = np.max(np.abs(original_preds - loaded_preds))
    print(f"  Max absolute prediction difference between original & loaded model: {max_diff:.8e}")
    np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-6, atol=1e-6)
    print("  SUCCESS: Original and loaded models produce 100% identical predictions!")

    print("\n" + "=" * 70)
    print(" Binary Classification Demo completed successfully! ")
    print("=" * 70)


if __name__ == "__main__":
    main()
