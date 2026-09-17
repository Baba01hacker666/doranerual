"""Convolutional Neural Network (CNN) Example with neural_lib.

Demonstrates spatial 2D convolutions on CPU using pure NumPy:
1. Synthetic 8x8 image classification task (Horizontal lines vs Vertical bars).
2. Mini CNN: Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dropout -> Dense -> Softmax.
3. Optimization using Adam with decoupled moments.
4. Categorical cross-entropy training loop.
5. Evaluation, prediction, and serialization verification.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_lib import (
    Sequential,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    Dense,
    ReLU,
    Softmax,
    CategoricalCrossEntropy,
    Adam,
    set_seed,
    train_test_split,
    one_hot_encode,
    load_model,
)


def generate_synthetic_patterns(n_samples: int = 400, img_size: int = 8, noise: float = 0.1):
    """Generate simple 8x8 synthetic images of horizontal lines (class 0) vs vertical bars (class 1)."""
    rng = np.random.default_rng(42)
    images = np.zeros((n_samples, 1, img_size, img_size), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)

    half = n_samples // 2
    # Class 0: Horizontal stripe across middle rows
    for i in range(half):
        row = rng.integers(2, img_size - 2)
        images[i, 0, row:row+2, :] = 1.0
        labels[i] = 0

    # Class 1: Vertical stripe across middle columns
    for i in range(half, n_samples):
        col = rng.integers(2, img_size - 2)
        images[i, 0, :, col:col+2] = 1.0
        labels[i] = 1

    # Add Gaussian noise
    if noise > 0:
        images += rng.normal(0, noise, images.shape).astype(np.float32)

    perm = rng.permutation(n_samples)
    return images[perm], labels[perm]


def main() -> None:
    print("=" * 70)
    print(" neural_lib: 2D Convolutional Network (CNN) Demo on CPU ")
    print("=" * 70)

    set_seed(42)

    # 1. Generate 400 synthetic 8x8 single-channel images
    X, y_raw = generate_synthetic_patterns(n_samples=400, img_size=8, noise=0.1)
    y = one_hot_encode(y_raw, num_classes=2)
    print(f"[1/5] Generated {len(X)} images (shape: {X.shape}) across 2 visual patterns.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=42)
    print(f"      Train: {X_train.shape[0]} images | Test: {X_test.shape[0]} images")

    # 2. Define CNN Architecture
    print("\n[2/5] Constructing CNN architecture:")
    model = Sequential([
        Conv2D(in_channels=1, out_channels=4, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),   # Output: (4 channels, 4x4)
        Flatten(),                          # Output: 4 * 4 * 4 = 64
        Dropout(drop_rate=0.1),
        Dense(in_features=64, out_features=2),
        Softmax(),
    ])
    model.summary()

    # 3. Compile with Adam and CategoricalCrossEntropy
    print("\n[3/5] Compiling with Adam optimizer (lr=0.005)...")
    model.compile(
        loss=CategoricalCrossEntropy(),
        optimizer=Adam(lr=0.005),
        metrics=["accuracy"],
    )

    # 4. Train
    print("\n[4/5] Training for 20 epochs...")
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        verbose=1,
        validation_data=(X_test, y_test),
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\n" + "-" * 50)
    print(f"Final Test Evaluation: Loss = {test_loss:.4f}, Accuracy = {test_acc * 100:.2f}%")
    print("-" * 50)

    assert test_acc >= 0.90, f"Expected test accuracy >= 90%, got {test_acc * 100:.2f}%"

    # 5. Serialization roundtrip
    print("\n[5/5] Verifying model persistence...")
    save_path = Path("saved_models") / "cnn_pattern_model"
    model.save(save_path)
    loaded = load_model(save_path)

    orig_p = model.predict_proba(X_test)
    load_p = loaded.predict_proba(X_test)
    np.testing.assert_allclose(orig_p, load_p, atol=1e-6)
    print("  SUCCESS: CNN saved, reloaded, and verified with identical predictions!")

    print("\n" + "=" * 70)
    print(" CNN Demo finished successfully! ")
    print("=" * 70)


if __name__ == "__main__":
    main()
