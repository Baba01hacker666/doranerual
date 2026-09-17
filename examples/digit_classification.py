"""0 to 9 Digit Classifier Example with neural_lib.

End-to-end multi-class classification for 8x8 handwritten/printed digits (0-9):
1. Pure NumPy synthetic digit generation with natural jitter and noise.
2. Train / test split.
3. Sequential MLP Architecture: 64 inputs -> 48 hidden -> 24 hidden -> 10 output classes.
4. Adam optimizer with Categorical Cross-Entropy.
5. Training loop with epoch progress.
6. Evaluation and test accuracy verification (> 95%).
7. ASCII Art terminal visualization showing actual digit samples and predictions.
8. Model persistence (save & reload).
"""

import sys
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_lib import (
    Sequential,
    Dense,
    ReLU,
    Softmax,
    CategoricalCrossEntropy,
    Adam,
    set_seed,
    train_test_split,
    one_hot_encode,
    make_digits,
    load_model,
)


def render_ascii_digit(image_flat: np.ndarray, threshold: float = 0.35) -> str:
    """Render an 8x8 flattened grayscale image as an ASCII character grid."""
    img = image_flat.reshape(8, 8)
    lines = []
    for row in img:
        chars = []
        for val in row:
            if val > 0.6:
                chars.append("██")
            elif val > threshold:
                chars.append("▒▒")
            else:
                chars.append("  ")
        lines.append("  " + "".join(chars))
    return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print(" neural_lib: 0 to 9 Digit Classifier (Pure NumPy / CPU) ")
    print("=" * 70)

    # 1. Reproducibility
    set_seed(42)

    # 2. Generate 1,500 samples of 8x8 digits (0 through 9)
    n_samples = 1500
    print(f"[1/6] Generating {n_samples} synthetic 8x8 digit images (0 to 9)...")
    X, y_labels = make_digits(n_samples=n_samples, noise=0.10, flatten=True, seed=42)
    y_one_hot = one_hot_encode(y_labels, num_classes=10)

    print(f"      Feature matrix X shape: {X.shape} (64 pixels per image)")
    print(f"      One-hot target y shape: {y_one_hot.shape} (10 classes)")

    # 3. Train / test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, shuffle=True, seed=42
    )
    y_test_labels = np.argmax(y_test, axis=-1)
    print(f"      Training set: {X_train.shape[0]} images | Test set: {X_test.shape[0]} images")

    # 4. Construct Neural Network Architecture
    # 64 input pixels -> 48 neurons -> 24 neurons -> 10 digit classes
    print("\n[2/6] Defining Neural Network Architecture:")
    model = Sequential([
        Dense(in_features=64, out_features=48, weight_init="he"),
        ReLU(),
        Dense(in_features=48, out_features=24, weight_init="he"),
        ReLU(),
        Dense(in_features=24, out_features=10, weight_init="xavier"),
        Softmax(),
    ])
    model.summary()

    # 5. Compile Model with Adam Optimizer
    print("\n[3/6] Compiling model with Adam optimizer (lr=0.01)...")
    model.compile(
        loss=CategoricalCrossEntropy(),
        optimizer=Adam(lr=0.01),
        metrics=["accuracy"],
    )

    # 6. Fit Model
    epochs = 20
    batch_size = 32
    print(f"\n[4/6] Training for {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=True,
        validation_data=(X_test, y_test),
    )

    # 7. Evaluate Performance
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\n" + "-" * 55)
    print(f"Final Test Evaluation: Loss = {test_loss:.4f}, Accuracy = {test_acc * 100:.2f}%")
    print("-" * 55)

    assert test_acc >= 0.95, f"Expected accuracy >= 95%, got {test_acc * 100:.2f}%"

    # 8. ASCII Terminal Demonstration on Random Test Digits
    print("\n[5/6] Demonstrating Predictions with ASCII Visualization:")
    sample_indices = [5, 12, 27, 42, 60]
    for idx in sample_indices:
        sample_img = X_test[idx]
        true_label = int(y_test_labels[idx])

        # Predict
        proba = model.predict_proba(sample_img.reshape(1, -1))[0]
        pred_label = int(np.argmax(proba))
        confidence = proba[pred_label] * 100.0

        print(f"\nSample #{idx} (True Digit: {true_label}):")
        print(render_ascii_digit(sample_img))
        print(f"  ──▶ Predicted: [{pred_label}] with {confidence:.1f}% confidence "
              f"({'CORRECT' if pred_label == true_label else 'WRONG'})")

    # 9. Model Persistence Roundtrip
    print("\n[6/6] Testing Save & Reload Roundtrip...")
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "digit_0_to_9_classifier"

    model.save(save_path)
    print(f"  Model saved to: {save_path}.json and {save_path}.npz")

    loaded_model = load_model(save_path)
    print("  Model loaded back into memory.")

    orig_p = model.predict_proba(X_test)
    load_p = loaded_model.predict_proba(X_test)
    np.testing.assert_allclose(orig_p, load_p, atol=1e-6)
    print("  SUCCESS: Original and loaded models produce 100% identical digit predictions!")

    print("\n" + "=" * 70)
    print(" 0 to 9 Digit Classifier Demo completed successfully! ")
    print("=" * 70)


if __name__ == "__main__":
    main()
