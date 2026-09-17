"""Interactive Digit Generator & Neural Network Predictor with Side-by-Side Noise Display.

Features:
- Side-by-side ASCII comparison: shows Clean Base Digit vs Noised Digit (what the network sees).
- Continuous interactive loop: prompt returns after every prediction.
- Noise control:
    * Enter a digit: '7' (uses current noise level)
    * Enter digit + noise: '7 0.35' (adds custom noise to this sample)
    * Change default noise: 'noise 0.2' or 'n 0.25'
    * Type 'q' or 'exit' to quit.

Usage:
  python examples/generate_and_predict_digit.py
  python examples/generate_and_predict_digit.py 7
  python examples/generate_and_predict_digit.py 7 0.35
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
    one_hot_encode,
    train_test_split,
    make_digits,
    load_model,
)

TEMPLATES = {
    0: [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
    1: [[0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 1, 1]],
    2: [[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]],
    3: [[1, 1, 1, 0], [0, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0]],
    4: [[1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
    5: [[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0]],
    6: [[0, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]],
    7: [[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
    8: [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]],
    9: [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 1, 0]],
}


def generate_digit_pair(digit: int, noise: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Generate both a clean digit image and its noised counterpart."""
    rng = np.random.default_rng()
    t = np.array(TEMPLATES[digit], dtype=np.float32)
    th, tw = t.shape

    img_clean = np.zeros((8, 8), dtype=np.float32)
    r = rng.integers(1, 8 - th)
    c = rng.integers(1, 8 - tw)
    intensity = rng.uniform(0.95, 1.05)
    img_clean[r : r + th, c : c + tw] = t * intensity

    # Apply noise to copy
    img_noised = img_clean.copy()
    if noise > 0.0:
        noise_arr = rng.normal(0.0, noise, img_noised.shape).astype(np.float32)
        img_noised = np.clip(img_noised + noise_arr, 0.0, 1.0)

    return img_clean, img_noised


def _format_row_chars(row: np.ndarray) -> str:
    """Format an 8-pixel row into 16-character ASCII block representation."""
    chars = []
    for val in row:
        if val > 0.65:
            chars.append("██")
        elif val > 0.35:
            chars.append("▒▒")
        elif val > 0.15:
            chars.append("··")
        else:
            chars.append("  ")
    return "".join(chars)


def render_side_by_side_ascii(img_clean: np.ndarray, img_noised: np.ndarray, noise: float) -> str:
    """Render clean and noised digit grids side-by-side with clear headers."""
    gap = "        "  # 8 spaces between columns
    output = []

    # Headers
    header_clean = "  [ Clean Base Digit ]  "
    header_noised = f"  [ Noised Input (σ={noise:.2f}) ]"
    output.append(f"{header_clean:<26}{gap}{header_noised}")

    output.append(f"  ┌────────────────┐{gap}  ┌────────────────┐")
    for row_c, row_n in zip(img_clean, img_noised):
        str_c = _format_row_chars(row_c)
        str_n = _format_row_chars(row_n)
        output.append(f"  │{str_c}│{gap}  │{str_n}│")
    output.append(f"  └────────────────┘{gap}  └────────────────┘")

    return "\n".join(output)


def render_probability_bars(probas: np.ndarray, predicted_digit: int) -> str:
    """Render a text bar chart showing model probabilities across all 10 digits."""
    lines = []
    max_bar_width = 24
    for d, p in enumerate(probas):
        bar_len = int(round(p * max_bar_width))
        bar = "█" * bar_len
        marker = " ◀ PREDICTED" if d == predicted_digit else ""
        lines.append(f"    Digit {d}: {p * 100:5.1f}% | {bar:<{max_bar_width}}{marker}")
    return "\n".join(lines)


def get_or_train_model() -> Sequential:
    """Load existing trained digit classifier or train one if not found."""
    model_path = Path("saved_models") / "digit_0_to_9_classifier"
    if Path(f"{model_path}.json").exists() and Path(f"{model_path}.npz").exists():
        return load_model(model_path)

    print("Pretrained model not found. Training 0-9 digit classifier (~2s)...")
    X, y_raw = make_digits(n_samples=1500, noise=0.1, flatten=True, seed=42)
    y = one_hot_encode(y_raw, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    model = Sequential([
        Dense(64, 48, weight_init="he"),
        ReLU(),
        Dense(48, 24, weight_init="he"),
        ReLU(),
        Dense(24, 10, weight_init="xavier"),
        Softmax(),
    ])
    model.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(lr=0.01), metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    model.save(model_path)
    return model


def process_digit(digit: int, noise: float, model: Sequential) -> None:
    """Generate clean & noised image, display side-by-side, and output model prediction."""
    print("\n" + "=" * 64)
    print(f" [Requested Digit]: '{digit}'   |   Applied Noise (σ): {noise:.2f}")
    print("=" * 64)

    # 1. Generate clean and noised images
    img_clean, img_noised = generate_digit_pair(digit, noise=noise)

    # 2. Display side-by-side ASCII art
    print("\nVisual Comparison (Clean vs Noised):")
    print(render_side_by_side_ascii(img_clean, img_noised, noise=noise))

    # 3. Feed NOSED image to neural network
    x_input = img_noised.reshape(1, 64)
    probas = model.predict_proba(x_input)[0]
    predicted_digit = int(np.argmax(probas))
    confidence = probas[predicted_digit] * 100.0

    # 4. Output prediction
    print("\nNetwork Prediction on Noised Input:")
    print(f"  ▶ Predicted Digit: {predicted_digit}")
    print(f"  ▶ Confidence:      {confidence:.2f}%")

    print("\nProbability Distribution (0 to 9):")
    print(render_probability_bars(probas, predicted_digit))
    print("=" * 64)


def main() -> None:
    print("Loading 0 to 9 Neural Network Classifier...")
    model = get_or_train_model()
    print("Ready!")

    # Check CLI arguments: e.g. 'python script.py 7' or 'python script.py 7 0.35'
    if len(sys.argv) > 1:
        digit_arg = sys.argv[1].strip()
        noise_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
        if digit_arg.isdigit() and 0 <= int(digit_arg) <= 9:
            process_digit(int(digit_arg), noise_arg, model)
            return
        else:
            print(f"Error: Argument must be a digit between 0 and 9. Got: {digit_arg}")
            sys.exit(1)

    # Interactive Loop
    current_noise = 0.15
    print("\n" + "-" * 64)
    print(" Interactive Mode:")
    print("   • Enter a digit:           e.g. '7' (uses current noise)")
    print("   • Enter digit + noise:     e.g. '7 0.35'")
    print("   • Change default noise:    e.g. 'n 0.2' or 'noise 0.3'")
    print("   • Exit:                    type 'q' or 'exit'")
    print("-" * 64)

    while True:
        try:
            prompt = f"\n[Noise: {current_noise:.2f}] Enter digit (0-9): "
            user_input = input(prompt).strip()

            if not user_input:
                continue

            if user_input.lower() in ("q", "quit", "exit"):
                print("Exiting interactive session. Goodbye!")
                break

            # Check if setting noise (e.g. 'n 0.25' or 'noise 0.3')
            tokens = user_input.split()
            if tokens[0].lower() in ("n", "noise"):
                if len(tokens) > 1:
                    try:
                        new_noise = float(tokens[1])
                        if new_noise < 0.0:
                            print("Noise must be non-negative (>= 0.0).")
                        else:
                            current_noise = new_noise
                            print(f"Default noise updated to: {current_noise:.2f}")
                    except ValueError:
                        print("Invalid noise number. Example: 'noise 0.25'")
                else:
                    print(f"Current noise is {current_noise:.2f}. Change it with: 'noise <value>'")
                continue

            # Check if user entered 'digit' or 'digit noise'
            if len(tokens) >= 1 and tokens[0].isdigit() and (0 <= int(tokens[0]) <= 9):
                digit = int(tokens[0])
                sample_noise = current_noise
                if len(tokens) >= 2:
                    try:
                        sample_noise = max(0.0, float(tokens[1]))
                    except ValueError:
                        print(f"Could not parse noise '{tokens[1]}', using default {current_noise:.2f}")

                process_digit(digit, sample_noise, model)
            else:
                print("Invalid input. Enter a single digit 0-9 (e.g. '7' or '7 0.3'), 'noise 0.2', or 'q'.")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive session. Goodbye!")
            break


if __name__ == "__main__":
    main()
