"""Workspace and project state manager for doraneural CLI.

Enables seamless workflow without long commands:
- `doraneural new`: Initialize a project model
- `doraneural train`: Train current project without tedious flags
- `doraneural status`: View current architecture, training history, and accuracy
- `doraneural predict`: Run inference on user-supplied numbers
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from .easy import create
from .model import Sequential
from .losses import BinaryCrossEntropy, CategoricalCrossEntropy
from .optimizers import Adam
from .serialization import save_model, load_model
from .teach import render_network_diagram, explain
from .utils import make_moons, make_blobs, make_digits, train_test_split, one_hot_encode

CONFIG_FILE = ".doraneural.json"
DEFAULT_MODEL_NAME = "current_model"


def load_workspace_config() -> Dict[str, Any]:
    """Load project configuration from current directory."""
    path = Path(CONFIG_FILE)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_workspace_config(config: Dict[str, Any]) -> None:
    """Save project configuration to current directory."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def init_workspace(
    inputs: int = 2,
    hidden: List[int] = [16, 8],
    outputs: int = 1,
    dataset: str = "moons",
) -> None:
    """Create a new model and dataset setup in the current workspace."""
    model = create(inputs=inputs, hidden=hidden, outputs=outputs)
    save_model(model, DEFAULT_MODEL_NAME)

    config = {
        "inputs": inputs,
        "hidden": hidden,
        "outputs": outputs,
        "dataset": dataset,
        "epochs_trained": 0,
        "best_accuracy": 0.0,
        "model_file": DEFAULT_MODEL_NAME,
    }
    save_workspace_config(config)

    print("\n🎉 New doraneural project initialized!")
    print(f"   Architecture: {inputs} inputs ──▶ {hidden} hidden ──▶ {outputs} outputs")
    print(f"   Default dataset: {dataset}")
    print("\nNext step: simply type 'doraneural train' to train this network!")


def get_dataset_for_task(name: str, n_samples: int = 1000):
    """Generate or retrieve dataset based on task name."""
    name = name.lower().strip()
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.15, seed=42)
        return X, y
    elif name == "blobs":
        X, y_raw = make_blobs(n_samples=n_samples, centers=3, seed=42)
        y = one_hot_encode(y_raw, num_classes=3)
        return X, y
    elif name in ("digits", "digit"):
        X, y_raw = make_digits(n_samples=n_samples, noise=0.1, flatten=True, seed=42)
        y = one_hot_encode(y_raw, num_classes=10)
        return X, y
    else:
        # Default to moons
        X, y = make_moons(n_samples=n_samples, noise=0.15, seed=42)
        return X, y


def train_current_workspace(epochs: int = 25, lr: float = 0.01) -> None:
    """Train the active project model on its configured dataset."""
    config = load_workspace_config()
    if not config:
        print("No active project found in this folder. Creating a standard starter project...")
        init_workspace(inputs=2, hidden=[16, 8], outputs=1, dataset="moons")
        config = load_workspace_config()

    model_path = Path(config.get("model_file", DEFAULT_MODEL_NAME))
    try:
        model = load_model(model_path)
    except Exception:
        model = create(
            inputs=config["inputs"],
            hidden=config["hidden"],
            outputs=config["outputs"],
            lr=lr,
        )

    # Automatically ensure model is compiled with appropriate loss and optimizer
    is_binary = (config.get("outputs", 1) == 1)
    loss_fn = BinaryCrossEntropy() if is_binary else CategoricalCrossEntropy()
    model.compile(loss=loss_fn, optimizer=Adam(lr=lr), metrics=["accuracy"])

    dataset_name = config.get("dataset", "moons")
    print(f"\nTraining current model on '{dataset_name}' dataset for {epochs} epochs...")
    X, y = get_dataset_for_task(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        verbose=1,
        validation_data=(X_test, y_test),
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    model.save(model_path)

    # Update config
    prev_trained = config.get("epochs_trained", 0)
    config["epochs_trained"] = prev_trained + epochs
    config["best_accuracy"] = max(config.get("best_accuracy", 0.0), float(test_acc))
    config["latest_loss"] = float(test_loss)
    save_workspace_config(config)

    print("\n" + "─" * 58)
    print(f"✨ Training complete!")
    print(f"   Accuracy: {test_acc * 100:.2f}%  |  Loss: {test_loss:.4f}")
    print(f"   Total Epochs Trained: {config['epochs_trained']}")
    print("─" * 58)
    print("\n💡 Learning Tip:")
    print("   Loss measures mistakes (lower is better). Accuracy is % correct.")
    print("   Type 'doraneural status' to inspect your network anytime!")


def show_workspace_status() -> None:
    """Display visual status of current model and project."""
    config = load_workspace_config()
    if not config:
        print("No active project in this directory.")
        print("Run 'doraneural new' to create one, or 'doraneural train' to start!")
        return

    inputs = config.get("inputs", 2)
    hidden = config.get("hidden", [16, 8])
    outputs = config.get("outputs", 1)

    print("\n" + "=" * 62)
    print(" 📊 doraneural Project Status")
    print("=" * 62)
    print("\nArchitecture Flow:")
    print(render_network_diagram(inputs, hidden, outputs))

    print("\nProject Details:")
    print(f"  • Dataset:              {config.get('dataset', 'moons')}")
    print(f"  • Total Epochs Trained: {config.get('epochs_trained', 0)}")
    best_acc = config.get("best_accuracy", 0.0)
    print(f"  • Best Test Accuracy:   {best_acc * 100:.2f}%")
    if "latest_loss" in config:
        print(f"  • Latest Loss:          {config['latest_loss']:.4f}")
    print(f"  • Saved Files:          {config.get('model_file', DEFAULT_MODEL_NAME)}.json / .npz")
    print("=" * 62)


def predict_with_current_workspace(values: List[float]) -> None:
    """Pass numerical values to the active model and print prediction."""
    config = load_workspace_config()
    if not config:
        print("No active project found. Run 'doraneural new' first.")
        return

    model_path = Path(config.get("model_file", DEFAULT_MODEL_NAME))
    model = load_model(model_path)

    expected_inputs = config.get("inputs", 2)
    if len(values) != expected_inputs:
        print(f"Error: Model expects {expected_inputs} input numbers, but you gave {len(values)}.")
        print(f"Usage example: doraneural predict " + " ".join(["1.0"] * expected_inputs))
        return

    x = np.array([values], dtype=np.float32)
    proba = model.predict_proba(x)[0]
    pred = model.predict(x)[0]

    print("\n" + "=" * 50)
    print(f" Input Values: {values}")
    print("=" * 50)
    if len(proba) == 1:
        # Binary
        conf = proba[0] if pred[0] == 1 else (1.0 - proba[0])
        print(f"  ▶ Prediction: Class {pred[0]} ({'Positive/Yes' if pred[0]==1 else 'Negative/No'})")
        print(f"  ▶ Confidence: {conf * 100:.2f}%")
    else:
        # Multiclass
        print(f"  ▶ Predicted Class: {pred[0]}")
        print(f"  ▶ Confidence:      {proba[pred[0]] * 100:.2f}%")
    print("=" * 50)
