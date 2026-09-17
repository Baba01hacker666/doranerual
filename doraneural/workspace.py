"""Workspace and project state manager for doraneural CLI.

Enables seamless workflow without long commands:
- `doraneural new`: Initialize a project model
- `doraneural train`: Train current project without tedious flags (shows ASCII loss drop)
- `doraneural status`: View current architecture, training history, and accuracy
- `doraneural predict`: Run inference on user-supplied numbers
- `doraneural plot`: View training curves in ASCII terminal graph
- `doraneural export`: Export trained model to zero-dependency pure Python script
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import numpy as np

from .easy import create
from .model import Sequential
from .losses import BinaryCrossEntropy, CategoricalCrossEntropy, MeanSquaredError
from .optimizers import Adam
from .serialization import save_model, load_model
from .teach import render_network_diagram, explain
from .utils import make_moons, make_blobs, make_digits, train_test_split, one_hot_encode
from .data import load_csv
from .plot import plot_history, plot_ascii_curve
from .export import export_to_standalone_python
from .layers import Dense

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
    inputs: Optional[int] = None,
    hidden: Union[int, List[int]] = [16, 8],
    outputs: Optional[int] = None,
    dataset: str = "moons",
    task: str = "auto",
    data_path: Optional[str] = None,
) -> None:
    """Create a new model and dataset setup in the current workspace."""
    hidden_list = [hidden] if isinstance(hidden, int) else list(hidden)

    if data_path:
        # Auto-configure architecture from CSV
        csv_path = Path(data_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV file not found: {data_path}")
        X, y, meta = load_csv(csv_path)
        inputs = meta["n_features"]
        dataset = csv_path.name
        if meta["label_names"]:
            # Multiclass or binary classification
            num_classes = len(meta["label_names"])
            if num_classes <= 2:
                outputs = 1
                task = "binary"
            else:
                outputs = num_classes
                task = "multiclass"
        else:
            # Continuous targets -> regression
            outputs = 1
            task = "regression"
    else:
        inputs = inputs if inputs is not None else 2
        outputs = outputs if outputs is not None else 1

    model = create(inputs=inputs, hidden=hidden_list, outputs=outputs, task=task)
    save_model(model, DEFAULT_MODEL_NAME)

    config = {
        "inputs": inputs,
        "hidden": hidden_list,
        "outputs": outputs,
        "task": task,
        "dataset": dataset,
        "data_path": data_path,
        "epochs_trained": 0,
        "best_accuracy": 0.0,
        "model_file": DEFAULT_MODEL_NAME,
        "history": {"loss": []},
    }
    save_workspace_config(config)

    print("\n🎉 New doraneural project initialized!")
    print(f"   Task:         {task.upper()}")
    print(f"   Architecture: {inputs} inputs ──▶ {hidden_list} hidden ──▶ {outputs} outputs")
    print(f"   Dataset:      {dataset}")
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


def train_current_workspace(
    epochs: int = 25,
    lr: float = 0.01,
    data_path: Optional[str] = None,
) -> None:
    """Train the active project model on its configured dataset."""
    config = load_workspace_config()
    if not config:
        print("No active project found in this folder. Creating a standard starter project...")
        init_workspace(inputs=2, hidden=[16, 8], outputs=1, dataset="moons")
        config = load_workspace_config()

    if data_path:
        config["data_path"] = data_path
        config["dataset"] = Path(data_path).name

    model_path = Path(config.get("model_file", DEFAULT_MODEL_NAME))
    try:
        model = load_model(model_path)
    except Exception:
        model = create(
            inputs=config["inputs"],
            hidden=config["hidden"],
            outputs=config["outputs"],
            task=config.get("task", "auto"),
            lr=lr,
        )

    task = config.get("task", "auto")
    outputs = config.get("outputs", 1)

    # Load dataset: from custom CSV or built-in generator
    if config.get("data_path"):
        csv_file = config["data_path"]
        print(f"\nLoading dataset from '{csv_file}'...")
        X, y_raw, meta = load_csv(csv_file)
        if task == "regression":
            y = y_raw.reshape(-1, 1).astype(np.float32)
        elif meta["label_names"]:
            if len(meta["label_names"]) > 2:
                y = one_hot_encode(y_raw, num_classes=len(meta["label_names"]))
            else:
                y = y_raw.reshape(-1, 1).astype(np.float32)
        else:
            y = y_raw.reshape(-1, 1).astype(np.float32)
    else:
        dataset_name = config.get("dataset", "moons")
        print(f"\nTraining current model on '{dataset_name}' dataset for {epochs} epochs...")
        X, y = get_dataset_for_task(dataset_name)

    # Ensure model is compiled with appropriate loss and optimizer
    if task == "regression":
        loss_fn = MeanSquaredError()
        metrics = ["mse"]
    elif outputs == 1:
        loss_fn = BinaryCrossEntropy()
        metrics = ["accuracy"]
    else:
        loss_fn = CategoricalCrossEntropy()
        metrics = ["accuracy"]

    model.compile(loss=loss_fn, optimizer=Adam(lr=lr), metrics=metrics)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        verbose=1,
        validation_data=(X_test, y_test),
    )

    eval_result = model.evaluate(X_test, y_test)
    test_loss = eval_result[0]
    test_metric = eval_result[1] if len(eval_result) > 1 else None
    model.save(model_path)

    # Update config and save history for plotting
    prev_trained = config.get("epochs_trained", 0)
    config["epochs_trained"] = prev_trained + epochs
    config["latest_loss"] = float(test_loss)
    if test_metric is not None and task != "regression":
        config["best_accuracy"] = max(config.get("best_accuracy", 0.0), float(test_metric))

    # Append to cumulative history
    curr_hist = config.get("history", {})
    new_losses = curr_hist.get("loss", []) + history.history.get("loss", [])
    new_accs = curr_hist.get("accuracy", []) + history.history.get("accuracy", [])
    config["history"] = {"loss": new_losses, "accuracy": new_accs}

    save_workspace_config(config)

    # Render Visual ASCII Training Graph
    print("\n" + plot_history(history))

    print("─" * 58)
    print(f"✨ Training complete!")
    if task == "regression":
        print(f"   Final Test MSE Loss: {test_loss:.4f}")
    else:
        acc_pct = (test_metric * 100) if test_metric is not None else 0.0
        print(f"   Accuracy: {acc_pct:.2f}%  |  Loss: {test_loss:.4f}")
    print(f"   Total Epochs Trained: {config['epochs_trained']}")
    print("─" * 58)
    print("\n💡 Learning Tip:")
    print("   Loss measures prediction error (lower is better).")
    print("   Type 'doraneural plot' to inspect your full learning curve anytime!")
    print("   Type 'doraneural export' to export to a zero-dependency Python script!")


def plot_current_workspace() -> None:
    """Display the full learning curve in ASCII terminal graphics."""
    config = load_workspace_config()
    if not config or "history" not in config or not config["history"].get("loss"):
        print("No training history found. Run 'doraneural train' first to train the network!")
        return
    history_dict = config["history"]
    print("\n" + plot_history(history_dict))


def export_current_workspace(output_path: str = "predict.py") -> Optional[Path]:
    """Export active model to a standalone Python script."""
    config = load_workspace_config()
    if not config:
        print("No active project found in this directory. Run 'doraneural new' first.")
        return None

    model_path = Path(config.get("model_file", DEFAULT_MODEL_NAME))
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Could not load model: {e}")
        return None

    out_file = export_to_standalone_python(model, output_path)
    print(f"\n🚀 Model exported successfully to: {out_file}")
    print(f"   Zero external dependencies! You can run it on ANY machine with pure Python:")
    print(f"   $ python {out_file} <inputs...>\n")
    return out_file


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
    task = config.get("task", "classification")

    print("\n" + "=" * 62)
    print(" 📊 doraneural Project Status")
    print("=" * 62)
    print("\nArchitecture Flow:")
    print(render_network_diagram(inputs, hidden, outputs))

    print("\nProject Details:")
    print(f"  • Task:                 {task.upper()}")
    print(f"  • Dataset:              {config.get('dataset', 'moons')}")
    print(f"  • Total Epochs Trained: {config.get('epochs_trained', 0)}")
    if task != "regression":
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
    task = config.get("task", "auto")

    # Check if regression model
    is_regression = (task == "regression") or (len(model.layers) > 0 and isinstance(model.layers[-1], Dense))

    print("\n" + "=" * 50)
    print(f" Input Values: {values}")
    print("=" * 50)

    if is_regression:
        raw_out = model.forward(x)[0]
        val = raw_out[0] if len(raw_out) == 1 else raw_out
        print(f"  ▶ Predicted Value: {val:.4f}")
    else:
        proba = model.predict_proba(x)[0]
        pred = model.predict(x)[0]
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

