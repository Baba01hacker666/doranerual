"""doraneural: Unified Command Line Interface (CLI).

Designed for beginners to create, train, inspect, and understand neural networks
with short, effortless commands:
  doraneural new       - Start a new model (interactive wizard or flags)
  doraneural train     - Train the current model with zero tedious flags
  doraneural status    - View ASCII architecture diagram and training metrics
  doraneural predict   - Test model on user inputs
  doraneural explain   - Plain-English deep learning concept explanations
  doraneural demo      - Run visual benchmarks (moons, blobs, digits, cnn)
  doraneural test      - Run test suite
  doraneural info      - View system and library info
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

import doraneural as dn
from .workspace import (
    init_workspace,
    train_current_workspace,
    show_workspace_status,
    predict_with_current_workspace,
)
from .teach import explain


def cmd_new(args: argparse.Namespace) -> None:
    """Initialize a new project model."""
    if args.interactive:
        print("\n🧙 doraneural Model Wizard")
        print("─" * 40)
        try:
            inp = input("How many inputs? (default: 2): ").strip()
            inputs = int(inp) if inp else 2

            hid = input("Hidden layer sizes? (e.g. 16 or 16,8) (default: 16,8): ").strip()
            hidden = [int(x.strip()) for x in hid.split(",")] if hid else [16, 8]

            out = input("How many outputs? (1 for yes/no, >1 for classes) (default: 1): ").strip()
            outputs = int(out) if out else 1

            dataset = input("Dataset? (moons, blobs, digits) (default: moons): ").strip() or "moons"
        except (ValueError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            return
    else:
        inputs = args.inputs
        hidden = args.hidden
        outputs = args.outputs
        dataset = args.dataset

    init_workspace(inputs=inputs, hidden=hidden, outputs=outputs, dataset=dataset)


def cmd_train(args: argparse.Namespace) -> None:
    """Train the active project model with effortless zero-flag command."""
    epochs = args.epochs
    lr = args.lr
    train_current_workspace(epochs=epochs, lr=lr)


def cmd_status(args: argparse.Namespace) -> None:
    """Display visual diagram and training progress of current model."""
    show_workspace_status()


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict using the current model."""
    if not args.values:
        print("Please provide input numbers to predict.")
        print("Example: doraneural predict 0.5 -1.2")
        return
    predict_with_current_workspace(args.values)


def cmd_explain(args: argparse.Namespace) -> None:
    """Plain-English machine learning teacher."""
    print(explain(args.topic))


def cmd_demo(args: argparse.Namespace) -> None:
    """Run an end-to-end demo script."""
    demo_name = args.name.lower()
    examples_dir = Path(__file__).resolve().parent.parent / "examples"

    mapping = {
        "digits": examples_dir / "digit_classification.py",
        "interactive": examples_dir / "generate_and_predict_digit.py",
        "moons": examples_dir / "binary_classification.py",
        "blobs": examples_dir / "multiclass_classification.py",
        "cnn": examples_dir / "cnn_image_classification.py",
    }

    if demo_name not in mapping:
        print(f"Unknown demo: {demo_name}. Available: {list(mapping.keys())}")
        return

    script = mapping[demo_name]
    print(f"Running demo '{demo_name}'...\n")
    os.system(f"{sys.executable} {script}")


def cmd_test(args: argparse.Namespace) -> None:
    """Run automated unit test suite."""
    test_path = Path(__file__).resolve().parent.parent / "tests" / "test_neural_lib.py"
    print("Running doraneural automated test suite...\n")
    ret = os.system(f"{sys.executable} {test_path}")
    if ret != 0:
        sys.exit(ret >> 8)


def cmd_info(args: argparse.Namespace) -> None:
    """Display doraneural system and environment information."""
    print("=" * 60)
    print(f" doraneural: Lightweight CPU Neural Network Library & CLI")
    print("=" * 60)
    print(f"Version:      {dn.__version__}")
    print(f"Python:       {sys.version.split()[0]} ({sys.platform})")
    print(f"NumPy:        {np.__version__}")
    print(f"Install Path: {Path(__file__).resolve().parent}")
    print("\nSupported Layers:")
    print("  • Dense, Conv2D, MaxPool2D, Flatten, LayerNorm, Dropout")
    print("\nSupported Activations:")
    print("  • ReLU, Sigmoid, Softmax")
    print("\nSupported Optimizers:")
    print("  • SGD (with Momentum), Adam, RMSprop")
    print("\nSupported Losses:")
    print("  • BinaryCrossEntropy, CategoricalCrossEntropy")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="doraneural",
        description="doraneural: The beginner-friendly neural network creation toolkit & CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # doraneural new
    sub_new = subparsers.add_parser("new", help="Start a new neural network model")
    sub_new.add_argument("-i", "--interactive", action="store_true", help="Interactive model creation wizard")
    sub_new.add_argument("--inputs", type=int, default=2, help="Number of inputs (default: 2)")
    sub_new.add_argument("--hidden", type=int, nargs="+", default=[16, 8], help="Hidden layer sizes (default: 16 8)")
    sub_new.add_argument("--outputs", type=int, default=1, help="Number of outputs (default: 1)")
    sub_new.add_argument("--dataset", choices=["moons", "blobs", "digits"], default="moons", help="Default dataset")
    sub_new.set_defaults(func=cmd_new)

    # doraneural train
    sub_train = subparsers.add_parser("train", help="Train the current model on the current dataset")
    sub_train.add_argument("-e", "--epochs", type=int, default=25, help="Number of epochs (default: 25)")
    sub_train.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    sub_train.set_defaults(func=cmd_train)

    # doraneural status
    sub_status = subparsers.add_parser("status", help="View current model architecture diagram & metrics")
    sub_status.set_defaults(func=cmd_status)

    # doraneural predict
    sub_pred = subparsers.add_parser("predict", help="Predict on input numbers using the current model")
    sub_pred.add_argument("values", type=float, nargs="*", help="Input numbers matching model inputs")
    sub_pred.set_defaults(func=cmd_predict)

    # doraneural explain
    sub_exp = subparsers.add_parser("explain", help="Learn deep learning concepts in plain English")
    sub_exp.add_argument("topic", nargs="?", default="help", help="Concept to explain (e.g. weights, epochs, backprop)")
    sub_exp.set_defaults(func=cmd_explain)

    # doraneural demo
    sub_demo = subparsers.add_parser("demo", help="Run interactive and visual benchmarks")
    sub_demo.add_argument("name", choices=["interactive", "digits", "moons", "blobs", "cnn"], default="interactive", nargs="?", help="Demo to run")
    sub_demo.set_defaults(func=cmd_demo)

    # doraneural test
    sub_test = subparsers.add_parser("test", help="Run automated unit test suite")
    sub_test.set_defaults(func=cmd_test)

    # doraneural info
    sub_info = subparsers.add_parser("info", help="Display environment and package info")
    sub_info.set_defaults(func=cmd_info)

    if len(sys.argv) == 1:
        # If run with no arguments, show friendly status or help
        config = Path(".doraneural.json")
        if config.exists():
            show_workspace_status()
            print("\n💡 Quick Commands:")
            print("   doraneural train       # Train current model")
            print("   doraneural status      # View model flow")
            print("   doraneural predict ... # Test on numbers")
            print("   doraneural explain ... # Learn concepts")
        else:
            parser.print_help()
            print("\n💡 Get started quickly:")
            print("   doraneural new         # Create your first neural network")
            print("   doraneural train       # Train it immediately")
            print("   doraneural explain     # Learn concepts in plain English")
        return

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
