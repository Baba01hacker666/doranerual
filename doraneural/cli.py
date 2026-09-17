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
import time
from pathlib import Path
import numpy as np

import doraneural as dn
from .workspace import (
    init_workspace,
    train_current_workspace,
    show_workspace_status,
    predict_with_current_workspace,
    plot_current_workspace,
    export_current_workspace,
)
from .teach import explain


def cmd_new(args: argparse.Namespace) -> None:
    """Initialize a new project model."""
    if args.interactive:
        print("\n🧙 doraneural Model Wizard")
        print("─" * 40)
        try:
            csv_in = input("Load from custom CSV file? (press Enter to skip): ").strip()
            if csv_in:
                init_workspace(data_path=csv_in)
                return

            task = input("Task type? (binary, multiclass, regression) (default: binary): ").strip() or "binary"
            inp = input("How many inputs? (default: 2): ").strip()
            inputs = int(inp) if inp else 2

            hid = input("Hidden layer sizes? (e.g. 16 or 16,8) (default: 16,8): ").strip()
            hidden = [int(x.strip()) for x in hid.split(",")] if hid else [16, 8]

            out = input("How many outputs? (default: 1): ").strip()
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
        task = args.task
        csv_in = args.data

    init_workspace(
        inputs=inputs,
        hidden=hidden,
        outputs=outputs,
        dataset=dataset,
        task=task,
        data_path=csv_in,
    )


def cmd_train(args: argparse.Namespace) -> None:
    """Train the active project model with effortless zero-flag command."""
    epochs = args.epochs
    lr = args.lr
    data_path = args.data
    train_current_workspace(epochs=epochs, lr=lr, data_path=data_path)


def cmd_plot(args: argparse.Namespace) -> None:
    """Display terminal ASCII learning curve for loss and accuracy."""
    plot_current_workspace()


def cmd_export(args: argparse.Namespace) -> None:
    """Export current model to a standalone, zero-dependency Python script."""
    export_current_workspace(output_path=args.output)


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
        "catdog": examples_dir / "cat_vs_dog_classification.py",
        "story": examples_dir / "run_huggingface_llm.py",
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


def cmd_story(args: argparse.Namespace) -> None:
    """Generate stories using pure NumPy LLaMA inference."""
    from .llm import load_pretrained_llm

    print("=" * 65)
    print(f"📖 doraneural LLM: {args.model} (Pure NumPy Pretrained LLaMA)")
    print("=" * 65)
    print("Loading pretrained weights and SentencePiece tokenizer from Hugging Face...")
    t0 = time.perf_counter()
    llm = load_pretrained_llm(model_name=args.model)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s! ({llm.config.dim} dim, {llm.config.n_layers} layers, {llm.config.vocab_size} vocab)")
    print(f"\nPrompt: \"{args.prompt}\"")
    print("-" * 65)
    sys.stdout.write(args.prompt)
    sys.stdout.flush()

    t_gen_start = time.perf_counter()
    count = 0
    for piece in llm.generate(prompt=args.prompt, max_tokens=args.tokens, temperature=args.temp, stream=True):
        sys.stdout.write(piece)
        sys.stdout.flush()
        count += 1

    gen_time = time.perf_counter() - t_gen_start
    speed = count / max(1e-4, gen_time)
    print("\n" + "-" * 65)
    print(f"Generated {count} tokens in {gen_time:.2f}s ({speed:.1f} tokens/sec on CPU)")
    print("=" * 65)


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
    print("\nSupported Layers & Blocks:")
    print("  • Dense, Conv2D, MaxPool2D, Flatten, LayerNorm, Dropout")
    print("  • SimpleRNN, LSTM, GRU, MultiHeadAttention, PositionalEncoding, TransformerBlock")
    print("\nSupported Activations:")
    print("  • ReLU, Sigmoid, Softmax")
    print("\nSupported Optimizers & Schedulers:")
    print("  • SGD, Adam, RMSprop (Weight Decay, L1/L2 Regularization)")
    print("  • StepLR, CosineAnnealingLR, WarmupCosineLR, Gradient Clipping")
    print("\nSupported Losses & Metrics:")
    print("  • BinaryCrossEntropy, CategoricalCrossEntropy, MeanSquaredError (MSE)")
    print("  • Accuracy, MSE, MAE")
    print("\nFramework & Engine Capabilities:")
    print("  • Autograd Engine: Dynamic reverse-mode autodiff DAG (dn.Tensor)")
    print("  • Static Graph Compiler: AOT operator fusion & static arena pool (0 GC allocs)")
    print("  • Versioned DNB Format: Portable binary spec (.dnb) with CRC32 integrity")
    print("  • Hardware JIT: Multi-threaded Numba im2col with pure NumPy fallback")
    print("  • Mixed Precision: float32/float64 toggling for 50% memory savings")
    print("  • Parallel DataLoader: Threaded batch prefetching with zero compute delay")
    print("  • Zero-dependency CSV loader & BMP image loader")
    print("  • Standalone zero-dependency Python code exporter")
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
    sub_new.add_argument("--task", choices=["auto", "binary", "multiclass", "regression"], default="auto", help="Task type")
    sub_new.add_argument("--data", type=str, default=None, help="Path to custom CSV dataset")
    sub_new.set_defaults(func=cmd_new)

    # doraneural train
    sub_train = subparsers.add_parser("train", help="Train the current model on the dataset")
    sub_train.add_argument("-e", "--epochs", type=int, default=25, help="Number of epochs (default: 25)")
    sub_train.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    sub_train.add_argument("--data", type=str, default=None, help="Path to custom CSV dataset")
    sub_train.set_defaults(func=cmd_train)

    # doraneural plot
    sub_plot = subparsers.add_parser("plot", help="Display visual ASCII learning curve in terminal")
    sub_plot.set_defaults(func=cmd_plot)

    # doraneural export
    sub_export = subparsers.add_parser("export", help="Export model to standalone zero-dependency Python file")
    sub_export.add_argument("-o", "--output", type=str, default="predict.py", help="Output .py file path (default: predict.py)")
    sub_export.set_defaults(func=cmd_export)

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

    # doraneural story
    sub_story = subparsers.add_parser("story", help="Generate stories using a pretrained LLaMA LLM in pure NumPy")
    sub_story.add_argument("--prompt", "-p", type=str, default="Once upon a time", help="Starting prompt for the story")
    sub_story.add_argument("--tokens", "-n", type=int, default=100, help="Number of tokens to generate (default: 100)")
    sub_story.add_argument("--temp", "-t", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    sub_story.add_argument("--model", "-m", choices=["stories260K", "stories15M"], default="stories260K", help="Pretrained model (default: stories260K)")
    sub_story.set_defaults(func=cmd_story)

    # doraneural demo
    sub_demo = subparsers.add_parser("demo", help="Run interactive and visual benchmarks")
    sub_demo.add_argument("name", choices=["story", "catdog", "interactive", "digits", "moons", "blobs", "cnn"], default="story", nargs="?", help="Demo to run")
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
            print("   doraneural plot        # Show ASCII training graph")
            print("   doraneural export      # Export to standalone Python script")
            print("   doraneural status      # View model flow")
            print("   doraneural predict ... # Test on numbers")
            print("   doraneural explain ... # Learn concepts")
        else:
            parser.print_help()
            print("\n💡 Get started quickly:")
            print("   doraneural new         # Create your first neural network")
            print("   doraneural train       # Train it immediately")
            print("   doraneural plot        # View ASCII training curve")
            print("   doraneural export      # Export standalone zero-dependency Python file")
            print("   doraneural explain     # Learn concepts in plain English")
        return

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
