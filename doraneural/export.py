"""Standalone zero-dependency Python code exporter.

Exports trained neural network models into a single, self-contained .py file
with weights baked in as standard Python lists. The exported script runs on
ANY standard Python 3 interpreter without requiring NumPy or any third-party packages.
"""

from pathlib import Path
from typing import Union, Optional
import numpy as np

from .model import Sequential
from .layers import Dense
from .activations import ReLU, Sigmoid, Softmax


def export_to_standalone_python(
    model: Sequential,
    output_path: Union[str, Path] = "predict.py",
) -> Path:
    """Export a trained model into a standalone, zero-dependency Python file.

    Args:
        model (Sequential): Trained Sequential model.
        output_path (Union[str, Path]): Target output file path.

    Returns:
        Path: Path to generated standalone Python script.
    """
    target = Path(output_path)
    if not str(target).endswith(".py"):
        target = Path(f"{target}.py")

    target.parent.mkdir(parents=True, exist_ok=True)

    # Inspect layers and extract parameters
    code_blocks = []
    code_blocks.append('"""Standalone Neural Network Predictor.')
    code_blocks.append('Generated automatically by doraneural.')
    code_blocks.append('Zero external dependencies: runs on pure Python Standard Library (no NumPy needed).')
    code_blocks.append('"""\n')
    code_blocks.append("import sys")
    code_blocks.append("import math\n")

    # Math helpers
    code_blocks.append("# ─────────────────────────────────────────────────────────────")
    code_blocks.append("# Pure Python Math Helpers (Zero Dependencies)")
    code_blocks.append("# ─────────────────────────────────────────────────────────────")
    code_blocks.append("def _dot_add(x, W, b):")
    code_blocks.append("    # x: 1D list, W: 2D list (in_dim, out_dim), b: 1D list (out_dim)")
    code_blocks.append("    in_dim, out_dim = len(W), len(W[0])")
    code_blocks.append("    out = [0.0] * out_dim")
    code_blocks.append("    for j in range(out_dim):")
    code_blocks.append("        val = sum(x[i] * W[i][j] for i in range(in_dim))")
    code_blocks.append("        if b is not None:")
    code_blocks.append("            val += b[j]")
    code_blocks.append("        out[j] = val")
    code_blocks.append("    return out\n")

    code_blocks.append("def _relu(x):")
    code_blocks.append("    return [max(0.0, v) for v in x]\n")

    code_blocks.append("def _sigmoid(x):")
    code_blocks.append("    return [1.0 / (1.0 + math.exp(-max(-88.0, min(88.0, v)))) for v in x]\n")

    code_blocks.append("def _softmax(x):")
    code_blocks.append("    max_v = max(x)")
    code_blocks.append("    exp_v = [math.exp(v - max_v) for v in x]")
    code_blocks.append("    sum_exp = sum(exp_v)")
    code_blocks.append("    return [v / sum_exp for v in exp_v]\n")

    # Model Weights Block
    code_blocks.append("# ─────────────────────────────────────────────────────────────")
    code_blocks.append("# Trained Model Parameters")
    code_blocks.append("# ─────────────────────────────────────────────────────────────")

    dense_idx = 0
    forward_steps = []

    for idx, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            dense_idx += 1
            w_list = layer.weights.tolist()
            b_list = layer.biases.tolist()[0] if layer.biases is not None else None

            code_blocks.append(f"W_{dense_idx} = {w_list}")
            code_blocks.append(f"b_{dense_idx} = {b_list}\n")
            forward_steps.append(f"current = _dot_add(current, W_{dense_idx}, b_{dense_idx})")
        elif isinstance(layer, ReLU):
            forward_steps.append("current = _relu(current)")
        elif isinstance(layer, Sigmoid):
            forward_steps.append("current = _sigmoid(current)")
        elif isinstance(layer, Softmax):
            forward_steps.append("current = _softmax(current)")

    # Determine model task from last layer
    last_layer = model.layers[-1] if model.layers else None
    if isinstance(last_layer, Softmax):
        task_kind = "multiclass"
    elif isinstance(last_layer, Sigmoid):
        task_kind = "binary"
    else:
        task_kind = "regression"

    # Forward Prediction Function
    code_blocks.append("# ─────────────────────────────────────────────────────────────")
    code_blocks.append("# Prediction API")
    code_blocks.append("# ─────────────────────────────────────────────────────────────")
    code_blocks.append("def predict(inputs):")
    code_blocks.append('    """Run inference on a single sample list of floats."""')
    code_blocks.append("    current = [float(v) for v in inputs]")
    for step in forward_steps:
        code_blocks.append(f"    {step}")

    if task_kind == "regression":
        code_blocks.append("    # Regression: return continuous value(s)")
        code_blocks.append("    return current[0] if len(current) == 1 else current\n")
    elif task_kind == "binary":
        code_blocks.append("    # Binary classification: return (class, probability)")
        code_blocks.append("    proba = current[0]")
        code_blocks.append("    pred = 1 if proba >= 0.5 else 0")
        code_blocks.append("    return pred, proba\n")
    else:
        code_blocks.append("    # Multiclass classification: return (class, probabilities)")
        code_blocks.append("    probas = current")
        code_blocks.append("    pred = max(range(len(probas)), key=lambda i: probas[i])")
        code_blocks.append("    return pred, probas\n")

    # Main block for CLI execution
    code_blocks.append("# ─────────────────────────────────────────────────────────────")
    code_blocks.append("# CLI Execution")
    code_blocks.append("# ─────────────────────────────────────────────────────────────")
    code_blocks.append("if __name__ == '__main__':")
    code_blocks.append("    if len(sys.argv) < 2:")
    code_blocks.append("        print('Usage: python predict.py <val1> <val2> ...')")
    code_blocks.append("        sys.exit(1)")
    code_blocks.append("    try:")
    code_blocks.append("        vals = [float(arg) for arg in sys.argv[1:]]")
    code_blocks.append("    except ValueError:")
    code_blocks.append("        print('Error: All input arguments must be numbers.')")
    code_blocks.append("        sys.exit(1)")

    if task_kind == "regression":
        code_blocks.append("    result = predict(vals)")
        code_blocks.append("    print(f'Prediction: {result}')")
    elif task_kind == "binary":
        code_blocks.append("    pred, proba = predict(vals)")
        code_blocks.append("    conf = proba if pred == 1 else (1.0 - proba)")
        code_blocks.append("    print(f'Prediction: Class {pred}')")
        code_blocks.append("    print(f'Confidence: {conf * 100:.2f}% (Probability: {proba:.4f})')")
    else:
        code_blocks.append("    pred, probas = predict(vals)")
        code_blocks.append("    print(f'Prediction: Class {pred}')")
        code_blocks.append("    print(f'Confidence: {probas[pred] * 100:.2f}%')")
        code_blocks.append("    print(f'All Probabilities: {[round(p, 4) for p in probas]}')")

    content = "\n".join(code_blocks) + "\n"

    with open(target, "w", encoding="utf-8") as f:
        f.write(content)

    return target
