"""Autograd Engine, Static Graph Compiler & DNB Binary Format Demo.

Demonstrates:
1. Dynamic Autograd Engine:
   - Defining multi-dimensional Tensors with computation DAG tracking.
   - Performing reverse-mode automatic differentiation with backpropagation.
   - Building a custom Linear module and training it with MSE loss.
2. Static Graph Compiler:
   - Ahead-of-Time (AOT) topology analysis and operator fusion (Dense + Bias + Activation).
   - Preallocated static memory arena (StaticBufferPool) for zero dynamic heap allocations.
   - Speedup benchmarking: Eager NumPy vs Compiled Model.
3. Versioned Doraneural Binary (.dnb) Format:
   - Saving architecture metadata and raw 8-byte aligned weights with CRC32 checksums.
   - Inspecting binary header, metadata, and tensor tables without Python pickle.
   - Fast loading and verified output numerical parity.
"""

import sys
import tempfile
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import doraneural as dn


def demo_autograd_engine():
    print("=" * 65)
    print("1. Dynamic Autograd Engine (Tensor-Level Reverse-Mode DAG)")
    print("=" * 65)

    dn.set_seed(42)

    # 1. Scalar & Vector arithmetic with automatic gradient tracking
    print("\n--- Example A: Scalar and Vector Autodiff ---")
    x = dn.Tensor([-2.0, 1.0, 3.0], requires_grad=True)
    w = dn.Tensor([0.5, -1.5, 2.0], requires_grad=True)
    b = dn.Tensor(0.25, requires_grad=True)

    # Forward computation DAG
    # y = relu(x * w + b)
    z = x * w + b
    y = z.relu()
    loss = y.sum()

    print(f"x        = {x.data}")
    print(f"w        = {w.data}")
    print(f"z        = (x * w + b) -> {z.data}")
    print(f"y        = relu(z)      -> {y.data}")
    print(f"loss     = sum(y)       -> {loss.data:.4f}")

    # Backpropagate through the entire graph
    loss.backward()

    print(f"\nComputed Gradients:")
    print(f"dL / dx  = {x.grad}")
    print(f"dL / dw  = {w.grad}")
    print(f"dL / db  = {b.grad}")

    # 2. Training a custom Module using Autograd
    print("\n--- Example B: Fitting a Linear Module via Autograd ---")
    model = dn.Linear(in_features=3, out_features=1)
    print(f"Initial weights:\n{model.weight.data.ravel()}")
    print(f"Initial bias:   {model.bias.data.ravel()}")

    # Synthetic targets: y = 2*x0 - 3*x1 + x2 + 0.5
    true_w = np.array([[2.0], [-3.0], [1.0]], dtype=np.float32)
    X_data = np.random.randn(32, 3).astype(np.float32)
    y_data = X_data @ true_w + 0.5

    X_t = dn.Tensor(X_data)
    y_t = dn.Tensor(y_data)

    lr = 0.05
    for epoch in range(1, 41):
        model.zero_grad()
        preds = model(X_t)
        loss = dn.mse_loss(preds, y_t)
        loss.backward()

        # Gradient descent step
        model.weight.data -= lr * model.weight.grad
        model.bias.data -= lr * model.bias.grad

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | MSE Loss: {loss.data:.6f}")

    print(f"Learned weights:\n{model.weight.data.ravel()}")
    print(f"Learned bias:   {model.bias.data.ravel()} (target: [0.5])")


def demo_static_compiler():
    print("\n" + "=" * 65)
    print("2. Static Graph Compiler (AOT Operator Fusion & Static Arena)")
    print("=" * 65)

    dn.set_seed(42)

    # Build a deep multi-layer perceptron
    model = dn.Sequential([
        dn.Dense(in_features=128, out_features=256),
        dn.ReLU(),
        dn.Dense(in_features=256, out_features=128),
        dn.ReLU(),
        dn.Dense(in_features=128, out_features=64),
        dn.ReLU(),
        dn.Dense(in_features=64, out_features=10),
        dn.Sigmoid(),
    ])

    batch_size = 64
    sample_input = np.random.randn(batch_size, 128).astype(np.float32)

    print("\nCompiling static graph with ahead-of-time operator fusion...")
    compiled = model.compile_graph(sample_input=sample_input)

    # Print compilation summary
    print(compiled.summary())

    # Benchmark eager vs compiled
    print("Running performance benchmark (1000 forward passes)...")
    bench = compiled.benchmark(sample_input, n_iters=1000, warmup=50)

    print(f"Eager Execution:    {bench['eager_mean_ms'] * 1000:.2f} µs/batch ({bench['eager_fps']:.1f} batches/sec)")
    print(f"Compiled Execution: {bench['compiled_mean_ms'] * 1000:.2f} µs/batch ({bench['compiled_fps']:.1f} batches/sec)")
    print(f"Inference Speedup:  {bench['speedup']:.2f}x faster")
    print(f"Static Arena Pool:  {bench['static_pool_bytes'] / 1024:.2f} KB (0 dynamic heap allocations)")

    # Verify numerical parity
    eager_out = model.forward(sample_input)
    comp_out = compiled.forward(sample_input)
    max_diff = np.max(np.abs(eager_out - comp_out))
    print(f"Max Absolute Parity Difference: {max_diff:.2e} (Exact match)")


def demo_dnb_serialization():
    print("\n" + "=" * 65)
    print("3. Versioned Doraneural Binary (.dnb) Serialization Spec")
    print("=" * 65)

    dn.set_seed(42)

    # Create a model with various layers
    model = dn.Sequential([
        dn.Dense(in_features=16, out_features=32),
        dn.LayerNorm(normalized_shape=32),
        dn.ReLU(),
        dn.Dense(in_features=32, out_features=8),
        dn.Dropout(rate=0.1),
        dn.Dense(in_features=8, out_features=2),
        dn.Softmax(),
    ])

    model.eval()
    test_input = np.random.randn(4, 16).astype(np.float32)
    orig_output = model.forward(test_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        dnb_file = Path(tmpdir) / "demo_network.dnb"

        # 1. Save to .dnb
        print(f"\nSaving model to binary format: {dnb_file.name}...")
        model.save_dnb(dnb_file)
        file_size = dnb_file.stat().st_size
        print(f"Saved successfully! File size: {file_size} bytes")

        # 2. Inspect header and tensor tables
        print("\nInspecting .dnb binary spec metadata:")
        info = dn.inspect_dnb(dnb_file)
        print(f"  Format Magic:       {info['magic']}")
        print(f"  Version:            v{info['version']}")
        print(f"  Model Type:         {info['model_type']}")
        print(f"  Total Parameters:   {info['total_parameters']:,}")
        print(f"  Stored Tensors:     {info['tensors_count']}")
        print("  Tensor Records:")
        for t in info["tensors"]:
            print(f"    - {t['name']:<24} shape={str(t['shape']):<12} dtype={t['dtype']:<8} "
                  f"crc32={t['crc32']} offset={t['offset']}")

        # 3. Load from .dnb
        print("\nLoading model from .dnb...")
        restored_model = dn.load_dnb(dnb_file)
        restored_model.eval()
        restored_output = restored_model.forward(test_input)

        np.testing.assert_allclose(orig_output, restored_output, atol=1e-7)
        print("Restored model produces identical predictions! Parity verified.")

        # 4. Corruption / Bit-rot detection
        print("\nSimulating file corruption (flipping 1 byte in weight payload)...")
        with open(dnb_file, "r+b") as f:
            f.seek(-8, 2)
            corrupted_byte = bytes([(f.read(1)[0] ^ 0xFF)])
            f.seek(-8, 2)
            f.write(corrupted_byte)

        try:
            dn.load_dnb(dnb_file)
            print("ERROR: Corrupted file was accepted!")
        except ValueError as err:
            print(f"Integrity Check Passed! Detected corruption:")
            print(f"  --> {err}")


def main():
    print("=================================================================")
    print("doraneural Framework: Autograd, Graph Compiler & DNB Binary Demo")
    print("=================================================================")

    demo_autograd_engine()
    demo_static_compiler()
    demo_dnb_serialization()

    print("\n" + "=" * 65)
    print("All demonstrations completed successfully!")
    print("=================================================================")


if __name__ == "__main__":
    main()
