# doraneural Python API Reference

`doraneural` provides both beginner one-liners and fully modular neural network components built strictly on NumPy.

---

## Table of Contents

- [Beginner Quickstart (`dn.create`, `dn.quick_train`)](#1-beginner-quickstart)
- [CSV Data Loader (`dn.load_csv`)](#2-csv-data-loader)
- [Standalone Python Exporter (`dn.export_to_standalone_python`)](#3-standalone-python-exporter)
- [ASCII Curves & Visuals (`dn.plot_history`, `dn.plot_ascii_curve`)](#4-ascii-curves--visuals)
- [Zero-Dependency Image Loader & Vision](#5-zero-dependency-image-loader--vision)
- [Sequence Modeling & Recurrent Networks (RNN, LSTM, GRU)](#6-sequence-modeling--recurrent-networks)
- [Attention & Transformer Blocks (Pure NumPy)](#7-attention--transformer-blocks)
- [Learning Rate Schedulers](#8-learning-rate-schedulers)
- [Core Modular Components](#9-core-modular-components)
  - [Layers](#layers)
  - [Activations](#activations)
  - [Loss Functions](#loss-functions)
  - [Optimizers & Regularizers](#optimizers--regularizers)
  - [Metrics](#metrics)
- [Model Serialization](#10-model-serialization)
- [Hardware Acceleration & JIT Conv2D](#11-hardware-acceleration--jit-conv2d)
- [Mixed-Precision & Memory Management (float32 vs float64)](#12-mixed-precision--memory-management)
- [Batch-Parallel DataLoader with Prefetching](#13-batch-parallel-dataloader-with-prefetching)
- [Computational Graph & Autograd Engine (`Tensor`, `requires_grad`)](#14-computational-graph--autograd-engine)
- [Static Graph Compile Step & Operator Fusion (`compile_model`)](#15-static-graph-compile-step--operator-fusion)
- [Portable Model Serialization Spec (`.dnb` v1.0)](#16-portable-model-serialization-spec-dnb)

---

## 1. Beginner Quickstart

### `dn.create`
Builds and compiles a `Sequential` network in a single line. Automatically assigns weight initializations (He/Xavier) and connects appropriate loss functions and output squashing.

```python
import doraneural as dn

# Binary Classification:
model_bin = dn.create(inputs=2, hidden=[16, 8], outputs=1, task="binary")

# Multiclass Classification:
model_multi = dn.create(inputs=4, hidden=[32, 16], outputs=3, task="multiclass")

# Continuous Regression:
model_reg = dn.create(inputs=5, hidden=[24, 12], outputs=1, task="regression")
```

### `dn.quick_train`
Inspects raw NumPy matrices, infers dimensions, creates a network, runs training, and returns `(model, history)`.

```python
import doraneural as dn

model, history = dn.quick_train(X, y, hidden=[32, 16], epochs=25)
```

---

## 2. CSV Data Loader

### `dn.load_csv`
Loads CSV files with zero external dependencies (no pandas required). Features:
- Auto-detects headers
- Converts categorical string labels to integers
- Applies standard z-score normalization ($Z = \frac{X - \mu}{\sigma}$)
- Handles missing or empty rows cleanly

```python
import doraneural as dn

X, y, meta = dn.load_csv(
    filepath="housing.csv",
    target_col="price",     # Column name or integer index (e.g. -1)
    normalize=True,         # Normalize feature columns
    has_header=True
)

print(f"Features: {meta['feature_names']}")
print(f"Target:   {meta['target_name']}")
print(f"Samples:  {meta['n_samples']}")
```

---

## 3. Standalone Python Exporter

### `dn.export_to_standalone_python`
Exports a trained model into a single `.py` script that runs on **any standard Python interpreter** without NumPy or any 3rd-party dependencies.

```python
import doraneural as dn

model = dn.create(inputs=4, hidden=[16, 8], outputs=3)
model.fit(X_train, y_train, epochs=15)

# Export standalone script
dn.export_to_standalone_python(model, output_path="predict.py")
```

You can now copy `predict.py` to microcontrollers, minimal Docker containers, or shared servers and run:
```bash
python3 predict.py 5.1 3.5 1.4 0.2
```

---

## 4. ASCII Curves & Visuals

Visualize your learning curve directly in the console:

```python
import doraneural as dn

# Plot from model History
print(dn.plot_history(history))

# Or plot any list of numbers directly
print(dn.plot_ascii_curve([1.0, 0.7, 0.4, 0.2, 0.05], title="Loss Drop"))
```

---

## 5. Zero-Dependency Image Loader & Vision

`doraneural` loads, resizes, and processes real images (`.bmp`, `.ppm`, `.pgm`) without requiring Pillow (PIL) or OpenCV:

### Load Image Folders (Cats vs Dogs)

```python
import doraneural as dn

# Scans dataset/cats and dataset/dogs, resizes to 24x24, and normalizes
X, y, class_names = dn.load_image_dataset(
    "path/to/dataset",
    target_size=(24, 24),
    grayscale=True
)
print(f"Loaded {len(X)} images of {class_names}")
```

### Render Images as Terminal ASCII Art

```python
import doraneural as dn

# Print visual grayscale ASCII representation directly in terminal
print(dn.render_image_ascii(X[0, 0], width=20, height=10))
```

### Save and Read BMP Files

```python
import doraneural as dn

# Read standard uncompressed 24-bit or 8-bit BMP
img_array = dn.read_bmp("cat.bmp")

# Write array back to BMP
dn.write_bmp("saved_image.bmp", img_array)
```

---

---

## 6. Sequence Modeling & Recurrent Networks

`doraneural` features pure NumPy sequence models with exact Backpropagation Through Time (BPTT):

### `SimpleRNN` / `RNN`
Standard Elman recurrent network layer:
$$h_t = \tanh(x_t W_{xh} + h_{t-1} W_{hh} + b_h)$$

```python
import doraneural as dn

# Output full sequence (batch, time, hidden_units)
rnn_seq = dn.SimpleRNN(in_features=8, hidden_units=16, return_sequences=True)

# Output last hidden state (batch, hidden_units)
rnn_last = dn.SimpleRNN(in_features=8, hidden_units=16, return_sequences=False)
```

### `LSTM` (Long Short-Term Memory)
Full gated LSTM cell with forget, input, cell candidate, and output gates:

```python
lstm = dn.LSTM(in_features=16, hidden_units=32, return_sequences=False)
```

### `GRU` (Gated Recurrent Unit)
Compact gated unit with reset and update gates:

```python
gru = dn.GRU(in_features=16, hidden_units=32, return_sequences=True)
```

---

## 7. Attention & Transformer Blocks (Pure NumPy)

Build and train transformers from scratch without PyTorch or JAX:

### `PositionalEncoding`
Sinusoidal positional embeddings:

```python
pe = dn.PositionalEncoding(d_model=32, max_len=500)
```

### `MultiHeadAttention`
Multi-head scaled dot-product attention with causal autoregressive masking support:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

```python
# Bidirectional encoder attention
mha = dn.MultiHeadAttention(d_model=32, num_heads=4, causal=False)

# Autoregressive causal decoder attention
mha_causal = dn.MultiHeadAttention(d_model=32, num_heads=4, causal=True)
```

### `TransformerBlock`
Pre-LN Transformer encoder block integrating `LayerNorm`, `MultiHeadAttention`, residual skip connections, `Dropout`, and a two-layer `Dense` feed-forward network:

```python
transformer = dn.TransformerBlock(
    d_model=32,
    num_heads=4,
    d_ff=64,
    dropout=0.1,
    causal=False
)
```

---

## 8. Learning Rate Schedulers

Dynamic learning rate scheduling integrated directly into `model.fit()`:

```python
opt = dn.Adam(lr=0.01)

# 1. Step decay
scheduler1 = dn.StepLR(opt, step_size=5, gamma=0.5)

# 2. Cosine Annealing
scheduler2 = dn.CosineAnnealingLR(opt, T_max=20, eta_min=1e-4)

# 3. Linear Warmup + Cosine Annealing
scheduler3 = dn.WarmupCosineLR(opt, warmup_epochs=3, total_epochs=20, eta_min=1e-4)

# Pass scheduler and gradient clipping to fit():
model.fit(
    X_train,
    y_train,
    epochs=20,
    scheduler=scheduler3,
    clip_norm=1.0  # Global gradient clipping
)
```

---

## 9. Core Modular Components

### Layers
- `Dense(in_features, out_features, weight_init="he", l1_reg=0.0, l2_reg=0.0)`: Fully connected layer. Supports 2D and 3D sequence inputs `(*, in_features)`.
- `Conv2D(in_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1, l1_reg=0.0, l2_reg=0.0)`: 2D Spatial Convolution with vectorized im2col, custom dilation, and `"same"` / `"valid"` padding.
- `MaxPool2D(pool_size=2, stride=2)`: 2D Spatial Max Pooling.
- `Flatten()`: Flattens N-D tensors into 2D batch matrices.
- `LayerNorm(normalized_shape, eps=1e-5)`: Layer Normalization across feature dimension.
- `Dropout(drop_rate=0.5)`: Inverted dropout with train/eval mode switching.
- `SimpleRNN(in_features, hidden_units, return_sequences=False)`: Elman RNN.
- `LSTM(in_features, hidden_units, return_sequences=False)`: Long Short-Term Memory.
- `GRU(in_features, hidden_units, return_sequences=False)`: Gated Recurrent Unit.
- `PositionalEncoding(d_model, max_len=5000)`: Sinusoidal embeddings.
- `MultiHeadAttention(d_model, num_heads=4, causal=False)`: Scaled dot-product MHA.
- `TransformerBlock(d_model, num_heads=4, d_ff=64, dropout=0.0)`: Transformer Block.

### Activations
- `ReLU()`: Rectified Linear Unit ($\max(0, x)$).
- `Sigmoid()`: Numerically stable logistic sigmoid ($[1 + e^{-x}]^{-1}$).
- `Softmax()`: Numerically stable exponent-shifted Softmax.

### Loss Functions
- `BinaryCrossEntropy()`: Binary classification loss.
- `CategoricalCrossEntropy()`: Multi-class cross entropy.
- `MeanSquaredError()` (alias `MSELoss`): Continuous regression loss.

### Optimizers & Regularizers
- `SGD(lr=0.01, momentum=0.9, weight_decay=0.0)`: Stochastic Gradient Descent with velocity momentum and decoupled weight decay.
- `Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)`: Adaptive Moment Estimation with bias correction and decoupled AdamW weight decay.
- `RMSprop(lr=0.001, alpha=0.9, eps=1e-8)`: Root Mean Square Propagation.
- `clip_grad_norm(layers_or_grads, max_norm)`: Global L2 gradient norm clipping.
- `clip_grad_value(layers_or_grads, clip_value)`: Element-wise gradient threshold clipping.

### Metrics
- `Accuracy()`: Binary and multiclass classification accuracy.
- `MSE()`: Mean Squared Error.
- `MAE()`: Mean Absolute Error.

---

## 10. Model Serialization

Save and reload architectures and weights:

```python
import doraneural as dn

# Save JSON architecture + compressed NPZ weights (supports Dense, Conv2D, RNN, LSTM, GRU, Transformers)
dn.save_model(model, "my_model")

# Reload fully restored model
loaded_model = dn.load_model("my_model")
```

---

## 11. Hardware Acceleration & JIT Conv2D

`doraneural` features an optional multi-threaded Numba JIT compiler accelerator for `im2col` spatial convolutions (delivering **8–10x faster CPU execution**) with transparent fallback to pure NumPy:

```python
import doraneural as dn

# Check whether Numba is installed
print("Numba available:", dn.is_numba_available())

# Check or change active convolution backend ('auto', 'numba', or 'numpy')
print("Current backend:", dn.get_im2col_backend())

# Explicitly force pure NumPy backend:
dn.set_im2col_backend("numpy")

# Explicitly force multi-threaded Numba JIT backend:
dn.set_im2col_backend("numba")

# Auto mode (uses Numba if installed, falls back to NumPy):
dn.set_im2col_backend("auto")
```

---

## 12. Mixed-Precision & Memory Management

Toggle between `float32` (single precision) and `float64` (double precision) dynamically:
- **`float32`**: 50% smaller memory footprint, higher CPU SIMD vectorized throughput.
- **`float64`**: Maximum numerical stability for stiff physical systems and gradient checks.

### Inspect Model RAM Consumption (`memory_summary`)
```python
import doraneural as dn

model = dn.create(inputs=128, hidden=[256, 128], outputs=10)

# Profile parameters and byte footprint
info = model.memory_summary()
print(info["formatted"])
# Example output: "67,210 parameters | 525.08 KB (float32) [50% smaller than float64]"
```

### Dynamic In-Place Precision Casting
```python
# Cast all weights, biases, gradients, and optimizer moments to float64
model.to_precision("float64")

# Cast back to float32 to cut memory by 50%
model.to_precision("float32")
```

### Global Precision & Context Scopes
```python
import doraneural as dn

# Set global default precision
dn.set_precision("float32")

# Temporarily run in double precision
with dn.precision_scope("float64"):
    high_precision_model = dn.create(inputs=4, hidden=[16], outputs=1)
```

---

## 13. Batch-Parallel DataLoader with Prefetching

Zero heavy dependencies (built entirely on Python's standard library `threading` and `queue`):

```python
import doraneural as dn

# Create thread-based batch-parallel loader with background prefetching
loader = dn.DataLoader(
    (X_train, y_train),
    batch_size=32,
    shuffle=True,
    prefetch_factor=2,  # Buffers 2 batches ahead on background thread
    drop_last=False
)

# 1. Manual iteration with zero batch-slicing compute delay:
for X_batch, y_batch in loader:
    preds = model.forward(X_batch)

# 2. Or pass directly into model.fit():
model.fit(loader, epochs=10)
```

---

## 14. Computational Graph & Autograd Engine

`doraneural` includes a reverse-mode automatic differentiation engine generalized to multi-dimensional NumPy arrays (`Tensor`), inspired by Karpathy's micrograd and PyTorch.

### Features
- **Dynamic Computation Graph (DAG)**: Tracks tensor operations, child nodes, and backward gradient closures automatically.
- **Broadcast Gradient Unbroadcasting**: Correctly reduces gradients when shapes differ due to NumPy broadcasting rules across arbitrary dimensions.
- **Operator Overloading**: `+`, `-`, `*`, `/`, `@` (matmul), `**`, `.relu()`, `.sigmoid()`, `.tanh()`, `.exp()`, `.log()`, `.sum()`, `.mean()`, `.reshape()`, `.transpose()`, `__getitem__` slicing.
- **Inference Context Manager**: `with dn.no_grad():` disables graph tracking for maximum speed and zero graph overhead.

### Autograd Example

```python
import doraneural as dn
import numpy as np

# Define tensors with gradient tracking
x = dn.Tensor([-2.0, 1.0, 3.0], requires_grad=True)
w = dn.Tensor([0.5, -1.5, 2.0], requires_grad=True)
b = dn.Tensor(0.25, requires_grad=True)

# Build dynamic computation tape
z = x * w + b
y = z.relu()
loss = y.sum()

# Automatic backpropagation
loss.backward()

print("dL/dx:", x.grad)
print("dL/dw:", w.grad)
print("dL/db:", b.grad)
```

### Modular Neural Modules with Autograd

```python
import doraneural as dn
import numpy as np

# Built-in linear module
fc = dn.Linear(in_features=4, out_features=1)

X = dn.Tensor(np.random.randn(16, 4))
y = dn.Tensor(np.random.randn(16, 1))

# Training step
preds = fc(X)
loss = dn.mse_loss(preds, y)
loss.backward()

# Parameter updates
for p in fc.parameters():
    p.data -= 0.01 * p.grad
```

---

## 15. Static Graph Compile Step & Operator Fusion

The ahead-of-time (AOT) static graph compiler optimizes inference by analyzing layer topology and fusing consecutive operators.

### Key Capabilities
- **Operator Fusion**: Fuses `Dense + Bias + Activation` (ReLU / Sigmoid) into unified in-place memory kernels.
- **Static Buffer Arena (`StaticBufferPool`)**: Precomputes all intermediate shapes and preallocates contiguous memory arenas. Zero dynamic heap memory allocations during inference loops.
- **Model Compilation API**: Call `dn.compile_model(model, sample_input)` or `model.compile_graph(sample_input)`.

### Compilation Example

```python
import doraneural as dn
import numpy as np

model = dn.Sequential([
    dn.Dense(in_features=64, out_features=128),
    dn.ReLU(),
    dn.Dense(in_features=128, out_features=10),
    dn.Softmax(),
])

sample = np.random.randn(32, 64).astype(np.float32)

# Compile static graph
compiled = model.compile_graph(sample_input=sample)

# Print execution plan summary
print(compiled.summary())

# Compare execution speed & latency
results = compiled.benchmark(sample, iterations=1000)
print(f"Speedup: {results['speedup']:.2f}x")
print(f"Pre-allocated Scratch RAM: {results['scratch_bytes']} bytes")
```

---

## 16. Portable Model Serialization Spec (`.dnb` v1.0)

A custom, versioned, architecture-neutral binary specification (**Doraneural Binary v1.0**) avoiding Python `pickle` security and portability pitfalls.

### Binary Layout Spec
- **32-Byte Fixed Header**:
  - `MAGIC` (`b"DNB\x01"`)
  - Format version `0x00010000` (v1.0)
  - Flags, metadata length, metadata offset, tensor count
- **Metadata Section**: UTF-8 JSON describing layer sequence, hyperparameters, layer types, and tensor offsets.
- **Tensor Storage Payload**:
  - 8-byte aligned raw binary float arrays for direct zero-copy memory mapping.
  - Per-tensor CRC32 checksums for guaranteed corruption and bit-rot detection.

### Python Usage

```python
import doraneural as dn

# 1. Save model to .dnb
dn.save_dnb(model, "my_model.dnb")
# Or via model method:
model.save_dnb("my_model.dnb")

# 2. Inspect file header and tensor table without full loading
info = dn.inspect_dnb("my_model.dnb")
print("Model type:", info["model_type"])
print("Parameter count:", info["total_parameters"])
print("Tensors:", info["tensors_count"])

# 3. Fast load with CRC32 verification
loaded_model = dn.load_dnb("my_model.dnb")
predictions = loaded_model.forward(X_test)
```

