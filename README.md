# doraneural

An intuitive, beginner-friendly neural network toolkit and CLI built purely with **NumPy** for **low-power CPU hardware** (Termux, Raspberry Pi, Chromebooks, micro-servers).

`doraneural` strips away the complexity of heavyweight frameworks like PyTorch and TensorFlow. It hides tedious math behind clean one-liners, displays real-time ASCII training curves, and exports models to zero-dependency standalone scripts.

---

## ⚡ Quick Installation

```bash
git clone https://github.com/Baba01hacker666/doranerual.git
cd doranerual
pip install -e .
```

*Note: Both `doraneural` and `doranerual` work interchangeably as CLI commands and Python packages.*

---

## 🚀 30-Second CLI Quickstart

You can create, train, and test a complete neural network in 3 simple commands:

```bash
# 1. Create a model (2 inputs -> 16,8 hidden -> 1 output)
doraneural new --inputs 2 --hidden 16 8 --outputs 1

# 2. Train it (Loss curves and accuracy plot right in your terminal!)
doraneural train -e 15

# 3. Predict on new numbers
doraneural predict 0.45 -0.82
```

### Visual ASCII Training Curves

No matplotlib or graphical desktop needed—`doraneural` draws loss drops directly in the console:

```text
┌────────────────────────────────────────────────────┐
│ 📈 Loss Progress (Initial: 0.2854 ──▶ Final: 0.0210) │
├────────────────────────────────────────────────────┤
│  0.285 ┤███                                        │
│  0.220 ┤   ████                                    │
│  0.155 ┤       ████                                │
│  0.090 ┤           ██████                          │
│  0.021 ┤                 █████████████████████████ │
│         └───────────────────────────────────────── │
│         Epoch 1                             Epoch 15 │
└────────────────────────────────────────────────────┘
```

---

## 🐍 30-Second Python Quickstart

### One-Liner Model Creation (`dn.create`)

```python
import doraneural as dn

# Auto-configures layers, initializations, activations, and losses:
model = dn.create(inputs=4, hidden=[16, 8], outputs=3, task="multiclass")
model.fit(X_train, y_train, epochs=20)

# Evaluate and predict
loss, acc = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
```

Supports `task="binary"`, `task="multiclass"`, and `task="regression"`.

---

## ✨ Key Features

- **Pretrained Hugging Face LLM (Pure NumPy LLaMA)**: Runs real pretrained transformer language models directly from Hugging Face Hub (`karpathy/tinyllamas`) in pure NumPy with RoPE, RMSNorm, KV-Cache, SwiGLU FFN, and SentencePiece BPE tokenizer, generating 70–100+ tokens/sec on CPU:
  ```bash
  doraneural story --prompt "Once upon a time, a brave puppy"
  ```
- **Computational Graph & Autograd Engine**: Reverse-mode automatic differentiation DAG on multi-dimensional NumPy arrays (`Tensor`) with automatic broadcast gradient reduction, operator overloading, and PyTorch-like dynamic tape execution.
- **Static Graph Compiler & Operator Fusion**: Ahead-of-Time (AOT) static graph compilation (`model.compile_graph()`) with fused kernels (`Dense + Bias + ReLU`) and preallocated contiguous scratch memory arenas (`StaticBufferPool`) eliminating dynamic heap allocations during inference.
- **Versioned Doraneural Binary Spec (`.dnb` v1.0)**: Architecture-neutral, versioned binary format with a 32-byte fixed header, UTF-8 JSON architecture metadata, 8-byte aligned raw tensor payloads, and CRC32 checksums for guaranteed corruption detection—completely free of Python `pickle`.
- **Multi-Threaded JIT Conv2D Acceleration**: Multi-threaded Numba JIT `im2col` convolution kernel delivering **8–10x faster execution on CPU**, with transparent fallback to pure NumPy.
- **Mixed-Precision & 50% Memory Savings**: Dynamic `float32` vs `float64` toggling (`model.to_precision("float32")`) cuts RAM consumption by exactly 50% on low-power devices with higher SIMD throughput.
- **Batch-Parallel Threaded DataLoader**: Overlaps data slicing, memory formatting, and augmentation on background threads with zero heavy dependencies (`threading` + `queue`).
- **Pure NumPy Sequence & Recurrent Models**: Full analytical BPTT support for `SimpleRNN`, `LSTM`, and `GRU` sequence modeling.
- **Pure NumPy Attention & Transformers**: Pre-LN `TransformerBlock`, `MultiHeadAttention` (with causal autoregressive masking), and `PositionalEncoding`—all in pure NumPy without PyTorch or JAX.
- **LR Schedulers & Gradient Clipping**: Dynamic `StepLR`, `CosineAnnealingLR`, `WarmupCosineLR`, along with `clip_grad_norm`, `clip_grad_value`, and decoupled weight decay.
- **Zero-Dependency Image Loader & Cat vs Dog Vision**: Load `.bmp`, `.ppm`, and `.pgm` images without Pillow or OpenCV. Run real CPU computer vision with `Conv2D` (stride, dilation, padding) and `MaxPool2D`:
  ```bash
  doraneural demo catdog
  ```
- **Zero-Dependency CSV Loader**: Load and auto-normalize tabular datasets with pure standard library (no pandas needed):
  ```bash
  doraneural new --data my_dataset.csv
  doraneural train -e 25
  ```
- **Standalone Zero-Dependency Exporter**: Bake your trained model into a single 30-line pure Python file that runs on **any computer without NumPy**:
  ```bash
  doraneural export -o predict.py
  python3 predict.py 0.5 -0.2
  ```
- **Plain-English Machine Learning Teacher**:
  ```bash
  doraneural explain backprop
  doraneural explain lr
  doraneural explain weights
  ```
- **Human-Readable Error Messages**: Friendly explanations showing *what happened* and *how to fix it* if shapes don't match.

---

## 📋 CLI Cheat Sheet

| Command | What it does |
|---|---|
| `doraneural` | Shows workspace status & quick action recommendations |
| `doraneural story` | Generates stories with a pretrained LLaMA LLM from Hugging Face |
| `doraneural new` | Creates a new model (`-i` interactive wizard, `--data <file.csv>`) |
| `doraneural train` | Trains active model on dataset (`-e` epochs, `--lr` rate) |
| `doraneural plot` | Displays terminal ASCII training curves for loss and accuracy |
| `doraneural export` | Exports model to a zero-dependency standalone `.py` script |
| `doraneural status` | Displays ASCII architecture flow and training stats |
| `doraneural predict <vals>` | Runs immediate inference on user input numbers |
| `doraneural explain <topic>`| Explains concepts (`weights`, `epochs`, `loss`, `backprop`) |
| `doraneural demo <name>` | Runs demos (`story`, `catdog`, `interactive`, `digits`, `moons`, `blobs`, `cnn`) |
| `doraneural test` | Runs the automated 67-test unit test suite |
| `doraneural info` | Displays system specs, NumPy version, and engine info |

---

## 📚 Documentation

For complete deep dives and advanced configurations, see the `docs/` folder:

- 📖 **[CLI Reference Guide](docs/cli_reference.md)**: Full wizard walkthroughs, custom dataset options, and parameter flags.
- 🧠 **[Python API Reference](docs/python_api.md)**: Pretrained LLaMA LLM, `Tensor` autograd engine, static graph compiler, `.dnb` binary spec, `Dense`, `Conv2D`, `LSTM`, `GRU`, `TransformerBlock`, `MultiHeadAttention`, JIT acceleration, mixed-precision, schedulers, image loading, custom losses, optimizers, and data preprocessing.
- 📐 **[Math & Backpropagation Under the Hood](docs/math_and_backprop.md)**: Numerical gradient checks, Jacobians, weight initialization, and backprop math.
- 🎮 **[Interactive Demos & Visualizer](docs/demos.md)**: Real-time 0–9 digit synthesizer, Cat vs Dog CNN vision benchmark, and ASCII previewer.

---

## 🧪 Running Tests

```bash
doraneural test
# or
python3 -m unittest discover tests/
```
All 67 unit tests pass on pure CPU with standard NumPy.

