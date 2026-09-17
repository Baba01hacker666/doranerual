# neural_lib

A minimalist, modular Python neural network library built purely with **NumPy** and the Python Standard Library. Designed specifically for education, hackability, and running on **low-power CPU hardware** (such as Raspberry Pi, Chromebooks, Android/Termux, or embedded micro-servers) without needing heavy frameworks like PyTorch, TensorFlow, JAX, or scikit-learn.

---

## Table of Contents

- [Project Purpose and Non-Goals](#project-purpose-and-non-goals)
- [Library Structure Overview](#library-structure-overview)
- [Installation](#installation)
- [Quickstart Snippet](#quickstart-snippet)
- [How Backpropagation Works Here](#how-backpropagation-works-here)
- [Supported Features](#supported-features)
- [Full Training Examples](#full-training-examples)
  - [1. Binary Classification (Moons Dataset)](#1-binary-classification-moons-dataset)
  - [2. Multi-class Classification (Blobs Dataset)](#2-multi-class-classification-blobs-dataset)
  - [3. 2D Convolutional Network (Visual Patterns)](#3-2d-convolutional-network-visual-patterns)
  - [4. 0 to 9 Digit Classifier](#4-0-to-9-handwrittenprinted-digit-classifier)
- [Model Persistence (Save & Load)](#model-persistence-save--load)
- [Performance & Low-Power Constraints](#performance--low-power-constraints)
- [Running the Test Suite](#running-the-test-suite)
- [Known Limitations](#known-limitations)

---

## Project Purpose and Non-Goals

### Purpose
- **Educational Clarity**: Clear, readable Python code where every layer, matrix multiplication, derivative, and update step is accessible in plain NumPy.
- **Ultra-Lightweight & CPU-Centric**: Zero gigabyte dependencies. Runs immediately on constrained CPUs with minimal RAM footprint.
- **Hackability**: Easily inspect, print, or modify weights, gradients, activations, and loss functions in minutes.

### Non-Goals
- **Not a replacement for PyTorch / JAX / TensorFlow**: No CUDA / GPU acceleration kernels, no distributed clusters, and no dynamic automatic differentiation graph engines.
- **Not for massive LLMs or computer vision transformers**: Focused on educational MLPs, CNNs, and classifiers.

---

## Library Structure Overview

```text
neural_lib/
├── __init__.py          # Public API exports (Sequential, Dense, Conv2D, Adam, etc.)
├── base.py              # Layer abstract interface contract (forward / backward / params / train / eval)
├── layers.py            # Dense, Dropout, LayerNorm, Flatten, Conv2D, MaxPool2D
├── activations.py       # ReLU, Sigmoid, and Softmax activation layers
├── losses.py            # BinaryCrossEntropy and CategoricalCrossEntropy
├── optimizers.py        # SGD (with Momentum), Adam, RMSprop
├── metrics.py           # Accuracy metric for binary and multiclass tasks
├── model.py             # Sequential container model and History logging
├── utils.py             # Seed management, dataset splitting, one-hot, synthetic generators
└── serialization.py     # JSON architecture and NumPy .npz weight persistence
examples/
├── binary_classification.py      # End-to-end 2D Moons demo
├── multiclass_classification.py  # End-to-end 3-class Blobs demo
└── cnn_image_classification.py   # End-to-end 2D CNN pattern recognition demo
tests/
└── test_neural_lib.py            # Numerical gradient checks, stability, & unit tests
requirements.txt                  # NumPy only (numpy>=1.20.0)
setup.py                          # Standard packaging
README.md                         # Comprehensive documentation
```

---

## Supported Features

- **Layers**:
  - `Dense(in_features, out_features, weight_init="he", use_bias=True)`
  - `Dropout(drop_rate=0.5)`
  - `LayerNorm(normalized_shape, eps=1e-5)`
  - `Flatten()`
  - `Conv2D(in_channels, out_channels, kernel_size, stride, padding)`
  - `MaxPool2D(pool_size, stride)`
- **Activations**:
  - `ReLU()`
  - `Sigmoid()`
  - `Softmax(axis=-1)`
- **Losses**:
  - `BinaryCrossEntropy(eps=1e-7)`
  - `CategoricalCrossEntropy(eps=1e-7)`
- **Optimizers**:
  - `SGD(lr=0.01, momentum=0.9, clip_norm=None)`
  - `Adam(lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0)`
  - `RMSprop(lr=0.001, alpha=0.99, momentum=0.0)`

---

## Installation

The only prerequisite is Python 3.8+ and NumPy.

### Option A: Using `uv` (Recommended)
```bash
# Create a clean virtual environment
uv venv .venv
source .venv/bin/activate

# Install requirements
uv pip install -r requirements.txt
```

### Option B: Standard `python -m venv`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart Snippet

```python
import numpy as np
from neural_lib import Sequential, Dense, ReLU, Sigmoid
from neural_lib.losses import BinaryCrossEntropy
from neural_lib.optimizers import SGD

# 1. Build a simple multi-layer perceptron
model = Sequential([
    Dense(in_features=2, out_features=16, weight_init="he"),
    ReLU(),
    Dense(in_features=16, out_features=1, weight_init="xavier"),
    Sigmoid(),
])

# 2. Compile model
model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=SGD(lr=0.1, momentum=0.9),
    metrics=["accuracy"],
)

# 3. Fit on your dataset
# X: (N, 2), y: (N, 1)
X = np.random.randn(200, 2).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32).reshape(-1, 1)

history = model.fit(X, y, epochs=20, batch_size=16, verbose=1)

# 4. Evaluate & Predict
loss, acc = model.evaluate(X, y)
predictions = model.predict(X)
print(f"Accuracy: {acc * 100:.1f}%")
```

---

## How Backpropagation Works Here

In this library, backpropagation is implemented without complicated computational graph abstractions. Instead, it follows a simple **chain of message passing** through the layer stack in reverse.

```text
[Forward Pass]
  Input X  ───▶  Dense ───▶  ReLU ───▶  Dense ───▶  Sigmoid ───▶ Loss L
                   │           │          │            │
                 Cache X     Cache Z    Cache H      Cache P
                   │           │          │            │
[Backward Pass]    ▼           ▼          ▼            ▼
  dL/dX   ◀─── dL/dW,b ◀── dL/dZ  ◀── dL/dH,W,b ◀── dL/dP  ◀── dL/dy_pred
```

### Step-by-Step Chain Rule Flow

1. **The Loss Layer (`losses.py`)**:
   - The loss function compares model predictions $\hat{y}$ with true labels $y$.
   - It calculates the scalar error and computes the gradient:
     $$\frac{\partial L}{\partial \hat{y}}$$
     *In plain English: "If the output probability $\hat{y}$ increases slightly, how much does the loss go up or down?"*
   - This gradient tensor is passed backward to the final layer.

2. **The Output Activation Layer (e.g. `Sigmoid` or `Softmax`)**:
   - The activation layer receives the gradient $\frac{\partial L}{\partial \hat{y}}$.
   - It calculates its local derivative:
     $$\text{for Sigmoid: } \frac{d\hat{y}}{dz} = \hat{y} \cdot (1 - \hat{y})$$
   - By the chain rule, multiplying incoming error by local derivative yields $\frac{\partial L}{\partial z}$:
     $$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{d\hat{y}}{dz}$$
   - This signal $\frac{\partial L}{\partial z}$ tells upstream layers how pre-activation logits contributed to the final error.

3. **The Dense Layer (`layers.py`)**:
   - A Dense layer computes $Z = X W + b$.
   - When receiving incoming gradient $G = \frac{\partial L}{\partial Z}$ from the next layer, it computes three things:
     1. **Weight Gradient ($\frac{\partial L}{\partial W}$)**:
        $$\frac{\partial L}{\partial W} = X^T \cdot G$$
        *(Each input neuron is correlated with each output error to see how strongly that connection caused the error).*
     2. **Bias Gradient ($\frac{\partial L}{\partial b}$)**:
        $$\frac{\partial L}{\partial b} = \sum_{\text{batch samples}} G$$
        *(The overall offset error across all batch samples).*
     3. **Input Gradient ($\frac{\partial L}{\partial X}$)**:
        $$\frac{\partial L}{\partial X} = G \cdot W^T$$
        *(This is returned from `backward()` and passed down as the error signal to preceding layers).*

4. **Hidden Activations (e.g. `ReLU`)**:
   - ReLU simply checks which inputs were active ($x > 0$) during the forward pass.
   - If $x > 0$, the gradient passes right through unchanged.
   - If $x \le 0$, the gradient is set to 0.

5. **The Optimizer (`optimizers.py`)**:
   - Once all layers have computed their parameter gradients (`dweights`, `dbiases`), the optimizer executes a step:
     $$\text{weights} \leftarrow \text{weights} - \text{learning\_rate} \times \text{dweights}$$
   - With momentum enabled, past update directions are smoothed into a velocity buffer, accelerating convergence through valleys and damping oscillations.

---

## Full Training Examples

### 1. Binary Classification (Moons Dataset)

Runs a 2-layer MLP on non-linearly separable interleaving moon shapes:

```bash
python examples/binary_classification.py
```

**Key Highlights**:
- Synthetic data generated purely with NumPy (`make_moons`).
- Reaches **> 98% accuracy** in under 50 epochs on CPU.
- Performs inference on test coordinates.
- Validates save and load roundtrip.

### 2. Multi-class Classification (Blobs Dataset)

Demonstrates 3-class classification with `Softmax` and `CategoricalCrossEntropy`:

```bash
python examples/multiclass_classification.py
```

### 3. 2D Convolutional Network (Visual Patterns)

Demonstrates spatial 2D convolutions, max pooling, dropout, and Adam optimization:

```bash
python examples/cnn_image_classification.py
```

**Key Highlights**:
- Vectorized `im2col` convolution forward and backward operations.
- `Conv2D` -> `ReLU` -> `MaxPool2D` -> `Flatten` -> `Dropout` -> `Dense` -> `Softmax`.
- Trains cleanly on CPU with `Adam` in a few seconds.

### 4. 0 to 9 Handwritten/Printed Digit Classifier

Full 10-class digit recognition on synthetic 8x8 pixel grids with ASCII art terminal rendering:

```bash
python examples/digit_classification.py
```

**Key Highlights**:
- 1,500 samples across digits 0-9 generated with pure NumPy (`make_digits`).
- Multi-layer MLP: 64 inputs -> 48 hidden -> 24 hidden -> 10 classes with Softmax.
- Reaches **> 99% accuracy** in ~2 seconds on CPU.
- Displays actual digit samples rendered in ASCII art in the terminal alongside prediction confidence.

---

## Model Persistence (Save & Load)

`neural_lib` provides clean, transparent persistence without Python `pickle`:
- **Architecture**: Saved as human-readable JSON (layer names, shapes, activations).
- **Weights**: Saved as standard compressed NumPy `.npz` archive.

```python
from neural_lib import load_model

# Save model
model.save("saved_models/my_classifier")
# Creates:
#   - saved_models/my_classifier.json
#   - saved_models/my_classifier.npz

# Load model anywhere without training code
restored_model = load_model("saved_models/my_classifier")
restored_predictions = restored_model.predict(X_test)
```

---

## Performance & Low-Power Constraints

- **CPU Efficient**: Hot loops are fully vectorized into BLAS matrix products (`x @ W`, `x.T @ G`).
- **In-Place Updates**: The optimizer modifies parameter and velocity arrays in-place (`param -= step`), eliminating continuous memory reallocations during training.
- **Float32 Precision**: Arrays default to `np.float32`, cutting memory bandwidth and RAM consumption in half compared to Python's default float64.
- **Zero Heavy C++ Dependencies**: No LLVM, CUDA, or 2GB wheel installations needed.

---

## Running the Test Suite

A complete unit test suite with finite-difference numerical gradient checks is included:

```bash
python -m unittest discover -s tests
```

Tests include:
- Dense layer forward/backward tensor dimensions.
- Finite-difference gradient checks comparing analytical backprop to numerical slopes.
- Sigmoid & Softmax extreme value numerical stability (preventing NaN / Inf).
- Loss calculation and metric edge cases.
- Input dimension validation and user-friendly error messages.
- Save/load serialization fidelity.

---

## Known Limitations

1. **Architecture Scope**: Supports sequential feed-forward stacks (`Sequential`). Branching (residual/skip connections) or multi-input graphs are not supported in v1.
2. **CPU Only**: Designed for learning and small-to-medium datasets. Not suited for multi-million parameter models.
3. **Layer Types**: v1 focuses on Dense (MLP) layers, ReLU, Sigmoid, and Softmax.

---

## Next Improvements

If you wish to expand `neural_lib` for future projects, recommended additions are:
- **Optimizers**: Adam and RMSprop adaptive learning rate optimizers.
- **Regularization**: Dropout layer and L2 weight decay.
- **Normalization**: Batch Normalization and Layer Normalization.
- **Convolution**: 1D and 2D Convolution layers (`Conv2D`, `MaxPool2D`) for image processing.
- **Mini-Autograd**: A tape-based or node-based scalar/tensor autograd engine (similar to micrograd) for arbitrary compute graphs.
