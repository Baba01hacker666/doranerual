# doraneural

An intuitive, beginner-friendly neural network creation toolkit and CLI built purely with **NumPy** for **low-power CPU hardware** (Raspberry Pi, Chromebooks, Android/Termux, micro-servers).

`doraneural` strips away the complexity of heavyweight frameworks like PyTorch and TensorFlow. It hides the tedious math behind clean one-liners, provides clear visual feedback, teaches core deep learning concepts in plain English, and produces friendly, actionable error messages when things go wrong.

---

## Table of Contents

- [Why doraneural?](#why-doraneural)
- [Installation](#installation)
- [CLI Quickstart: No Complex Commands](#cli-quickstart-no-complex-commands)
  - [1. Start a New Model](#1-start-a-new-model-doraneural-new)
  - [2. Train with Zero Flags](#2-train-with-zero-flags-doraneural-train)
  - [3. View Architecture & Progress](#3-view-architecture--progress-doraneural-status)
  - [4. Predict on Inputs](#4-predict-on-inputs-doraneural-predict)
  - [5. Learn Machine Learning in Plain English](#5-learn-machine-learning-in-plain-english-doraneural-explain)
- [Python Library API: One-Liner Creation](#python-library-api-one-liner-creation)
- [Beginner-Friendly Error System](#beginner-friendly-error-system)
- [Full Command Reference](#full-command-reference)
- [Interactive Visualizer & Demos](#interactive-visualizer--demos)
- [Under the Hood: How Backprop Works Here](#under-the-hood-how-backprop-works-here)

---

## Why doraneural?

- **Hides the Math, Teaches the Concepts**: Create layers and train networks without deriving Jacobians or writing backpropagation loops manually.
- **Short, Effortless Commands**: Train your active model with just `doraneural train`—no 10-flag commands needed.
- **Human-Readable Error Logs**: No cryptic dimension errors. If a shape doesn't match, `doraneural` explains *what happened* and *how to fix it*.
- **Pure NumPy / CPU Native**: Runs immediately on minimal hardware with zero GPU drivers or gigabyte wheels.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Baba01hacker666/doranerual.git
cd doranerual

# Using venv or uv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Once installed, both `doraneural` and `doranerual` are available directly as CLI commands in your terminal.

---

## CLI Quickstart: No Complex Commands

### 1. Start a New Model: `doraneural new`

Create a new network using an interactive wizard or quick arguments:

```bash
# Interactive setup: asks inputs, hidden layers, outputs, and dataset
doraneural new -i

# Or quick setup: 2 inputs -> [16, 8] hidden -> 1 output (binary classifier)
doraneural new --inputs 2 --hidden 16 8 --outputs 1 --dataset moons
```

### 2. Train with Zero Flags: `doraneural train`

Train the active model on your dataset with a single short command:

```bash
doraneural train
```

```text
Training current model on 'moons' dataset for 25 epochs...
Epoch   1/25 - loss: 0.2854 - accuracy: 0.8662 - val_loss: 0.3279 - val_accuracy: 0.8200
...
Epoch  25/25 - loss: 0.0210 - accuracy: 0.9950 - val_loss: 0.0310 - val_accuracy: 0.9900

──────────────────────────────────────────────────────────
✨ Training complete!
   Accuracy: 99.00%  |  Loss: 0.0310
   Total Epochs Trained: 25
──────────────────────────────────────────────────────────

💡 Learning Tip:
   Loss measures mistakes (lower is better). Accuracy is % correct.
   Type 'doraneural status' to inspect your network anytime!
```

### 3. View Architecture & Progress: `doraneural status`

Inspect your network's layers, parameter counts, and accuracy anytime:

```bash
doraneural status
```

```text
==============================================================
 📊 doraneural Project Status
==============================================================

Architecture Flow:
  ┌───────┐ ──▶ ┌──────────┐ ──▶ ┌──────────┐ ──▶ ┌────────┐
  │ Input │ ──▶ │ Hidden 1 │ ──▶ │ Hidden 2 │ ──▶ │ Output │
  │ (2)   │ ──▶ │ (16)     │ ──▶ │ (8)      │ ──▶ │ (1)    │
  └───────┘ ──▶ └──────────┘ ──▶ └──────────┘ ──▶ └────────┘

Project Details:
  • Dataset:              moons
  • Total Epochs Trained: 25
  • Best Test Accuracy:   99.00%
  • Latest Loss:          0.0310
  • Saved Files:          current_model.json / .npz
==============================================================
```

### 4. Predict on Inputs: `doraneural predict`

Pass values directly to the active model and receive predictions and confidence scores:

```bash
doraneural predict 0.5 -0.2
```

```text
==================================================
 Input Values: [0.5, -0.2]
==================================================
  ▶ Prediction: Class 1 (Positive/Yes)
  ▶ Confidence: 98.45%
==================================================
```

### 5. Learn Machine Learning in Plain English: `doraneural explain`

An interactive teacher explaining deep learning concepts with analogies:

```bash
doraneural explain backprop
doraneural explain lr
doraneural explain weights
```

```text
┌──────────────────────────────────────────────────────────────┐
│ 📖 Topic: Backpropagation (Chain Rule)                       │
├──────────────────────────────────────────────────────────────┤
│ 📌 Quick Summary:                                            │
│    The mechanism that traces errors backward to find out     │
│    which weights caused the mistake.                         │
│                                                              │
│ 🔍 Details:                                                  │
│    1. Forward Pass: Data moves left to right.                │
│    2. Error Calculation: Guess compared to ground truth.     │
│    3. Backward Pass: Error travels in reverse. Each weight   │
│       gets a 'gradient' showing how to fix it.               │
│                                                              │
│ 💡 Analogy:                                                  │
│    Like a detective working backward from the crime scene to │
│    identify who contributed to the outcome.                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Python Library API: One-Liner Creation

For writing Python scripts, `doraneural` provides a beginner one-liner API alongside standard modular layers.

### One-Liner Model Creation (`dn.create`)

You don't need to manually configure activations, loss functions, or weight initializations:

```python
import doraneural as dn

# Automatically connects Dense layers, adds ReLUs, configures Softmax & CrossEntropy
model = dn.create(inputs=4, hidden=[16, 8], outputs=3)

# Train in one line
history = model.fit(X_train, y_train, epochs=20)

# Evaluate and predict
loss, acc = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
```

### Data-to-Model Auto Training (`dn.quick_train`)

Pass raw NumPy arrays, and `doraneural` will infer input features, count classes, split train/test subsets, build the architecture, and train it:

```python
import doraneural as dn

model, history = dn.quick_train(X, y, hidden=[32, 16], epochs=25)
```

---

## Beginner-Friendly Error System

Instead of cryptic matrix multiplication traces, `doraneural` catches shape errors and provides actionable advice:

```text
┌────────────────────────────────────────────────────────────┐
│ ❌ Input Shape Mismatch in Dense                           │
├────────────────────────────────────────────────────────────┤
│ 💡 What happened:                                          │
│    Layer expected inputs with 4 features,                  │
│    but received input data with shape (32, 2).             │
│                                                            │
│ 🛠️  How to fix it:                                          │
│    Adjust the layer's in_features to match your data, or   │
│    ensure your input has 4 columns.                        │
└────────────────────────────────────────────────────────────┘
```

---

## Full Command Reference

| Command | Description |
|---|---|
| `doraneural` | Shows current project status and quick suggestions |
| `doraneural new` | Creates a new model project (`-i` for interactive wizard) |
| `doraneural train` | Trains active model on current dataset without flags |
| `doraneural status` | Prints visual ASCII architecture diagram and training stats |
| `doraneural predict <vals...>` | Feeds numbers to the current model and prints prediction |
| `doraneural explain <topic>` | Plain-English explanations (`weights`, `epochs`, `lr`, `loss`, `backprop`) |
| `doraneural demo <name>` | Runs demos: `digits`, `interactive`, `moons`, `blobs`, `cnn` |
| `doraneural test` | Runs the automated 14-test suite |
| `doraneural info` | Displays environment specs and supported layers |

---

## Interactive Visualizer & Demos

Launch the interactive 0–9 digit playground with real-time ASCII art and noise tuning:

```bash
doraneural demo interactive
```

```text
Visual Comparison (Clean vs Noised):
  [ Clean Base Digit ]              [ Noised Input (σ=0.28) ]
  ┌────────────────┐          ┌────────────────┐
  │                │          │▒▒▒▒▒▒··    ····│
  │      ████████  │          │      ██████··  │
  │      ██        │          │      ██    ····│
  │      ██████    │          │      ██████    │
  │            ██  │          │··    ··    ██  │
  │      ██████    │          │      ██████  ··│
  │                │          │  ▒▒▒▒      ··██│
  │                │          │  ··      ····▒▒│
  └────────────────┘          └────────────────┘

Network Prediction on Noised Input:
  ▶ Predicted Digit: 5
  ▶ Confidence:      77.98%
```

---

## Under the Hood: How Backprop Works Here

Even though the high-level API hides the math, all underlying components follow clean, modular object-oriented contracts:

1. **Forward Pass**:
   - Dense layers compute $Z = X W + b$ using vectorized BLAS matrix multiplications.
   - Activations apply non-linear maps (e.g. $\text{ReLU}(z) = \max(0, z)$).
2. **Backward Pass**:
   - The loss layer computes $\frac{\partial L}{\partial \hat{y}}$ and sends it upstream.
   - Activations apply local derivatives (e.g. ReLU passes gradient where $x > 0$).
   - Dense layers compute parameter gradients ($X^T \cdot G$) and propagate input gradients ($G \cdot W^T$).
3. **Parameter Updates**:
   - `Adam`, `SGD` (with Momentum), or `RMSprop` update weights in-place with zero memory allocation churn.
