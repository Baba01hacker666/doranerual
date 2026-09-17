# doraneural CLI Reference Guide

The `doraneural` (and alias `doranerual`) CLI is designed so beginners never need to remember dozens of complex flags.

---

## Command Overview

| Command | Description | Example |
|---|---|---|
| `doraneural` | Shows current project status and next logical steps | `doraneural` |
| `doraneural new` | Initialize a new model in current folder | `doraneural new --task binary` |
| `doraneural train` | Train the model with auto ASCII curve display | `doraneural train -e 25` |
| `doraneural plot` | View cumulative learning curve in terminal ASCII | `doraneural plot` |
| `doraneural export` | Export model to zero-dependency standalone Python file | `doraneural export -o predict.py` |
| `doraneural status` | Inspect ASCII network diagram & metrics | `doraneural status` |
| `doraneural predict` | Pass values directly to active model | `doraneural predict 0.5 -1.2` |
| `doraneural explain` | Plain-English ML teacher with analogies | `doraneural explain backprop` |
| `doraneural demo` | Run visual interactive demos & benchmarks | `doraneural demo interactive` |
| `doraneural test` | Run automated unit test suite (22 tests) | `doraneural test` |
| `doraneural info` | Display environment, NumPy, and engine info | `doraneural info` |

---

## 1. `doraneural new`

Initializes a workspace configuration file (`.doraneural.json`) and starter model weights (`current_model.json` / `.npz`).

### Options
- `-i`, `--interactive`: Launch the step-by-step interactive wizard.
- `--inputs <N>`: Number of input features per sample (default: `2`).
- `--hidden <N...>`: Hidden layer node counts (default: `16 8`).
- `--outputs <N>`: Output units (default: `1`).
- `--task <type>`: `auto`, `binary`, `multiclass`, or `regression` (default: `auto`).
- `--dataset <name>`: Built-in dataset generator (`moons`, `blobs`, `digits`).
- `--data <file.csv>`: Automatically configure inputs and tasks from a CSV dataset.

### Examples
```bash
# Interactive setup
doraneural new -i

# Binary classification for 2 features
doraneural new --inputs 2 --hidden 16 8 --outputs 1

# Regression model (e.g. predicting continuous price from 4 features)
doraneural new --inputs 4 --hidden 32 16 --outputs 1 --task regression

# Auto-configure directly from a CSV file
doraneural new --data dataset.csv
```

---

## 2. `doraneural train`

Trains the active workspace model on its dataset.

### Options
- `-e`, `--epochs <N>`: Number of training epochs (default: `25`).
- `--lr <float>`: Learning rate (default: `0.01`).
- `--data <file.csv>`: Train on a specific custom CSV file.

### Examples
```bash
# Zero-flag training (uses project defaults)
doraneural train

# Train for 50 epochs with custom learning rate
doraneural train -e 50 --lr 0.005

# Train on a custom CSV file
doraneural train --data sales.csv -e 30
```

---

## 3. `doraneural plot`

Renders high-contrast ASCII graphs of your training loss and validation accuracy directly inside your terminal.

```bash
doraneural plot
```

Output:
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
│         Epoch 1                             Epoch 25 │
└────────────────────────────────────────────────────┘
```

---

## 4. `doraneural export`

Bakes model weights into a clean, zero-dependency Python script. The resulting script runs on **pure vanilla Python 3** without needing NumPy or any package installations.

```bash
doraneural export -o my_predictor.py
```

Run inference with standard Python:
```bash
python3 my_predictor.py 0.45 -0.82
```

---

## 5. `doraneural predict`

Sends numbers directly into the active model and displays predicted classes or continuous values.

```bash
doraneural predict 1.2 0.35
```

---

## 6. `doraneural explain`

Machine learning concepts explained in plain English without confusing mathematical jargon.

```bash
doraneural explain weights
doraneural explain backprop
doraneural explain epochs
doraneural explain lr
doraneural explain loss
```

---

## 7. `doraneural story`

Autoregressively generates text / stories using a pretrained LLaMA model from Hugging Face Hub directly on pure CPU:

```bash
# Generate with default stories260K model
doraneural story --prompt "Once upon a time, in a magical forest" --tokens 80 --temp 0.7

# Generate with 10M-parameter Tiny-LLM
doraneural story --model arnir0/Tiny-LLM --prompt "According to all known laws of aviation" --tokens 60
```

---

## 8. `doraneural chat`

Starts an interactive terminal chat REPL with conversational memory, lookahead stop-sequence detection, and automatic sliding context window management:

```bash
# Launch interactive chat (defaults to arnir0/Tiny-LLM on CPU)
doraneural chat

# Chat with specific parameters or smaller model
doraneural chat --model stories260K --tokens 50 --temp 0.6 --system "You are a concise tutor."
```

### Available Chat Slash Commands:
- `/clear`, `/reset` — Clears conversation history and model KV cache
- `/context`, `/info` — Displays token usage bar and memory statistics
- `/temp <float>` — Dynamically adjusts sampling temperature (e.g. `/temp 0.5`)
- `/tokens <int>` — Sets maximum new tokens per response turn (e.g. `/tokens 80`)
- `/system <text>` — Updates assistant system prompt / persona
- `/stats` — Displays total turns and token generation throughput
- `/exit`, `/quit` — Exits chat session

---

## 9. `doraneural finetune`

Fine-tunes a pretrained LLaMA model on custom text with cross-entropy loss and native AdamW updates:

```bash
# Fine-tune base model on a text corpus:
doraneural finetune --data my_data.txt --epochs 5 --lr 0.0005 --output my_finetuned.bin
```

