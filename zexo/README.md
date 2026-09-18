# 🤖 Zexo AI: The Autonomous Conversational AI Model

Welcome to the **Zexo AI** sub-project! This directory houses the standalone architecture, configuration tiers, training pipeline, and conversational interface for our custom AI, **Zexo**.

---

## 🧭 Architectural Tiers & Roadmap

Zexo is engineered across five scalable tiers, allowing seamless migration from lightweight CPU development to large-scale conversational reasoning:

| Tier | Parameters | Dimensions | Hidden Dim | Layers | Heads / KV | Vocab | Context | Target Hardware |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`micro`** | **280K** | 64 | 172 | 5 | 4 / 4 | 512 | 256 | Instant local test / CI runners |
| **`mini`** | **6.8M** | 192 | 1024 | 1 | 2 / 1 | 32,000 | 1,024 | Fast CPU local chat |
| **`chat`** | **25.3M** | 384 | 1024 | 8 | 8 / 4 (GQA) | 32,000 | 1,024 | Multi-turn conversational dialogue |
| **`base`** | **109.5M** | 768 | 2048 | 12 | 12 / 12 | 32,000 | 2,048 | Full reasoning & deep knowledge |
| **`large`** | **221.5M** | 1024 | 2816 | 16 | 16 / 8 (GQA) | 32,000 | 4,096 | Scaled long-context conversational AI |

---

## 📂 Project Structure

```
zexo/
├── README.md               # You are here! Complete documentation & roadmap
├── config.py               # ZexoConfig: Architectural tiers and hyperparameter specs
├── persona.py              # System prompt persona, dialogue formatting & stop tokens
├── model.py                # ZexoModel: High-level conversational wrapper
├── train.py                # Dedicated conversational training runner
├── chat.py                 # Interactive terminal chat REPL
├── init_weights.py         # Base checkpoint initializer (.bin)
├── data/
│   ├── quality_dialogues.jsonl     # Curated, validated gold conversations
│   ├── zexo_quality_v1_train.txt  # Built training corpus (gold + technical data)
│   ├── zexo_quality_v1_eval.txt   # Held-out evaluation corpus (never trained)
│   ├── dataset_tools.py            # Validator, deduplicator, splitter, renderer
│   ├── assemble_dataset.py         # Legacy technical corpus builder
│   ├── identity_conversations.txt  # Small legacy persona set
│   └── multi_turn_chats.txt        # Small legacy multi-turn set
└── checkpoints/
    └── .gitkeep            # Directory for trained Zexo weights (.bin) & metadata
```

---

## 🚀 Quick Start: Chatting & Training

### 1. Chat with Zexo in Terminal:
```bash
python zexo/chat.py
```
- Type your message and hit **Enter**.
- Commands available inside chat:
  - `/clear`: Reset conversation context.
  - `/stats`: Display context window token usage.
  - `/quit`: Exit.

### 2. Fine-Tune Zexo on Custom Dialogues:
```bash
python -m zexo.data.dataset_tools
python zexo/train.py \
  --data zexo/data/zexo_quality_v1_train.txt \
  --eval-data zexo/data/zexo_quality_v1_eval.txt \
  --tier micro \
  --epochs 5 \
  --lr 5e-4 \
  --seq-len 64
```

### 3. Fine-Tune Zexo on Hugging Face Datasets:
```bash
# Auto-downloads and trains Zexo directly:
python zexo/train.py \
  --hf-dataset roneneldan/TinyStories \
  --hf-samples 20 \
  --epochs 2
```

### 4. Initialize Fresh Base Weights for Any Tier:
```bash
# Create a fresh 25.3M parameter Zexo-Chat base checkpoint:
python zexo/init_weights.py --tier chat
```

## 🧪 Quality Dataset Workflow

The quality corpus uses JSONL records with explicit `user` and `assistant` roles. Each record is validated, deduplicated, balanced across categories, and split deterministically so evaluation examples never enter training:

```bash
python -m zexo.data.dataset_tools
```

The builder creates:

- `zexo_quality_v1_train.txt`: curated conversations plus the existing technical corpus.
- `zexo_quality_v1_eval.txt`: held-out conversations covering behavior, safety, coding, ML, math, and systems.
- `zexo_quality_v1_manifest.json`: hashes, category counts, and split provenance.

When adding data, prefer short, correct examples that demonstrate the desired behavior: ask clarifying questions when requirements are incomplete, state uncertainty, show safe alternatives, and avoid claiming tools or facts that were not verified. Keep evaluation prompts out of training data.

The trainer supports shuffled windows, overlapping windows via `--stride`, held-out loss, deterministic seeds, and JSONL input. Training loss is a signal—not a substitute for held-out behavioral tests.

The default trainer remains the fast head-only path. For real CPU NumPy backpropagation through every decoder layer, use `--full-backprop`; this updates embeddings, attention projections, RoPE-connected attention, RMSNorm scales, SwiGLU projections, and the output head. It is intentionally slower and is best suited to micro models, smoke tests, and small fine-tuning runs. Training metadata records which mode was used.

The inference engine now uses the optimized C++ path automatically when available. Set `DORANEURAL_NUM_THREADS` or call `llm.set_num_threads(n)` to tune native parallelism. Tokenizers use heap-based merges and bounded caches, and HuggingFace single-file, BF16, and sharded safetensors checkpoints are supported without PyTorch.

```bash
# Full transformer backpropagation (CPU/NumPy reference trainer)
python zexo/train.py \
  --tier micro \
  --data zexo/data/zexo_quality_v1_train.txt \
  --eval-data zexo/data/zexo_quality_v1_eval.txt \
  --full-backprop \
  --epochs 1 \
  --seq-len 64
```

---

## 🔁 Continuous Cloud Training with GitHub Actions

Zexo is fully wired into `.github/workflows/train_llm.yml`. When you push new dialogues into `zexo/data/` or trigger via GitHub Actions:
1. It automatically restores the previous Zexo checkpoint from cache (`zexo/checkpoints/latest.bin`).
2. Trains Zexo on the new dataset using native OpenMP C++ acceleration.
3. Tests conversational answers before and after training.
4. Uploads the newly updated weights as an artifact (`zexo-checkpoint-<run_number>`)!
