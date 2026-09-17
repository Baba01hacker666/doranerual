# 🤖 Zexo AI: The Autonomous Conversational AI Model

Welcome to the **Zexo AI** sub-project! This directory houses the standalone architecture, configuration tiers, training pipeline, and conversational interface for our custom AI, **Zexo**.

---

## 🧭 Architectural Tiers & Roadmap

Zexo is engineered across five scalable tiers, allowing seamless migration from lightweight CPU development to large-scale conversational reasoning:

| Tier | Parameters | Dimensions | Hidden Dim | Layers | Heads / KV | Vocab | Context | Target Hardware |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`micro`** | **280K** | 64 | 172 | 5 | 4 / 4 | 512 | 256 | Instant local test / CI runners |
| **`mini`** | **15.2M** | 288 | 768 | 6 | 6 / 6 | 32,000 | 512 | Fast CPU local chat |
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
│   ├── identity_conversations.txt  # Core persona & identity dialogue dataset
│   └── multi_turn_chats.txt        # Multi-turn coding & reasoning dataset
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
python zexo/train.py \
  --data zexo/data/identity_conversations.txt \
  --epochs 3 \
  --lr 5e-4
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

---

## 🔁 Continuous Cloud Training with GitHub Actions

Zexo is fully wired into `.github/workflows/train_llm.yml`. When you push new dialogues into `zexo/data/` or trigger via GitHub Actions:
1. It automatically restores the previous Zexo checkpoint from cache (`zexo/checkpoints/latest.bin`).
2. Trains Zexo on the new dataset using native OpenMP C++ acceleration.
3. Tests conversational answers before and after training.
4. Uploads the newly updated weights as an artifact (`zexo-checkpoint-<run_number>`)!
