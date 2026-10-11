# 🤖 Zexo: A Trainable LLaMA-Style Language Model

Welcome to the **Zexo** sub-project! This directory houses the standalone model architecture, configuration tiers, staged training pipeline, and conversational interface. The tier names describe model architectures; they do not imply pretrained weights or autonomous reasoning.

---

## 🧭 Architectural Tiers & Roadmap

Zexo defines five architecture presets for experiments ranging from tiny CPU tests to larger pretraining runs. Parameter counts describe model size only; conversational quality depends on the checkpoint, tokenizer, and training data:

| Tier | Parameters | Dimensions | Hidden Dim | Layers | Heads / KV | Vocab | Context | Target Hardware |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`micro`** | **280K** | 64 | 172 | 5 | 4 / 4 | 512 | 256 | Instant local test / CI runners |
| **`mini`** | **6.8M** | 192 | 1024 | 1 | 2 / 1 | 32,000 | 1,024 | CPU smoke tests / small adaptation |
| **`chat`** | **25.3M** | 384 | 1024 | 8 | 8 / 4 (GQA) | 32,000 | 1,024 | Small-corpus SFT experiments |
| **`base`** | **109.5M** | 768 | 2048 | 12 | 12 / 12 | 32,000 | 2,048 | Larger pretraining experiment |
| **`large`** | **221.5M** | 1024 | 2816 | 16 | 16 / 8 (GQA) | 32,000 | 4,096 | Long-context scale-up; GPU recommended |

These rows are **architecture sizes**, not claims that trained weights are shipped.
Zexo-Chat has about 25M parameters and Zexo-Base about 110M; LLaMA-1's smallest
public model was about 7B parameters. The included text is only a small
smoke-test/SFT corpus, so neither this repository nor one short CPU run can produce
LLaMA-1-level knowledge or fluency. The realistic first milestone is a small model
that learns clean next-token continuation from a much larger permitted corpus,
then follows a modest set of high-quality instructions.

### Vocabulary-Free Byte Tiers

Research Paper 15 measured the embedding ("vocabulary") tax on the ladder above and
shipped parallel **byte-level** tiers that keep the transformer body *identical* while
using a 258-token vocabulary (raw UTF-8 bytes + BOS/EOS):

| Tier | Parameters | Body vs BPE tier | Embedding share |
| :--- | ---: | :--- | ---: |
| `micro-byte` | 264,256 | identical | 6.2% |
| `mini-byte` | **750,528** | identical to `mini` | 6.6% |
| `chat-byte` | **13,081,728** | identical to `chat` | 0.8% |
| `base-byte` | 85,152,000 | identical to `base` | 0.2% |

```bash
# Byte tiers need no tokenizer file and need a larger --seq-len (1 token per byte)
python zexo/train.py --tier chat-byte --data zexo/data/zexo_quality_v1_train.txt \
  --epochs 5 --seq-len 1024
```

`zexo-mini` spends **89.8%** of its parameters on its 32k embedding matrix; `mini-byte`
holds the same 700,992 transformer parameters in 750K total. Measured throughput on
an AVX2 CPU: `mini-byte` is **37.8x** faster and `chat-byte` **1.81x** faster than their
BPE twins, because single-token decoding is bound by streaming the embedding matrices.

See [`research/15_llm_efficiency_audit.md`](../research/15_llm_efficiency_audit.md).

---

## Experimental DeepSeek-Flash-inspired reference architecture

I reviewed the [DeepSeek-V4.1-Flash paper](https://arxiv.org/html/2609.19969v1). It describes a 552B multimodal MoE model with a 20/20 causal encoder/decoder split, SWA plus Compressed Sparse Attention 2 (CSA2), cross-layer global KV/index reuse, MoE, and specialized training/deployment systems. Reproducing its scale or kernels is not realistic in this NumPy/CPU repository. I added `zexo/flash_lite.py` as a separate, small full-backprop reference—not a DeepSeek reproduction or a source of pretrained weights.

Its default six-layer layout demonstrates the feasible architectural ideas:

- Causal encoder/decoder split: decoder global KV is projected from the encoder boundary state.
- Local sliding-window attention at every layer; the first two encoder layers are local-only.
- CSA2-inspired `Full → Reindex → Reuse` decoder groups. Reuse layers share the preceding global KV and selected indices; the small reference omits the paper's hierarchical sparse indexer and KV compression.
- A shared SwiGLU expert plus top-k routed SwiGLU experts, with an auxiliary router-balancing loss.
- Byte-level UTF-8 tokens and full-transformer backpropagation. The reference uses NumPy autograd and recomputes the prefix during generation, so it is intended for small experiments, not high-throughput serving.

The reference does **not** implement the paper's multimodal vision stack, 552B-scale distributed MoE, Engram, single-pass mHC, FP4/FP8 cache kernels, bounded-replay serving, or DSpark. It uses a separate `.npz` checkpoint format; do not pass it to the existing `.bin` Zexo chat CLI.

```bash
# Raw-text pretraining; use substantial, licensed corpora for useful language quality.
python zexo/train_flash.py \
  --data /path/to/general_text_train.txt \
  --eval-data /path/to/general_text_eval.txt \
  --output zexo/checkpoints/flash_lite.npz \
  --epochs 1 --seq-len 128

# Random initial weights will not chat fluently; this is an inference smoke test.
python zexo/chat_flash.py --checkpoint zexo/checkpoints/flash_lite.npz

# Optional: apply reviewed DPO preferences in the FlashLite format.
python zexo/refine_flash.py \
  --checkpoint zexo/checkpoints/flash_lite.npz \
  --preferences preferences/train.jsonl \
  --eval-preferences preferences/eval.jsonl \
  --output zexo/checkpoints/flash_lite_refined.npz
```

---

## 🧪 Data Packing

The SFT trainer packs **all dialogues into one EOS-delimited token stream** and windows
it once, instead of rounding every dialogue up to a whole number of windows. With the
default `--seq-len 256` this cuts the shipped quality corpus from 172 windows/epoch to
**113 (-34% forward passes)** and raises supervised-token density from 55.8% to 84.4%.

Pass `--no-pack` to restore the old per-dialogue windowing.

---

## 📂 Project Structure

```
zexo/
├── __init__.py             # Installable Zexo package
├── README.md               # You are here! Complete documentation & roadmap
├── config.py               # ZexoConfig: Architectural tiers and hyperparameter specs
├── persona.py              # System prompt persona, dialogue formatting & stop tokens
├── model.py                # ZexoModel: High-level conversational wrapper
├── train.py                # Base pretraining and conversational SFT runner
├── preferences.py          # Offline DPO loss, preference JSONL, validation selection
├── refine.py               # Preference-refinement CLI (DPO)
├── flash_lite.py           # Small CED/SWA/CSA2-lite/MoE reference architecture
├── train_flash.py          # FlashLite full-backprop text pretraining CLI
├── refine_flash.py         # Full-model FlashLite offline DPO CLI
├── chat_flash.py           # FlashLite NPZ checkpoint smoke-test chat
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

### 1. Chat with a trained checkpoint:
```bash
python zexo/chat.py --checkpoint zexo/checkpoints/sft/latest.bin
```
With no trained checkpoint, the CLI warns that it is loading random weights; such a
model is not expected to produce fluent text. Type a message and hit **Enter** once
a trained checkpoint is loaded.
- Commands available inside chat:
  - `/clear`: Reset conversation context.
  - `/stats`: Display context window token usage.
  - `/quit`: Exit.

### 2. Initialize and pretrain a language-model base:

A random checkpoint is **not** a language model yet. Pretrain on a large, clean,
licensed general-text corpus before teaching it chat behavior. The repository's
small examples are smoke-test data, not enough to learn English fluency.

```bash
# Create an 8-layer, 25M-parameter LLaMA-style starting point.
python zexo/init_weights.py --tier chat

# Train next-token prediction on your own plain-text corpus, with a separate holdout.
python zexo/train.py \
  --stage pretrain \
  --checkpoint zexo/checkpoints/zexo_chat_base.bin \
  --data /path/to/general_text_train.txt \
  --eval-data /path/to/general_text_eval.txt \
  --tier chat \
  --epochs 1 \
  --seq-len 512 \
  --lr 3e-4 \
  --output-dir zexo/checkpoints/pretrained
```

`--stage pretrain` requires a corpus, disables chat replay and prompt masking, and
selects full-transformer backpropagation. Use a substantially larger corpus and
many more training tokens/steps than the tiny bundled examples for useful language
modeling. Keep validation documents disjoint from training.

### 3. Supervised fine-tune on conversations:

```bash
python -m zexo.data.dataset_tools
python zexo/train.py \
  --stage sft \
  --checkpoint zexo/checkpoints/pretrained/latest.bin \
  --data zexo/data/zexo_quality_v1_train.txt \
  --eval-data zexo/data/zexo_quality_v1_eval.txt \
  --tier chat \
  --epochs 3 \
  --lr 5e-5 \
  --seq-len 256 \
  --output-dir zexo/checkpoints/sft
```

Full-transformer training is the CLI default; `--head-only` is the explicit
fast-but-limited legacy path. For an already-trained base, `--lora-rank 8` is a
lower-storage alternative. The SFT trainer masks user/prompt targets so the model
learns to produce assistant responses.

### 4. Refine with reviewed preference pairs (DPO):

Create JSONL records with `prompt`, `chosen`, and `rejected` fields. Keep a separate
held-out preference file; do not let the trainer see evaluation pairs.

```json
{"id":"helpful-1","prompt":"How do I start learning Python?","chosen":"Start with variables, conditionals, loops, and small projects. Tell me your experience level and I can suggest a first exercise.","rejected":"Python is impossible to learn. Give up."}
```

```bash
python zexo/refine.py \
  --checkpoint zexo/checkpoints/sft/latest.bin \
  --preferences preferences/train.jsonl \
  --eval-preferences preferences/eval.jsonl \
  --output zexo/checkpoints/refined.bin \
  --lora-rank 8 \
  --epochs 1
```

This is offline Direct Preference Optimization (DPO): it raises the probability of
reviewed chosen responses relative to rejected ones while regularizing against a
frozen copy of the input model. With the default LoRA mode it saves an adapter
and a merged checkpoint; `--full-model` updates all parameters. When held-out pairs
are supplied, it restores the best eval-loss epoch. To iterate, review new outputs,
add fresh preference pairs, and use the previous refined checkpoint as the next input.
It is **not** autonomous self-RL: the project has no trustworthy reward model or
environment, and training on unreviewed self-generated critiques can reinforce
errors.

### 5. Try a tiny Hugging Face pretraining smoke run:
```bash
# This validates the download/training path only; 20 samples will not teach fluency.
python zexo/train.py \
  --stage pretrain \
  --hf-dataset roneneldan/TinyStories \
  --hf-samples 20 \
  --epochs 1 \
  --max-steps 1 \
  --seq-len 128 \
  --tier micro \
  --from-scratch \
  --output-dir zexo/checkpoints/hf-smoke
```
Check dataset licensing and quality before using it for real training.

### 6. Initialize Fresh Base Weights for Any Tier:
```bash
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

The CLI and the `LlamaLLM.train` / `ZexoModel.train` Python APIs default to full-transformer backpropagation so a randomly initialized model does not silently train only its output head. `--head-only` in the CLI or `full_backprop=False` in Python explicitly selects the legacy fast path; it is suitable only for quick experiments, not for learning language from scratch. Full backprop updates embeddings, attention projections, RoPE-connected attention, RMSNorm scales, SwiGLU projections, and the output head. `--numpy-full` forces the slower reference implementation; otherwise the native C++ trainer is used when available. Training metadata records the selected mode.

The inference and native-training engine uses the optimized C++ path automatically when available. Set `DORANEURAL_NUM_THREADS` or call `llm.set_num_threads(n)` to tune native parallelism. The CLI also accepts `--threads N` / `--num-threads N`, including values such as 200 or 3000; omitted values keep normal OpenMP CPU auto-selection. Values above the CPUs exposed to the process are accepted but may oversubscribe and become slower. Tokenizers use heap-based merges and bounded caches, and HuggingFace single-file, BF16, and sharded safetensors checkpoints are supported without PyTorch.

For pretrained-checkpoint adaptation, LoRA keeps the base weights frozen and saves only small adapter matrices:

```bash
python zexo/train.py \
  --tier micro \
  --checkpoint path/to/pretrained.bin \
  --data zexo/data/zexo_quality_v1_train.txt \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-targets q,v \
  --adapter-output zexo/checkpoints/zexo_style.npz
```

`--full-backprop` now selects the native C++ full-transformer trainer automatically when the C++ engine is available. Use `--numpy-full` only when you specifically want the reference autograd implementation for debugging or numerical experiments.

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
