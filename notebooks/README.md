# 📓 DoraNeural & Dora-X2 Google Colab Notebooks

This directory provides pre-configured, one-click interactive Jupyter Notebooks for training and experimenting with `doraneural` architectures on Google Colab.

---

## ⚡ Available Notebooks

### 1. Dora-X2 16-Layer MH-RTU Chat Model
Train and chat with the 16-layer, 768-dimension, 12-head Multi-Head Recurrent Trace Unit (MH-RTU) model with pure SwiGLU:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Baba01hacker666/doranerual/blob/main/notebooks/train_dora_x2_colab.ipynb)

**Features:**
- **Zero-Setup Cloud Training**: Installs dependencies and builds the native C++ SIMD engine automatically.
- **Configurable Hyperparameters**: Run multi-epoch training over curated conversational datasets.
- **Interactive Chat Terminal**: Stream multi-turn conversations directly in the Colab output cell.
- **Google Drive Export**: Persist `.npz` tensors and `.json` manifests directly to your Google Drive.

---

## 🚀 How to Run in Google Colab

1. Click the **Open In Colab** badge above.
2. In Colab, select **Runtime ➜ Change runtime type** (Standard CPU or High-RAM).
3. Run the cells in sequential order using `Shift + Enter` or click **Runtime ➜ Run all**.
4. Chat with the trained model in Section 4!
