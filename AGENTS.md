# AGENTS.md — Instructions for Autonomous AI Agents

Welcome, Agent. This document establishes the mandatory architecture principles, coding standards, commit policies, and testing requirements for working in the **`doraneural`** repository.

---

## 1. Project Overview & Architecture

`doraneural` is a lightweight, educational, zero-heavy-dependency neural network and LLM framework built with:
- **NumPy Frontend**: Full neural layers, custom/exotic neurons, autograd tape, static operator fusion compiler, schedulers, and metrics.
- **Native C++ Inference & Training Engine (`doraneural/csrc/`)**: High-performance CPU engine implementing LLaMA forward pass, KV-cache, RoPE, multi-head attention, SwiGLU, cross-entropy loss, AdamW backpropagation, and SIMD/OpenMP acceleration.
- **Zexo Subproject (`zexo/`)**: In-repo mini-LLM development suite including custom BPE/WordPiece tokenizers, dataset curations, SFT masking, and CLI training workflows.
- **Research Lab (`research/`)**: Derivations, empirical benchmarks, and architectural investigations (e.g., Chebyshev KAN, novel neurons, TPS optimization).

---

## 2. Mandatory Rules & Hard Constraints

### 🚫 Rule 1: NEVER Include Co-Author Tags
- **NEVER** append `Co-authored-by: arena-agent ...`, `Co-authored-by: ...` or any AI-assistant attribution trailers to commit messages or PR descriptions.
- Commits must be authored cleanly and attributed to the repository owner (`baba01hacker666`).
- If rebasing or pulling remote branches created by other agents that contain co-authored tags, **always sanitize the commit history** before merging into `main`.

### 🛡️ Rule 2: 100% Test Suite Pass Required
- **Never push breaking code to `main`**.
- Before every commit and push, run the test suite:
  ```bash
  python3 -m pytest
  ```
  All tests (currently 114+ tests) must pass with zero failures.

### 🌐 Rule 3: Cross-Platform Portability (ARM64 + x86)
- The codebase runs on both **ARM64** (e.g. Cortex-A, Apple Silicon, Neoverse) and **x86_64** (Intel/AMD) machines.
- **NEVER blindly include x86-specific headers**:
  ```cpp
  // ❌ BAD — breaks ARM compilation:
  #include <immintrin.h>

  // ✅ GOOD — guarded:
  #if defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>
  #elif defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #endif
  ```
- Always provide a scalar C++ fallback path for any new vectorized kernel.

### 🎯 Rule 4: Numerical Parity (Float32 Reference vs. Quantization)
- Standard forward passes (`llama_forward`) must maintain exact mathematical parity with NumPy reference forward passes (`atol < 1e-4`).
- **Lossy INT8 / FP16 quantization must be opt-in**:
  - Controlled via `DORANEURAL_USE_INT8=1` or `DORANEURAL_FAST=1`.
  - **Do NOT** enable lossy quantization as the silent default inside `llama_create()`.

---

## 3. C++ Native Engine & Modular Kernel Dispatch

The C++ acceleration layer in `doraneural/csrc/` is structured modularly:

| File | Purpose |
| :--- | :--- |
| [`doraneural/csrc/cpu_features.h`](doraneural/csrc/cpu_features.h) | Runtime & compile-time CPU architecture and SIMD probing (AVX-512 VNNI, AVX2, ARM NEON, DotProd). |
| [`doraneural/csrc/kernel_dispatch.h`](doraneural/csrc/kernel_dispatch.h) | Modular function pointer table (`KernelDispatch`) separating compute kernels from transformer model logic. |
| [`doraneural/csrc/llm_engine.cpp`](doraneural/csrc/llm_engine.cpp) | Transformer forward pass, KV-cache management, RoPE rotation, attention, full backpropagation, and AdamW. |
| [`doraneural/csrc/llm_engine.h`](doraneural/csrc/llm_engine.h) | C-compatible interface exported to Python ctypes. |
| [`doraneural/cpp_backend.py`](doraneural/cpp_backend.py) | Dynamic JIT compilation wrapper with automatic fallback compile flags. |

### How to Rebuild C++ Backend
```python
import doraneural as dn
dn.build_cpp_library(force=True)
```
Or via terminal:
```bash
python3 -c "import doraneural as dn; dn.build_cpp_library(force=True)"
```

### Probing CPU Features in Python
```python
import doraneural as dn

print("CPU Architecture:", dn.get_cpu_arch())      # e.g., "ARM64", "x86_64"
print("Optimal Backend: ", dn.get_cpu_backend())   # e.g., "ARM_NEON_DOTPROD", "AVX512_VNNI"
dn.print_cpu_features()
```

---

## 4. Dataset & Training Guidelines

All curated and held-out evaluation datasets live under [`zexo/data/`](zexo/data/):

- **Primary SFT Training Corpus**: `zexo/data/zexo_quality_v1_train.txt`
- **Evaluation Corpus**: `zexo/data/zexo_quality_v1_eval.txt`
- **Structured JSONL Dialogues**: `zexo/data/quality_dialogues.jsonl`
- **Continual Replay Anchor**: `zexo/data/step1_conversational_expanded.txt`
- **Dataset Manifest**: `zexo/data/zexo_quality_v1_manifest.json`

### Training Rules:
1. **SFT Prompt Masking**: Target tokens with value `< 0` (e.g. `-1`) represent masked prompt tokens. The forward pass must run to compute KV cache, but loss calculation and gradient backprop must be skipped for masked tokens.
2. **Native Thread Scaling**: Honor `--threads <N>`, `--num-threads <N>`, and `DORANEURAL_NUM_THREADS=<N>` environment variables. Do not hardcode thread counts.

---

## 5. Research & Novel Neurons

The repository contains novel neural architectures exploring alternatives to standard MLPs:
- `ChebyshevKAN`: Kolmogorov-Arnold Networks using Chebyshev polynomial basis functions.
- `ReflectiveDense` & `InvertedDense`: Reflection and inversion gating activations.
- `Tensor4DDense`: 4D hyperdimensional spatial projection layers.
- `TunnelingDense`: Quantum-tunneling inspired non-monotonic activation layers.

### Novel Neuron Extension Checklist:
When modifying or adding new neuron layers:
1. Implement forward and backward passes supporting autograd tape (`doraneural/autograd.py`).
2. Implement serialization round-trip in `.dnb` format (`doraneural/dnb.py`).
3. Add corresponding unit tests in `tests/test_neural_lib.py`.

---

## 6. Pre-Commit Verification Checklist

Before reporting completion or pushing changes:

- [ ] Rebuilt native engine: `dn.build_cpp_library(force=True)`.
- [ ] Executed full test suite: `python3 -m pytest` passes with 100% green.
- [ ] Checked `git diff` for accidental debug code, temporary logs, or hardcoded paths.
- [ ] Verified commit message contains **NO** `Co-authored-by` tags.
- [ ] Verified cross-platform compatibility (no unguarded x86/ARM intrinsics).
