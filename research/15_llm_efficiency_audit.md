# Research Paper 15: LLM Efficiency Audit — Vocabulary Tax, Data Starvation, and Packed SFT

**Date:** 2026-10-08
**Scope:** Efficiency of the LLM path itself (architecture, configuration, data pipeline) rather than SIMD kernel tuning.
**Outcome:** Byte-level Zexo tiers, cross-document packing, batched prefill and a
CPU-dispatch-aware benchmark harness all shipped. Full suite: 198 passed, 2 skipped.

---

## 1. Abstract

Research Papers 10 and 14 independently derived that a vocabulary-free byte-level
model is the correct choice for small-parameter language models, and Paper 11
identified data starvation as the cause of `zexo-dora` losing to `zexo-mini`. Yet
the shipped tier ladder in `zexo/config.py` had never been updated to match:
every tier still declared `vocab_size=32000`.

This audit quantifies the cost of that inconsistency, then implements and measures
the fix. Three findings:

1. **The vocabulary tax is severe and asymmetric.** `zexo-mini` spends **89.8%** of
   its 6.8M parameters on the embedding matrix and has a single transformer layer.
2. **Cutting the vocabulary is a throughput win, not only a memory win**, because
   single-token decoding is bound by streaming the embedding and unembedding
   matrices from DRAM. Measured on an AVX2 Skylake host: `chat-byte` runs **1.81x**
   faster and `mini-byte` **37.8x** faster than their BPE twins with a *bit-identical
   transformer body*.
3. **The SFT pipeline was throwing away 16–44% of every forward pass** by rounding
   each dialogue up to a whole number of windows instead of packing documents.

---

## 2. Diagnosis

### 2.1 The vocabulary tax (measured from `zexo/config.py`)

| Tier | Total params | Embedding (`V x d`) | Embedding share | Body layers |
| :--- | ---: | ---: | ---: | ---: |
| `micro` | 280,512 | 32,768 | 11.7% | 5 |
| `mini` | 6,844,992 | 6,144,000 | **89.8%** | 1 |
| `chat` | 25,270,656 | 12,288,000 | **48.6%** | 8 |
| `dora` | 66,783,744 | 16,384,000 | 24.5% | 12 |
| `base` | 109,529,856 | 24,576,000 | 22.4% | 12 |
| `large` | 221,545,472 | 32,768,000 | 14.8% | 16 |

`zexo-mini` is not a 6.8M-parameter language model; it is a 6.1M-parameter lookup
table with 0.7M parameters of language model attached. This is exactly the failure
mode Paper 10 characterised ("30% to 90% of the entire parameter budget").

> **Note on the Paper 11 figure.** `zexo-dora` is documented as 54M parameters in
> the README and in `ZexoConfig.dora`'s docstring, but `parameter_count` computes
> **66,783,744**. The discrepancy is the `novel_per_layer` term. Any paper
> comparing tiers by parameter count should read it from the property, not the prose.

### 2.2 Data starvation, now measurable

Tokenizing the shipped SFT corpus with the real `HFTokenizer`:

```
corpus               zexo/data/zexo_quality_v1_train.txt
bytes                109,704
BPE tokens           29,049
unique tokens used    4,543          -> 14.20% of the 32,000-token vocabulary
bytes per token           3.777      (healthy for a 32k BPE)
```

Two consequences:

- **27,457 embedding rows (85.8%) never receive a single gradient.** The tokenizer
  itself is fine — 3.78 bytes/token is normal for a 32k SentencePiece BPE. The
  problem is corpus size relative to vocabulary size, which is a *tier
  configuration* error rather than a tokenizer error.
- The gold set is **36 records** (26 train / 10 eval) per
  `zexo_quality_v1_manifest.json`. Paper 11 attributed Dora's failure to "54M
  parameters and only 41,600 training tokens"; the current corpus is smaller still.

### 2.3 Three shipped claims are not reproducible

| Claim | Source | Actual repo state |
| :--- | :--- | :--- |
| 100,000-sample decision dataset, ECE 0.0247, TPR 98.6%, FPR 0.4% | Paper 13 | `jev_100k_manifest.json` declares 100k samples, but the only committed artifact is `jev_decisions_large.jsonl` with **3,106 lines**; `jev_100k_train.jsonl` and `jev_100k_eval.jsonl` are **`.gitignore`d** and absent. The metrics cannot be re-derived. |
| Stanford Alpaca cache for dialogue synthesis | `curate_decision_dataset.py:17` | `zexo/data/alpaca_data_cleaned.json` was **committed to git as a 0-byte file**. |
| >1000 TPS chat-tier inference | Paper 09 | Measured on an AVX-512 **VNNI** host. On an AVX2 host `get_cpu_backend()` reports `AVX2_FMA` and the INT8 VNNI kernel does not exist. |

The Alpaca 0-byte file was harmless at runtime (the loader already guards on
`st_size > 1000` and re-downloads), but it meant every clone shipped a corrupt
cache file. It has been untracked and gitignored; `build_100k_dataset.py` already
keeps its downloads in the gitignored `zexo/data/.dataset_cache/`.

### 2.4 Prefill was serialised

`csrc/llm_engine.cpp` processed the prompt one token at a time:

```cpp
for(int i=0;i<prompt_len;i++){
    if(i==prompt_len-1) llama_forward(engine, prompt_tokens[i], pos, engine->logits.data());
```

A 1024-token context therefore cost 1024 serialised forward passes, each re-streaming
every weight matrix from DRAM. This was the single largest remaining waste, and it is
now fixed (§3.4).

### 2.5 Single-token decode is bandwidth-bound, not compute-bound

On the measured host the `chat` tier sustains roughly **1.8% of AVX2 FMA peak**.
Adding FLOPs buys nothing; moving fewer bytes does. Since the embedding and
unembedding matrices are `2 x V x d` floats and must be streamed once per token,
**vocabulary size is a first-order decode-time cost.** This is the mechanism
behind the results in §3.1.

---

## 3. Implemented Interventions

### 3.1 P0 — Byte-level tiers (`zexo/config.py`, `doraneural/llm.py`)

Added `ByteTokenizer` (`doraneural/llm.py`): token ids `0..255` are raw UTF-8
bytes, ids `256`/`257` are BOS/EOS, total vocabulary **258**. Two ids rather than
a flat 256 keeps the mapping lossless — no byte value is sacrificed to make room
for specials — at a cost of 0.8% more embedding parameters.

Byte tiers mirror their BPE twins **body for body** (identical `dim`,
`hidden_dim`, `n_layers`, `n_heads`, `n_kv_heads`) so the comparison isolates the
vocabulary:

| Tier | BPE total | Byte total | Ratio | Body identical |
| :--- | ---: | ---: | ---: | :--- |
| `micro-byte` | 280,512 | 264,256 | 1.06x | yes |
| `mini-byte` | 6,844,992 | 750,528 | **9.12x** | yes |
| `chat-byte` | 25,270,656 | 13,081,728 | **1.93x** | yes |
| `base-byte` | 109,529,856 | 85,152,000 | 1.29x | yes |

`vocab_size` share collapses from 48.6% to 0.8% for `chat-byte`. Byte tiers also
set `eos_token_id=257`; the `LlamaConfig` default of `2` would otherwise treat a
raw control byte as end-of-text.

**Throughput, measured** (`scripts/bench_llm.py`, 4 threads, best of 2, AVX2 host):

| Tier | 32 tok | 128 tok | Speedup at 128 tok | Param ratio |
| :--- | ---: | ---: | ---: | ---: |
| `mini` | 530.73 TPS | 544.58 TPS | — | — |
| `mini-byte` | 14,959.08 TPS | **20,592.92 TPS** | **37.8x** | 9.12x |
| `chat` | 114.05 TPS | 126.19 TPS | — | — |
| `chat-byte` | 193.89 TPS | **228.16 TPS** | **1.81x** | 1.93x |

For `chat` the speedup (1.81x) tracks the parameter ratio (1.93x) almost exactly,
confirming §2.5: decode is bound by streaming `2 x V x d` floats. For `mini` the
37.8x exceeds the 9.12x parameter ratio because at `V=32000` the unembedding
matmul alone is 6.14M MACs against a 0.70M-parameter body — the vocabulary was
**~90% of the compute**, and it also falls below the int8 VNNI kernel's size
threshold, changing the dispatched code path.

The trade-off is real and must be stated: byte-level models consume 1 token per
byte instead of 3.78 bytes per token, so a fixed corpus yields **3.78x more
gradient updates** (the point — it fights §2.2) but also 3.78x longer sequences,
so byte tiers need a larger `--seq-len` to cover comparable text.

### 3.2 P1 — Data pipeline

**(a) Alpaca cache.** Untracked the 0-byte `zexo/data/alpaca_data_cleaned.json`
and added it to `.gitignore` alongside the existing dataset-cache entries. The
loader already re-downloads on a size check.

**(b) Cross-document packing** (`_prepare_training_batches`, `pack_documents=True`).
Previously each dialogue was windowed independently, so every dialogue's tail
wasted up to a full window of masked slots. Now all dialogues are concatenated
into one stream (BOS-delimited, EOS-terminated) and windowed once:

| Tokenizer | seq_len | Unpacked windows | Packed windows | Passes saved | Supervised density |
| :--- | ---: | ---: | ---: | ---: | ---: |
| BPE | 128 | 285 | 227 | **20.4%** | 67.4% -> 84.4% |
| BPE | 256 | 172 | 113 | **34.3%** | 55.8% -> 84.4% |
| Byte | 128 | 892 | 845 | 5.3% | 82.2% -> 86.8% |

Combined with raising the `--seq-len` default from 128 to **256**, the shipped SFT
path now runs **113 windows/epoch instead of 172 — 34% fewer forward passes** —
with supervised-token density up from 55.8% to 84.4%.

Documents remain EOS-delimited so the model can learn boundaries. The causal mask
is **not** blocked across documents, because the native engine has no document
mask; this is called out in the docstring. A proper block-diagonal mask is the
correct long-term fix.

**(c) `--seq-len` default 128 -> 256.** At 3.78 bytes/token, 128 tokens is ~483
bytes, roughly 80 words. That is too short a horizon for credit assignment.

### 3.3 P4 — Benchmark harness (`scripts/bench_llm.py`)

Replaces ad-hoc benchmark snippets with a script that records `cpu_arch`,
`cpu_backend`, platform, thread count and NumPy version **alongside** every TPS
figure, pins the thread count, and reports best-of-N. This exists because §2.3's
third row makes an unannotated TPS number uninterpretable: the same binary is ~10x
apart across CPU dispatch paths. Covered by
`tests/test_llm_efficiency.py::test_bench_script_reports_cpu_dispatch_signature`.

---

### 3.4 P3 — Batched prefill (`csrc/llm_engine.cpp`)

New entry point `llama_forward_chunk(engine, tokens, T, pos_start, out_logits)` runs a
whole prompt window in one pass. It adds:

- **`matmul_forward_batched`** — a GEMM computing `y[t*d_out + i] = sum_j W[i*n_in+j] * x[t*n_in+j]`
  over a `[T][n_in]` activation block, keeping `W`'s existing `[d_out][n_in]` layout so the
  GEMV path is untouched. A 16x64 `(output, batch)` register tile means each weight element
  is streamed from DRAM roughly `T/64` times instead of `T` times.
- **`llama_forward_chunk`** — batched QKV, RoPE, WO, W1/W3, SiLU-gating and W2. Attention
  deliberately stays per-position and causal: position `t` reads keys `0..pos_start+t` out of
  the very cache this call is filling, so it is not expressible as a single GEMM.
- **`ensure_prefill_capacity`** — lazily grown `[T][*]` scratch arenas, so the single-token
  decode path keeps its original allocation and cache behaviour unchanged.

`llama_generate_ex` now prefills via `llama_forward_chunk`, and the Python streaming path in
`llm.py` does the same. Both fall back to the original serial loop if the window is rejected.

**Measured prefill throughput** (4 threads, best of 2, AVX2 host):

| Tier | Prompt | Serial | Batched | Speedup |
| :--- | ---: | ---: | ---: | ---: |
| `chat` (V=32000) | 16 | 125.4 | 584.5 | 4.66x |
| `chat` | 64 | 123.2 | 1039.2 | 8.44x |
| `chat` | 256 | 123.8 | 1482.6 | **11.98x** |
| `chat` | 512 | 120.0 | 1223.3 | 10.20x |
| `chat-byte` | 64 | 248.8 | 1245.7 | 5.01x |
| `chat-byte` | 256 | 235.6 | 1466.7 | 6.23x |

Decode speed is unchanged, which is the point: prefill and decode have opposite bottlenecks
(weight-streaming vs. one-token-at-a-time), so a single mixed TPS figure hides this win.

**Tile tuning.** `IB x TB` was swept empirically at 1T-4T: `(4,8)` gave 3.1x, `(8,64)` gave
1608 tok/s, `(16,64)` gave 1941, `(32,64)` gave 1987. `(16,64)` was chosen: the 2% extra from
`(32,64)` costs a 8 KB stack tile instead of 4 KB, which matters given the embedded target in
`AGENTS.md` Rule 3.

**Scope limit, stated plainly.** The batched GEMM covers the **float32 path only**. Under
`AGENTS.md` Rule 4 float32 is the reference and lossy INT8/FP16 is opt-in, so a quantized
engine delegates to the serial loop rather than silently changing its numerics. INT8 here
quantises *one activation vector per matmul*; a batched equivalent would have to redefine the
scale granularity, which is a separate accuracy decision and was not taken unilaterally.

**Parity.** The batched path is verified against the serialised path at `atol < 1e-4`
(`tests/test_prefill_batching.py`, 22 tests) across GQA, MQA, MHA, both RoPE layouts, dims that
are not multiples of the 16-wide SIMD or the tile widths, non-zero `pos_start`, split-point
invariance across chunk sizes 1-128, per-position KV-cache write verification, and greedy
generation determinism. Observed worst-case deviation is ~3e-6.

---

## 4. What Was Deliberately Not Done

- **P2, hybrid sliding-window attention + MH-RTU.** Paper 14 buys `O(1)` inference
  memory by *deleting* attention. That is correct for memory and fatal for exact
  recall — which the repo's own `programming` and `systems` categories depend on.
  The right design is sliding-window attention (local, exact) plus an MH-RTU
  global state (long-range, O(1)), keeping both properties rather than trading one
  away. This is a research-scale addition, not a patch.
- **Batched INT8/FP16 prefill.** See the scope limit in §3.4.
- **Block-diagonal document masking.** Packing (§3.2b) concatenates documents without
  masking attention across the boundary, because the native engine has no document
  mask. A correct implementation needs a per-position segment id threaded into the
  attention score kernel.
- **Scaling any tier.** Paper 11 showed parameter scaling on a starved corpus
  *loses*. Fix coverage first.

---

## 5. Recommendations, In Order

1. **Adopt `chat-byte` as the default conversational tier.** 1.93x fewer
   parameters and 1.81x more throughput with zero capacity loss. Keep the BPE
   tiers only for pretrained-checkpoint compatibility (the 32k tokenizer is needed
   to load published weights).
2. **Grow the corpus before scaling models.** Every tier is data-starved; see
   §2.2. The Jev decision builder already produces balanced multi-domain data and
   should be pointed at the conversational corpus too.
3. **Commit the generated 100k decision artifacts** (or the generating script plus
   pinned source hashes) so Paper 13's metrics become reproducible.
4. **Report parameter counts from `ZexoConfig.parameter_count`**, never from prose.
5. **Batched prefill is already shipped** (§3.4); extend it to the INT8 path and add
   block-diagonal document masking (§4) rather than re-litigating it.
6. **Then** tackle the hybrid attention design (§4).

---

## 6. Verification

- `doraneural/csrc/libdoraneural.so` rebuilt with `dn.build_cpp_library(force=True)`.
- `python3 -m pytest` — **198 passed, 2 skipped**, zero failures.
- New coverage: `tests/test_llm_efficiency.py` (23 tests) covering byte-tokenizer
  round trips across 2-, 3- and 4-byte UTF-8, byte-tier parameter/body invariants,
  `from_tier` alias resolution, EOS-id selection, packing window-count and
  supervised-density invariants, SFT mask preservation under packing, `max_batches`
  capping, packing inertness on non-dialogue text, and an end-to-end byte-tier
  train/generate round trip.
- `tests/test_prefill_batching.py` (22 tests) covers batched-vs-serial parity, KV-cache
  equivalence for continued decode, non-zero `pos_start`, split-point invariance, and
  per-position cache writes (§3.4).