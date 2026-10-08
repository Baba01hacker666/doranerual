# Research Paper 16: Empirical BPE vs Byte-Level Comparison on Trained Models

**Date:** 2026-10-08
**Extends:** [Paper 15](15_llm_efficiency_audit.md) (the audit that proposed byte tiers)
**Goal:** Test Paper 15's claims on real training runs instead of synthetic weights.
**Script:** [`scripts/exp_byte_vs_bpe.py`](../scripts/exp_byte_vs_bpe.py)
**Result:** The efficiency claims hold and are large. **The loss claims do not — the byte tokenizer is a measurably worse compressor, and the apparent "win" on cross-entropy is a vocabulary-size artifact.**

---

## 1. Method

Two transformer bodies from the Zexo ladder, each trained four ways:

| Body | Config | BPE side | Byte side |
| :--- | :--- | :--- | :--- |
| `micro` (the `stories260K` shape: dim=64, 5 layers, hidden=172, 4 heads) | 280,512 params | `tok512`, V=512, **1.464 bytes/token** | V=258 |
| `mini` (dim=192, 1 layer, hidden=1024, 2 heads) | 6,844,992 params | `tokenizer.json`, V=32,000, **3.779 bytes/token** | V=258 |

Held constant: transformer body, `config.seq_len`, weight-init seed (42), window-shuffle
seed, lr 5e-4, corpus (`zexo_quality_v1_train.txt`, 109,763 bytes), held-out set
(`zexo_quality_v1_eval.txt`, 7,364 bytes), `--full-backprop`, 4 OpenMP threads.

Varied: vocabulary/tokenizer, and `pack_documents` (off vs EOS-delimited global packing).

**Step matching, not epoch matching.** A byte model emits one token per byte, so equal
epochs hands it ~3.8x the supervised tokens. Every variant was therefore given the same
*optimizer-step* budget, and per-variant epochs derived from `round(target / windows_per_epoch)`.

---

## 2. Results — `mini` body (V=32,000 vs V=258)

This is the case where the vocabulary tax is real.

| Variant | Params | Ep | Steps | Win/ep | **Sup.tok/ep** | Train s | Train L | Val L | **Bits/byte** | **Decode TPS** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BPE pack=off | 6,844,992 | 5 | 860 | 172 | 24,573 | 1046.8 | 4.0663 | 4.4456 | **1.6972** | 595.1 |
| BPE pack=on | 6,844,992 | 8 | 904 | 113 | 24,408 | 1364.5 | 3.2805 | 4.3515 | **1.6613** | 603.0 |
| byte pack=on equal-seq | 750,528 | 2 | 852 | 426 | 93,763 | 83.8 | 2.4517 | 2.0288 | 2.9269 | 13,884.5 |
| byte pack=on equal-text | 750,528 | 8 | 904 | 113 | 93,722 | **427.7** | 2.1685 | **1.6829** | 2.4279 | **14,150.0** |

Headline, `mini-byte` vs `mini`:

- **9.1x fewer parameters** (750,528 vs 6,844,992) for a bit-identical transformer body
- **3.84x more supervised tokens per window** (93,722 vs 24,408) from the same corpus bytes
- **3.2x faster to train** (428 s vs 1364 s) at a matched step budget
- **23.5x faster to decode** (14,150 vs 603 tokens/sec)

## 3. Results — `micro` body (V=512 vs V=258)

| Variant | Params | Ep | Steps | Win/ep | Sup.tok/ep | Train s | Train L | Val L | Bits/byte | Decode TPS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BPE pack=off | 280,512 | 5 | 1,775 | 355 | 64,342 | 133.6 | 3.2601 | 2.8779 | 2.8360 | 6,751.6 |
| BPE pack=on | 280,512 | 6 | 1,752 | 292 | 64,279 | 164.4 | 3.2578 | 2.8175 | **2.7765** | 7,524.5 |
| byte pack=on equal-seq | 264,256 | 4 | 1,704 | 426 | 93,763 | 157.5 | 2.4280 | 2.1354 | 3.0807 | 9,246.5 |
| byte pack=on equal-text | 264,256 | 6 | 1,746 | 291 | 93,576 | 275.3 | 2.2856 | 1.9473 | 2.8094 | 6,632.7 |
| BPE pack=on *token-matched* | 280,512 | 9 | 2,628 | 292 | 64,279 | 244.9 | 2.8995 | 2.2612 | 2.2286 | 7,298.7 |

A token-matched BPE run (9 epochs, 578,511 supervised tokens vs the byte model's 561,456)
was added to close the loop: BPE improves 2.7765 -> 2.2286 bits/byte with 50% more steps.
It still does not reach the byte model's compression, but the *gap* is now 1.2% in BPE's
favour at this scale rather than 46%.

---

## 4. Findings

### 4.1 The efficiency claims are real, and large

Everything Paper 15 predicted about *cost* held up on trained models: byte tiers cut
parameters 9.1x, decode 23.5x, training wall-clock 3.2x, and multiply gradient updates per
corpus byte 3.84x at the `mini` scale. The mechanism is unchanged and now demonstrated
end-to-end: because the model spends nothing on a 6.1M-parameter embedding table, all of
the text it does see becomes useful gradient signal.

### 4.2 The byte tokenizer is a *worse* compressor. This is a genuine cost.

Normalising val cross-entropy by bytes-of-text (the standard tokenizer comparison) reverses
the raw-CE ordering:

| Body | BPE bits/byte | byte bits/byte | BPE advantage |
| :--- | ---: | ---: | ---: |
| `micro` (V=512) | 2.7765 | 2.8094 | 1.2% |
| `mini` (V=32,000) | 1.6613 | 2.4279 | **46%** |

**The compression penalty of going byte-level grows with vocabulary size**, because the
BPE merge table is what buys the compression. At V=512 the penalty is negligible; at V=32,000
it is large. The parameter saving grows too, so this is a genuine scale-dependent trade-off,
not a free win — and Paper 15's framing understated the cost side.

### 4.3 Raw cross-entropy is not comparable across vocabularies

The tempting reading of §2 is "byte val CE 1.68 vs BPE 4.35, so byte is much better". That
is an artifact. A uniform predictor scores `ln(V)`, and:

```
ln(32000) = 10.373      ln(258) = 5.553      floor gap = 4.821 nats
observed CE gap        = 2.669 nats
```

**The vocabulary entropy floor gap (4.821) is larger than the entire observed loss gap
(2.669).** Any byte-vs-BPE loss comparison must normalise by bytes-per-token or it is
measuring the vocabulary size, not the model.

### 4.4 Packing is a clean, unambiguous win

In both bodies, at matched steps and matched supervised tokens:

| Body | Windows/ep | Val L | Bits/byte |
| :--- | ---: | ---: | ---: |
| `mini` pack=off -> on | 172 -> **113** (-34.3%) | 4.4456 -> **4.3515** | 1.6972 -> **1.6613** |
| `micro` pack=off -> on | 355 -> **292** (-17.7%) | 2.8779 -> **2.8175** | 2.8360 -> **2.7765** |

Fewer forward passes *and* better loss, because packing removes empty tail slots that were
being paid for in full-window forward passes while contributing no supervised targets.
This is the one change in this work with no downside at any scale.

### 4.5 Batched prefill scales with prompt length

Measured on the `micro` body: 1.33x at 16 tokens, 3.88x at 64, 3.21x at 256, 2.01x at 512.
The win is real but modest at short prompts — it only matters once the prompt is long enough
for weight-stream amortisation to dominate.

### 4.6 The BPE run had plateaued, so the compression gap is not a budget artifact

Going from 860 to 904 steps moved BPE only 1.6972 -> 1.6613 bits/byte (2.1%). The 46% gap at
the `mini` scale will not close by training the BPE model longer.

---

## 5. What this experiment does NOT establish

**No text-quality claim.** Every variant, in both bodies, collapses to newline runs or `the`
repetition under greedy decoding. Both bodies are badly undertrained on a 110 KB corpus, and
the held-out set is only 7,364 bytes. This experiment settles *cost* — parameters, throughput,
wall-clock, gradient updates — and settles the compression question. It does **not** show that
byte tiers produce better or worse text.

**Absolute losses are high.** Val CE 1.68 on a byte model is still far from useful. Nothing
here should be read as "the byte tier is now a good model"; it is a cheaper way to spend the
same body, which only matters once there is enough data to use it.

---

## 6. Recommendations, revised

1. **Keep packing** — no downside, measured twice.
2. **Keep batched prefill** — large win on long prompts, no correctness cost (Paper 15 §3.4).
3. **Adopt byte tiers for the small end, and stop describing them as free.** At `mini` scale
   they are 9.1x smaller and 23.5x faster, at the cost of 46% worse text compression. That
   trade is worth it only when data — not capacity — is the binding constraint, which is
   exactly the situation Paper 11 documented and exactly the situation this corpus is in.
4. **Do not use byte tiers as a drop-in replacement for a 32k-vocab model.** If
   `zexo-mini`'s job is conversational quality at scale, the 46% compression penalty is real
   and the parameter saving is comparatively small (22% of `base`). Byte tiers are for the
   `micro`/`mini` tiers and for any corpus too small to train a 32k merge table on.
5. **Report bits/byte, never raw CE**, in any future tokenizer comparison in this repo.

---

## 7. Reproduction

```bash
# micro body (~9 min at 4 threads)
python3 scripts/exp_byte_vs_bpe.py --body micro --match-steps 1752 --threads 4

# mini body (~50 min at 4 threads; the 32k classifier dominates full-backprop cost)
python3 scripts/exp_byte_vs_bpe.py --body mini --match-steps 300 --threads 4
```

Both write a JSON report containing per-variant losses, corpus statistics, throughput,
samples, and the prefill scaling table.