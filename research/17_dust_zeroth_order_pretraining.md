# Research Paper 17: Dust — Zeroth-Order Pretraining in Pure NumPy

**Date:** 2026-10-09
**Reference:** Samip Dahal, Bishwas Mandal, Serdar Gülbahar, Akshay Vegesna, *"Dust: Pretraining Transformers Without Backpropagation"*, Q Labs, October 2026. https://qlabs.sh/research/dust
**Module:** [`doraneural/dust.py`](../doraneural/dust.py)
**Result:** The paper's central claim reproduces — useful gradients emerge from population alone, with no chain rule. But on this repository's compute budget Dust is roughly **180x less efficient per optimizer step than the backprop we already have**, exactly as the authors themselves warn.

---

## 1. Abstract

Dust perturbs the *output of each linear layer at every token* rather than the weights, so one forward pass evaluates a population of one member per token instead of one member per forward pass. The per-token loss change credits the perturbation, and the reward-weighted noise averaged over draws estimates the layer's output error; its outer product with the layer input is the same weight gradient backpropagation forms.

Two claims were tested on our own Zexo `micro` body:

1. **Does the estimator converge to backprop's gradient?** Yes. Cosine rises monotonically
   from **+0.163 at K=8 to +0.682 at K=512**, fitting the paper's Eq. 5 law
   `cos(K) = c_max/√(1+c/K)` with **RMSE 0.016–0.025** (paper reports < 0.06).
2. **Is that useful here?** No. At matched wall-clock Dust completed **4 optimizer steps
   where backprop completed 707**, reaching held-out loss 6.25 against backprop's 2.38.

The headline is the *ratio*: the mechanism is real and the diagnostics work, and the
arithmetic is hopeless on an 8-core CPU.

---

## 2. What Was Implemented

A faithful slice of the method, not the full algorithm:

- **Forward-only NumPy pass** ([`dust_forward`](../doraneural/dust.py)) with no autodiff
  tape at all — the point of the method. Verified to match the existing autograd model to
  **1.3e-6**, which is the main correctness guard on the reimplementation.
- **Activation-space perturbation** at each site: `y_t → y_t + σ·a_t`, `a_t ~ N(0, I)`.
- **Per-token credit**: `c_s = ℓ_s − ℓ̃_s` (perturbed minus clean), then
  `r_t = Σ_{s≥t} γ^(s−t) c_s` as a reverse decayed cumsum.
- **Estimated output error** (paper eq. 2), contracted over draws only:
  `ĝ[t,c] = (1/Kσ) Σ_i r_t^(i) a_{t,c}^(i)`.
- **Weight gradient** (paper eq. 3): `Ĝ_W = Σ_t ĝ_t x_tᵀ`.
- **Sites**: `wo` (attention output projection) and `w2` (FFN down projection) — the two
  per-block linear outputs whose credit needs no attention internals, both with **γ = 0**.
- **Trainer** with SGD+momentum and Adam, splitting the population across sites.

**Not implemented**, and therefore *not* claimed: the attention internals (q, k, v, gate,
value embedding), which the paper credits through `c_s = −⟨ĝ_s, Δo_s⟩` (eq. 4); the
logit-space LM head population; and the 2L residual mixing scalars that the paper hands to
ordinary weight-space ES. A complete Dust would do strictly better than what follows.

---

## 3. Estimator Quality: the claim reproduces

Cosine to the true backprop gradient on the same batch, `micro` body (280K params),
T = 14, σ = 0.01, γ = 0, layer 0:

| K (draws) | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| site `wo` | +0.163 | +0.290 | +0.345 | +0.444 | +0.559 | +0.632 | **+0.682** |
| site `w2` | +0.195 | +0.275 | +0.354 | +0.391 | +0.560 | +0.641 | **+0.690** |

Fitted Eq. 5: `c_max ≈ 0.76`, `c ≈ 115`, **RMSE 0.016 (`wo`) / 0.025 (`w2`)**.

Monotone in K, with nothing from the chain rule anywhere in the estimator. This is the
load-bearing result and it holds.

### 3.1 The sign convention, and a note on the paper

The paper's eq. 2 carries a leading minus because its `r` is a *reward*. Our `c` is the
loss **increase**, so the directional derivative along `a_i` is `c_i/σ` and the estimate
needs no minus; it is the same quantity with the paper's reward negated. Measured directly:
with the literal sign, cosine is **−0.44**; flipped, **+0.44**. `test_dust.py` pins the
positive value so a future sign flip fails loudly.

### 3.2 Two findings the paper does not emphasise

**σ cancels in the estimator.** The estimator divides by σ and is linear in the
perturbation, so the cosine is invariant to σ up to second order:

| σ | 0.003 | 0.01 | 0.03 | 0.1 |
| :--- | ---: | ---: | ---: | ---: |
| `micro` cosine @ K=256 | +0.6320 | +0.6321 | +0.6322 | +0.6263 |
| `mini` cosine @ K=256 | +0.7621 | +0.7621 | +0.7621 | +0.7605 |

σ only matters once it is large enough for curvature to bite (0.1). So the paper's §2.3
suggestion of tuning σ *by maximising cosine* cannot discriminate between those settings —
that tuning has to be done against training loss, not the diagnostic. This is a real
limitation of their stated tuning procedure as applied to our implementation.

**Bigger models are more population-efficient** — their §4.1 result, reproduced:

| Model | Params | cosine @ K=256 |
| :--- | ---: | ---: |
| `micro` | 280K | +0.632 |
| `mini` | 6.8M | **+0.762** |

24x the parameters, *higher* cosine at the same population.

---

## 4. Efficiency: where it falls down

### 4.1 Matched wall-clock, 30 s budget each

[`scripts/exp_dust_vs_backprop.py`](../scripts/exp_dust_vs_backprop.py), `micro` body, S=64
byte-level windows, held-out `zexo_quality_v1_eval.txt`:

| Method | Steps | Tokens | Forwards/step | Seconds | Held-out loss |
| :--- | ---: | ---: | ---: | ---: | ---: |
| backprop | **707** | 44,541 | 1 | 30.0 | **2.3798** |
| Dust (K=1000) | 4 | 252 | 1,001 | 37.2 | 6.2473 |

**177x fewer optimizer steps.** Each Dust step costs 1000 forwards (100 draws × 10
layer-site pairs, plus a clean pass); each backprop step costs one backward pass.

### 4.2 It does optimize — just far too slowly

Holding steps fixed instead of time, 25 Dust steps (K=100, Adam) on the `micro` body:

| lr | start | +5 | +10 | +15 | +20 | +25 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3e-3 | 6.230 | 6.087 | 5.908 | 5.840 | 5.774 | **5.663** |
| 1e-2 | 6.230 | 6.058 | 5.961 | 5.855 | 5.794 | 5.686 |
| 3e-2 | 6.230 | 6.131 | 5.916 | 5.899 | 5.876 | 5.813 |

Held-out loss falls monotonically and the gradient is usable at only ~10 draws per site
(cosine ≈ 0.17 at K=10). Adam outperforms SGD, matching the paper's §3.3. lr = 3e-2 is
worst: the estimate is noisy enough that a large step amplifies its error.

So Dust is **correct but starved**: 4 steps cannot move a 280K-parameter model, 707 can.

### 4.3 A self-inflicted cost worth recording

Our first trainer gave every (layer, site) the full K, making a step cost
`K × n_layers × n_sites` = 320 forwards instead of 100 — a 10x–30x tax that made Dust look
3x worse than it is and made any wall-clock comparison meaningless. The paper splits the
population across layers ("the share of the population each layer gets", §2.3). With the
split, 3 steps became 19 in the same budget. Recorded here because the failure mode — a
missing normalisation in an estimator's accounting — is easy to repeat.

---

## 5. Honest Assessment

**What is genuinely valuable:**

- **A working diagnostic.** Cosine-to-backprop measures how many draws a zeroth-order
  estimator needs, with no training and one backward pass. That is a reusable tool for any
  black-box optimiser, and it is now in the repo.
- **Proof that the mechanism works at our scale.** Useful zeroth-order gradients emerge at
  280K parameters, not only at 1B.
- **The σ finding** (§3.2), which corrects a tuning procedure in the source paper.

**What is not:**

- **It is not a faster trainer here, and it will not become one on this hardware.** The
  authors are explicit: "We must have orders of magnitude more compute efficiency before
  Dust becomes a practical alternative to backprop at current levels of compute." Our
  measurement puts that at ~180x on a single 8-core CPU socket for a 280K model. Closing a
  180x gap needs roughly 180x the parallelism plus a C++ population kernel, i.e. the GPU
  cluster their experiments assume.

**The one case where it might still earn its place here:** the paper's real thesis is that
Dust needs no differentiability, so it can train networks backprop cannot. This repository
is full of exactly those — the tanh-saturated Chebyshev KAN and dendritic gating that
papers 09 and 11 blame for `zexo-dora`'s failure to train, and integer-only weights that
EGGROLL could not stabilise. **For that, the correct implementation is a C++ population
kernel riding the batched GEMM from Paper 15, not this NumPy reference.** Worth attempting
only if someone wants to revisit the novel-neuron tiers.

---

## 6. What Was Not Done

- **Attention-internal credit** (eq. 4), the LM-head population, and weight-space ES for the
  residual scalars. A full Dust would beat these numbers.
- **A C++ population kernel.** The batched GEMM in §3.4 of Paper 15 is exactly the right
  primitive — a population is a batch dimension — but it is a separate piece of work.
- **Larger models / longer sequences.** Stopped deliberately: the K=1024 mini-tier ladder
  pegged the machine's load average at 6.29 and was aborted. The cost was not measured
  before it was launched, which is the actual lesson.

---

## 7. Reproduction

```bash
# Estimator quality (fast: micro tier, K <= 256)
python3 scripts/exp_dust_vs_backprop.py --budget-seconds 10 --draws 64

# Matched wall-clock comparison
python3 scripts/exp_dust_vs_backprop.py --budget-seconds 30 --draws 1000 --skip-ladder

python3 -m pytest tests/test_dust.py -q     # 16 tests
```

`tests/test_dust.py` covers forward/tape parity (including GQA), credit decay at γ=0 and
γ>0, estimator shapes, the sign convention, cosine-rises-with-population, the Eq. 5 fit, σ
invariance, population splitting, and that the trainer reduces loss with no autodiff.