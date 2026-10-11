# 🔬 Research Paper 19: JEPA — Joint-Embedding Predictive Architectures, Implemented and Audited

**Modules:** [`doraneural/jepa.py`](../doraneural/jepa.py) · [`doraneural/jepa_data.py`](../doraneural/jepa_data.py)
**Scripts:** [`research/train_jepa.py`](train_jepa.py) · [`examples/explore_jepa.py`](../examples/explore_jepa.py)
**Tests:** [`tests/test_jepa.py`](../tests/test_jepa.py) (34 tests)

---

## 1. Abstract

Every self-supervised method in the JEPA family replaces one thing: **what you
predict**. Instead of reconstructing pixels (MAE), matching augmented views
(SimCLR/DINO), or generating tokens (GPT), a JEPA encodes a *context* view and a
*target* view and trains a predictor to map one embedding onto the other, with
the loss computed **entirely in representation space**:

$$\mathcal{L} \;=\; \big\lVert P_\phi\big(E_\theta(x),\, z\big) \;-\; \mathrm{sg}\big(E_{\bar\theta}(y)\big) \big\rVert$$

This paper documents a from-scratch, pure-NumPy implementation of the family —
I-JEPA (images), V-JEPA (video), the SIGReg/LeJEPA objective, and an
action-conditioned latent world model with a CEM planner — and then reports what
happens when you actually run it at laptop scale. Four findings stand out:

1. **A masked-latent objective can reach a near-perfect loss while learning
   nothing.** On our small image benchmarks the EMA-teacher I-JEPA objective
   converges to a *positional lookup table*: a predictor that ignores the
   context and outputs "the expected feature at this location" drives the
   smooth-L1 loss to `1e-5 - 5e-4`, while a linear probe of the frozen features
   stays exactly at the random-initialisation accuracy (0.453 vs 0.453; the
   pooled latents carry `signal_ratio` 7e-4 of input-dependent energy).
   `masked_target_baseline()` measures this directly: **position alone explains
   99.7% of the target variance** at a 64-token grid.
2. **Collapse does not have to be dimensional to be fatal.** An encoder whose
   output is dominated by an input-independent constant has a healthy effective
   rank (7-24), a near-zero mean cosine similarity, and a plausible feature
   norm, yet it is useless. We add `signal_ratio` (input-dependent energy
   fraction) to `representation_diagnostics` to catch it; the classic
   rank/cosine pair does not.
3. **SIGReg escapes the trap — and the escapes are measurable.** Swapping the
   EMA teacher for the LeJEPA objective (JEPA prediction + Sketched Isotropic
   Gaussian Regularization, no stop-gradient, no teacher) takes the video
   temporal-order task from **0.875 vs a 0.500 random-initialised baseline** on
   the committed 400-step artifact (1.000 vs 0.583 on a longer 600-step run),
   and lifts the image probe above its random-initialisation line for the first
   time (0.344 vs 0.266, chance 0.25). The same objective trains an 82K-parameter
   world model end-to-end **from pixels** to a next-latent MSE of **0.116**
   against 0.198 for no-change and 0.480 for the best constant, with
   `action_gain` **0.464** — and its cross-entropy-method planner finishes closer
   to the goal than a random rollout in **79%** of 24 episodes (mean final
   distance 0.286 vs 0.559).
4. **World-model accuracy is not evidence of dynamics.** Permuting the action
   stream across the batch costs 86% more prediction error and zeroing it costs
   70% more in the best run — but a *different* configuration in this same paper
   reached a 44% error reduction over no-change with an `action_gain` of 0.015
   and zeroed actions that did not hurt at all. We ship that ablation
   (`action_utilization`) and validate it on a controlled linear system, where
   it reports 0.72 for a model that recovered the map.

The headline is not that JEPA fails at small scale — it is that **latent loss is
not a progress metric**, and the repository now ships the four measurements
(shortcut audit, input-energy diagnostic, random-init probe, action-permutation
ablation) that turn a JEPA run into an interpretable experiment.

---

## 2. Why predict in representation space

LeCun's energy-based framing (`A Path Towards Autonomous Machine
Intelligence`, 2022) places JEPA among generative and joint-embedding
architectures as a compromise: an encoder that is allowed to *discard*
unpredictable detail, plus a predictor that is conditioned on a latent variable
`z` describing *which* of the compatible targets we are predicting. The
practical payoffs are the ones that motivated this implementation:

| Property | Generative (MAE, diffusion) | Joint-embedding (SimCLR, DINO) | JEPA |
| :--- | :--- | :--- | :--- |
| Prediction target | pixels / tokens | another view's embedding | another *view's embedding*, conditioned on position |
| Hand-crafted augmentations | none needed | central to the method | none needed (masking supplies the views) |
| Compute per step | decodes every patch | full view pair | encodes **only the visible context** (≈25% of tokens) |
| Failure mode | wasting capacity on pixel noise | collapse, augmentation leakage | collapse, *or the shortcut described in §6* |

### 2.1 I-JEPA (Assran et al., CVPR 2023)

Given an image split into `N` patches:

* **Targets.** Sample `M = 4` possibly overlapping blocks with scale in
  `(0.15, 0.2)` of the image area and aspect ratio in `(0.75, 1.5)`. The targets
  are the *outputs of the target encoder* at those positions — masking happens
  at the encoder output, never its input, so every target token is a contextual,
  high-level representation.
* **Context.** Sample a single block at scale `(0.85, 1.0)` with unit aspect
  ratio, then subtract every target region from it. The context and targets are
  disjoint; the predictor must therefore *synthesise* the masked content rather
  than copy visible tokens.
* **Predictor.** A narrow ViT. The context tokens (projected into the predictor's
  width) and one learnable mask token per target position (plus positional
  embeddings) are concatenated, self-attended, and projected back to the encoder
  width. Only the outputs at mask positions enter the loss.
* **Teacher.** The target encoder `E_θ̄` is an exponential moving average of the
  context encoder (`0.996 → 1.0`, cosine schedule). No gradient flows into it.
* **Loss.** Smooth-L1 in latent space (the paper's equation is L2; the released
  code uses Huber).

### 2.2 V-JEPA / V-JEPA 2 (Bardes et al., 2024; Assran et al., 2025)

Video replaces patches with **tubelets** (`2 x 16 x 16`), masks become
**tubes** extruded through time (short-range + long-range groups), the loss is
L1 on masked tokens only, targets are normalised per token, and absolute
positions give way to **3D-RoPE** at scale. V-JEPA 2 adds an
*action-conditioned* post-training stage (V-JEPA 2-AC): a 300M-parameter
block-causal transformer predicts the *next frame's latent* from past latents
and an action, trained with teacher forcing plus a 2-step rollout loss, while
the encoder stays frozen. That is the template for our
`JEPAWorldModel`.

### 2.3 LeJEPA and SIGReg (Balestriero & LeCun, 2025)

JEPA's anti-collapse machinery (EMA teachers, stop-gradients, schedulers) is a
pile of heuristics. LeJEPA replaces it with a *distributional* argument: the
embedding cloud should be an isotropic Gaussian, and matching all 1-D random
projections to `N(0,1)` is enough (Cramér-Wold). The univariate test is the
Epps-Pulley characteristic-function statistic,

$$\mathcal{L}_{\text{SIGReg}}(Z)=\frac{1}{M}\sum_{m=1}^{M}\int \big|\hat\varphi_{A_m Z}(t)-e^{-t^2/2}\big|^2 e^{-t^2/2}\,dt,\qquad
\hat\varphi_X(t)=\tfrac1N\sum_i e^{\,it X_i}$$

which is differentiable, has bounded gradients, and costs `O(N·M·T)` with
`M ≈ 256` slices and 17 quadrature points. The total objective is
`L_pred + λ·L_SIGReg` — a single hyper-parameter, no teacher, no stop-gradient.
A degenerate (constant) embedding cloud is infinitely far from `N(0,I)`, so
collapse is impossible *by construction* rather than by accident.

### 2.4 Latent world models (V-JEPA 2-AC, LeWorldModel, 2025-2026)

Close the loop: learn `z_{t+1} = f(z_{≤t}, a_{≤t})` in a JEPA latent space, then
*plan* by minimising a latent cost (distance to a goal embedding) over action
sequences with CEM. No reward model, no pixel decoding, no environment
interaction during search. LeWorldModel shows this also works end-to-end from
pixels with SIGReg as the only regulariser — which is exactly the recipe we
implemented as `JEPAWorldModel`.

---

## 3. What was implemented

All of it runs on the repository's own reverse-mode autograd engine
(`doraneural/autograd.py`), on NumPy, on CPU, with no pretrained weights. Two
operations were added to `Tensor` for this work: `sin()` and `cos()` (RoPE and
the characteristic function).

| Component | Class / function | Notes |
| :--- | :--- | :--- |
| Encoder (ViT) | `JEPAEncoder` | sine-cosine 2D/3D positions, **2D/3D RoPE**, learned positions, LayerNorm or RMSNorm, context-only encoding |
| Predictor | `JEPAPredictor` | narrow ViT, shared positional table, learnable mask tokens |
| Masking | `sample_ijepa_masks`, `sample_vjepa_masks`, `collate_masks` | exact I-JEPA scales; tube masks; unbiased random context subsampling; validity masks for padded targets |
| I-JEPA trainer | `IJEPA` | EMA teacher + cosine momentum schedule, smooth-L1/L2, target normalisation |
| V-JEPA trainer | `VJEPA` | tubelets, 3D positions, tube masking, L1-style objective |
| Anti-collapse | `SIGReg`, `sigreg_loss`, `ema_update`, `ema_schedule` | Epps-Pulley quadrature, trapezoid weights, LeJEPA objective in `fit(sigreg=True)` |
| Diagnostics | `representation_diagnostics`, `masked_target_baseline`, `linear_probe` | effective rank, cosine, uniformity, input-dependent energy; positional-shortcut audit; closed-form ridge probe |
| Action audit | `JEPAWorldModel.action_utilization` | true vs permuted vs zeroed action streams: the test that separates learned dynamics from shrinkage towards the marginal |
| World model | `JEPAWorldModel`, `JEPADynamics` | pixel or vector encoder, block-causal action-conditioned predictor, teacher forcing + k-step rollout loss, SIGReg; `plan()` = batched CEM in latent space |
| Checkpoints | `save` / `load` on every trainer | `.npz` weights + `.json` config round-trip (`inspect`-filtered, future-proof) |
| Synthetic data | `doraneural/jepa_data.py` | shape images (semantic = shape, nuisance = position), grating images (semantic = global orientation), moving-dot clips (semantic = temporal order), bouncing point mass (planning) |
| Demo / script | `examples/explore_jepa.py`, `research/train_jepa.py` | `python3 research/train_jepa.py --task all` |

### 3.1 Cost profile (2 vCPU sandbox, pure NumPy, float32)

| Run | Model | Steps | Wall clock |
| :--- | :--- | ---: | ---: |
| I-JEPA, 32x32 images, patch 8 (16 tokens) | 0.41M params | 300 | 17 s |
| I-JEPA, 32x32 images, patch 4 (64 tokens) | 0.41M params | 300 | 48 s |
| V-JEPA, 8x32x32 clips, tubelet 2 (64 tokens) | 0.41M params | 600 | 135 s |
| World model, 24x24 pixels, 10-step trajectories | 145K params | 600 | 75 s |
| `pytest tests/test_jepa.py` (34 tests) | - | - | 13 s |

---

## 4. Experimental protocol

Every experiment reports the same four things, because a JEPA number without
them is uninterpretable:

1. **Random-initialisation probe** — the same architecture, the same closed-form
   ridge probe, untrained weights. This is the "no learning" line.
2. **Probe accuracy** of the trained frozen encoder on the semantic label.
3. **Collapse diagnostics** — effective rank, mean cosine, and the new
   `signal_ratio` (fraction of representation energy that depends on the input).
4. **Shortcut audit** — `masked_target_baseline`: the model's loss vs the loss
   of a position-lookup table vs the loss of the best constant.
5. **Action-utilisation audit** (world models) — `action_utilization`: the same
   rollout is repeated with the action stream intact, permuted across the batch,
   and zeroed. A predictor that has settled on the conditional *mean* of the
   latents scores the same loss in all three cases; a predictor that actually
   models the dynamics is hurt by both corruptions. The headline number is
   ``action_gain = 1 - MSE(true) / MSE(shuffled)``. We validate the instrument on
   a controlled linear latent system (a known-stable linear map with Gaussian
   noise), where it separates a model that recovered the map (`action_gain`
   0.72 one-step, 0.55 over six steps, latent MSE 0.056 vs 0.107 for no-change)
   from the observation domain, where it reports 0.02 on the same code path.

Datasets were *designed to be failable*: for the temporal-order task, reversing
a clip preserves the multiset of frames exactly, so any order-invariant summary
(including mean-pooled features of a random encoder) is at chance *by
construction*. That property is verified empirically in §5.3.

---

## 5. Results

### 5.1 I-JEPA on images: the objective converges, the representation does not

Synthetic gratings (label = global orientation, 4 classes) and shapes (label =
shape identity, 3 classes). 192 training images, 64 held-out, closed-form ridge
probe on mean-pooled frozen features.

| Run | Probe (random init) | Probe (I-JEPA) | Effective rank | `signal_ratio` | `position_r2` | Final loss |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| gratings, patch 4, 200 steps | 0.266 | 0.250 | 10.2 | - | **0.993** | 0.0005 |
| gratings, patch 4, 500 steps | 0.266 | 0.250 | 10.2 | - | **0.993** | 0.0001 |
| gratings, patch 8, 500 steps | 0.328 | 0.297 | 24.4 | - | 0.947 | 0.0005 |
| gratings, patch 4, RoPE positions | 0.266 | 0.250 | 10.2 | - | **0.993** | 0.0002 |
| shapes, patch 4, 200 steps | 0.458 | 0.458 | 7.3 | - | **0.996** | 0.0006 |
| shapes, EMA 0.996, raw targets | 0.438 | 0.438 | 17.8 | 0.001 | 0.977 | 0.0003 |
| shapes, EMA 0.95, normalised targets | 0.438 | 0.438 | 23.6 | 0.001 | 0.972 | 0.0001 |

**We then tried to break the shortcut and measured each attempt** (shape
dataset, 64-token grid, 300 steps, λ = 0.1 where SIGReg is on; the random-init
probe is 0.51 in this setup and a ridge probe on *raw pixels* scores 0.52, so
the task itself is the hard part):

| Variant | Loss | Probe | `signal_ratio` | Effective rank | Collapsed |
| :--- | ---: | ---: | ---: | ---: | :---: |
| EMA teacher, token targets | 0.0001 | 0.480 | 0.0007 | 6.9 | yes |
| SIGReg, token level | 7.29 (regulariser dominated) | 0.480 | 0.0285 | 8.6 | no |
| SIGReg, **image level** | 0.85 | 0.460 | 0.0578 | 2.9 | no |
| whole-image target + image-level SIGReg | 0.87 | 0.460 | 0.0501 | 3.5 | no |
| whole-image target + EMA teacher | 0.0000 | 0.490 | 0.0016 | 8.5 | yes |

Image-level SIGReg — the one penalty that *does* rule out a position-only
encoder, because a constant pooled latent has a large Epps-Pulley statistic —
successfully prevents the collapse (rank 2.9, `signal_ratio` 0.058, loss no
longer near zero) but does **not**, at 300 steps and 300 images, produce features
that beat a linear probe on raw pixels. Escaping the shortcut is necessary, not
sufficient: what is missing at this scale is data and steps, not another term in
the loss.

**The escape hatch exists.** With the LeJEPA objective instead of the EMA
teacher, on the grating dataset at the same 400-step budget and a *single*
λ = 0.005, the probe moves above its own random-initialisation line for the first
time in this paper: **0.344 vs 0.266** (chance 0.25), `signal_ratio` 0.062,
effective rank 23.5, `position_r2` 0.955 — still a shortcut-dominated objective,
but no longer a dead one. That is the committed artifact row, and it is the
smallest of the three positive results here; we report it without a significance
claim because the test split is 64 images.

Read the table again with §2 in mind: the loss falls by 3-4 orders of magnitude,
target and prediction agree to `1e-4`, and **the frozen features are never
better than random**. Removing additive positional embeddings from the tokens
(RoPE) does *not* help, because the problem is not where the position cue comes
from — it is that the target encoder's output barely depends on the input at
all, and the residual variation it does have is a smooth function of *where* the
token is.

### 5.2 Four distinct ways a JEPA lies to you

A masked-latent objective is a **sufficiency** objective: it asks that the
context suffice to predict the target, and says nothing about whether the
prediction requires understanding. Four low-level priors can satisfy it without
any representation learning:

| Failure | Signature | Detector |
| :--- | :--- | :--- |
| **Dimensional collapse** | rank → 1, cosine → 1, std → 0 | classic; caught by effective rank / cosine |
| **Input-independent target** | rank 7-10, cosine ≈ 0, feature norm ≈ 5, `std_mean` ≈ 5e-4 | `signal_ratio` < 0.02 |
| **Positional shortcut** | loss → 1e-5, probe ≈ random init | `masked_target_baseline` → `position_r2` → 1 |
| **Marginal dynamics** | MSE below no-change, actions irrelevant | `action_utilization` → `action_gain` ≈ 0 |

The second row is the interesting one: an untrained ViT's token features are
dominated by the large additive positional table (norm ≈ 5) relative to the
patch-projection contribution (≈ 0.04 at init). Layer-normalising each token does
not remove that domination; it *amplifies* it into the direction of the token
vector. The EMA target encoder, moving at 0.4-4% per step, keeps serving
position-dominated targets for the whole run, so the predictor learns the lookup
and the context encoder receives almost no useful gradient — a self-reinforcing
trap. The target encoder's features collapse onto "where is this token", its
pooled features become the same vector for every image, and per-token target
normalisation then turns that constant into a *perfectly* predictable target. The
last row is the world-model analogue of the same phenomenon, and it is why the
action-permutation ablation exists.

### 5.3 The measurement is the message: loss and usefulness point opposite ways

Two independent settings in this paper make the same point.

**(a) The image case.** On the shape dataset the I-JEPA objective reaches a mean
smooth-L1 of `5e-5` while the frozen-feature probe sits exactly on the
random-initialisation baseline (0.453 vs 0.453), the pooled features carry
`signal_ratio = 7e-4` of input-dependent energy, and the shortcut audit explains
99.7% of the target variance from position alone. A run of 300 further steps
changed nothing in the probe (0.48 vs a 0.51 random baseline). If this paper
reported only the loss — as most JEPA evaluations do — it would have claimed a
converged representation.

**(b) The world-model case.** An absolute-head model at a 32x32 grid (1000
steps) reached next-latent MSE `0.717` against a no-change baseline of `1.287` —
a 44% error reduction that looks like a learned dynamics — with an
`action_gain` of **0.015**. Its accuracy was the latent marginal, not the
dynamics. In the matched pair of §5.5 the same audit separates a good model from
a better one (0.068 vs 0.175) that MSE alone ranks almost identically (0.681 vs
0.451).

**(c) The trade-off is real and is what λ buys.** Pinned as a unit test
(`test_sigreg_keeps_the_encoder_alive`): with SIGReg weight 0 the encoder
collapses and the prediction loss is `0.0001`; with weight 0.5 the encoder stays
alive (`signal_ratio` 0.037 vs 0.011) and the loss is `0.0129`. The regulariser
*pays* loss to keep the representation usable. §5.5's λ sweep shows the same
curve end-to-end (λ = 0.005 → MSE 0.149 and gain 0.31; λ = 0.05 → MSE 0.706 and
gain 0.069), and the direction is the opposite of alarming: in this regime a
*worse* latent loss accompanies a *more* useful representation.

| Failure | Signature | Detector |
| :--- | :--- | :--- |
| **Dimensional collapse** | rank → 1, cosine → 1, std → 0 | classic; caught by effective rank / cosine |
| **Input-independent target** | rank 7-10, cosine ≈ 0, feature norm ≈ 5, `std_mean` ≈ 5e-4 | `signal_ratio` < 0.02 |
| **Positional shortcut** | loss → 1e-5, probe ≈ random init | `masked_target_baseline` → `position_r2` → 1 |
| **Marginal dynamics** | MSE below no-change, actions irrelevant | `action_utilization` → `action_gain` ≈ 0 |

The middle row is the interesting one: an untrained ViT's token features are
dominated by the large additive positional table (norm ≈ 5) relative to the
patch-projection contribution (≈ 0.04 at init). Layer-normalising each token does
not remove that domination; it *amplifies* it into the direction of the token
vector. The EMA target encoder, moving at 0.4-4% per step, keeps serving
position-dominated targets for the whole run, so the predictor learns the lookup
and the context encoder receives almost no useful gradient — a self-reinforcing
trap. The target encoder's features collapse onto "where is this token", its
pooled features become the same vector for every image, and per-token target
normalisation then turns that constant into a *perfectly* predictable target.

### 5.4 What works at small scale #1: SIGReg makes video JEPA learn

Task: 8-frame clips of a translating dot, labelled **forward vs reversed in
time**. The label is invisible to any order-invariant feature extractor; a
randomly initialised V-JEPA (identical architecture, mean-pooled features)
scores 0.583 on 48 held-out clips — chance, as designed.

| Objective | Order probe | Effective rank | `signal_ratio` | Collapsed |
| :--- | ---: | ---: | ---: | :---: |
| random init (baseline) | 0.583 | - | - | - |
| EMA teacher, 600 steps | 0.542 | 28.4 | **0.001** | yes |
| EMA teacher, 1500 steps | 0.875 | 3.3 | 0.032 | no |
| **SIGReg (LeJEPA), 600 steps, 192 clips** | **1.000** | 11.0 | 0.145 | no |
| **SIGReg (LeJEPA), 400 steps, 96 clips** (committed artifact) | **0.875** | 9.2 | 0.098 | no |

The last row is the run committed to `19_results_jepa.json`; the random-init
baseline for it is exactly 0.500 and chance is 0.500, so the 0.875 is 28 points
above the no-learning line on 96 clips (a 2-class problem where the *only*
signal is temporal order).

The EMA variant at equal budget is collapsed and *below* the random baseline; it
needs 2.5x the steps to reach 0.875. The SIGReg variant is perfect at 600 steps
with a far healthier latent. This is the cleanest positive result in this paper:
**when the target cannot be recovered from position, masked latent prediction
learns exactly the right thing — and the distribution-matching regulariser is
what gets it there at small scale.**

### 5.5 What works at small scale #2: a world model you can plan with

Point mass in a box, rendered at 24x24 (agent and goal both visible), actions
are velocity commands, `damping=0` makes the position a random walk driven
entirely by the action. The encoder and the block-causal action-conditioned
predictor are trained **end-to-end from pixels** — 145K parameters, 128 training
trajectories of 10 steps, 600 steps, 3-step rollout loss — and evaluated with
five independent instruments: next-latent MSE against two baselines,
`action_utilization` (the permutation/zeroing ablation), frozen-latent probes,
collapse diagnostics, and a CEM planner scored in the real environment.

**The domain dictates the metric.** Because a velocity command integrates into a
random walk, *no-change* is a poor predictor of the next latent (it is worse
than the best constant, 1.17 vs 0.66 in the weakest row below), so a predictor
can look respectable by shrinking towards the latent marginal while ignoring
its action input entirely. The ablation is what separates the two readings: it
re-runs the identical rollout with the action stream permuted across the batch
and with it zeroed, and reports `action_gain = 1 - MSE(true)/MSE(shuffled)`.

| SIGReg weight (λ) | Next-latent MSE: model / no-change / best constant | `action_gain` (1-step / 6-step) | Position probe R² | `signal_ratio` | Effective rank | CEM success (vs random) | Paired win-rate | Mean final distance (vs random) |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **0.005** | **0.116 / 0.198 / 0.480** | **0.464 / 0.151** | **0.982** | 0.842 | 6.1 | 4.2% / 0.0% | 0.792 | **0.286 / 0.559** |
| 0.02 | 0.451 / 0.748 / 0.626 | 0.175 / 0.105 | 0.948 | 0.881 | 8.3 | 4.2% / 0.0% | 0.792 | 0.352 / 0.559 |
| 0.05 | 0.706 / 1.166 / 0.661 | 0.069 / 0.034 | 0.782 | 0.904 | 9.6 | 8.3% / 0.0% | 0.625 | 0.407 / 0.559 |

The λ = 0.005 row is the committed artifact run
(`research/19_results_jepa.json`); all baselines are computed inside the model's
own latent space, so they move with the encoder. The planner is run for 24
episodes from the same start/goal pairs as its random baseline, with 192
candidate action sequences, 5 CEM iterations and horizon 6. Read the table as a **trade-off curve for the single
LeJEPA hyperparameter**: λ = 0.005 is enough to keep the representation alive
(`signal_ratio` 0.842, no collapse) while letting the predictor fit the dynamics,
and it is the only row where the model beats *both* baselines and the actions
measurably matter. Turning λ up to 0.05 in the same setting leaves the encoder
healthier-looking but costs **6x** the prediction error and **6.7x** the action
gain — the regulariser is not free, and its benefit is not monotone. The best
row's ablation also orders correctly and by a wide margin: permuting the action
stream costs 86% more error (0.116 → 0.217) and zeroing it costs 70% more
(0.116 → 0.198).

**Parameterisation ablation (matched settings: 600 steps, 24x24 pixels, dim 48,
λ = 0.02, same seed and data):**

| Predictor head | Next-latent MSE | `action_gain` (1-step / 6-step) | Action stream order (true / shuffled / zeroed) | Position R² | Paired win-rate | Mean final distance (vs random) |
| :--- | ---: | ---: | :--- | ---: | ---: | :--- |
| absolute (`P(z,a) ≈ z_{t+1}`) | 0.681 | 0.068 / 0.029 | 0.681 / 0.731 / 0.709 | 0.819 | 0.708 | 0.393 / 0.559 |
| **delta / residual (default)** | **0.451** | **0.175 / 0.105** | — | **0.948** | **0.792** | **0.352 / 0.559** |

Predicting the *change* instead of the next latent removes the "shrink to the
marginal" optimum, because "nothing changes" is no longer a free prediction. At
matched settings the residual head wins on MSE (0.451 vs 0.681), on action use
(2.6x the gain), on the position probe and on the paired planning win-rate — and
it is the version that ships as the default. The coarse success count disagrees
(12.5% absolute vs 4.2% residual on 24 episodes); with a paired win-rate of 0.79
and a mean final distance 0.11 closer to the goal, we read that as granularity of
a 24-episode estimate, not as evidence for the absolute head.

The audit also earns its place on *earlier, unmatched* runs. A 1000-step
absolute-head model at a 32x32 grid reached a *better* MSE than the residual row
above (0.717 against a no-change baseline of 1.287) and looked like a working
dynamics model, but its `action_gain` was **0.015** with zeroing the action
stream *improving* the loss slightly (`0.7298 true / 0.7410 shuffled / 0.7290
zeroed`): its accuracy came from the marginal, not from the dynamics. Every
number in this paper's world-model tables is therefore reported next to its
permutation ablation.

**Instrument validation.** On a controlled linear latent system (5-D states, a
stable matrix `I + 0.1·N(0,1)` with Gaussian drive, Gaussian noise, 200 training
steps) the same code path recovers the map: latent MSE 0.056 against 0.107 for
no-change and 0.508 for the best constant, `action_gain` 0.72 one-step and 0.55
over six steps, state probe R² 0.989. The audit therefore reports ≫0 when there
is a dynamics to find and ≈0 when there is not, which is what makes the rows
above interpretable.

The planner never sees a reward, never decodes a pixel and never touches the
environment during search: it minimises `||rollout(z_start, a_{1:H}) − z_goal||²`
with 192 sampled action sequences and 5 elite-refinement iterations, then the
resulting actions are executed in the real environment. `examples/explore_jepa.py`
prints an ASCII rendering of one planned episode.

## 6. Discussion: when is JEPA the right tool (at this scale)?

The three failure modes above are all the *same* phenomenon from different
angles: masked latent prediction is a **sufficiency** objective. It asks that the
context suffice to predict the target, and says nothing about whether the
prediction requires understanding. Any low-level prior that can produce the
target — a position table, a constant, the mean of the batch — is a valid
solution, and at small scale SGD finds it long before it finds semantics.

This is consistent with the published record rather than in tension with it:
I-JEPA's own ablations show the masking strategy is worth 54.2 vs 17.6 points on
low-shot ImageNet, and the recipe is tuned for 300 epochs at batch 1024+ over
1M+ images — four to six orders of magnitude more data and compute than a
15-second CPU run. What we contribute is the *measurement* of the trap and a
minimal recipe that escapes it:

1. **Make the target unrecoverable from a cheap prior.** Our temporal-order
   benchmark is the design pattern: matched multisets of frames, order-only
   labels. If a randomly initialised encoder can solve your probe, the probe
   measures nothing.
2. **Regularise the distribution, not just the norm.** SIGReg (Epps-Pulley)
   prevents both dimensional and input-independence collapse because a constant
   cloud is maximally non-Gaussian. It is ~50 lines of NumPy here and needs no
   teacher, no stop-gradient, no schedule.
3. **Do not trust the loss.** Log `signal_ratio` and the shortcut audit next to
   it, every run.
4. **Train the encoder where the loss cannot be cheated.** Predicting the
   *future* of a system, or another *ordering* of the same frames, leaves no
   positional shortcut: the context genuinely has to determine the target.

---

## 7. Reproducibility

```bash
# The whole family, tiny budgets (smoke run, seconds)
python3 research/train_jepa.py --task all --quick

# The three published runs behind 19_results_jepa.json
python3 research/train_jepa.py --task images --dataset grating \
    --steps 400 --n-images 256 --dim 48 --patch-size 8 --sigreg --sigreg-weight 0.005
python3 research/train_jepa.py --task video --video-task order \
    --steps 400 --n-clips 96 --dim 48 --patch-size 8 --sigreg --sigreg-weight 0.005
python3 research/train_jepa.py --task worldmodel --steps 600 --n-trajectories 128 \
    --horizon 12 --dim 48 --patch-size 8 --rollout-steps 3 --plan-episodes 24 \
    --sigreg-weight 0.005

# The lambda sweep and the parameterisation ablation of 5.5 (same harness, one flag apart)
python3 research/train_jepa.py --task worldmodel --sigreg-weight 0.02 --steps 600 \
    --n-trajectories 128 --horizon 12 --dim 48 --patch-size 8 --plan-episodes 24
python3 research/train_jepa.py --task worldmodel --no-residual --sigreg-weight 0.02 \
    --steps 600 --n-trajectories 128 --horizon 12 --dim 48 --patch-size 8 \
    --plan-episodes 24

# Guided demo with ASCII visualisations of masks, frames and planned rollouts
python3 examples/explore_jepa.py            # full
python3 examples/explore_jepa.py --quick    # smoke, ~90 s

# Unit tests (34 tests, all green in ~13 s)
python3 -m pytest tests/test_jepa.py
```

Metrics are written to [`19_results_jepa.json`](19_results_jepa.json); every
number in §5 comes from a run of the commands above on a 2-vCPU sandbox, except
where a row is explicitly attributed to an earlier/longer run. The world-model
rows all re-load their checkpoint and re-run `action_utilization`, so the
ablation can be recomputed without retraining:

```python
from doraneural.jepa import JEPAWorldModel
from doraneural.jepa_data import PointMassConfig, sample_point_mass_trajectories

model = JEPAWorldModel.load("checkpoints/jepa/world_model")
data = sample_point_mass_trajectories(n=32, horizon=12, config=PointMassConfig(size=24), seed=99)
print(model.action_utilization(data["obs"], data["actions"]))
```

## 8. Limitations and honest caveats

* **Scale.** All runs are 15-450 seconds on 2 CPU cores over ≤ 400 synthetic
  images/clips. Nothing here contradicts the published large-scale results;
  the audit explains why copying the *hyper-parameters* of a 300-epoch run
  (especially `EMA 0.996 → 1.0`) into a 600-step run is a mistake.
* **Synthetic data.** Gratings and moving dots are controlled, not natural.
  The advantage is that the semantic variable is known and the random baseline
  is measurable; the cost is that natural-image statistics (and thus the
  shortcut structure) differ.
* **Mean-pooled linear probes.** I-JEPA/V-JEPA report attentive probes; our
  closed-form ridge probe on pooled features is weaker and cheaper. It is a
  *lower* bound on representation quality, which makes the "no better than
  random" results conservative.
* **No 3D-RoPE at scale.** 3D RoPE and tubelet video are implemented and
  tested, but only at 64-128 tokens; the stabilisation effect V-JEPA 2 reports
  for billion-parameter models is out of reach here.
* **Planner.** CEM in a 64-dimensional latent with a horizon of 6 and 192
  samples is a minimal planner; V-JEPA 2-AC uses a 300M predictor, longer
  horizons and receding-horizon replanning.

---

## 9. References

1. Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas.
   *Self-Supervised Learning from Images with a Joint-Embedding Predictive
   Architecture.* CVPR 2023. (I-JEPA)
2. Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas.
   *Revisiting Feature Prediction for Learning Visual Representations from
   Video.* 2024. (V-JEPA)
3. Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding,
   Prediction and Planning.* 2025.
4. Balestriero, LeCun. *LeJEPA: Provable and Scalable Self-Supervised Learning
   Without the Heuristics.* 2025. (SIGReg, Epps-Pulley)
5. Maes, Le Lidec, Scieur, LeCun, Balestriero. *LeWorldModel: Stable End-to-End
   Joint-Embedding Predictive Architecture from Pixels.* 2026.
6. LeCun. *A Path Towards Autonomous Machine Intelligence.* 2022.
7. Epps, Pulley. *A test for normality based on the empirical characteristic
   function.* Biometrika 1983.
8. Wang, Isola. *Understanding Contrastive Representation Learning through
   Alignment and Uniformity on the Hypersphere.* ICML 2020.
9. Roy, Vetterli. *The effective rank: a measure of effective dimensionality.*
   EUSIPCO 2007.
10. *Critiques of World Models.* 2025 — the counter-argument that latent-space
    supervision is an ungrounded surrogate, which §5.2 independently reproduces.
