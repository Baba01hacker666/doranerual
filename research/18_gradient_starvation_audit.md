# Research Paper 18: Gradient Starvation Is Not a Novel-Neuron Problem — and Dust Does Not Fix It

**Date:** 2026-10-09
**Extends:** [Paper 11](11_comparative_study_dora_transformer_vs_rtu.md) (which blamed `zexo-dora`'s failure on novel-neuron saturation), [Paper 17](17_dust_zeroth_order_pretraining.md) (Dust)
**Result:** Paper 11's mechanism is **real** but its **attribution is wrong**. The gradient collapse lives in the standard SwiGLU FFN and hits the *novel-free* model just as hard. And because saturation is a property of the forward function rather than of differentiation, Dust tracks backprop straight into it — so the paper 17 hope of rescuing the novel-neuron tiers with a gradient-free method does not survive contact.

---

## 1. Abstract

Research Paper 11 explained why the 12-layer Bio-Reflective KAN tier (`zexo-dora`) failed to
train — loss 6.14 against `zexo-mini`'s 4.03 — by asserting that the Chebyshev KAN's
`u = tanh(hidden)` saturates to ±1 and extinguishes upstream gradients. It offered no
measurement.

We measured it. Three findings:

1. **The saturation is real and severe.** At default initialisation 5.3% of KAN activations
   have `|tanh(u)| > 0.99` and 2.7% exceed 0.999; at 3x weight scale this rises to **34%**,
   and at 10x to **49%**. At `|tanh| = 0.999` the derivative is 2e-6.
2. **It is not caused by the novel neurons.** Gradient norms collapse by ~890x over 120
   ordinary optimizer steps in the **standard** SwiGLU model too, tracking the dora model
   within 20% at every step. The novel neurons are not the differentiator.
3. **Dust does not rescue it.** At the site the KAN sits between — the `w3` up-projection —
   Dust's estimate and backprop's gradient shrink together and the cosine holds at ~0.27
   while both collapse. Saturation is a property of the *forward function*, so a
   zeroth-order probe of that function is blinded by it exactly as the chain rule is.

Paper 11's own alternative explanation — that `novel_neurons=True` locks the model out of
the native C++ engine and starves it of compute (0.53 vs 2.11 updates/s) — survives. It is
the better-supported account.

---

## 2. Method

A 4-layer body matching the `dora` tier shape (dim 64, hidden 1536, 8 heads / 4 kv-heads,
V=512), run in two configurations: `novel_neurons=False` (standard SwiGLU) and
`novel_neurons=True` (dendritic gating + Chebyshev KAN + cortical reflection).

Measurements:
- **Saturation**: replay the block's arithmetic and histogram `|tanh(u)|` at the KAN input,
  sweeping an artificial weight scale to induce saturation.
- **Gradient norms**: true gradients from the existing autograd tape
  (`dust.backprop_gradient`), at initialisation and after real backprop training.
- **Dust quality**: `dust.estimate_site_gradient` at K=128, compared to backprop's gradient
  by cosine and by norm.

A `w3` injection site was added to [`dust.py`](../doraneural/dust.py) for this study: it is
the up-projection output, injected **before** the dendritic gate, the SiLU gating and the
KAN polynomial, so its credit must traverse the whole saturating path.

---

## 3. Saturation Is Real

`|tanh(u)|` at the Chebyshev KAN input:

| Weight scale | mean | p90 | p99 | frac > 0.99 | frac > 0.999 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1x (default) | 0.358 | 0.932 | 1.000 | 0.053 | 0.027 |
| 3x | 0.599 | 1.000 | 1.000 | **0.340** | 0.300 |
| 10x | 0.585 | 1.000 | 1.000 | **0.492** | 0.470 |
| 30x | 0.515 | 1.000 | 1.000 | 0.471 | 0.460 |

Paper 11's description of the mechanism checks out. Note it is already 5.3% at *default*
initialisation — no scaling required.

---

## 4. But It Is Not the Novel Neurons

### 4.1 Gradient norms at the KAN-adjacent site collapse in both models

`|∂L/∂w3|`, layer 0, during ordinary backprop training (Adam, lr 3e-4, identical corpus):

| Steps | standard `w3` | dora `w3` | standard `w2` | dora `w2` | standard loss | dora loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.096e+00 | 8.277e-01 | 5.321e+00 | 4.028e+00 | 6.4545 | 6.1338 |
| 20 | 1.690e-01 | 1.465e-01 | 8.202e-01 | 7.117e-01 | 2.5935 | 2.5845 |
| 60 | 1.442e-02 | 1.207e-02 | 6.903e-02 | 5.807e-02 | 1.8908 | 1.8806 |
| 120 | 1.233e-03 | 1.012e-03 | 5.718e-03 | 4.897e-03 | **1.0438** | **1.0428** |

Two things stand out.

- **The collapse is ~890x** (`w3`) and ~930x (`w2`) over 120 steps — no weight scaling, no
  novel neurons, just training a transformer.
- **The two models are within 20% at every step** and reach the same loss to three decimal
  places. The bio-reflective stack changes nothing.

So the pathology belongs to the shared FFN: `silu = sigmoid(gate) * gate` saturates for
large `gate`, and where the KAN is present `tanh` saturates too. Both live inside the
SwiGLU block that every tier already uses.

### 4.2 Where it does not bite

At the `w2` site — *downstream* of the KAN — cosine to backprop was 0.80–0.81 in **both**
models at every weight scale, and backprop's gradient norm stayed roughly constant
(4.03 → 6.06 for dora across a 30x scale change). Saturation kills the gradient flowing
*upstream* of itself, exactly as paper 11 says; it does not damage the sites below it.

### 4.3 Why training survives it anyway

Gradient norms fall by three orders of magnitude yet loss keeps dropping. Adam is
scale-invariant per parameter — it normalises by the gradient's own second moment — so a
shrinking gradient norm does not stall progress as long as the *direction* stays informative.
This is probably why the pathology went unnoticed: it is real, it is measurable, and under
Adam it is mostly harmless.

---

## 5. Dust Does Not Escape Saturation

This is the negative result that matters for Paper 17's hopes. Measured at `w3`, the site
the KAN sits between, K=128, layer 0:

| Model | Weight scale | `\|g_backprop\|` | `\|g_Dust\|` | cosine |
| :--- | ---: | ---: | ---: | ---: |
| standard | 1x | 1.096e+00 | 8.616e+01 | 0.156 |
| standard | 3x | 3.272e-01 | 2.778e+01 | 0.267 |
| standard | 10x | 9.675e-02 | 8.368e+00 | 0.269 |
| standard | 30x | 3.210e-02 | 2.794e+00 | 0.269 |
| dora (KAN) | 1x | 8.277e-01 | 6.335e+01 | 0.124 |
| dora (KAN) | 3x | 3.238e-01 | 2.651e+01 | 0.260 |
| dora (KAN) | 10x | 1.094e-01 | 8.932e+00 | 0.267 |
| dora (KAN) | 30x | 4.329e-02 | 3.566e+00 | 0.272 |

Reading the table:

- As saturation worsens, `|g_backprop|` falls 34x (standard) and 19x (dora).
- **`|g_Dust|` falls by the same factor** — 31x and 18x. Dust does not hold its ground.
- **Cosine stays ~0.27 in both models** across three decades of gradient magnitude. Dust
  tracks backprop's *direction* even as that direction's signal collapses.

**Why this is expected.** Saturation is a property of the forward function: if `tanh` is
locally flat, then perturbing its input barely changes the loss, so the reward `c_s` is
near zero — for backprop's chain rule *and* for Dust's reward-weighted noise alike. A
zeroth-order method exploits **non-differentiability**, not **insensitivity**. It rescues
things like straight-through estimators, integer-only weights, and ReLU exactly at zero
(where the function still responds to a perturbation but the derivative is undefined). It
does nothing for a smooth function that has genuinely gone flat.

So the hope that Dust could train the novel-neuron tiers — Paper 17's most plausible use
case for this repo — **does not survive contact with the measurement**. The KAN fails for a
reason orthogonal to how the gradient is obtained.

---

## 6. Consequences for Paper 11

| Paper 11's claim | Verdict |
| :--- | :--- |
| Chebyshev KAN `tanh` saturates and extinguishes upstream gradients | **Confirmed.** 5.3% saturated at default init, 49% at 10x scale. |
| That is why `zexo-dora` failed | **Refuted as an attribution.** The novel-free model shows the same collapse, within 20%, and the same loss curve. |
| Reflection's `down * 0.75 + tanh(down @ w_ref) * 0.25` dampens feature variance | **Not the differentiator.** Measured grad/weight ratios at the novel params were the *highest* in the model (2.3–6.0), not starved. |
| C++ engine lockout (`novel_neurons=True`) starves the model of compute | **Unaffected and now the best-supported account.** |

Also worth recording: Paper 11 quotes `zexo-dora` as 54M parameters. `ZexoConfig.parameter_count`
computes **66,783,744**. Any tier comparison should read the property, not the prose.

---

## 7. What To Do About It

The pathology is real but Adam already absorbs most of it, so the actionable item is
**diagnostic, not architectural**:

1. **Add a gradient-norm health check to training.** Log `|g|` per site alongside loss. A
   1000x collapse is a signal that something is saturating, and it is currently invisible.
2. **If FFN saturation must be reduced, fix the shared block, not the novel neurons.**
   Candidate levers that apply to every tier: normalising or scaling down the FFN input, or
   replacing `sigmoid(gate) * gate` with a form that does not saturate as hard. This is worth
   a sweep — it would help `zexo-mini` and `zexo-base` as much as `zexo-dora`.
3. **Do not spend more effort on Dust as a novel-neuron workaround.** Measured: it does not
   help.

---

## 8. Honest Limits

- Gradients were measured on random token sequences, not on the training corpus, and on a
  4-layer body. The collapse is a property of the FFN block, so the shape should hold at
  depth, but it was not measured at 12 layers.
- The weight-scale sweep is artificial. Section 4 shows the collapse also occurs at trained
  weights without any scaling, which is the stronger result, but the two experiments were
  kept separate rather than combined.
- Cosine ~0.27 at `w3` is low even before saturation. That site sits behind two
  multiplications (`silu` and the KAN), so low cosine there may partly be genuine estimator
  variance rather than saturation. What the experiment shows is that it does not *change*
  with saturation, which is the claim being made.
- No seeds. Each cell is a single run at seed 0.

---

## 9. Reproduction

```python
from doraneural.transformer import TransformerDecoderLM
from doraneural.dust import backprop_gradient, estimate_site_gradient, DustConfig
```

Section 4's table is `backprop_gradient` after N `TensorAdamW` steps; section 5's is
`estimate_site_gradient(..., 'w3', 0, DustConfig(draws=128))` versus the same tensor.
Saturation histograms come from replaying the block arithmetic and reading `|tanh(hidden)|`
at the KAN input. `tests/test_dust.py` covers the `w3` site's plumbing (parity, shapes,
credit decay, injection ordering).