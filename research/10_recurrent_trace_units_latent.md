# 🧠 Research Paper 10: Recurrent Trace Units (RTU) & Byte-Level Latent Prediction

## 1. Abstract & Motivation
Classical Large Language Models (LLMs) rely on two fundamental architectural assumptions:
1. **Large Subword Vocabularies ($V \approx 32{,}000 - 128{,}000$):** In small models (e.g. 1M to 20M parameters), token embedding and unembedding matrices consume 70% to 90% of the entire parameter budget, starving the hidden layers of representational capacity.
2. **Quadratic Attention & Unbounded KV Caches:** Standard Transformers have an $O(N^2)$ sequence compute complexity and $O(N)$ inference memory footprint, making streaming real-time learning and infinite-context processing memory-prohibitive.

Inspired by novel connectionist explorations such as [`test-model-thing`](https://github.com/jrz97619761/test-model-thing/), this paper formulates **Recurrent Trace Units (RTUs)**:
- **Tokenizer-Free Byte-Level Modeling ($V = 256$):** Raw UTF-8 bytes eliminate out-of-vocabulary (OOV) tokens and reduce embedding parameters by $>95\%$, allocating parameter budget directly into sequence dynamics.
- **$O(1)$ State Memory with Learned Decay Traces:** Linear state accumulation with per-channel learned decay allows infinite continuous context without KV cache growth.
- **Online Real-Time Recurrent Learning (RTRL) Eligibility Traces:** Forward-mode analytical gradient propagation eliminates the need for Backpropagation Through Time (BPTT) memory buffers.
- **JEPA-Style Multi-Objective Latent Prediction:** Jointly training next-byte prediction and next-latent state prediction with VICReg variance regularization prevents representation collapse and builds robust world representations.

---

## 2. Tokenizer-Free Byte Architecture ($V = 256$)

### Parameter Efficiency Comparison
In a conventional transformer with hidden dimension $D = 512$:
- Subword Tokenizer ($V = 32{,}000$):
  $$P_{\text{embed}} = 32{,}000 \times 512 = 16{,}384{,}000 \text{ parameters}$$
- Raw Byte Vocabulary ($V = 256$):
  $$P_{\text{embed}} = 256 \times 512 = 131{,}072 \text{ parameters}$$

**Savings:** A byte-level vocabulary frees up **16.25 million parameters** (a 99.2% reduction in embedding size), allowing compact edge models to allocate virtually their entire parameter capacity to deep non-linear recurrence, associative memory, and multi-layer abstraction.

```
Classical Small LLM (18M Params):
[ 32k Vocab Embeddings: 16.4M params (91%) ] [ 2 Tiny Layers: 1.6M params (9%) ]

RTU Byte Model (18M Params):
[ 256 Bytes: 0.13M (0.7%) ] [ Deep Recurrent Trace Engine: 17.87M params (99.3%) ]
```

---

## 3. Mathematical Formulation of Recurrent Trace Units

### Channel Decay & State Accumulation
Each layer $l$ maintains an internal continuous hidden state $s_t^{(l)} \in \mathbb{R}^D$.
The decay vector $d \in (0, 1)^D$ is parameterized via learned unconstrained logits $w_{\text{decay}}$:
$$d = \sigma(w_{\text{decay}}) = \frac{1}{1 + e^{-w_{\text{decay}}}}$$

At each time step $t$, the state is updated linearly:
$$s_t = d \odot s_{t-1} + x_t$$
where $\odot$ denotes element-wise multiplication and $x_t$ is the input vector (token embedding for layer 0, or previous layer output).

### Non-Linear Readout & Layer Normalization
To prevent linear state explosion and provide non-linear capacity, the state is normalized and projected:
$$\mu_t = \frac{1}{D} \sum_{i=1}^D s_{t, i}, \quad \sigma_t^2 = \frac{1}{D} \sum_{i=1}^D (s_{t, i} - \mu_t)^2$$
$$\hat{s}_t = \frac{s_t - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}}$$
$$h_t = \gamma \odot \hat{s}_t + \beta$$
$$y_t = \text{GELU}(h_t W + b)$$

Because the state transition $s_t = d \odot s_{t-1} + x_t$ is linear, long-term memory does not suffer from vanishing or exploding non-linear Jacobians across long sequence contexts.

---

## 4. Online RTRL & Forward-Mode Eligibility Traces

### Eliminating BPTT Memory Buffers
Classical RNN training uses Backpropagation Through Time (BPTT), which requires storing activations for all $T$ steps in memory:
$$\text{Memory}_{\text{BPTT}} = O(T \cdot L \cdot D)$$
For streaming audio, robotics, or infinite-context text, BPTT must truncate context windows (TBPTT), destroying dependencies beyond the horizon.

In RTUs, gradients with respect to the input embedding $E$ and decay logits $w_{\text{decay}}$ can be accumulated forward in time:
$$\frac{\partial s_t}{\partial x_{\tau}} = \prod_{k=\tau+1}^t d = d^{t - \tau}$$
$$\text{embed\_trace}_t = d \odot \text{embed\_trace}_{t-1} + 1$$

Similarly, for the decay parameter $w_{\text{decay}}$:
$$\frac{\partial d}{\partial w_{\text{decay}}} = d \odot (1 - d)$$
$$\text{decay\_trace}_t = d \odot \text{decay\_trace}_{t-1} + s_{t-1} \odot d \odot (1 - d)$$

### Constant $O(1)$ Online Step Update
Given upstream gradient $\delta_t = \frac{\partial L}{\partial s_t}$ computed at the current step:
$$\nabla_E L = \delta_t \odot \text{embed\_trace}_t$$
$$\nabla_{w_{\text{decay}}} L = \delta_t \odot \text{decay\_trace}_t$$

Memory overhead during training is strictly $O(L \cdot D)$—**completely invariant to sequence length $T$**. The network can train continuously on multi-megabyte streams (such as raw Wikipedia dumps) without ever running out of RAM.

---

## 5. Multi-Objective Latent Training (JEPA + VICReg)

Pure next-token cross-entropy forces the network to spend capacity memorizing surface token distributions rather than underlying concepts. To guide representations toward semantic world models, RTU implements a Joint Embedding Predictive Architecture (JEPA) objective.

### Latent Prediction Loss
Let $z_t = h_t^{(L)}$ be the top-layer latent vector at step $t$. A prediction head $W_{\text{pred}}$ predicts the future latent state $z_{t+1}$:
$$\hat{z}_{t+1} = z_t W_{\text{pred}}$$
$$L_{\text{JEPA}} = \|\hat{z}_{t+1} - \text{stop\_gradient}(z_{t+1})\|_2^2$$

### Preventing Collapse via VICReg Variance Penalty
If trained only with MSE, the network trivializes the loss by collapsing all latents to a constant vector: $z_t \to \mathbf{c}$.
To prevent representation collapse, we enforce an information-preserving variance constraint across the feature dimensions:
$$\text{std}(\hat{z}) = \sqrt{\frac{1}{D} \sum_{i=1}^D (\hat{z}_i - \bar{z})^2 + \epsilon}$$
$$L_{\text{var}} = \max(0, 1 - \text{std}(\hat{z}))$$

### Total Multi-Task Objective
$$L = L_{\text{xent}} + \lambda_{\text{latent}} L_{\text{JEPA}} + \lambda_{\text{var}} L_{\text{var}} + \lambda_{\text{stop}} L_{\text{stop}}$$
where:
- $L_{\text{xent}} = - \log P(x_{t+1} \mid z_t)$ ensures grammatical text generation.
- $L_{\text{JEPA}}$ aligns internal representations with future dynamics.
- $L_{\text{var}}$ guarantees feature diversity and non-collapse.
- $L_{\text{stop}}$ predicts sequence termination boundaries.

---

## 6. Continuous Wikipedia Pretraining Strategy

In reference to [`test-model-thing/issues/9`](https://github.com/jrz97619761/test-model-thing/issues/9), training byte-level recurrent models on encyclopedic corpora (e.g. Simple English Wikipedia) provides an ideal testbed:
1. **High Information Density**: Encyclopedic definitions, facts, and relational structure without conversational noise.
2. **UTF-8 Byte Stream Processing**: Articles can be fed as continuous UTF-8 byte streams. The model automatically learns character spellings, morphemes, words, and multi-word semantic bindings directly from raw bytes.
3. **State Persistence Across Articles**: The recurrent trace state $s_t$ can be preserved across related articles (e.g. "Physics" $\to$ "Quantum Mechanics"), enabling continual in-weights associative recall.

---

## 7. Sandbox Implementation & Verification

The complete reference implementation is maintained in the research sandbox:
- **Model Definition:** [`research/rtu_sandbox/rtu.py`](file:///root/doranerual/research/rtu_sandbox/rtu.py)
  - `RTUConfig`: Hyperparameter specification (dimensions, layers, learning rates, loss weights).
  - `RTULayerState`: Persistent linear state, eligibility traces, and LayerNorm cache.
  - `RTULanguageModel`: Full forward pass, real-time RTRL training loop, byte-level generation, and `.npz` serialization.
- **Verification Suite:** [`research/rtu_sandbox/verify_rtu.py`](file:///root/doranerual/research/rtu_sandbox/verify_rtu.py)
  - Unit tests for exact parameter counts, recurrence state propagation, loss convergence on sequence fitting, streaming generation, and state round-trip serialization.
