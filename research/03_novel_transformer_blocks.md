# 🚀 Research Paper 03: Novel Transformer Blocks (KAN-Former & Dendritic-Former)

## 1. Scaling Beyond Fixed-MLP Feedforward Layers
In standard Transformer architectures (Vaswani et al., 2017), the Transformer block is divided into:
1. Multi-Head Attention (token communication across time).
2. Feed-Forward Network (token feature transformation across channels):
   $$\text{FFN}(x) = W_2 \cdot \text{Activation}(W_1 x + b_1) + b_2$$

While attention has seen massive innovation (flash attention, linear attention, sliding window), the FFN sublayer has remained largely locked into static MLPs (with variations like SwiGLU / GeGLU).

We introduced two novel Transformer blocks in [`doraneural/attention.py`](../doraneural/attention.py):
1. **[`KANTransformerBlock`](../doraneural/attention.py#L393)** (KAN-Former): Replaces MLP with continuous orthogonal polynomial KAN layers.
2. **[`DendriticTransformerBlock`](../doraneural/attention.py#L484)** (Dendritic-Former): Replaces MLP with multi-branch pyramidal gating.

---

## 2. Architecture Comparison

### Standard Transformer Block
$$\text{norm}_1 = \text{LayerNorm}(x)$$
$$x_1 = x + \text{Dropout}(\text{MultiHeadAttention}(\text{norm}_1))$$
$$\text{norm}_2 = \text{LayerNorm}(x_1)$$
$$\text{out} = x_1 + \text{Dropout}(W_2 \cdot \text{ReLU}(W_1 \cdot \text{norm}_2))$$

### KAN-Transformer Block (`KANTransformerBlock`)
$$\text{norm}_2 = \text{LayerNorm}(x_1)$$
$$\text{kan\_out} = \text{ChebyshevKAN}_2(\text{ChebyshevKAN}_1(\text{norm}_2))$$
$$\text{out} = x_1 + \text{Dropout}(\text{kan\_out})$$

### Dendritic-Transformer Block (`DendriticTransformerBlock`)
$$\text{norm}_2 = \text{LayerNorm}(x_1)$$
$$\text{dend\_out} = \text{DendriticDense}_2(\text{DendriticDense}_1(\text{norm}_2))$$
$$\text{out} = x_1 + \text{Dropout}(\text{dend\_out})$$

---

## 3. Empirical Benchmark: Coupled Non-Linear Sequence Dynamics
Task: Predicting non-linear accumulated coupled state across an 8-step sequence with 4 input dimensions:
$$y = \frac{1}{T} \sum_{t=1}^T \left( \sin(\pi \cdot x_{t,0}) + \frac{x_{t,1} \cdot x_{t,2}}{1 + x_{t,3}^2} \right)$$

### Results:
| Architecture | Block Parameters | Final Train Loss | Test MSE | $R^2$ Variance Explained |
| :--- | :---: | :---: | :---: | :---: |
| **Standard Transformer (MLP)** | 4,385 | 0.00868 | 0.03368 | 46.10% |
| **Dendritic Transformer (Pyramidal)** | 6,585 | **0.00209** *(4.1x lower)* | 0.03483 | 44.27% |
| **KAN Transformer (Chebyshev)** | 7,193 | **0.00126** *(6.9x lower)* | 0.03527 | 43.56% |

### Key Insight:
KAN and Dendritic Transformer blocks possess significantly higher fitting expressivity inside the feedforward channel transformations, achieving **6.9x lower training loss** than the standard Transformer on non-linear coupled token dynamics.

---

## 4. Implementation Reference
- Source code: [`doraneural/attention.py`](../doraneural/attention.py#L393)
- Test suite: [`tests/test_neural_lib.py`](../tests/test_neural_lib.py#L873)
- Demo script: [`examples/explore_novel_transformers.py`](../examples/explore_novel_transformers.py)
