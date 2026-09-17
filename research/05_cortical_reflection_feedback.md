# 🔄 Research Paper 05: Cortical Reflection & Iterative Feedback Loops

## 1. Motivation: The Feedforward Limitation
Conventional deep learning architectures are strictly unidirectional feedforward directed acyclic graphs (DAGs):
$$x \longrightarrow h_1 \longrightarrow h_2 \longrightarrow \dots \longrightarrow y$$

In contrast, biological cortical sensory processing (e.g. human visual areas V1 $\leftrightarrow$ V2 $\leftrightarrow$ V4 $\leftrightarrow$ IT) involves **massive recurrent top-down feedback loops** where feedback axons equal or outnumber feedforward axons (Felleman & Van Essen, 1991; Rao & Ballard, 1999). 

Top-down feedback provides a predictive prior that reflects back to the earlier representation, resolving ambiguous, noisy, or occluded sensory signals through iterative relaxation.

## 2. Mathematical Formulation
[`ReflectiveDense`](file:///root/doranerual/doraneural/layers.py#L1101) models recurrent predictive coding within an individual layer over $K$ reflection passes:

For input $x^{(0)} \in \mathbb{R}^{B \times D_{\text{in}}}$ and iteration $k \in \{0, 1, \dots, K-1\}$:

1. **Feedforward Hypothesis:**
   $$h_{\text{raw}}^{(k)} = x^{(k)} W_{\text{fwd}} + b_{\text{fwd}}$$
   $$h^{(k)} = \text{SiLU}(h_{\text{raw}}^{(k)})$$
2. **Top-Down Reflection (for $k < K-1$):**
   $$r_{\text{raw}}^{(k)} = h^{(k)} W_{\text{ref}} + b_{\text{ref}}$$
   $$r^{(k)} = \tanh(r_{\text{raw}}^{(k)})$$
3. **Hypothesis-Driven Input Refinement:**
   $$x^{(k+1)} = x^{(0)} + \alpha \cdot r^{(k)}$$

The final emitted output is the refined state $h^{(K-1)}$.

Where:
- $W_{\text{fwd}} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ is the bottom-up feedforward projection.
- $W_{\text{ref}} \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$ is the top-down cortical reflection projection.
- $\alpha \in [0, 1]$ is the feedback blending rate (default: 0.4 - 0.5).
- $K \ge 1$ is the number of reflection steps (default: 2 or 3).

## 3. Exact Unrolled Backpropagation
Gradients unroll through the recurrence in reverse chronological order:
At step $k = K-1$: $\delta h^{(K-1)} = dL/dy$.
For $k = K-1, \dots, 0$:
1. $\delta h_{\text{raw}}^{(k)} = \delta h^{(k)} \odot \text{SiLU}'(h_{\text{raw}}^{(k)})$
2. $dW_{\text{fwd}} += (x^{(k)})^T \delta h_{\text{raw}}^{(k)}$
3. $\delta x^{(k)} = \delta h_{\text{raw}}^{(k)} W_{\text{fwd}}^T$
4. If $k > 0$:
   $$\delta x^{(0)} += \delta x^{(k)}$$
   $$\delta r^{(k-1)} = \alpha \cdot \delta x^{(k)}$$
   $$\delta r_{\text{raw}}^{(k-1)} = \delta r^{(k-1)} \odot (1 - (r^{(k-1)})^2)$$
   $$dW_{\text{ref}} += (h^{(k-1)})^T \delta r_{\text{raw}}^{(k-1)}$$
   $$\delta h^{(k-1)} = \delta r_{\text{raw}}^{(k-1)} W_{\text{ref}}^T$$
Total input gradient: $\frac{\partial L}{\partial x} = \delta x^{(0)} + \delta x^{(0)}|_{\text{accumulated}}$.

Verified via finite-difference gradient checks with relative error $< 10^{-4}$.

## 4. Empirical Benchmark: Noisy Ambiguity Resolution
- When inputs contain coupled non-linear noise ($\sigma = 0.15$), a single feedforward pass is forced to guess from the noisy input.
- Reflective feedback ($K=3$) iteratively relaxes the input representation against its own intermediate hypothesis, lowering error and enhancing signal clarity.

## 5. Implementation Reference
- Source code: [`doraneural/layers.py`](file:///root/doranerual/doraneural/layers.py#L1101)
- Test suite: [`tests/test_neural_lib.py`](file:///root/doranerual/tests/test_neural_lib.py#L465)
- Demo script: [`examples/explore_reflection_and_gating.py`](file:///root/doranerual/examples/explore_reflection_and_gating.py)
