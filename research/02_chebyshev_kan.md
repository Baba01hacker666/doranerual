# 📐 Research Paper 02: Chebyshev Kolmogorov-Arnold Networks (KAN)

## 1. Background & The Kolmogorov-Arnold Theorem
In 1957, mathematicians Andrey Kolmogorov and Vladimir Arnold proved that any continuous multi-variable function $f(x_1, \dots, x_n)$ on a bounded domain can be represented as a finite composition of continuous 1D functions and addition:
$$f(x_1, \dots, x_n) = \sum_{q=1}^{2n+1} \Phi_q \left( \sum_{p=1}^n \phi_{q, p}(x_p) \right)$$

In contrast to Multi-Layer Perceptrons (MLPs), which place fixed activation functions on nodes and learn linear weights on edges:
- **MLP:** $y = \sigma(W x)$ (Fixed node functions, scalar edge weights)
- **KAN:** $y = \sum_i \phi_i(x_i)$ (Linear node summation, **learnable continuous non-linear functions on edges**)

## 2. Orthogonal Chebyshev Basis Representation
Instead of expensive B-splines which require non-uniform grid lookups and GPU/CPU branch divergences, [`ChebyshevKAN`](file:///root/doranerual/doraneural/layers.py#L797) utilizes **orthogonal Chebyshev polynomials of the first kind** $T_k(u)$:

1. **Input Normalization to $[-1, 1]$:**
   $$u = \tanh(x)$$
2. **Chebyshev Polynomial Recurrence:**
   $$T_0(u) = 1, \quad T_1(u) = u$$
   $$T_k(u) = 2u T_{k-1}(u) - T_{k-2}(u) \quad \text{for } k \ge 2$$
   And their exact derivatives:
   $$T'_0(u) = 0, \quad T'_1(u) = 1$$
   $$T'_k(u) = 2 T_{k-1}(u) + 2u T'_{k-1}(u) - T'_{k-2}(u)$$
3. **Dual-Branch Edge Transformation:**
   $$y_j = \underbrace{\sum_i \text{SiLU}(x_i) W_{\text{base}, i, j}}_{\text{Base Linear Residual Branch}} + \underbrace{\sum_i \sum_{k=0}^K c_{i, j, k} T_k(\tanh(x_i))}_{\text{Chebyshev Polynomial Non-Linear Branch}} + \text{Bias}_j$$

## 3. The Spectral Bias Breakthrough
Standard neural networks suffer from the **F-Principle (Spectral Bias)**: gradient descent learns low-frequency components rapidly, but struggles to learn high-frequency oscillations without massive overparameterization.

### Empirical Benchmark: High-Frequency Wave Fitting
$$f(x) = \sin(3\pi x) + 0.5 \cos(9\pi x), \quad x \in [-1, 1]$$

Under matched parameter budgets (~170-205 parameters):

| Architecture | Parameters | Test MSE | Max $L_\infty$ Error | $R^2$ (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Classical MLP (Dense + SiLU)** | 205 | 0.12525 | 1.1636 | 79.88% |
| **Dendritic Net (DendriticDense)** | 83 | 0.15552 | 0.9129 | 75.02% |
| **Chebyshev KAN** | **171** | **0.00690** | **0.2243** | **98.89%** |

### Key Result:
Chebyshev KAN delivered an **18.2x reduction in Test MSE** and a **5.2x reduction in peak error** compared to the Classical MLP while using fewer parameters!

## 4. Implementation Reference
- Source code: [`doraneural/layers.py`](file:///root/doranerual/doraneural/layers.py#L797)
- Test suite: [`tests/test_neural_lib.py`](file:///root/doranerual/tests/test_neural_lib.py#L358)
- Demo script: [`examples/benchmark_neuron_arena.py`](file:///root/doranerual/examples/benchmark_neuron_arena.py)
