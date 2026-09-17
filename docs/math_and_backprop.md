# Under the Hood: Math & Backpropagation

`doraneural` is designed to be educational and mathematically transparent. All gradients are calculated analytically and verified with finite-difference numerical checks.

---

## 1. Dense Layer Forward & Backward

Given input batch $X \in \mathbb{R}^{N \times D_{in}}$, weights $W \in \mathbb{R}^{D_{in} \times D_{out}}$, and biases $b \in \mathbb{R}^{1 \times D_{out}}$:

### Forward Pass:
$$Z = X W + b$$

### Backward Pass:
Given incoming upstream gradient $\frac{\partial L}{\partial Z} \in \mathbb{R}^{N \times D_{out}}$:
- **Weight Gradient**: $\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Z}$
- **Bias Gradient**: $\frac{\partial L}{\partial b} = \sum_{i=1}^N \frac{\partial L}{\partial Z}_{i, :}$
- **Input Gradient (propagated downstream)**: $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \cdot W^T$

---

## 2. Activation Derivatives

### ReLU:
$$f(x) = \max(0, x)$$
$$\frac{df}{dx} = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}$$

### Sigmoid (Numerically Stable):
$$\sigma(x) = \frac{1}{1 + e^{-\text{clip}(x, -88, 88)}}$$
$$\frac{d\sigma}{dx} = \sigma(x) \cdot (1 - \sigma(x))$$

### Softmax (Vector-Jacobian Product):
$$p_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$
Combined with Categorical Cross-Entropy, the downstream gradient simplifies cleanly to:
$$\frac{\partial L}{\partial z} = \frac{p - y}{N}$$

---

## 3. Loss Functions

### Mean Squared Error (MSE):
$$L = \frac{1}{N} \sum_{i=1}^N (y_{\text{pred}, i} - y_{\text{true}, i})^2$$
$$\frac{\partial L}{\partial y_{\text{pred}}} = \frac{2}{N} (y_{\text{pred}} - y_{\text{true}})$$

### Binary Cross-Entropy (BCE):
$$L = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i + \epsilon) + (1 - y_i) \log(1 - \hat{y}_i + \epsilon) \right]$$
$$\frac{\partial L}{\partial \hat{y}} = \frac{1}{N} \left[ \frac{\hat{y} - y}{\hat{y}(1 - \hat{y}) + \epsilon} \right]$$

---

## 4. Weight Initializations

- **He (Kaiming) Normal**: Suitable for layers followed by ReLU:
  $$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{D_{in}}}\right)$$
- **Xavier (Glorot) Uniform**: Suitable for Sigmoid / Softmax output layers:
  $$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{D_{in} + D_{out}}}, \sqrt{\frac{6}{D_{in} + D_{out}}}\right)$$
