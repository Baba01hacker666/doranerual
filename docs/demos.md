# Interactive Demos & Examples

`doraneural` includes built-in interactive benchmarks and terminal visualizations.

---

## 1. Interactive Digit Generator & Classifier

Generate synthetic handwritten digits (0–9), introduce Gaussian noise, inspect clean vs. noised ASCII renderings, and watch the network predict in real-time.

```bash
doraneural demo interactive
```

Preview:
```text
Visual Comparison (Clean vs Noised):
  [ Clean Base Digit ]              [ Noised Input (σ=0.28) ]
  ┌────────────────┐          ┌────────────────┐
  │                │          │▒▒▒▒▒▒··    ····│
  │      ████████  │          │      ██████··  │
  │      ██        │          │      ██    ····│
  │      ██████    │          │      ██████    │
  │            ██  │          │··    ··    ██  │
  │      ██████    │          │      ██████  ··│
  │                │          │  ▒▒▒▒      ··██│
  │                │          │  ··      ····▒▒│
  └────────────────┘          └────────────────┘

Network Prediction on Noised Input:
  ▶ Predicted Digit: 5
  ▶ Confidence:      77.98%
```

---

## 2. Other Included Demos

Run any demo directly using `doraneural demo <name>`:

| Demo Name | Command | Description |
|---|---|---|
| `interactive` | `doraneural demo interactive` | Interactive digit drawing with noise controls |
| `digits` | `doraneural demo digits` | Full 10-class handwritten digit classifier |
| `moons` | `doraneural demo moons` | Non-linear 2D two moons classification |
| `blobs` | `doraneural demo blobs` | 3-cluster Gaussian blob classification |
| `cnn` | `doraneural demo cnn` | End-to-end 2D Convolutional Neural Network |
