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

## 2. Cat vs Dog Computer Vision Benchmark

Demonstrates that real computer vision (Cat vs Dog) runs blazingly fast on CPU without GPUs, PyTorch, TensorFlow, or OpenCV.

```bash
doraneural demo catdog
```

Loads 24x24 BMP animal images, trains a 2-stage CNN (`Conv2D` -> `MaxPool2D` -> `Conv2D` -> `MaxPool2D` -> `Dense`), and prints ASCII visualizations of cats and dogs with real-time predictions:

```text
Test Sample #1 [Ground Truth: CATS]:
┌──────────────────────────────────────────┐
│           ▒▒████▓▓    ▓▓████▓▓           │
│           ░░░░▒▒▓▓▒▒▓▓▓▓▓▓░░░░           │
│             ▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓           │
│     ▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▓▓     │
│     ▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒     │
│               ░░▓▓▓▓▓▓▓▓▓▓░░             │
└──────────────────────────────────────────┘
  ▶ Prediction: 🐱 CATS
  ▶ Confidence: 98.89% (P(dog) = 0.0111)
```

---

## 3. Other Included Demos

Run any demo directly using `doraneural demo <name>`:

| Demo Name | Command | Description |
|---|---|---|
| `catdog` | `doraneural demo catdog` | Cat vs Dog CNN image classifier with ASCII visualization |
| `interactive` | `doraneural demo interactive` | Interactive digit drawing with noise controls |
| `digits` | `doraneural demo digits` | Full 10-class handwritten digit classifier |
| `moons` | `doraneural demo moons` | Non-linear 2D two moons classification |
| `blobs` | `doraneural demo blobs` | 3-cluster Gaussian blob classification |
| `cnn` | `doraneural demo cnn` | End-to-end 2D Convolutional Neural Network |
