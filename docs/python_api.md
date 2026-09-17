# doraneural Python API Reference

`doraneural` provides both beginner one-liners and fully modular neural network components built strictly on NumPy.

---

## Table of Contents

- [Beginner Quickstart (`dn.create`, `dn.quick_train`)](#1-beginner-quickstart)
- [CSV Data Loader (`dn.load_csv`)](#2-csv-data-loader)
- [Standalone Python Exporter (`dn.export_to_standalone_python`)](#3-standalone-python-exporter)
- [ASCII Curves & Visuals (`dn.plot_history`, `dn.plot_ascii_curve`)](#4-ascii-curves--visuals)
- [Core Modular Components](#5-core-modular-components)
  - [Layers](#layers)
  - [Activations](#activations)
  - [Loss Functions](#loss-functions)
  - [Optimizers](#optimizers)
  - [Metrics](#metrics)
- [Model Serialization](#6-model-serialization)

---

## 1. Beginner Quickstart

### `dn.create`
Builds and compiles a `Sequential` network in a single line. Automatically assigns weight initializations (He/Xavier) and connects appropriate loss functions and output squashing.

```python
import doraneural as dn

# Binary Classification:
model_bin = dn.create(inputs=2, hidden=[16, 8], outputs=1, task="binary")

# Multiclass Classification:
model_multi = dn.create(inputs=4, hidden=[32, 16], outputs=3, task="multiclass")

# Continuous Regression:
model_reg = dn.create(inputs=5, hidden=[24, 12], outputs=1, task="regression")
```

### `dn.quick_train`
Inspects raw NumPy matrices, infers dimensions, creates a network, runs training, and returns `(model, history)`.

```python
import doraneural as dn

model, history = dn.quick_train(X, y, hidden=[32, 16], epochs=25)
```

---

## 2. CSV Data Loader

### `dn.load_csv`
Loads CSV files with zero external dependencies (no pandas required). Features:
- Auto-detects headers
- Converts categorical string labels to integers
- Applies standard z-score normalization ($Z = \frac{X - \mu}{\sigma}$)
- Handles missing or empty rows cleanly

```python
import doraneural as dn

X, y, meta = dn.load_csv(
    filepath="housing.csv",
    target_col="price",     # Column name or integer index (e.g. -1)
    normalize=True,         # Normalize feature columns
    has_header=True
)

print(f"Features: {meta['feature_names']}")
print(f"Target:   {meta['target_name']}")
print(f"Samples:  {meta['n_samples']}")
```

---

## 3. Standalone Python Exporter

### `dn.export_to_standalone_python`
Exports a trained model into a single `.py` script that runs on **any standard Python interpreter** without NumPy or any 3rd-party dependencies.

```python
import doraneural as dn

model = dn.create(inputs=4, hidden=[16, 8], outputs=3)
model.fit(X_train, y_train, epochs=15)

# Export standalone script
dn.export_to_standalone_python(model, output_path="predict.py")
```

You can now copy `predict.py` to microcontrollers, minimal Docker containers, or shared servers and run:
```bash
python3 predict.py 5.1 3.5 1.4 0.2
```

---

## 4. ASCII Curves & Visuals

Visualize your learning curve directly in the console:

```python
import doraneural as dn

# Plot from model History
print(dn.plot_history(history))

# Or plot any list of numbers directly
print(dn.plot_ascii_curve([1.0, 0.7, 0.4, 0.2, 0.05], title="Loss Drop"))
```

---

## 5. Zero-Dependency Image Loader & Vision

`doraneural` loads, resizes, and processes real images (`.bmp`, `.ppm`, `.pgm`) without requiring Pillow (PIL) or OpenCV:

### Load Image Folders (Cats vs Dogs)

```python
import doraneural as dn

# Scans dataset/cats and dataset/dogs, resizes to 24x24, and normalizes
X, y, class_names = dn.load_image_dataset(
    "path/to/dataset",
    target_size=(24, 24),
    grayscale=True
)
print(f"Loaded {len(X)} images of {class_names}")
```

### Render Images as Terminal ASCII Art

```python
import doraneural as dn

# Print visual grayscale ASCII representation directly in terminal
print(dn.render_image_ascii(X[0, 0], width=20, height=10))
```

### Save and Read BMP Files

```python
import doraneural as dn

# Read standard uncompressed 24-bit or 8-bit BMP
img_array = dn.read_bmp("cat.bmp")

# Write array back to BMP
dn.write_bmp("saved_image.bmp", img_array)
```

---

## 6. Core Modular Components

### Layers
- `Dense(in_features, out_features, weight_init="he")`: Fully connected layer.
- `Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0)`: 2D Spatial Convolution with vectorized im2col.
- `MaxPool2D(pool_size=2, stride=2)`: 2D Spatial Max Pooling.
- `Flatten()`: Flattens N-D tensors into 2D batch matrices.
- `LayerNorm(normalized_shape, eps=1e-5)`: Layer Normalization.
- `Dropout(drop_rate=0.5)`: Inverted dropout with train/eval switching.

### Activations
- `ReLU()`: Rectified Linear Unit ($\max(0, x)$).
- `Sigmoid()`: Numerically stable logistic sigmoid ($[1 + e^{-x}]^{-1}$).
- `Softmax()`: Numerically stable exponent-shifted Softmax.

### Loss Functions
- `BinaryCrossEntropy()`: Binary classification loss.
- `CategoricalCrossEntropy()`: Multi-class cross entropy.
- `MeanSquaredError()` (alias `MSELoss`): Continuous regression loss.

### Optimizers
- `SGD(lr=0.01, momentum=0.9)`: Stochastic Gradient Descent with velocity momentum.
- `Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)`: Adaptive Moment Estimation with bias correction.
- `RMSprop(lr=0.001, alpha=0.9, eps=1e-8)`: Root Mean Square Propagation.

### Metrics
- `Accuracy()`: Binary and multiclass classification accuracy.
- `MSE()`: Mean Squared Error.
- `MAE()`: Mean Absolute Error.

---

## 7. Model Serialization

Save and reload architectures and weights:

```python
import doraneural as dn

# Save JSON architecture + compressed NPZ weights
dn.save_model(model, "my_model")

# Reload fully restored model
loaded_model = dn.load_model("my_model")
```
