"""CSV dataset loader and preprocessor for custom user data.

Loads CSV files with zero third-party dependencies (no pandas required),
handles automatic string label encoding, feature normalization, and train/test splitting.
"""

import csv
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional, Any
import numpy as np


def load_csv(
    filepath: Union[str, Path],
    target_col: Union[int, str] = -1,
    normalize: bool = True,
    has_header: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a CSV file into NumPy feature and target matrices.

    Automatically detects column headers, encodes string categorical labels
    into integer classes, and normalizes feature columns.

    Args:
        filepath (Union[str, Path]): Path to CSV file.
        target_col (Union[int, str]): Target column index (e.g. -1 for last column)
            or header name (e.g. 'species', 'price').
        normalize (bool): Whether to normalize features via standard z-score.
        has_header (bool): Whether the first row contains column names.

    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
            - X: Feature matrix of shape (samples, features).
            - y: Target array of shape (samples,).
            - metadata: Dictionary with 'feature_names', 'label_names', 'norm_stats'.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]

    if not rows:
        raise ValueError(f"CSV file '{filepath}' is empty.")

    # 1. Parse header
    if has_header:
        headers = [h.strip() for h in rows[0]]
        data_rows = rows[1:]
    else:
        headers = [f"col_{i}" for i in range(len(rows[0]))]
        data_rows = rows

    # Resolve target column index
    if isinstance(target_col, str):
        if target_col not in headers:
            raise ValueError(f"Target column '{target_col}' not found in headers: {headers}")
        target_idx = headers.index(target_col)
    else:
        target_idx = target_col if target_col >= 0 else (len(headers) + target_col)

    feature_indices = [i for i in range(len(headers)) if i != target_idx]
    feature_names = [headers[i] for i in feature_indices]
    target_name = headers[target_idx]

    # 2. Extract features and targets
    X_raw = []
    y_raw = []
    for r_idx, row in enumerate(data_rows):
        # Extract features
        feat_vals = []
        for fi in feature_indices:
            try:
                feat_vals.append(float(row[fi].strip()))
            except ValueError:
                raise ValueError(
                    f"Non-numeric feature value '{row[fi]}' in row {r_idx + 1}, column '{headers[fi]}'."
                )
        X_raw.append(feat_vals)
        y_raw.append(row[target_idx].strip())

    X = np.array(X_raw, dtype=np.float32)

    # 3. Process targets (Check if string labels or numeric)
    label_map = {}
    label_names = []
    is_numeric = False
    try:
        y_numeric = np.array([float(val) for val in y_raw], dtype=np.float32)
        is_numeric = True
        y = y_numeric
    except ValueError:
        is_numeric = False

    if not is_numeric:
        # String categorical labels -> map to integers
        unique_labels = sorted(list(set(y_raw)))
        label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        label_names = unique_labels
        y = np.array([label_map[lbl] for lbl in y_raw], dtype=np.int64)

    # 4. Feature Normalization (Z-score)
    norm_stats = {}
    if normalize:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0.0] = 1.0  # Prevent division by zero
        X = (X - mean) / std
        norm_stats = {"mean": mean.tolist(), "std": std.tolist()}

    metadata = {
        "feature_names": feature_names,
        "target_name": target_name,
        "label_names": label_names,
        "label_map": label_map,
        "norm_stats": norm_stats,
        "n_samples": len(X),
        "n_features": len(feature_names),
    }

    return X, y, metadata


def create_sample_classification_csv(filepath: Union[str, Path] = "sample_iris.csv") -> Path:
    """Create a clean 3-class sample CSV file for instant practice."""
    path = Path(filepath)
    headers = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    # 6 typical representative samples
    rows = [
        [5.1, 3.5, 1.4, 0.2, "setosa"],
        [4.9, 3.0, 1.4, 0.2, "setosa"],
        [7.0, 3.2, 4.7, 1.4, "versicolor"],
        [6.4, 3.2, 4.5, 1.5, "versicolor"],
        [6.3, 3.3, 6.0, 2.5, "virginica"],
        [5.8, 2.7, 5.1, 1.9, "virginica"],
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def create_sample_regression_csv(filepath: Union[str, Path] = "sample_housing.csv") -> Path:
    """Create a sample continuous regression CSV file (predicting house prices)."""
    path = Path(filepath)
    headers = ["size_sqft", "bedrooms", "age_years", "price_k"]
    rows = [
        [1200, 2, 10, 250.0],
        [1500, 3, 5, 320.0],
        [850, 1, 20, 180.0],
        [2200, 4, 2, 450.0],
        [1800, 3, 15, 360.0],
        [2800, 5, 1, 580.0],
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    return path
