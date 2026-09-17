"""Data utilities, dataset generation, and reproducibility helpers.

Contains lightweight helper routines for splitting data, encoding labels,
mini-batch generation, and generating synthetic classification benchmarks without
any external dependencies outside NumPy.
"""

import random
from typing import Tuple, Generator, Optional
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set global random seed across Python's random and NumPy for deterministic reproducibility.

    Args:
        seed (int): Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into random train and test subsets.

    Args:
        X (np.ndarray): Feature array of shape (N, features).
        y (np.ndarray): Target array of shape (N, ...).
        test_size (float): Proportion of the dataset to include in the test split (0.0 to 1.0).
        shuffle (bool): Whether to shuffle data before splitting.
        seed (Optional[int]): Random seed for reproducible splitting.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (X_train, X_test, y_train, y_test)
    """
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y)

    if len(X_arr) != len(y_arr):
        raise ValueError(
            f"Length mismatch: X has {len(X_arr)} samples, but y has {len(y_arr)} samples."
        )

    n_samples = len(X_arr)
    n_test = int(round(n_samples * test_size))
    n_train = n_samples - n_test

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


def one_hot_encode(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """Convert an array of class indices to a one-hot encoded matrix.

    Args:
        y (np.ndarray): 1D or 2D column array of class integer indices.
        num_classes (Optional[int]): Total number of distinct classes. If None,
            inferred from max(y) + 1.

    Returns:
        np.ndarray: One-hot encoded matrix of shape (N, num_classes) with float32 type.
    """
    labels = np.asarray(y).ravel().astype(np.int64)
    if len(labels) == 0:
        return np.zeros((0, num_classes or 0), dtype=np.float32)

    inferred_classes = int(np.max(labels)) + 1
    c = int(num_classes) if num_classes is not None else inferred_classes
    if c < inferred_classes:
        raise ValueError(
            f"num_classes={c} is smaller than highest label index ({inferred_classes - 1})"
        )

    one_hot = np.zeros((len(labels), c), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return one_hot


# Alias matching Keras API
to_categorical = one_hot_encode


def batch_iterator(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate mini-batches from feature and label arrays.

    Args:
        X (np.ndarray): Features array of shape (N, ...).
        y (np.ndarray): Targets array of shape (N, ...).
        batch_size (int): Size of each mini-batch.
        shuffle (bool): Whether to shuffle samples at start of iteration.
        seed (Optional[int]): Random seed for shuffling.

    Yields:
        Tuple[np.ndarray, np.ndarray]: Mini-batch slice (X_batch, y_batch).
    """
    n_samples = len(X)
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    for start_idx in range(0, n_samples, batch_size):
        batch_idx = indices[start_idx : start_idx + batch_size]
        yield X[batch_idx], y[batch_idx]


def make_moons(
    n_samples: int = 1000,
    noise: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two interleaving half circles (toy binary classification problem).

    Pure NumPy implementation of the popular 2D benchmark.

    Args:
        n_samples (int): Total number of points to generate.
        noise (float): Standard deviation of Gaussian noise added to the data.
        seed (Optional[int]): Random seed.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: 2D feature coordinates, shape (n_samples, 2).
            - y: Binary class targets {0, 1}, shape (n_samples, 1).
    """
    rng = np.random.default_rng(seed)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Upper moon (class 0)
    theta_out = np.linspace(0, np.pi, n_samples_out)
    x_out = np.stack([np.cos(theta_out), np.sin(theta_out)], axis=1)

    # Lower moon (class 1)
    theta_in = np.linspace(0, np.pi, n_samples_in)
    x_in = np.stack([1.0 - np.cos(theta_in), 1.0 - np.sin(theta_in) - 0.5], axis=1)

    X = np.vstack([x_out, x_in]).astype(np.float32)
    y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)]).reshape(-1, 1).astype(np.float32)

    if noise > 0.0:
        X += rng.normal(scale=noise, size=X.shape).astype(np.float32)

    # Shuffle
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def make_blobs(
    n_samples: int = 1000,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate isotropic Gaussian blobs for multi-class classification.

    Args:
        n_samples (int): Total number of points divided equally among clusters.
        n_features (int): Number of input features for each sample.
        centers (int): Number of cluster centers / classes.
        cluster_std (float): Standard deviation of the clusters.
        seed (Optional[int]): Random seed.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: Feature array of shape (n_samples, n_features).
            - y: Class label indices of shape (n_samples,).
    """
    rng = np.random.default_rng(seed)
    samples_per_center = n_samples // centers

    # Place cluster centers evenly along a circle or randomly
    center_coords = []
    angle_step = 2.0 * np.pi / centers
    radius = 3.5
    for i in range(centers):
        cx = radius * np.cos(i * angle_step)
        cy = radius * np.sin(i * angle_step)
        extra = [0.0] * (n_features - 2) if n_features > 2 else []
        center_coords.append([cx, cy] + extra)
    center_coords = np.array(center_coords, dtype=np.float32)

    X_list = []
    y_list = []
    for i, center in enumerate(center_coords):
        n = samples_per_center if i < centers - 1 else (n_samples - samples_per_center * (centers - 1))
        cluster_points = rng.normal(loc=center, scale=cluster_std, size=(n, n_features)).astype(np.float32)
        X_list.append(cluster_points)
        y_list.append(np.full((n,), fill_value=i, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def make_digits(
    n_samples: int = 1500,
    noise: float = 0.1,
    flatten: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 8x8 handwritten/printed digit images (digits 0 to 9).

    Pure NumPy generator producing 8x8 grayscale digit images with realistic
    stroke variations, random spatial translations, intensity jitter, and noise.
    Ideal for lightweight CPU benchmarks without external datasets.

    Args:
        n_samples (int): Total number of samples across digits 0-9.
        noise (float): Gaussian noise standard deviation.
        flatten (bool): If True, returns X with shape (n_samples, 64).
            If False, returns X with shape (n_samples, 1, 8, 8) suitable for Conv2D.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: Digit images, shape (n_samples, 64) or (n_samples, 1, 8, 8).
            - y: Integer class labels in range 0..9, shape (n_samples,).
    """
    templates = {
        0: [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]],
        1: [[0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 1, 1, 1]],
        2: [[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]],
        3: [[1, 1, 1, 0], [0, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0]],
        4: [[1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
        5: [[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0]],
        6: [[0, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]],
        7: [[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
        8: [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]],
        9: [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 1, 0]],
    }

    rng = np.random.default_rng(seed)
    samples_per_digit = n_samples // 10

    X_list = []
    y_list = []

    for digit in range(10):
        t = np.array(templates[digit], dtype=np.float32)
        th, tw = t.shape
        n_count = samples_per_digit if digit < 9 else (n_samples - samples_per_digit * 9)

        for _ in range(n_count):
            img = np.zeros((8, 8), dtype=np.float32)
            # Random positioning
            r = rng.integers(1, 8 - th)
            c = rng.integers(1, 8 - tw)
            intensity = rng.uniform(0.85, 1.15)
            img[r : r + th, c : c + tw] = t * intensity

            # Add small stroke variations
            if rng.random() > 0.5:
                # Add minor random stroke jitter
                jr = rng.integers(1, 7)
                jc = rng.integers(1, 7)
                img[jr, jc] = max(img[jr, jc], rng.uniform(0.5, 0.9))

            if noise > 0.0:
                img += rng.normal(0.0, noise, img.shape).astype(np.float32)

            img = np.clip(img, 0.0, 1.0)

            if flatten:
                X_list.append(img.reshape(-1))
            else:
                X_list.append(img.reshape(1, 8, 8))

            y_list.append(digit)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    perm = rng.permutation(len(X))
    return X[perm], y[perm]

