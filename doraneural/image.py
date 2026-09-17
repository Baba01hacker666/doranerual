"""Zero-dependency image file loader, preprocessor, and ASCII visualizer.

Loads, saves, and processes standard uncompressed BMP, PPM, and PGM image files
using pure Python standard library (`struct`, `os`, `pathlib`) and NumPy.
No Pillow (PIL), OpenCV, or external image libraries required.
"""

import os
import struct
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict, Any
import numpy as np


def read_bmp(filepath: Union[str, Path]) -> np.ndarray:
    """Read an uncompressed 24-bit RGB or 8-bit grayscale BMP image file.

    Args:
        filepath: Path to the .bmp file.

    Returns:
        np.ndarray: uint8 array of shape (height, width, 3) for RGB or (height, width) for grayscale.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"BMP image file not found: {filepath}")

    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 54 or data[:2] != b"BM":
        raise ValueError(f"File '{filepath}' is not a valid BMP image (missing 'BM' signature).")

    offset = struct.unpack_from("<I", data, 10)[0]
    dib_size = struct.unpack_from("<I", data, 14)[0]
    width = struct.unpack_from("<i", data, 18)[0]
    height = struct.unpack_from("<i", data, 22)[0]
    planes = struct.unpack_from("<H", data, 26)[0]
    bpp = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]

    if compression != 0:
        raise NotImplementedError(
            f"Only uncompressed BMP files (compression=0) are supported, got {compression}."
        )

    is_top_down = height < 0
    height = abs(height)

    if bpp == 24:
        row_bytes = width * 3
        padding = (4 - (row_bytes % 4)) % 4
        stride = row_bytes + padding

        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        for r in range(height):
            file_row = r if is_top_down else (height - 1 - r)
            start = offset + file_row * stride
            row_data = data[start : start + row_bytes]
            if len(row_data) < row_bytes:
                break
            bgr = np.frombuffer(row_data, dtype=np.uint8).reshape(width, 3)
            pixels[r, :, 0] = bgr[:, 2]  # Red
            pixels[r, :, 1] = bgr[:, 1]  # Green
            pixels[r, :, 2] = bgr[:, 0]  # Blue
        return pixels

    elif bpp == 8:
        # Grayscale or palette
        row_bytes = width
        padding = (4 - (row_bytes % 4)) % 4
        stride = row_bytes + padding

        # Color palette is between header (54) and offset
        palette = data[54:offset]
        pixels = np.zeros((height, width), dtype=np.uint8)
        for r in range(height):
            file_row = r if is_top_down else (height - 1 - r)
            start = offset + file_row * stride
            row_data = data[start : start + row_bytes]
            if len(row_data) < row_bytes:
                break
            pixels[r, :] = np.frombuffer(row_data, dtype=np.uint8)
        return pixels

    else:
        raise NotImplementedError(f"BMP with bit depth {bpp} is not supported. Please use 24-bit RGB or 8-bit BMP.")


def write_bmp(filepath: Union[str, Path], img: np.ndarray) -> Path:
    """Save an RGB or grayscale NumPy array as a standard 24-bit uncompressed BMP file.

    Args:
        filepath: Target destination file path.
        img: Array of shape (H, W, 3) or (H, W). Values should be uint8 (0-255) or float (0.0-1.0).

    Returns:
        Path: Path to written BMP file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        # Expand grayscale to 3 channels for universal 24-bit BMP compatibility
        arr = np.stack([arr, arr, arr], axis=-1)

    height, width, channels = arr.shape
    row_bytes = width * 3
    padding = (4 - (row_bytes % 4)) % 4
    image_size = (row_bytes + padding) * height
    file_size = 54 + image_size

    # 14-byte File Header + 40-byte DIB Header
    header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
    dib_header = struct.pack(
        "<IIIHHIIIIII",
        40,          # DIB Header size
        width,       # Width
        height,      # Height (positive = bottom-up)
        1,           # Planes
        24,          # Bits per pixel
        0,           # Compression (0 = uncompressed BI_RGB)
        image_size,  # Image payload size
        2835,        # Horiz resolution (pixels/meter ~ 72 DPI)
        2835,        # Vert resolution
        0,           # Colors in palette
        0,           # Important colors
    )

    pad_bytes = b"\x00" * padding
    with open(path, "wb") as f:
        f.write(header)
        f.write(dib_header)
        # BMP bottom-up: write rows from bottom to top
        for r in reversed(range(height)):
            row = arr[r]
            # Convert RGB to BGR
            bgr = np.empty_like(row)
            bgr[:, 0] = row[:, 2]
            bgr[:, 1] = row[:, 1]
            bgr[:, 2] = row[:, 0]
            f.write(bgr.tobytes())
            if padding > 0:
                f.write(pad_bytes)

    return path


def read_ppm(filepath: Union[str, Path]) -> np.ndarray:
    """Read a Netpbm PPM (P3/P6 RGB) or PGM (P2/P5 grayscale) image file.

    Args:
        filepath: Path to .ppm or .pgm file.

    Returns:
        np.ndarray: uint8 image array.
    """
    path = Path(filepath)
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic not in (b"P2", b"P3", b"P5", b"P6"):
            raise ValueError(f"Not a valid Netpbm image. Magic signature: {magic}")

        # Helper to get next non-comment token
        def get_token():
            while True:
                line = f.readline()
                if not line:
                    return None
                line = line.strip()
                if not line or line.startswith(b"#"):
                    continue
                for token in line.split():
                    yield token

        tokens = get_token()
        width = int(next(tokens))
        height = int(next(tokens))
        max_val = int(next(tokens))

        if magic == b"P6":
            # Binary RGB
            raw_bytes = f.read(width * height * 3)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, width, 3)
            return arr
        elif magic == b"P5":
            # Binary Grayscale
            raw_bytes = f.read(width * height)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, width)
            return arr
        elif magic == b"P3":
            # ASCII RGB
            nums = [int(t) for t in tokens]
            arr = np.array(nums, dtype=np.uint8).reshape(height, width, 3)
            return arr
        else:
            # ASCII Grayscale (P2)
            nums = [int(t) for t in tokens]
            arr = np.array(nums, dtype=np.uint8).reshape(height, width)
            return arr


def resize_image(img: np.ndarray, target_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """Resize an image to target (height, width) using fast bilinear interpolation.

    Args:
        img (np.ndarray): Image array of shape (H, W) or (H, W, C).
        target_size (Tuple[int, int]): (target_height, target_width).

    Returns:
        np.ndarray: Resized image float32 array.
    """
    orig_h, orig_w = img.shape[:2]
    th, tw = target_size

    if orig_h == th and orig_w == tw:
        return img.astype(np.float32)

    row_coords = np.linspace(0, orig_h - 1, th)
    col_coords = np.linspace(0, orig_w - 1, tw)

    r0 = np.floor(row_coords).astype(int)
    r1 = np.clip(r0 + 1, 0, orig_h - 1)
    rf = (row_coords - r0)[:, None]

    c0 = np.floor(col_coords).astype(int)
    c1 = np.clip(c0 + 1, 0, orig_w - 1)
    cf = (col_coords - c0)[None, :]

    if img.ndim == 2:
        top = img[r0[:, None], c0] * (1.0 - cf) + img[r0[:, None], c1] * cf
        bot = img[r1[:, None], c0] * (1.0 - cf) + img[r1[:, None], c1] * cf
        return (top * (1.0 - rf) + bot * rf).astype(np.float32)
    elif img.ndim == 3:
        top = (
            img[r0[:, None], c0, :] * (1.0 - cf[:, :, None])
            + img[r0[:, None], c1, :] * cf[:, :, None]
        )
        bot = (
            img[r1[:, None], c0, :] * (1.0 - cf[:, :, None])
            + img[r1[:, None], c1, :] * cf[:, :, None]
        )
        return (top * (1.0 - rf[:, :, None]) + bot * rf[:, :, None]).astype(np.float32)
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")


def rgb_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using perceptual luminance weighting (Y = 0.299 R + 0.587 G + 0.114 B)."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] >= 3:
        return (
            0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        ).astype(img.dtype)
    return img


def load_image(
    filepath: Union[str, Path],
    target_size: Tuple[int, int] = (32, 32),
    grayscale: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Load a single image file (.bmp, .ppm, .pgm) and prepare it for neural network input.

    Args:
        filepath: Path to image.
        target_size: Desired (height, width) output size.
        grayscale: If True, converts image to single channel grayscale.
        normalize: If True, rescales pixel intensities from [0, 255] to [0.0, 1.0].

    Returns:
        np.ndarray: float32 array shaped (1, H, W) for grayscale or (3, H, W) for RGB.
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".bmp":
        raw = read_bmp(path)
    elif ext in (".ppm", ".pgm"):
        raw = read_ppm(path)
    else:
        raise ValueError(f"Unsupported image format '{ext}'. Supported formats: .bmp, .ppm, .pgm")

    # Grayscale conversion
    if grayscale and raw.ndim == 3:
        raw = rgb_to_grayscale(raw)

    # Resize
    resized = resize_image(raw, target_size=target_size)

    # Normalize
    if normalize:
        resized = resized / 255.0

    # Ensure channel-first format (C, H, W)
    if resized.ndim == 2:
        return resized[np.newaxis, :, :]  # (1, H, W)
    elif resized.ndim == 3:
        return np.transpose(resized, (2, 0, 1))  # (C, H, W)
    return resized


def load_image_dataset(
    root_dir: Union[str, Path],
    target_size: Tuple[int, int] = (32, 32),
    grayscale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load an image classification dataset organized in subfolders by class name.

    Expected directory structure:
        dataset/
          ├── cats/
          │     ├── cat_01.bmp
          │     └── cat_02.bmp
          └── dogs/
                ├── dog_01.bmp
                └── dog_02.bmp

    Args:
        root_dir: Root directory containing class subfolders.
        target_size: Target image dimensions (height, width).
        grayscale: Whether to convert images to single-channel grayscale.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]:
            - X: Image batch tensor of shape (N, C, H, W)
            - y: Label vector of shape (N,) with integer class indices
            - class_names: List of class names corresponding to the indices
    """
    path = Path(root_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Image dataset root folder not found: {root_dir}")

    # Discover class subdirectories
    subdirs = sorted([d for d in path.iterdir() if d.is_dir()])
    if not subdirs:
        raise ValueError(f"No class subfolders found inside '{root_dir}'.")

    class_names = [d.name for d in subdirs]
    images_list = []
    labels_list = []

    valid_exts = {".bmp", ".ppm", ".pgm"}

    for class_idx, class_dir in enumerate(subdirs):
        file_paths = [
            f for f in class_dir.iterdir() if f.suffix.lower() in valid_exts
        ]
        for f in file_paths:
            try:
                img_tensor = load_image(f, target_size=target_size, grayscale=grayscale)
                images_list.append(img_tensor)
                labels_list.append(class_idx)
            except Exception as e:
                print(f"Warning: Could not read image '{f.name}': {e}")

    if not images_list:
        raise ValueError(f"No valid images (.bmp, .ppm, .pgm) found in '{root_dir}'.")

    X = np.stack(images_list, axis=0).astype(np.float32)
    y = np.array(labels_list, dtype=np.int64)

    return X, y, class_names


def render_image_ascii(
    img: np.ndarray,
    width: int = 24,
    height: int = 12,
    invert: bool = False,
) -> str:
    """Render any 2D image matrix as crisp terminal ASCII grayscale art.

    Uses luminance gradient characters: ' ', '░', '▒', '▓', '█'.

    Args:
        img: 2D image array of intensities in [0, 1] or [0, 255].
        width: Output character width.
        height: Output character height.
        invert: Invert luminance if needed.

    Returns:
        str: Formatted ASCII art string.
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            # (C, H, W) -> (H, W)
            arr = np.mean(arr, axis=0)
        else:
            # (H, W, C) -> (H, W)
            arr = np.mean(arr, axis=-1)

    # Normalize to [0.0, 1.0]
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)

    # Resize to terminal aspect ratio
    resized = resize_image(arr, (height, width))

    ramp = " ░▒▓█"
    if invert:
        ramp = ramp[::-1]

    n_chars = len(ramp)
    lines = []
    border = "─" * (width * 2 + 2)
    lines.append(f"┌{border}┐")

    for r in range(height):
        chars = []
        for c in range(width):
            val = float(resized[r, c])
            idx = int(val * (n_chars - 1))
            idx = max(0, min(n_chars - 1, idx))
            # Double characters horizontally to correct 2:1 terminal font aspect ratio
            chars.append(ramp[idx] * 2)
        lines.append(f"│ {''.join(chars)} │")

    lines.append(f"└{border}┘")
    return "\n".join(lines)


def create_sample_cat_dog_dataset(
    output_dir: Union[str, Path] = "sample_cat_dog_dataset",
    samples_per_class: int = 15,
) -> Path:
    """Create a miniature Cat vs Dog BMP image dataset for instant training & practice.

    Generates synthetic 24x24 pixel images:
    - Cats have pointed triangular ears and horizontal whisker contours.
    - Dogs have floppy drooping ears and prominent snout silhouettes.

    Args:
        output_dir: Destination folder.
        samples_per_class: Number of sample images per class.

    Returns:
        Path: Path to created dataset folder.
    """
    base = Path(output_dir)
    cats_dir = base / "cats"
    dogs_dir = base / "dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    size = 24

    for i in range(samples_per_class):
        # 1. Generate Cat Pattern (Pointed ears at top, whiskers)
        cat_img = np.zeros((size, size), dtype=np.float32)
        # Head oval
        y_grid, x_grid = np.ogrid[:size, :size]
        head_mask = ((x_grid - 12) ** 2 + (y_grid - 13) ** 2) <= 36
        cat_img[head_mask] = 0.8

        # Pointed triangular ears
        # Left ear: apex around (8, 4)
        cat_img[4:8, 7:10] += 0.9
        cat_img[5:8, 6:11] += 0.8
        # Right ear: apex around (16, 4)
        cat_img[4:8, 14:17] += 0.9
        cat_img[5:8, 13:18] += 0.8

        # Whiskers (horizontal stripes)
        cat_img[13, 2:8] = 0.95
        cat_img[15, 2:8] = 0.95
        cat_img[13, 16:22] = 0.95
        cat_img[15, 16:22] = 0.95

        # Add slight variation and noise
        cat_img += rng.normal(0, 0.05, cat_img.shape).astype(np.float32)
        cat_img = np.clip(cat_img, 0.0, 1.0)
        write_bmp(cats_dir / f"cat_{i+1:02d}.bmp", cat_img)

        # 2. Generate Dog Pattern (Floppy ears on sides, snout in center)
        dog_img = np.zeros((size, size), dtype=np.float32)
        # Head
        dog_head = ((x_grid - 12) ** 2 + (y_grid - 11) ** 2) <= 40
        dog_img[dog_head] = 0.75

        # Floppy ears drooping down the sides
        dog_img[7:16, 4:7] += 0.9  # Left floppy ear
        dog_img[7:16, 17:20] += 0.9  # Right floppy ear

        # Prominent lower snout / muzzle
        snout_mask = ((x_grid - 12) ** 2 + (y_grid - 16) ** 2) <= 12
        dog_img[snout_mask] = 0.95

        # Dog nose
        dog_img[15:17, 11:13] = 0.3

        # Add variation
        dog_img += rng.normal(0, 0.05, dog_img.shape).astype(np.float32)
        dog_img = np.clip(dog_img, 0.0, 1.0)
        write_bmp(dogs_dir / f"dog_{i+1:02d}.bmp", dog_img)

    return base
