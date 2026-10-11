"""JEPA: Joint-Embedding Predictive Architectures, implemented in pure NumPy.

A JEPA learns by *predicting representations* instead of reconstructing inputs.
Given a context view ``x`` and a target view ``y`` of the same signal, an encoder
``E`` maps both to a latent space and a predictor ``P`` is trained to map the
context embedding onto the target embedding::

    L = || P(z_x, z) - sg(E(y)) ||^2      z_x = E(x)

The loss never touches pixels, so the encoder is free to discard unpredictable
low-level detail and keep semantic content. This module implements the family:

* :class:`IJEPA` - image JEPA (Assran et al., 2023). Multi-block masking, an
  exponential-moving-average target encoder, a narrow predictor ViT, and a
  smooth-L1 loss in latent space.
* :class:`VJEPA` - video JEPA (Bardes et al., 2024; Assran et al., 2025).
  Tubelet patchification, tube masking, 3D position information.
* :class:`JEPAWorldModel` - action-conditioned latent world model in the style of
  V-JEPA 2-AC (Assran et al., 2025) and LeWorldModel (Maes et al., 2026), with a
  cross-entropy-method planner that searches action sequences *in latent space*.
* :func:`sigreg_loss` - Sketched Isotropic Gaussian Regularization (Balestriero &
  LeCun, 2025), a distribution-matching anti-collapse objective that lets a JEPA
  train end-to-end with no EMA, no stop-gradient and a single trade-off
  hyperparameter ``lambda``.
* :func:`representation_diagnostics` - the collapse detectors every JEPA needs:
  effective rank, per-dimension variance, mean pairwise cosine similarity and
  the uniformity/alignment pair of Wang & Isola.

Everything runs on the repository's NumPy autograd engine, so the same code path
trains on a laptop CPU and needs no GPU, no PyTorch and no pretrained weights.

References
----------
Assran et al. "Self-Supervised Learning from Images with a Joint-Embedding
Predictive Architecture." CVPR 2023. (I-JEPA)
Bardes et al. "Revisiting Feature Prediction for Learning Visual Representations
from Video." 2024. (V-JEPA)
Assran et al. "V-JEPA 2: Self-Supervised Video Models Enable Understanding,
Prediction and Planning." 2025.
Balestriero & LeCun. "LeJEPA: Provable and Scalable Self-Supervised Learning
Without the Heuristics." 2025. (SIGReg)
Maes et al. "LeWorldModel: Stable End-to-End Joint-Embedding Predictive
Architecture from Pixels." 2026.
"""

from dataclasses import dataclass, field
import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .autograd import Module, Parameter, Tensor, no_grad


__all__ = [
    "IJEPA",
    "VJEPA",
    "JEPAWorldModel",
    "JEPAEncoder",
    "JEPAPredictor",
    "JEPADynamics",
    "SIGReg",
    "sigreg_loss",
    "ema_update",
    "ema_schedule",
    "patchify_images",
    "patchify_video",
    "unpatchify_images",
    "sincos_2d_pos_embed",
    "sincos_3d_pos_embed",
    "sample_block", 
    "sample_ijepa_masks",
    "sample_vjepa_masks",
    "collate_masks",
    "smooth_l1_loss",
    "latent_l2_loss",
    "representation_diagnostics",
    "linear_probe",
    "named_parameters",
    "save_jepa",
    "load_jepa",
]


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _f32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _as_tensor(x: Any, dtype: Any = np.float32) -> Tensor:
    """Wrap NumPy input in a ``Tensor``; pass existing tensors through."""
    return x if isinstance(x, Tensor) else Tensor(_f32(x), dtype=dtype)


def _constant(x: Any) -> Tensor:
    """A detached tensor used for targets, positional tables and masks."""
    return Tensor(_f32(x), requires_grad=False, dtype=np.float32)


def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the keyword arguments ``fn`` actually accepts (stable configs)."""
    accepted = set(inspect.signature(fn).parameters) - {"self"}
    return {k: v for k, v in kwargs.items() if k in accepted}


def _init_weight(shape: Tuple[int, ...], std: float = 0.02, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Truncated-normal initialisation as used by ViT/i-JEPA (0.02 std, 2-sigma cut)."""
    gen = rng if rng is not None else np.random
    if gen is np.random:
        w = np.random.randn(*shape) * std
    else:
        w = gen.standard_normal(shape) * std
    return np.clip(w, -2.0 * std, 2.0 * std).astype(np.float32)


def named_parameters(module: Module, prefix: str = "") -> List[Tuple[str, Parameter]]:
    """Deterministic, nested listing of a module's parameters.

    ``Module.parameters()`` returns a flat list without names (good for
    optimizers); serialization and EMA updates need stable names instead.
    """
    out: List[Tuple[str, Parameter]] = []
    for name, value in module.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, Parameter):
            out.append((f"{prefix}{name}", value))
        elif isinstance(value, Module):
            out.extend(named_parameters(value, f"{prefix}{name}."))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, Parameter):
                    out.append((f"{prefix}{name}.{i}", item))
                elif isinstance(item, Module):
                    out.extend(named_parameters(item, f"{prefix}{name}.{i}."))
    return out


# ---------------------------------------------------------------------------
# Patchification and position embeddings
# ---------------------------------------------------------------------------

def patchify_images(images: np.ndarray, patch_size: int) -> np.ndarray:
    """Split ``(B, C, H, W)`` images into ``(B, N, C*P*P)`` patch vectors."""
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"patchify_images expects (B, C, H, W), got {images.shape}")
    b, c, h, w = images.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"image size {(h, w)} must be divisible by patch size {patch_size}")
    gh, gw = h // patch_size, w // patch_size
    x = images.reshape(b, c, gh, patch_size, gw, patch_size)
    x = x.transpose(0, 2, 4, 1, 3, 5)
    return x.reshape(b, gh * gw, c * patch_size * patch_size).astype(np.float32)


def unpatchify_images(patches: np.ndarray, patch_size: int, channels: int = 1) -> np.ndarray:
    """Inverse of :func:`patchify_images` (used by demo visualisations)."""
    patches = np.asarray(patches)
    b, n, d = patches.shape
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"cannot unpatchify {n} tokens into a square grid")
    x = patches.reshape(b, side, side, channels, patch_size, patch_size)
    x = x.transpose(0, 3, 1, 4, 2, 5)
    return x.reshape(b, channels, side * patch_size, side * patch_size).astype(np.float32)


def patchify_video(videos: np.ndarray, tubelet: int, patch_size: int) -> np.ndarray:
    """Split ``(B, C, T, H, W)`` clips into ``(B, N, C*tubelet*P*P)`` tubelets."""
    videos = np.asarray(videos)
    if videos.ndim != 5:
        raise ValueError(f"patchify_video expects (B, C, T, H, W), got {videos.shape}")
    b, c, t, h, w = videos.shape
    if t % tubelet or h % patch_size or w % patch_size:
        raise ValueError("clip shape must be divisible by the tubelet and patch size")
    gt, gh, gw = t // tubelet, h // patch_size, w // patch_size
    x = videos.reshape(b, c, gt, tubelet, gh, patch_size, gw, patch_size)
    x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(b, gt * gh * gw, c * tubelet * patch_size * patch_size).astype(np.float32)


def sincos_2d_pos_embed(dim: int, grid_h: int, grid_w: int) -> np.ndarray:
    """Fixed 2D sine-cosine table of shape ``(grid_h*grid_w, dim)``."""
    if dim % 4:
        raise ValueError("sincos_2d_pos_embed requires dim divisible by 4")
    q = dim // 4
    omega = 1.0 / (10000.0 ** (np.arange(q, dtype=np.float64) / max(q, 1)))
    grid_y, grid_x = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    out_y = np.outer(grid_y.reshape(-1), omega)
    out_x = np.outer(grid_x.reshape(-1), omega)
    emb = np.concatenate([np.sin(out_y), np.cos(out_y), np.sin(out_x), np.cos(out_x)], axis=1)
    return emb.astype(np.float32)


def sincos_3d_pos_embed(dim: int, grid_t: int, grid_h: int, grid_w: int) -> np.ndarray:
    """Fixed 3D sine-cosine table of shape ``(grid_t*grid_h*grid_w, dim)``.

    The embedding dimension is split into three even bands for the (time,
    height, width) axes, following standard video-transformer practice. Any
    remainder (``dim % 3 != 0``) is padded with zeros so every even dimension
    from 6 upwards is usable.
    """
    if dim % 2 or dim < 6:
        raise ValueError("sincos_3d_pos_embed requires an even dim >= 6")
    band = dim // 3
    band -= band % 2
    sizes = [band, band, dim - 2 * band]
    sizes[2] -= sizes[2] % 2
    gt, gh, gw = np.meshgrid(
        np.arange(grid_t), np.arange(grid_h), np.arange(grid_w), indexing="ij"
    )
    bands = []
    for axis, size in zip((gt, gh, gw), sizes):
        q = size // 2
        omega = 1.0 / (10000.0 ** (np.arange(q, dtype=np.float64) / max(q, 1)))
        proj = np.outer(axis.reshape(-1), omega)
        bands.append(np.sin(proj))
        bands.append(np.cos(proj))
    tail = dim - sum(sizes)
    if tail > 0:
        bands.append(np.zeros((gt.size, tail), dtype=np.float64))
    return np.concatenate(bands, axis=1).astype(np.float32)


def rope_frequencies(dim: int, positions: np.ndarray, base: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Rotary table ``(cos, sin)`` of shape ``(len(positions), dim)`` for RoPE."""
    if dim % 2:
        raise ValueError("rope_frequencies requires an even dimension")
    half = dim // 2
    omega = 1.0 / (base ** (np.arange(half, dtype=np.float64) / max(half, 1)))
    angles = np.outer(np.asarray(positions, dtype=np.float64), omega)
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=1)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=1)
    return cos.astype(np.float32), sin.astype(np.float32)


def rope3d_tables(
    head_dim: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    base: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """3D rotary tables for spatio-temporal tokens, shape ``(N, head_dim)``.

    The head dimension is split into three even bands for (time, height, width).
    A remainder of 0, 2 or 4 dimensions (when ``head_dim`` is not a multiple of
    6) is left unrotated, which keeps the scheme valid for every head size.
    """
    band = head_dim // 3
    band -= band % 2
    sizes = [band, band, head_dim - 2 * band]
    sizes[2] -= sizes[2] % 2
    gt, gh, gw = np.meshgrid(
        np.arange(grid_t), np.arange(grid_h), np.arange(grid_w), indexing="ij"
    )
    cos_parts, sin_parts = [], []
    for axis, size in zip((gt, gh, gw), sizes):
        c, s = rope_frequencies(size, axis.reshape(-1), base=base)
        cos_parts.append(c)
        sin_parts.append(s)
    tail = head_dim - sum(sizes)
    if tail > 0:
        ones = np.ones((gt.size, tail), dtype=np.float32)
        zeros = np.zeros((gt.size, tail), dtype=np.float32)
        cos_parts.append(ones)
        sin_parts.append(zeros)
    return np.concatenate(cos_parts, axis=1), np.concatenate(sin_parts, axis=1)


# ---------------------------------------------------------------------------
# Masking (the part of I-JEPA that carries most of the signal)
# ---------------------------------------------------------------------------

def sample_block(
    grid_h: int,
    grid_w: int,
    scale: Tuple[float, float],
    aspect_ratio: Tuple[float, float],
    rng: np.random.Generator,
) -> Tuple[int, int, int, int]:
    """Sample a rectangular block ``(top, left, height, width)`` on a token grid."""
    area = grid_h * grid_w * rng.uniform(*scale)
    ar = rng.uniform(*aspect_ratio)
    bh = int(round(math.sqrt(area * ar)))
    bw = int(round(math.sqrt(area / ar)))
    bh = int(np.clip(bh, 1, grid_h))
    bw = int(np.clip(bw, 1, grid_w))
    top = int(rng.integers(0, grid_h - bh + 1))
    left = int(rng.integers(0, grid_w - bw + 1))
    return top, left, bh, bw


def _block_mask(top: int, left: int, bh: int, bw: int, grid_h: int, grid_w: int) -> np.ndarray:
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    mask[top : top + bh, left : left + bw] = True
    return mask


def sample_ijepa_masks(
    grid_h: int,
    grid_w: int,
    n_targets: int = 4,
    target_scale: Tuple[float, float] = (0.15, 0.2),
    context_scale: Tuple[float, float] = (0.85, 1.0),
    aspect_ratio: Tuple[float, float] = (0.75, 1.5),
    min_context: int = 4,
    rng: Optional[np.random.Generator] = None,
    max_tries: int = 20,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Sample I-JEPA multi-block context/target masks.

    Following Assran et al. (2023): ``M`` target blocks are sampled with a scale
    of 15-20% of the image and an aspect ratio in [0.75, 1.5]; a single context
    block with 85-100% scale and unit aspect ratio is sampled, then any region
    overlapping a target block is removed from it. Targets are *disjoint from*
    the context, which is what stops the predictor from solving the task by
    copying visible tokens.

    Returns ``(context_idx, [target_idx_1, ..., target_idx_M])`` as flat token
    indices into a row-major ``grid_h x grid_w`` grid.
    """
    rng = np.random.default_rng() if rng is None else rng
    mask: Optional[np.ndarray] = None
    for _ in range(max_tries):
        targets = [
            _block_mask(*sample_block(grid_h, grid_w, target_scale, aspect_ratio, rng), grid_h, grid_w)
            for _ in range(n_targets)
        ]
        union = np.zeros((grid_h, grid_w), dtype=bool)
        for t in targets:
            union |= t
        ctx_block = _block_mask(*sample_block(grid_h, grid_w, context_scale, (1.0, 1.0), rng), grid_h, grid_w)
        ctx = ctx_block & ~union
        if ctx.sum() >= min_context:
            mask = ctx
            break

    if mask is None:  # fallback: everything that is not the first target block
        union = np.zeros((grid_h, grid_w), dtype=bool)
        for t in targets:
            union |= t
        ctx = ~union
        if ctx.sum() == 0:  # pathological grid: keep at least one token
            ctx = np.zeros((grid_h, grid_w), dtype=bool)
            ctx[0, 0] = True
            targets = [np.ones((grid_h, grid_w), dtype=bool) for _ in range(n_targets)]
            targets[0][0, 0] = False

    context_idx = np.flatnonzero(mask.reshape(-1)).astype(np.int64)
    target_idx = [np.flatnonzero(t.reshape(-1)).astype(np.int64) for t in targets]
    return context_idx, target_idx


def sample_vjepa_masks(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    n_targets: int = 2,
    target_scale: Tuple[float, float] = (0.15, 0.4),
    aspect_ratio: Tuple[float, float] = (0.75, 1.5),
    temporal_scale: Tuple[float, float] = (1.0, 1.0),
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Sample V-JEPA tube masks (Bardes et al., 2024).

    Two mask groups (short-range and long-range) are sampled as spatial blocks
    that are extruded through the temporal axis ("tubes"). ``temporal_scale``
    controls how much of the clip each tube covers; the default of
    ``(1.0, 1.0)`` reproduces full tube masking, which is the V-JEPA default
    and the variant that avoids every target being recoverable by a single
    visible frame.

    Returns flat indices into a row-major ``grid_t x grid_h x grid_w`` grid.
    """
    rng = np.random.default_rng() if rng is None else rng
    spat = np.zeros((grid_h, grid_w), dtype=np.int64)
    for i in range(n_targets):
        top, left, bh, bw = sample_block(grid_h, grid_w, target_scale, aspect_ratio, rng)
        spat[top : top + bh, left : left + bw] = i + 1

    t0 = int(rng.integers(0, grid_t)) if temporal_scale != (1.0, 1.0) else 0
    t_span = grid_t if temporal_scale == (1.0, 1.0) else max(1, int(round(grid_t * rng.uniform(*temporal_scale))))
    time_mask = np.zeros((grid_t, 1, 1), dtype=bool)
    if t_span >= grid_t:
        time_mask[:] = True
    else:
        t_idx = np.arange(t0, min(grid_t, t0 + t_span))
        time_mask[t_idx] = True

    occupied = np.broadcast_to(spat[None] > 0, (grid_t, grid_h, grid_w)) & np.broadcast_to(time_mask, (grid_t, grid_h, grid_w))
    context_idx = np.flatnonzero(~occupied.reshape(-1)).astype(np.int64)
    target_idx: List[np.ndarray] = []
    for i in range(n_targets):
        m = np.broadcast_to(spat[None] == (i + 1), (grid_t, grid_h, grid_w)) & np.broadcast_to(time_mask, (grid_t, grid_h, grid_w))
        idx = np.flatnonzero(m.reshape(-1)).astype(np.int64)
        if idx.size == 0:
            idx = np.array([int(np.flatnonzero(~occupied.reshape(-1))[0])], dtype=np.int64)
        target_idx.append(idx)

    if context_idx.size < 4:  # degenerate mask, keep the first few unoccupied tokens
        free = np.flatnonzero(~occupied.reshape(-1)).astype(np.int64)
        context_idx = free if free.size else np.array([0], dtype=np.int64)
    return context_idx, target_idx


def collate_masks(
    context_idx: Sequence[np.ndarray],
    target_idx: Sequence[Sequence[np.ndarray]],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad variable-length per-sample masks into rectangular batch arrays.

    Context masks are reduced to the smallest context in the batch by drawing a
    *random* subset, so every index is a real token and no spatial bias is
    introduced (taking the first ``k`` sorted indices would systematically strip
    the bottom of the image). Target masks are padded by repeating an existing
    index and the returned ``valid`` mask records which entries are real, so
    padded positions contribute neither loss nor gradient.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_ctx = min(len(c) for c in context_idx)
    rows = []
    for c in context_idx:
        if len(c) == n_ctx:
            rows.append(np.asarray(c, dtype=np.int64))
        else:
            rows.append(np.sort(rng.choice(c, size=n_ctx, replace=False)).astype(np.int64))
    ctx = np.stack(rows).astype(np.int64)

    flat_targets = [np.concatenate(list(t)) if len(t) else np.array([0], dtype=np.int64) for t in target_idx]
    n_tgt = max(len(t) for t in flat_targets)
    tgt = np.zeros((len(flat_targets), n_tgt), dtype=np.int64)
    valid = np.zeros((len(flat_targets), n_tgt), dtype=np.float32)
    for i, t in enumerate(flat_targets):
        tgt[i, : len(t)] = t
        valid[i, : len(t)] = 1.0
        if len(t) < n_tgt:  # repeat a real index, flagged as invalid
            tgt[i, len(t) :] = t[-1]
    return ctx, tgt, valid


# ---------------------------------------------------------------------------
# Normalisation, attention and transformer blocks
# ---------------------------------------------------------------------------

class JEPALayerNorm(Module):
    """LayerNorm without bias-free shortcuts: ``y = (x - mu) / sqrt(var + eps) * g + b``."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = Parameter(np.ones(dim, dtype=np.float32))
        self.bias = Parameter(np.zeros(dim, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        centered = x - mu
        var = (centered * centered).mean(axis=-1, keepdims=True)
        return centered / (var + self.eps) ** 0.5 * self.weight + self.bias


class JEPARMSNorm(Module):
    """RMSNorm: cheaper than LayerNorm and equally stable for ViT towers."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = Parameter(np.ones(dim, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        ms = (x * x).mean(axis=-1, keepdims=True)
        return x / (ms + self.eps) ** 0.5 * self.weight


def _make_norm(kind: str, dim: int) -> Module:
    return JEPARMSNorm(dim) if kind == "rmsnorm" else JEPALayerNorm(dim)


def _split_heads(x: Tensor, n_heads: int) -> Tensor:
    b, n, d = x.shape
    return x.reshape(b, n, n_heads, d // n_heads).transpose(0, 2, 1, 3)


def _merge_heads(x: Tensor) -> Tensor:
    b, h, n, hd = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, n, h * hd)


def tensor_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax.

    The max is subtracted as a *detached constant*, which is exact: softmax is
    shift-invariant, so the gradient is unchanged while the exponential stays
    inside float32 range.
    """
    shifted = x - _constant(np.max(x.data, axis=axis, keepdims=True))
    e = shifted.exp()
    return e / e.sum(axis=axis, keepdims=True)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return Tensor.cat([-x[..., half:], x[..., :half]], axis=-1)


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    n_heads: int,
    attn_bias: Optional[np.ndarray] = None,
    rope: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tensor:
    """Multi-head scaled dot-product attention.

    ``attn_bias`` is an additive mask (``-inf`` blocks) broadcast over batch and
    heads; ``rope`` is an optional ``(cos, sin)`` table applied to q and k.
    """
    head_dim = q.shape[-1] // n_heads
    qh, kh, vh = _split_heads(q, n_heads), _split_heads(k, n_heads), _split_heads(v, n_heads)
    if rope is not None:
        cos, sin = rope
        if cos.ndim == 3:  # per-sample tables, already gathered to the visible tokens
            cos_t, sin_t = _constant(cos)[:, None], _constant(sin)[:, None]
        else:
            cos_t, sin_t = _constant(cos)[None, None], _constant(sin)[None, None]
        qh = qh * cos_t + _rotate_half(qh) * sin_t
        kh = kh * cos_t + _rotate_half(kh) * sin_t
    logits = qh @ kh.transpose(0, 1, 3, 2) * (1.0 / math.sqrt(head_dim))
    if attn_bias is not None:
        logits = logits + attn_bias
    return _merge_heads(tensor_softmax(logits) @ vh)


class JEPAEncoderBlock(Module):
    """Pre-norm transformer block: attention then MLP, both residual."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        norm: str = "layernorm",
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.norm1 = _make_norm(norm, dim)
        self.qkv = Parameter(_init_weight((dim, 3 * dim), init_std))
        self.proj = Parameter(_init_weight((dim, dim), init_std))
        self.norm2 = _make_norm(norm, dim)
        hidden = int(round(dim * mlp_ratio))
        self.fc1 = Parameter(_init_weight((dim, hidden), init_std))
        self.fc2 = Parameter(_init_weight((hidden, dim), init_std))

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[np.ndarray] = None,
        rope: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        idx: Optional[np.ndarray] = None,
    ) -> Tensor:
        if rope is not None and idx is not None:
            idx = np.asarray(idx)
            rope = (rope[0][idx], rope[1][idx])
        h = self.norm1(x)
        qkv = h @ self.qkv
        d = self.dim
        q, k, v = qkv[..., :d], qkv[..., d : 2 * d], qkv[..., 2 * d :]
        x = x + attention(q, k, v, self.n_heads, attn_bias=attn_bias, rope=rope) @ self.proj
        h = self.norm2(x)
        h = (h @ self.fc1).relu()
        return x + h @ self.fc2


class _MLPBlock(Module):
    """A pre-norm 2-layer MLP branch used by the vector encoder."""

    def __init__(self, dim: int, hidden: int, norm: str = "layernorm", init_std: float = 0.02) -> None:
        super().__init__()
        self.norm = _make_norm(norm, dim)
        self.fc1 = Parameter(_init_weight((dim, hidden), init_std))
        self.fc2 = Parameter(_init_weight((hidden, dim), init_std))

    def forward(self, x: Tensor) -> Tensor:
        return x + (self.norm(x) @ self.fc1).relu() @ self.fc2


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class JEPAEncoder(Module):
    """Vision-transformer encoder over pre-patchified tokens.

    The encoder deliberately accepts *NumPy* patches plus an optional per-sample
    index array: a JEPA only ever differentiates with respect to its weights, so
    gathering the visible tokens on the NumPy side keeps the graph small and the
    encoder fast (this is what makes context-only encoding, I-JEPA's central
    efficiency trick, cheap).
    """

    def __init__(
        self,
        patch_dim: int,
        dim: int = 192,
        depth: int = 6,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        n_tokens: int = 64,
        grid: Optional[Tuple[int, int]] = None,
        grid_t: Optional[int] = None,
        pos_embed: str = "sincos",
        norm: str = "layernorm",
        init_std: float = 0.02,
        pos_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.pos_scale = float(pos_scale)
        self.patch_dim = int(patch_dim)
        self.dim = int(dim)
        self.depth = int(depth)
        self.n_heads = int(n_heads)
        self.n_tokens = int(n_tokens)
        self.grid = grid
        self.grid_t = grid_t
        self.pos_embed_kind = pos_embed
        self.norm_kind = norm

        self.patch_proj = Parameter(_init_weight((patch_dim, dim), init_std))
        self.blocks = [
            JEPAEncoderBlock(dim, n_heads, mlp_ratio, norm=norm, init_std=init_std) for _ in range(depth)
        ]
        self.final_norm = _make_norm(norm, dim)

        self.rope: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.pos_table: Optional[np.ndarray] = None
        self.learned_pos: Optional[Parameter] = None
        if pos_embed == "learned":
            self.learned_pos = Parameter(_init_weight((n_tokens, dim), init_std))
        elif pos_embed in ("rope3d",):
            if grid is None or grid_t is None:
                raise ValueError("rope3d needs grid=(h, w) and grid_t")
            self.rope = rope3d_tables(dim // n_heads, grid_t, grid[0], grid[1])
        elif pos_embed in ("sincos", "rope2d"):
            if grid is None:
                raise ValueError(f"{pos_embed} needs grid=(h, w)")
            gh, gw = grid
            if grid_t is None:
                self.pos_table = sincos_2d_pos_embed(dim, gh, gw) * self.pos_scale
            else:
                self.pos_table = sincos_3d_pos_embed(dim, grid_t, gh, gw) * self.pos_scale
            if pos_embed == "rope2d":
                self.rope = rope_frequencies(dim // n_heads, np.arange(n_tokens))
        elif pos_embed != "none":
            raise ValueError(f"unknown pos_embed kind: {pos_embed}")

    def forward(self, patch_tokens: np.ndarray, idx: Optional[np.ndarray] = None) -> Tensor:
        """Encode patches ``(B, N, patch_dim)``; ``idx`` selects tokens per sample."""
        patches = np.asarray(patch_tokens, dtype=np.float32)
        if idx is not None:
            idx = np.asarray(idx)
            b = patches.shape[0]
            patches = patches[np.arange(b)[:, None], idx]
        x = _constant(patches) @ self.patch_proj
        if self.learned_pos is not None:
            table = self.learned_pos[None] if idx is None else self.learned_pos[np.asarray(idx)]
            x = x + (table if isinstance(table, Tensor) else _constant(table))
        elif self.pos_table is not None:
            table = self.pos_table if idx is None else self.pos_table[np.asarray(idx)]
            x = x + _constant(table)
        for block in self.blocks:
            x = block(x, rope=self.rope, idx=idx)
        return self.final_norm(x)

    def token_table(self) -> Optional[Tensor]:
        """Positional table for the predictor: a Tensor for learned, constant otherwise."""
        if self.learned_pos is not None:
            return self.learned_pos
        if self.pos_table is not None:
            return _constant(self.pos_table)
        return None


class VectorEncoder(Module):
    """MLP encoder for low-dimensional observations (state vectors, tabular data)."""

    def __init__(
        self,
        obs_dim: int,
        dim: int = 96,
        depth: int = 3,
        mlp_ratio: float = 2.0,
        norm: str = "layernorm",
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.dim = int(dim)
        self.input = Parameter(_init_weight((obs_dim, dim), init_std))
        self.blocks = [_MLPBlock(dim, int(round(dim * mlp_ratio)), norm, init_std) for _ in range(depth)]
        self.final_norm = _make_norm(norm, dim)

    def forward(self, obs: np.ndarray) -> Tensor:
        x = _constant(np.asarray(obs, dtype=np.float32)) @ self.input
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class JEPAPredictor(Module):
    """Narrow predictor ViT conditioned on mask tokens (I-JEPA, Section 3).

    The predictor consumes the context encoder's output, adds learnable mask
    tokens at the positions the model must predict, and returns embeddings in
    the encoder's latent space.  Position information is shared with the encoder
    when it has a fixed table, exactly as in the reference implementation.
    """

    def __init__(
        self,
        encoder_dim: int,
        dim: int = 384,
        depth: int = 6,
        n_heads: int = 12,
        mlp_ratio: float = 4.0,
        pos_table: Optional[np.ndarray] = None,
        pos_parameter: Optional[Parameter] = None,
        norm: str = "layernorm",
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"predictor dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.embed = Parameter(_init_weight((encoder_dim, dim), init_std))
        self.mask_token = Parameter(_init_weight((1, 1, dim), init_std))
        self.blocks = [
            JEPAEncoderBlock(dim, n_heads, mlp_ratio, norm=norm, init_std=init_std) for _ in range(depth)
        ]
        self.final_norm = _make_norm(norm, dim)
        self.proj = Parameter(_init_weight((dim, encoder_dim), init_std))
        self.pos_table = pos_table
        self.pos_parameter = pos_parameter
        if pos_table is None and pos_parameter is None:
            raise ValueError("JEPAPredictor needs a positional table or parameter")

    def _pos(self, idx: np.ndarray) -> Union[Tensor, np.ndarray]:
        if self.pos_parameter is not None:
            return self.pos_parameter[np.asarray(idx)]
        return self.pos_table[np.asarray(idx)]

    def forward(self, context: Tensor, context_idx: np.ndarray, target_idx: np.ndarray) -> Tensor:
        """Return predictions for the target tokens, shape ``(B, n_target, D)``."""
        x = context @ self.embed + self._pos(context_idx)
        n_target = target_idx.shape[1]
        mask_tokens = self.mask_token + self._pos(target_idx)
        seq = Tensor.cat([x, mask_tokens], axis=1)
        for block in self.blocks:
            seq = block(seq)
        seq = self.final_norm(seq)
        return seq[:, -n_target:] @ self.proj


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def smooth_l1_loss(
    pred: Tensor,
    target: Tensor,
    beta: float = 1.0,
    valid: Optional[np.ndarray] = None,
) -> Tensor:
    """Huber / smooth-L1 loss used by the official I-JEPA and V-JEPA recipes.

    ``valid`` masks padded target slots so they contribute neither loss nor
    gradient; the reduction is per-element over the surviving entries, which
    makes the value comparable across differently sized batches.
    """
    diff = pred - target
    absd = (diff * diff + 1e-12) ** 0.5
    quad = diff * diff * (0.5 / beta)
    lin = absd - (0.5 * beta)
    quad_w = _constant(absd.data < beta)
    lin_w = _constant(absd.data >= beta)
    element = quad * quad_w + lin * lin_w
    if valid is None:
        return element.mean()
    w = _constant(valid)[..., None]
    denom = max(1.0, float(np.sum(valid)) * pred.shape[-1])
    return (element * w).sum() * (1.0 / denom)


def latent_l2_loss(
    pred: Tensor,
    target: Tensor,
    valid: Optional[np.ndarray] = None,
) -> Tensor:
    """Mean squared error in latent space (the L2 objective of the I-JEPA paper)."""
    diff = pred - target
    element = diff * diff
    if valid is None:
        return element.mean()
    w = _constant(valid)[..., None]
    denom = max(1.0, float(np.sum(valid)) * pred.shape[-1])
    return (element * w).sum() * (1.0 / denom)


def normalize_latents(z: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-token standardisation of target features (V-JEPA target normalisation)."""
    mu = z.mean(axis=-1, keepdims=True)
    sd = z.std(axis=-1, keepdims=True)
    return ((z - mu) / (sd + eps)).astype(np.float32)


# ---------------------------------------------------------------------------
# EMA target encoder
# ---------------------------------------------------------------------------

def ema_update(target: Module, source: Module, momentum: float) -> None:
    """``theta_target <- m * theta_target + (1 - m) * theta_source`` (in place)."""
    src = dict(named_parameters(source))
    for name, param in named_parameters(target):
        if name in src:
            param.data = (
                momentum * param.data + (1.0 - momentum) * src[name].data
            ).astype(param.data.dtype, copy=False)


def ema_schedule(step: int, total_steps: int, start: float = 0.996, end: float = 1.0) -> float:
    """Cosine momentum schedule from the I-JEPA reference implementation."""
    if total_steps <= 0:
        return end
    progress = min(1.0, max(0.0, step / float(total_steps)))
    return float(end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0)


# ---------------------------------------------------------------------------
# SIGReg: distribution matching anti-collapse regularizer (LeJEPA)
# ---------------------------------------------------------------------------

def _trapezoid_weights(t: np.ndarray) -> np.ndarray:
    """Quadrature weights of the trapezoidal rule for an evenly spaced grid."""
    n = t.size
    w = np.full(n, float(t[1] - t[0]) if n > 1 else 1.0, dtype=np.float64)
    if n > 1:
        w[0] *= 0.5
        w[-1] *= 0.5
    return w.astype(np.float32)


class SIGReg(Module):
    """Sketched Isotropic Gaussian Regularization (Balestriero & LeCun, 2025).

    The regularizer draws ``num_slices`` random unit directions, projects the
    embeddings onto them (Cramer-Wold: matching every 1-D projection implies
    matching the full distribution) and compares each projection's empirical
    characteristic function against the standard Gaussian's using the
    Epps-Pulley statistic, integrated with a trapezoidal quadrature.

    Matching an isotropic Gaussian is what makes JEPA training collapse-free
    *by construction*: a degenerate (constant) embedding cloud is infinitely far
    from ``N(0, I)``, so the objective cannot be minimised by mapping everything
    to one point. No EMA teacher, no stop-gradient, a single trade-off lambda.
    """

    def __init__(
        self,
        num_slices: int = 64,
        num_points: int = 17,
        t_max: float = 3.0,
        scale_by_n: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_slices = int(num_slices)
        self.num_points = int(num_points)
        self.t_max = float(t_max)
        self.scale_by_n = bool(scale_by_n)
        self.rng = np.random.default_rng(seed)
        self.t = np.linspace(-self.t_max, self.t_max, self.num_points).astype(np.float32)
        self.quad = _trapezoid_weights(self.t) * np.exp(-0.5 * self.t ** 2).astype(np.float32)
        self.target_cf = np.exp(-0.5 * self.t ** 2).astype(np.float32)

    def new_projections(self, dim: int) -> np.ndarray:
        """Sample fresh unit-norm projection directions, shape ``(dim, num_slices)``."""
        a = self.rng.standard_normal((dim, self.num_slices)).astype(np.float32)
        return a / np.maximum(np.linalg.norm(a, axis=0, keepdims=True), 1e-8)

    def forward(self, embeddings: Tensor, projections: Optional[np.ndarray] = None) -> Tensor:
        z = _as_tensor(embeddings)
        n, dim = z.shape
        a = self.new_projections(dim) if projections is None else np.asarray(projections, dtype=np.float32)
        proj = z @ _constant(a)                                   # (N, M)
        arg = proj.reshape(n, a.shape[1], 1) * self.t.reshape(1, 1, -1)  # (N, M, T)
        ecf = arg.cos().mean(axis=0)                              # (M, T)
        err = (ecf - _constant(self.target_cf)) ** 2 * self.quad  # (M, T) gradient flows through ecf
        stat = err.sum(axis=-1).mean()                            # mean over slices
        if self.scale_by_n:
            stat = stat * float(n)
        return stat


def sigreg_loss(
    embeddings: Tensor,
    num_slices: int = 64,
    num_points: int = 17,
    t_max: float = 3.0,
    scale_by_n: bool = True,
    seed: Optional[int] = None,
    projections: Optional[np.ndarray] = None,
) -> Tensor:
    """Functional shortcut for :class:`SIGReg`."""
    return SIGReg(num_slices, num_points, t_max, scale_by_n, seed)(
        embeddings, projections=projections
    )


# ---------------------------------------------------------------------------
# Collapse diagnostics
# ---------------------------------------------------------------------------

def effective_rank(features: np.ndarray, eps: float = 1e-12) -> float:
    """Entropy-based effective rank of Roy & Vetterli (2007).

    ``exp(H(p))`` with ``p`` the normalised singular-value spectrum. A healthy
    representation uses many directions (rank ~ min(N, D)); a collapsed one has
    rank -> 1.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2:
        return float(min(x.shape))
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    total = s.sum()
    if total <= eps:
        return 0.0
    p = s / total
    p = p[p > eps]
    return float(np.exp(-np.sum(p * np.log(p))))


def representation_diagnostics(
    features: np.ndarray,
    max_pairs: int = 512,
    dead_dim_std: float = 1e-3,
    seed: int = 0,
) -> Dict[str, float]:
    """Measure how alive a representation is.

    Returns per-dimension spread (``std_mean``, ``std_min``), ``effective_rank``,
    mean pairwise cosine similarity, the uniformity/alignment pair of Wang &
    Isola, the fraction of dead dimensions, the input-dependent energy fraction
    (``signal_ratio``, which catches a representation swamped by a constant
    offset) and a boolean ``collapsed`` verdict. A constant encoder scores
    ``std_mean ~ 0``, ``cosine ~ 1``, ``rank ~ 1``; an encoder whose output is
    dominated by a fixed bias scores ``signal_ratio ~ 0``.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("representation_diagnostics expects (N, D) features")
    n, d = x.shape
    std = x.std(axis=0)
    centered = x - x.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normed = centered / np.maximum(norms, 1e-12)

    rng = np.random.default_rng(seed)
    if n < 2:
        cosine = 0.0
        uniformity = 0.0
    elif norms.mean() < 1e-12:
        # Every vector is identical: the direction is undefined but the *meaning*
        # for a collapse detector is unambiguous, so report perfect alignment.
        cosine = 1.0
        uniformity = 0.0
    else:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        cosine = float(np.mean(np.sum(normed[i[keep]] * normed[j[keep]], axis=1)))
        dist2 = np.sum((centered[i[keep]] - centered[j[keep]]) ** 2, axis=1)
        uniformity = float(np.log(np.mean(np.exp(-2.0 * dist2)) + 1e-12))

    rank = effective_rank(x)
    degenerate = float(np.mean(std < dead_dim_std))
    # What fraction of the representation's energy is input-dependent? A target
    # encoder that is dominated by a large constant offset (e.g. an
    # under-trained positional table) can have a healthy effective rank, a
    # variance of 1e-3 and a near-zero mean cosine, yet still be useless: the
    # masked-latent loss becomes a positional lookup table. `signal_ratio` is
    # the scale-free detector for that failure mode.
    signal_ratio = float(np.linalg.norm(centered) / max(np.linalg.norm(x), 1e-12))
    collapsed = bool(
        rank < 2.0 or cosine > 0.9 or std.mean() < 1e-6 or signal_ratio < 0.02
    )
    return {
        "n": int(n),
        "dim": int(d),
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "effective_rank": rank,
        "rank_ratio": float(rank / max(1, min(n, d))),
        "mean_cosine": cosine,
        "uniformity": uniformity,
        "degenerate_dim_fraction": degenerate,
        "feature_norm_mean": float(np.linalg.norm(x, axis=1).mean()),
        "signal_ratio": signal_ratio,
        "collapsed": collapsed,
    }


def masked_target_baseline(
    model: "_JEPAPretrainer",
    inputs: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    batch_size: int = 16,
    sample_limit: Optional[int] = None,
    compare_with_model: bool = True,
) -> Dict[str, float]:
    """Quantify how much of the masked-prediction target is *position alone*.

    This is the shortcut every JEPA can fall into at small scale: if the target
    encoder is (close to) frozen, its features are dominated by the positional
    embedding table, so a predictor that ignores the context entirely can still
    drive the latent loss to zero by learning "target position -> expected
    feature vector". The loss looks healthy while nothing semantic is learned.

    The report compares three quantities in the model's own (optionally
    normalised) target space:

    * ``model_mse`` - the loss the model actually achieves on these masks;
    * ``position_mse`` - SSE of a lookup table that predicts the per-position
      mean target and ignores the context completely;
    * ``constant_mse`` - SSE of the single best constant vector.

    ``position_r2 = 1 - position_mse / constant_mse`` is the share of target
    variance explained by position alone: values near 1 mean the objective has
    collapsed onto the positional shortcut.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    patches = model.patchify(inputs)
    if sample_limit is not None:
        patches = patches[:sample_limit]
    dim = model.dim
    sums = np.zeros((model.n_tokens, dim), dtype=np.float64)
    counts = np.zeros(model.n_tokens, dtype=np.float64)
    total_sum = np.zeros(dim, dtype=np.float64)
    total_count = 0.0
    sse_model = 0.0
    n_model = 0.0

    for start in range(0, patches.shape[0], batch_size):
        chunk = patches[start : start + batch_size]
        ctx_idx, tgt_idx, valid = model.sample_masks(chunk.shape[0], rng)
        with no_grad():
            tokens = model.target_encoder(chunk).data
        tgt = np.take_along_axis(tokens, tgt_idx[..., None], axis=1)
        if model.normalize_targets:
            tgt = normalize_latents(tgt)
        if compare_with_model:
            context = model.context_encoder(chunk, ctx_idx)
            pred = model.predictor(context, ctx_idx, tgt_idx).data
            diff = (pred - tgt) ** 2
            sse_model += float(np.sum(diff * valid[..., None]))
            n_model += float(valid.sum()) * dim
        flat = tgt.reshape(-1, dim)
        pos = tgt_idx.reshape(-1)
        w = valid.reshape(-1)
        keep = w > 0
        np.add.at(sums, pos[keep], flat[keep])
        np.add.at(counts, pos[keep], 1.0)
        total_sum += flat[keep].sum(axis=0)
        total_count += float(keep.sum())

    if total_count == 0:
        return {"model_mse": float("nan"), "position_mse": float("nan"),
                "constant_mse": float("nan"), "position_r2": float("nan")}
    pos_means = sums / np.maximum(counts[:, None], 1.0)
    global_mean = total_sum / total_count

    sse_pos = 0.0
    sse_const = 0.0
    sse_within_pos = 0.0
    for start in range(0, patches.shape[0], batch_size):
        chunk = patches[start : start + batch_size]
        ctx_idx, tgt_idx, valid = model.sample_masks(chunk.shape[0], rng)
        with no_grad():
            tokens = model.target_encoder(chunk).data
        tgt = np.take_along_axis(tokens, tgt_idx[..., None], axis=1)
        if model.normalize_targets:
            tgt = normalize_latents(tgt)
        flat = tgt.reshape(-1, dim)
        pos = tgt_idx.reshape(-1)
        keep = valid.reshape(-1) > 0
        sse_pos += float(np.sum((flat[keep] - pos_means[pos[keep]]) ** 2))
        sse_const += float(np.sum((flat[keep] - global_mean) ** 2))

    out = {
        "model_mse": sse_model / n_model if n_model else float("nan"),
        "position_mse": sse_pos / (total_count * dim),
        "constant_mse": sse_const / (total_count * dim),
        "position_r2": 1.0 - sse_pos / max(sse_const, 1e-12),
        "n_target_tokens": total_count,
    }
    return out


def linear_probe(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    l2: float = 1e-2,
    mode: str = "auto",
) -> Dict[str, Any]:
    """Closed-form ridge probe on frozen features.

    Classification (integer or one-hot targets) is solved as ridge regression on
    the one-hot code; regression targets are solved directly. Closed form keeps
    the probe deterministic and instant on CPU, so the reported number measures
    the *features* rather than probe hyper-parameter luck.
    """
    x = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_targets)
    if mode == "auto":
        mode = "regression" if y.dtype.kind == "f" and y.ndim > 1 else "classification"
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    if mode == "classification":
        y_int = np.argmax(y, axis=1) if y.ndim > 1 else y.astype(np.int64)
        k = int(num_classes or (y_int.max() + 1))
        onehot = np.eye(k)[y_int]
        w = _ridge_solve(x, onehot, l2)
    else:
        y_flat = y.reshape(len(y), -1)
        w = _ridge_solve(x, y_flat, l2)

    te_x = np.concatenate([np.asarray(test_features, dtype=np.float64), np.ones((len(test_features), 1))], axis=1)
    pred = te_x @ w
    out: Dict[str, Any] = {"mode": mode, "weights": w}
    if mode == "classification":
        if test_targets is not None:
            y_true = np.asarray(test_targets)
            y_true = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true.astype(np.int64)
            acc = float(np.mean(np.argmax(pred, axis=1) == y_true))
            per_class = []
            for c in range(k):
                tp = int(np.sum((np.argmax(pred, axis=1) == c) & (y_true == c)))
                fp = int(np.sum((np.argmax(pred, axis=1) == c) & (y_true != c)))
                fn = int(np.sum((np.argmax(pred, axis=1) != c) & (y_true == c)))
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                per_class.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
            out.update({"accuracy": acc, "macro_f1": float(np.mean(per_class)), "num_classes": k})
        return out
    if test_targets is not None:
        y_true = np.asarray(test_targets, dtype=np.float64).reshape(len(test_targets), -1)
        ss_res = float(np.sum((y_true - pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
        out.update({"r2": 1.0 - ss_res / max(ss_tot, 1e-12), "mse": ss_res / y_true.size})
    return out


def _ridge_solve(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    d = x.shape[1]
    a = x.T @ x + l2 * np.eye(d)
    return np.linalg.solve(a, x.T @ y)


# ---------------------------------------------------------------------------
# Shared masked-pretraining engine
# ---------------------------------------------------------------------------

class _JEPAPretrainer(Module):
    """Common machinery for I-JEPA and V-JEPA.

    Subclasses provide ``patchify`` / ``sample_masks``; the base class owns the
    context encoder, the EMA target encoder, the narrow predictor, the
    latent-space loss, diagnostics and serialization.
    """

    def __init__(
        self,
        patch_dim: int,
        n_tokens: int,
        grid: Tuple[int, int],
        dim: int = 192,
        depth: int = 6,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        pred_dim: int = 192,
        pred_depth: int = 4,
        pred_heads: int = 6,
        pred_mlp_ratio: float = 4.0,
        pos_embed: str = "sincos",
        norm: str = "layernorm",
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        loss: str = "smooth_l1",
        normalize_targets: bool = True,
        target_mode: str = "tokens",
        grid_t: Optional[int] = None,
        init_std: float = 0.02,
        pos_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if target_mode not in ("tokens", "global"):
            raise ValueError("target_mode must be 'tokens' or 'global'")
        self.pos_scale = float(pos_scale)
        self.target_mode = target_mode
        self.patch_dim = int(patch_dim)
        self.n_tokens = int(n_tokens)
        self.grid = (int(grid[0]), int(grid[1]))
        self.grid_t = grid_t
        self.dim = int(dim)
        self.depth = int(depth)
        self.n_heads = int(n_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.pred_dim = int(pred_dim)
        self.pred_depth = int(pred_depth)
        self.pred_heads = int(pred_heads)
        self.pred_mlp_ratio = float(pred_mlp_ratio)
        self.pos_embed = pos_embed
        self.norm = norm
        self.ema_start = float(ema_start)
        self.ema_end = float(ema_end)
        self.loss_kind = loss
        self.normalize_targets = bool(normalize_targets)
        self.init_std = float(init_std)

        self.context_encoder = JEPAEncoder(
            patch_dim, dim, depth, n_heads, mlp_ratio, n_tokens, grid, grid_t, pos_embed, norm, init_std,
            pos_scale=pos_scale,
        )
        self.target_encoder = JEPAEncoder(
            patch_dim, dim, depth, n_heads, mlp_ratio, n_tokens, grid, grid_t, pos_embed, norm, init_std,
            pos_scale=pos_scale,
        )
        self.target_encoder.eval()
        for target_param, source_param in zip(
            named_parameters(self.target_encoder), named_parameters(self.context_encoder)
        ):
            target_param[1].data = source_param[1].data.copy()
        predictor_table = self.context_encoder.pos_table
        predictor_param = self.context_encoder.learned_pos
        if predictor_table is None and predictor_param is None:
            predictor_table = np.zeros((n_tokens, pred_dim), dtype=np.float32)
        self.predictor = JEPAPredictor(
            dim,
            pred_dim,
            pred_depth,
            pred_heads,
            pred_mlp_ratio,
            pos_table=predictor_table,
            pos_parameter=predictor_param,
            norm=norm,
            init_std=init_std,
        )
        self.target_encoder_is_ema = True

    # -- to be provided by subclasses ------------------------------------
    def patchify(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample_masks(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    # -- shared behaviour -------------------------------------------------
    def update_target(self, momentum: float) -> None:
        ema_update(self.target_encoder, self.context_encoder, momentum)

    def trainable_parameters(self) -> List[Parameter]:
        """Encoder + predictor parameters; the target encoder is never trained."""
        return [p for _, p in named_parameters(self.context_encoder)] + [
            p for _, p in named_parameters(self.predictor)
        ]

    def loss_fn(self, pred: Tensor, target: Tensor, valid: Optional[np.ndarray] = None) -> Tensor:
        if self.loss_kind == "l2":
            return latent_l2_loss(pred, target, valid)
        return smooth_l1_loss(pred, target, valid=valid)

    def forward(
        self,
        inputs: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        teacher: str = "ema",
    ) -> Dict[str, Any]:
        """One masked-prediction forward pass; returns loss, masks and latents.

        ``teacher="ema"`` uses the exponential-moving-average target encoder
        (I-JEPA / V-JEPA). ``teacher="self"`` uses a stop-gradient copy of the
        live encoder over the *unmasked* signal, which is the teacher-free
        variant that pairs with :class:`SIGReg` (LeJEPA).
        """
        rng = np.random.default_rng() if rng is None else rng
        patches = self.patchify(inputs)
        ctx_idx, tgt_idx, valid = self.sample_masks(patches.shape[0], rng)
        context = self.context_encoder(patches, ctx_idx)
        pred = self.predictor(context, ctx_idx, tgt_idx)
        with no_grad():
            target_encoder = self.target_encoder if teacher == "ema" else self.context_encoder
            target_tokens = target_encoder(patches)
        if self.target_mode == "global":
            # Predict the *whole-image* latent from the masked view: the target
            # is the mean-pooled embedding of the unmasked signal, so the model
            # cannot satisfy the loss with a position lookup table. Combined
            # with image-level SIGReg this is the LeJEPA objective, where the
            # thing being predicted is a view embedding rather than a token.
            target = target_tokens.data.mean(axis=1)
            if self.normalize_targets:
                target = normalize_latents(target[:, None])[:, 0]
            pred = pred.mean(axis=1)
            loss = self.loss_fn(pred[:, None], _constant(target[:, None]), None)
            return {
                "loss": loss,
                "predictions": pred[:, None],
                "context": context,
                "targets": target[:, None],
                "context_idx": ctx_idx,
                "target_idx": tgt_idx,
                "valid": np.ones((patches.shape[0], 1), dtype=bool),
                "patches": patches,
            }
        target = np.take_along_axis(target_tokens.data, tgt_idx[..., None], axis=1)
        if self.normalize_targets:
            target = normalize_latents(target)
        loss = self.loss_fn(pred, _constant(target), valid)
        return {
            "loss": loss,
            "predictions": pred,
            "context": context,
            "targets": target,
            "context_idx": ctx_idx,
            "target_idx": tgt_idx,
            "valid": valid,
            "patches": patches,
        }

    def loss_only(self, inputs: np.ndarray, rng: Optional[np.random.Generator] = None) -> float:
        return float(self.forward(inputs, rng)["loss"].data)

    def encode(
        self,
        inputs: np.ndarray,
        batch_size: int = 32,
        pool: str = "mean",
        use_target: bool = False,
        normalize: bool = False,
        progress: bool = False,
    ) -> np.ndarray:
        """Frozen features for downstream probes / world models.

        Runs the encoder over *all* tokens (no masking) and pools. Following the
        I-JEPA paper, the context encoder is the artefact worth keeping; the
        target encoder is only needed during pretraining.
        """
        encoder = self.target_encoder if use_target else self.context_encoder
        patches = self.patchify(inputs)
        chunks = []
        for start in range(0, patches.shape[0], batch_size):
            with no_grad():
                tokens = encoder(patches[start : start + batch_size])
            if pool == "mean":
                chunk = tokens.data.mean(axis=1)
            elif pool == "max":
                chunk = tokens.data.max(axis=1)
            elif pool == "tokens":
                chunk = tokens.data
            else:
                raise ValueError(f"unknown pool: {pool}")
            chunks.append(chunk)
            if progress and (start // batch_size) % 10 == 0:
                print(f"    encoded {min(start + batch_size, patches.shape[0])}/{patches.shape[0]}", flush=True)
        feats = np.concatenate(chunks, axis=0)
        if normalize:
            feats = feats / np.maximum(np.linalg.norm(feats, axis=-1, keepdims=True), 1e-8)
        return feats.astype(np.float32)

    def fit(
        self,
        inputs: np.ndarray,
        steps: int = 500,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        rng: Optional[np.random.Generator] = None,
        verbose_every: int = 50,
        sigreg: bool = False,
        sigreg_weight: float = 0.1,
        sigreg_slices: int = 64,
        sigreg_level: str = "token",
        on_step: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Train the encoder + predictor; the target encoder follows by EMA.

        ``sigreg=True`` swaps the EMA teacher for the LeJEPA objective: the
        target branch is detached and the embeddings are pushed towards an
        isotropic Gaussian instead, which is collapse-free by construction.
        """
        from .transformer import TensorAdamW  # local import avoids a cycle

        rng = np.random.default_rng(0) if rng is None else rng
        params = self.trainable_parameters()
        optimizer = TensorAdamW(params, lr=lr, weight_decay=weight_decay)
        regularizer = SIGReg(num_slices=sigreg_slices, seed=int(rng.integers(1 << 30))) if sigreg else None
        teacher = "self" if sigreg else "ema"
        n = len(inputs)
        history: List[float] = []
        for step in range(steps):
            idx = rng.integers(0, n, size=min(batch_size, n))
            batch = inputs[idx]
            out = self.forward(batch, rng, teacher=teacher)
            loss = out["loss"]
            if regularizer is not None:
                # SIGReg acts on every latent the encoder produces: the context
                # tokens and the predicted target tokens. Pushing that cloud
                # towards an isotropic Gaussian is what removes the need for the
                # EMA teacher and the stop-gradient.
                #
                # ``sigreg_level="token"`` regularises the token cloud, which is
                # what the reference LeJEPA implementation does for patch
                # encoders. ``sigreg_level="image"`` regularises the *pooled*
                # per-image latents instead, and that distinction decides whether
                # a position-only encoder survives: its token cloud is diverse
                # (every position gets its own vector) but its pooled latents are
                # a constant, so only the image-level penalty rules it out.
                d = out["predictions"].shape[-1]
                if sigreg_level == "image":
                    latents = Tensor.cat(
                        [out["context"].mean(axis=1), out["predictions"].mean(axis=1)], axis=0
                    )
                else:
                    latents = Tensor.cat(
                        [out["context"].reshape(-1, d), out["predictions"].reshape(-1, d)], axis=0
                    )
                loss = loss + regularizer(latents) * sigreg_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.target_encoder_is_ema and not sigreg:
                self.update_target(ema_schedule(step, steps, self.ema_start, self.ema_end))
            history.append(float(loss.data))
            if verbose_every and (step + 1) % verbose_every == 0:
                print(f"    step {step + 1:5d}/{steps} | loss {np.mean(history[-verbose_every:]):.4f}", flush=True)
            if on_step is not None:
                on_step(step, float(loss.data))
        return {"history": history, "final_loss": float(np.mean(history[-20:])) if history else 0.0}

    def diagnostics(self, inputs: np.ndarray, sample: int = 128, batch_size: int = 32) -> Dict[str, float]:
        """Collapse diagnostics (rank, cosine, input-dependent energy) of the encoder."""
        feats = self.encode(inputs[:sample], batch_size=batch_size)
        return representation_diagnostics(feats)

    def shortcut_audit(
        self,
        inputs: np.ndarray,
        sample_limit: int = 128,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        """Is the latent loss just a positional lookup table? See :func:`masked_target_baseline`."""
        return masked_target_baseline(
            self, inputs, rng=rng, sample_limit=sample_limit
        )

    # -- serialization ----------------------------------------------------
    def config(self) -> Dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "patch_dim": self.patch_dim,
            "n_tokens": self.n_tokens,
            "grid": list(self.grid),
            "grid_t": self.grid_t,
            "dim": self.dim,
            "depth": self.depth,
            "n_heads": self.n_heads,
            "mlp_ratio": self.mlp_ratio,
            "pred_dim": self.pred_dim,
            "pred_depth": self.pred_depth,
            "pred_heads": self.pred_heads,
            "pred_mlp_ratio": self.pred_mlp_ratio,
            "pos_embed": self.pos_embed,
            "norm": self.norm,
            "ema_start": self.ema_start,
            "ema_end": self.ema_end,
            "loss": self.loss_kind,
            "normalize_targets": self.normalize_targets,
            "target_mode": self.target_mode,
            "init_std": self.init_std,
            "pos_scale": self.pos_scale,
        }

    def state_dict(self) -> Dict[str, np.ndarray]:
        tensors: Dict[str, np.ndarray] = {}
        for part in ("context_encoder", "target_encoder", "predictor"):
            module = getattr(self, part)
            for name, param in named_parameters(module, prefix=f"{part}."):
                tensors[name] = param.data
        return tensors

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        npz = p.with_suffix(".npz")
        json_path = p.with_suffix(".json")
        np.savez(npz, **self.state_dict())
        json_path.write_text(json.dumps(self.config(), indent=2), encoding="utf-8")
        return npz

    @classmethod
    def load(cls, path: Union[str, Path]) -> "_JEPAPretrainer":
        p = Path(path)
        cfg = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        cls = globals()[cfg.pop("kind")]
        cfg = _filter_kwargs(cls.__init__, cfg)
        module = cls.from_config(cfg)
        weights = np.load(p.with_suffix(".npz"))
        for name, param in named_parameters(module):
            if name in weights:
                param.data = weights[name].astype(np.float32, copy=True)
        return module


save_jepa = _JEPAPretrainer.save
load_jepa = _JEPAPretrainer.load


class IJEPA(_JEPAPretrainer):
    """Image Joint-Embedding Predictive Architecture (I-JEPA, CVPR 2023).

    Example
    -------
    >>> import numpy as np
    >>> from doraneural.jepa import IJEPA, linear_probe
    >>> images = np.random.rand(32, 1, 32, 32).astype(np.float32)   # (B, C, H, W)
    >>> model = IJEPA(img_size=32, patch_size=8, dim=64, depth=2, n_heads=4)
    >>> stats = model.fit(images, steps=50, batch_size=8)
    >>> features = model.encode(images)          # (32, 64) frozen features
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        in_chans: int = 1,
        dim: int = 192,
        depth: int = 6,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        pred_dim: int = 192,
        pred_depth: int = 4,
        pred_heads: int = 6,
        pred_mlp_ratio: float = 4.0,
        n_targets: int = 4,
        target_scale: Tuple[float, float] = (0.15, 0.2),
        context_scale: Tuple[float, float] = (0.85, 1.0),
        aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        pos_embed: str = "sincos",
        norm: str = "layernorm",
        loss: str = "smooth_l1",
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        normalize_targets: bool = True,
        target_mode: str = "tokens",
        init_std: float = 0.02,
        pos_scale: float = 1.0,
    ) -> None:
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        grid = (img_size // patch_size, img_size // patch_size)
        super().__init__(
            patch_dim=in_chans * patch_size * patch_size,
            n_tokens=grid[0] * grid[1],
            grid=grid,
            dim=dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            pred_dim=pred_dim,
            pred_depth=pred_depth,
            pred_heads=pred_heads,
            pred_mlp_ratio=pred_mlp_ratio,
            pos_embed=pos_embed,
            norm=norm,
            ema_start=ema_start,
            ema_end=ema_end,
            loss=loss,
            normalize_targets=normalize_targets,
            target_mode=target_mode,
            init_std=init_std,
            pos_scale=pos_scale,
        )
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        self.n_targets = int(n_targets)
        self.target_scale = tuple(target_scale)
        self.context_scale = tuple(context_scale)
        self.aspect_ratio = tuple(aspect_ratio)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "IJEPA":
        return cls(**cfg)

    def patchify(self, inputs: np.ndarray) -> np.ndarray:
        return patchify_images(inputs, self.patch_size)

    def sample_masks(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        contexts, targets = [], []
        for _ in range(batch_size):
            ctx, tgt = sample_ijepa_masks(
                self.grid[0],
                self.grid[1],
                n_targets=self.n_targets,
                target_scale=self.target_scale,
                context_scale=self.context_scale,
                aspect_ratio=self.aspect_ratio,
                rng=rng,
            )
            contexts.append(ctx)
            targets.append(tgt)
        return collate_masks(contexts, targets, rng=rng)

    def config(self) -> Dict[str, Any]:
        cfg = super().config()
        cfg.update(
            {
                "kind": "IJEPA",
                "img_size": self.img_size,
                "patch_size": self.patch_size,
                "in_chans": self.in_chans,
                "n_targets": self.n_targets,
                "target_scale": list(self.target_scale),
                "context_scale": list(self.context_scale),
                "aspect_ratio": list(self.aspect_ratio),
            }
        )
        for key in ("patch_dim", "n_tokens", "grid", "grid_t"):
            cfg.pop(key, None)
        return cfg


class VJEPA(_JEPAPretrainer):
    """Video JEPA (V-JEPA / V-JEPA 2 style) with tubelet tokens and tube masking.

    Inputs are clips ``(B, C, T, H, W)``. Tokens are tubelets of
    ``tubelet x patch x patch``; masks are sampled spatially and extruded over
    time, and the target encoder's *output* is masked (never its input), which is
    the design choice I-JEPA/V-JEPA both report as load-bearing for semantic
    targets.

    >>> clips = np.random.rand(8, 1, 6, 32, 32).astype(np.float32)
    >>> vj = VJEPA(frames=6, img_size=32, patch_size=8, tubelet=2, dim=64, depth=2, n_heads=4)
    >>> _ = vj.fit(clips, steps=20, batch_size=4)
    """

    def __init__(
        self,
        frames: int = 8,
        img_size: int = 32,
        patch_size: int = 8,
        tubelet: int = 2,
        in_chans: int = 1,
        dim: int = 192,
        depth: int = 6,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        pred_dim: int = 192,
        pred_depth: int = 4,
        pred_heads: int = 6,
        pred_mlp_ratio: float = 4.0,
        n_targets: int = 2,
        target_scale: Tuple[float, float] = (0.15, 0.4),
        aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        temporal_scale: Tuple[float, float] = (1.0, 1.0),
        pos_embed: str = "sincos",
        norm: str = "layernorm",
        loss: str = "smooth_l1",
        ema_start: float = 0.998,
        ema_end: float = 1.0,
        normalize_targets: bool = True,
        target_mode: str = "tokens",
        init_std: float = 0.02,
        pos_scale: float = 1.0,
    ) -> None:
        if frames % tubelet:
            raise ValueError("frames must be divisible by tubelet")
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        grid_t, gh, gw = frames // tubelet, img_size // patch_size, img_size // patch_size
        super().__init__(
            patch_dim=in_chans * tubelet * patch_size * patch_size,
            n_tokens=grid_t * gh * gw,
            grid=(gh, gw),
            grid_t=grid_t,
            dim=dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            pred_dim=pred_dim,
            pred_depth=pred_depth,
            pred_heads=pred_heads,
            pred_mlp_ratio=pred_mlp_ratio,
            pos_embed=pos_embed,
            norm=norm,
            ema_start=ema_start,
            ema_end=ema_end,
            loss=loss,
            normalize_targets=normalize_targets,
            target_mode=target_mode,
            init_std=init_std,
            pos_scale=pos_scale,
        )
        self.frames = int(frames)
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.tubelet = int(tubelet)
        self.in_chans = int(in_chans)
        self.n_targets = int(n_targets)
        self.target_scale = tuple(target_scale)
        self.aspect_ratio = tuple(aspect_ratio)
        self.temporal_scale = tuple(temporal_scale)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "VJEPA":
        return cls(**cfg)

    def patchify(self, inputs: np.ndarray) -> np.ndarray:
        return patchify_video(inputs, self.tubelet, self.patch_size)

    def sample_masks(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        contexts, targets = [], []
        for _ in range(batch_size):
            ctx, tgt = sample_vjepa_masks(
                self.grid_t,
                self.grid[0],
                self.grid[1],
                n_targets=self.n_targets,
                target_scale=self.target_scale,
                aspect_ratio=self.aspect_ratio,
                temporal_scale=self.temporal_scale,
                rng=rng,
            )
            contexts.append(ctx)
            targets.append(tgt)
        return collate_masks(contexts, targets, rng=rng)

    def config(self) -> Dict[str, Any]:
        cfg = super().config()
        cfg.update(
            {
                "kind": "VJEPA",
                "frames": self.frames,
                "img_size": self.img_size,
                "patch_size": self.patch_size,
                "tubelet": self.tubelet,
                "in_chans": self.in_chans,
                "n_targets": self.n_targets,
                "target_scale": list(self.target_scale),
                "aspect_ratio": list(self.aspect_ratio),
                "temporal_scale": list(self.temporal_scale),
            }
        )
        for key in ("patch_dim", "n_tokens", "grid", "grid_t"):
            cfg.pop(key, None)
        return cfg


# ---------------------------------------------------------------------------
# Action-conditioned latent world model (V-JEPA 2-AC / LeWorldModel style)
# ---------------------------------------------------------------------------

class JEPADynamics(Module):
    """Block-causal transformer over interleaved ``[z_0, a_0, z_1, a_1, ...]`` tokens.

    Attention is block-causal: the action token at step ``t`` may attend to
    everything up to and including step ``t``, so the output at that position is
    a prediction of ``z_{t+1}`` that never sees the future. This is the
    V-JEPA 2-AC predictor design in miniature.

    With ``residual=True`` (the default) the head predicts the *change*
    ``z_{t+1} - z_t`` and the last latent is added back, so an untrained
    predictor behaves exactly like the no-change baseline. This matters more
    than it looks: in high-entropy domains the absolute-prediction objective has
    a strong degenerate optimum that shrinks the output towards the marginal
    distribution of the latents and ignores the action input entirely (the
    action-permutation audit in :meth:`JEPAWorldModel.action_utilization`
    measures it directly). The residual form removes that optimum, because
    predicting "nothing changes" is no longer free.
    """

    def __init__(
        self,
        dim: int = 128,
        action_dim: int = 2,
        depth: int = 2,
        n_heads: int = 4,
        mlp_ratio: float = 2.0,
        norm: str = "layernorm",
        init_std: float = 0.02,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.residual = bool(residual)
        self.z_proj = Parameter(_init_weight((dim, dim), init_std))
        self.a_proj = Parameter(_init_weight((action_dim, dim), init_std))
        self.blocks = [
            JEPAEncoderBlock(dim, n_heads, mlp_ratio, norm=norm, init_std=init_std) for _ in range(depth)
        ]
        self.final_norm = _make_norm(norm, dim)
        self.out = Parameter(_init_weight((dim, dim), init_std))

    @staticmethod
    def causal_bias(seq_len: int) -> np.ndarray:
        """Additive block-causal mask over ``2T`` interleaved tokens."""
        steps = np.arange(seq_len) // 2
        blocked = steps[None, :] > steps[:, None]
        bias = np.zeros((seq_len, seq_len), dtype=np.float32)
        bias[blocked] = -1e9
        return bias

    def forward(self, z_seq: Tensor, a_seq: Tensor) -> Tensor:
        """``z_seq``: (B, T, D), ``a_seq``: (B, Ta, A) -> next-latent predictions.

        The two sequences are aligned on their common length, so a trajectory of
        ``T`` latents with ``T-1`` actions yields ``T-1`` predictions of
        ``z_{1..T-1}``.
        """
        t = min(z_seq.shape[1], a_seq.shape[1])
        if z_seq.shape[1] != t:
            z_seq = z_seq[:, :t]
        if a_seq.shape[1] != t:
            a_seq = a_seq[:, :t]
        b, t, d = z_seq.shape
        z_tok = z_seq @ self.z_proj
        a_tok = a_seq @ self.a_proj
        seq = Tensor.cat([z_tok, a_tok], axis=1)                    # (B, 2T, D)
        perm = np.empty(2 * t, dtype=np.int64)
        perm[0::2] = np.arange(t)
        perm[1::2] = np.arange(t, 2 * t)
        seq = seq[:, perm]                                          # (B, 2T, D) interleaved
        bias = self.causal_bias(2 * t)
        for block in self.blocks:
            seq = block(seq, attn_bias=bias)
        seq = self.final_norm(seq)
        preds = (seq @ self.out)[:, 1::2]                           # position t predicts z_{t+1}
        if self.residual:
            preds = preds + z_seq                                    # delta head + last latent
        return preds

    def predict_next(self, z_hist: Tensor, a_hist: Tensor) -> Tensor:
        """Single-step prediction of the next latent from a history window."""
        return self.forward(z_hist, a_hist)[:, -1]


class JEPAWorldModel(Module):
    """Action-conditioned JEPA world model learned from pixels (or state vectors).

    Two-stage recipe (V-JEPA 2 style): a JEPA-pretrained encoder supplies the
    latent space, then this module learns ``z_{t+1} = f(z_{<=t}, a_{<=t})`` with a
    latent prediction loss plus :class:`SIGReg`. The optional end-to-end mode
    (``freeze_encoder=False``, ``sigreg_weight > 0``) is the LeWorldModel recipe:
    no EMA, no stop-gradient, one trade-off hyperparameter.

    Planning happens entirely in latent space with the cross-entropy method: see
    :meth:`plan`.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...] = (1, 32, 32),
        action_dim: int = 2,
        encoder: Optional[Module] = None,
        dim: int = 96,
        encoder_depth: int = 3,
        encoder_heads: int = 4,
        patch_size: int = 8,
        predictor_dim: int = 128,
        predictor_depth: int = 2,
        predictor_heads: int = 4,
        predictor_mlp_ratio: float = 2.0,
        context_len: int = 4,
        freeze_encoder: bool = True,
        detach_targets: bool = True,
        sigreg_slices: int = 64,
        norm: str = "layernorm",
        init_std: float = 0.02,
        residual: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.action_dim = int(action_dim)
        self.dim = int(dim)
        self.context_len = int(context_len)
        self.freeze_encoder = bool(freeze_encoder)
        self.detach_targets = bool(detach_targets)
        self.patch_size = int(patch_size)
        self.pixels = len(self.obs_shape) == 3
        # keep the construction arguments so a checkpoint can be rebuilt exactly
        self.encoder_depth = int(encoder_depth)
        self.encoder_heads = int(encoder_heads)
        self.predictor_depth = int(predictor_depth)
        self.predictor_heads = int(predictor_heads)
        self.predictor_mlp_ratio = float(predictor_mlp_ratio)
        self.norm = norm
        self.init_std = float(init_std)
        self.sigreg_slices = int(sigreg_slices)
        self.residual = bool(residual)

        if encoder is not None:
            self.encoder = encoder
            self.dim = int(getattr(encoder, "dim", dim))
            self.pixels = isinstance(encoder, JEPAEncoder)
        elif self.pixels:
            c, h, w = self.obs_shape
            if h % patch_size or w % patch_size:
                raise ValueError("obs spatial size must be divisible by patch_size")
            grid = (h // patch_size, w // patch_size)
            self.encoder = JEPAEncoder(
                c * patch_size * patch_size,
                dim,
                encoder_depth,
                encoder_heads,
                mlp_ratio=2.0,
                n_tokens=grid[0] * grid[1],
                grid=grid,
                pos_embed="sincos",
                norm=norm,
                init_std=init_std,
            )
        else:
            self.encoder = VectorEncoder(self.obs_shape[0], dim, encoder_depth, mlp_ratio=2.0, norm=norm, init_std=init_std)

        self.dynamics = JEPADynamics(
            self.dim, action_dim, predictor_depth, predictor_heads, predictor_mlp_ratio, norm, init_std,
            residual=residual,
        )
        self.sigreg = SIGReg(num_slices=sigreg_slices, seed=seed)
        if self.freeze_encoder:
            for _, param in named_parameters(self.encoder):
                param.requires_grad = False

    # -- encoding ---------------------------------------------------------
    def encode(self, obs: np.ndarray) -> Tensor:
        """Encode observations ``(B, ...)`` into a single latent vector per sample."""
        obs = np.asarray(obs, dtype=np.float32)
        if self.pixels:
            tokens = self.encoder(patchify_images(obs, self.patch_size))
        else:
            tokens = self.encoder(obs.reshape(len(obs), -1))
        if tokens.ndim == 3:
            return tokens.mean(axis=1)
        return tokens

    def encode_np(self, obs: np.ndarray, batch_size: int = 64) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        chunks = []
        for start in range(0, len(obs), batch_size):
            with no_grad():
                chunks.append(self.encode(obs[start : start + batch_size]).data)
        return np.concatenate(chunks, axis=0)

    # -- prediction / rollout --------------------------------------------
    def predict_next(self, z_hist: Tensor, a_hist: Tensor) -> Tensor:
        """Predict the next latent from a history window ``(B, T, D)`` + ``(B, T, A)``."""
        if z_hist.shape[1] > self.context_len:
            z_hist = z_hist[:, -self.context_len :]
            a_hist = a_hist[:, -self.context_len :]
        return self.dynamics.predict_next(z_hist, a_hist)

    def rollout(self, z0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Autoregressively roll the model forward in latent space (no gradients).

        ``z0``: (B, D) start latent, ``actions``: (B, H, A). Returns the final
        latent of each rollout plus the full latent trajectory.
        """
        actions = np.asarray(actions, dtype=np.float32)
        z = np.asarray(z0, dtype=np.float32)
        if z.ndim == 1:
            z = z[None]
        horizon = actions.shape[1]
        history = [z]
        with no_grad():
            for t in range(horizon):
                z_hist = np.stack(history[-(self.context_len + 1) : -1] or [z], axis=1)
                a_hist = actions[:, max(0, t - self.context_len + 1) : t + 1]
                if z_hist.shape[1] > a_hist.shape[1]:
                    z_hist = z_hist[:, -a_hist.shape[1] :]
                elif a_hist.shape[1] > z_hist.shape[1]:
                    a_hist = a_hist[:, -z_hist.shape[1] :]
                z_next = self.predict_next(_constant(z_hist), _constant(a_hist)).data
                history.append(z_next)
                z = z_next
        return z, np.stack(history, axis=1)

    # -- action utilisation ----------------------------------------------
    def action_utilization(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        seed: int = 0,
        max_horizon: Optional[int] = None,
    ) -> Dict[str, float]:
        """Measure how much the predictor actually *uses* the action sequence.

        A latent predictor can score a decent next-latent MSE while ignoring its
        action input entirely: in high-entropy domains (a point mass whose action
        is a pure velocity command is a random walk) the optimal deterministic
        prediction of the next latent is close to the marginal distribution, and
        shrinkage towards that marginal already halves the error of a no-change
        predictor. This method separates the two stories by re-running the very
        same rollout with the action stream (a) intact, (b) permuted across the
        batch - same distribution, wrong pairing - and (c) set to zero.
        A predictor that uses the actions loses accuracy under (b) and (c); one
        that has collapsed onto the marginal does not care.

        Returns ``one_step`` and ``multi_step`` blocks; the headline number is
        ``action_gain`` = 1 - true/shuffled, i.e. the fraction of the
        shuffled-action error that the correct action pairing removes.
        """
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        horizon = int(max_horizon or (obs.shape[1] - 1))
        horizon = max(1, min(horizon, obs.shape[1] - 1))
        z = self.encode_np(obs.reshape(-1, *self.obs_shape)).reshape(len(obs), obs.shape[1], -1)
        rng = np.random.default_rng(seed)

        def one_step_variant(act: np.ndarray) -> float:
            errs = []
            for t in range(horizon):
                pred = self.rollout(z[:, t], act[:, t : t + 1])[0]
                errs.append(float(np.mean((pred - z[:, t + 1]) ** 2)))
            return float(np.mean(errs))

        def multi_step_variant(act: np.ndarray) -> float:
            _, traj = self.rollout(z[:, 0], act[:, :horizon])
            return float(np.mean((traj[:, 1 : horizon + 1] - z[:, 1 : horizon + 1]) ** 2))

        perm = rng.permutation(len(actions))
        zero = np.zeros_like(actions)
        out: Dict[str, float] = {}
        for name, fn in (("one_step", one_step_variant), ("multi_step", multi_step_variant)):
            true = fn(actions)
            shuffled = fn(actions[perm])
            zeroed = fn(zero)
            out[f"{name}_true_action_mse"] = round(true, 4)
            out[f"{name}_shuffled_action_mse"] = round(shuffled, 4)
            out[f"{name}_zero_action_mse"] = round(zeroed, 4)
            out[f"{name}_action_gain"] = round(1.0 - true / max(shuffled, 1e-12), 4)
        return out

    # -- training ---------------------------------------------------------
    def train_step(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        optimizer: Any,
        sigreg_weight: float = 0.1,
        rollout_steps: int = 1,
        rng: Optional[np.random.Generator] = None,
        lambda_pred: float = 1.0,
    ) -> Dict[str, float]:
        """One optimisation step on a batch of trajectories.

        ``obs``: (B, T, ...) observations, ``actions``: (B, T-1, A).
        Loss = smooth-L1 next-latent prediction (+ optional k-step rollout term)
        + ``sigreg_weight`` * SIGReg(z). SIGReg is what keeps the latent cloud
        from collapsing when the encoder is trained jointly.
        """
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        b, t = obs.shape[0], obs.shape[1]
        z = self.encode(obs.reshape((b * t,) + self.obs_shape)).reshape(b, t, self.dim)

        a_seq_full = _constant(actions)
        pred = self.dynamics(z, a_seq_full)                     # (B, T-1, D) predicts z_{1..T-1}
        target = z[:, 1:]
        if self.detach_targets:
            target = _constant(target.data)
        pred_loss = smooth_l1_loss(pred, target)

        rollout_loss = _constant(np.zeros((), dtype=np.float32))
        if rollout_steps > 1 and t > rollout_steps:
            # Train on the model's own predictions: roll the latent forward with
            # a growing history so the k-step prediction is conditioned on the
            # same (partly imagined) context it will see at plan time. This is
            # the rollout term that fights compounding error during planning.
            horizon = min(rollout_steps, t - 1)
            history = [z[:, 0:1]]
            history_a: List[Tensor] = []
            for k in range(horizon):
                a_k = a_seq_full[:, k : k + 1]
                z_hist = Tensor.cat(history[-self.context_len :], axis=1)
                a_hist = Tensor.cat(history_a[-self.context_len :] + [a_k], axis=1)
                if z_hist.shape[1] > a_hist.shape[1]:
                    z_hist = z_hist[:, -a_hist.shape[1] :]
                z_next = self.predict_next(z_hist, a_hist)
                history.append(z_next[:, None])
                history_a.append(a_k)
            rollout_loss = smooth_l1_loss(history[-1], target[:, horizon - 1])
            pred_loss = pred_loss + rollout_loss

        reg = self.sigreg(z.reshape(b * t, self.dim))
        loss = pred_loss * lambda_pred + reg * sigreg_weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {
            "loss": float(loss.data),
            "pred_loss": float(pred_loss.data),
            "sigreg": float(reg.data),
            "rollout_loss": float(rollout_loss.data),
        }

    def fit(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        steps: int = 300,
        batch_size: int = 8,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        sigreg_weight: float = 0.1,
        rollout_steps: int = 1,
        rng: Optional[np.random.Generator] = None,
        verbose_every: int = 50,
        on_step: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Train the world model on trajectories ``(N, T, ...)`` + ``(N, T-1, A)``."""
        from .transformer import TensorAdamW

        rng = np.random.default_rng(0) if rng is None else rng
        params = [p for _, p in named_parameters(self.encoder)] + [
            p for _, p in named_parameters(self.dynamics)
        ]
        optimizer = TensorAdamW([p for p in params if p.requires_grad], lr=lr, weight_decay=weight_decay)
        history: List[Dict[str, float]] = []
        n = len(obs)
        for step in range(steps):
            idx = rng.integers(0, n, size=min(batch_size, n))
            stats = self.train_step(
                obs[idx],
                actions[idx],
                optimizer,
                sigreg_weight=sigreg_weight,
                rollout_steps=rollout_steps,
                rng=rng,
            )
            history.append(stats)
            if verbose_every and (step + 1) % verbose_every == 0:
                recent = history[-verbose_every:]
                print(
                    f"    step {step + 1:5d}/{steps} | loss {np.mean([h['loss'] for h in recent]):.4f}"
                    f" | pred {np.mean([h['pred_loss'] for h in recent]):.4f}"
                    f" | sigreg {np.mean([h['sigreg'] for h in recent]):.4f}",
                    flush=True,
                )
            if on_step is not None:
                on_step(step, stats)
        return {"history": history}

    # -- planning ---------------------------------------------------------
    def plan(
        self,
        z_start: np.ndarray,
        z_goal: np.ndarray,
        horizon: int = 6,
        n_samples: int = 192,
        n_elite: int = 16,
        n_iters: int = 4,
        action_low: float = -1.0,
        action_high: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        init_std: float = 1.0,
        min_std: float = 0.05,
    ) -> Dict[str, Any]:
        """Cross-entropy-method planning in latent space.

        Minimises ``|| rollout(z_start, a_{1:H}) - z_goal ||^2`` over action
        sequences. Every candidate rollout is a batched forward pass of the
        learned dynamics, so no environment interaction and no decoding to
        pixels is required.
        """
        rng = np.random.default_rng(0) if rng is None else rng
        z_start = np.atleast_2d(np.asarray(z_start, dtype=np.float32))
        z_goal = np.atleast_2d(np.asarray(z_goal, dtype=np.float32))
        mu = np.zeros((horizon, self.action_dim), dtype=np.float32)
        sigma = np.full((horizon, self.action_dim), float(init_std), dtype=np.float32)
        best_cost, best_actions, costs = np.inf, mu, None
        for _ in range(n_iters):
            samples = rng.normal(mu, sigma, size=(n_samples, horizon, self.action_dim)).astype(np.float32)
            samples = np.clip(samples, action_low, action_high)
            z0 = np.repeat(z_start, n_samples, axis=0)
            z_final, _ = self.rollout(z0, samples)
            costs = np.sum((z_final - z_goal) ** 2, axis=-1)
            order = np.argsort(costs)
            elite = samples[order[:n_elite]]
            mu = elite.mean(axis=0)
            sigma = np.maximum(elite.std(axis=0), min_std)
            if float(costs[order[0]]) < best_cost:
                best_cost = float(costs[order[0]])
                best_actions = samples[order[0]].copy()
        return {
            "actions": best_actions,
            "cost": best_cost,
            "final_cost": float(np.sum((self.rollout(z_start, best_actions[None])[0] - z_goal) ** 2)),
            "mean_cost": float(np.mean(costs)) if costs is not None else np.inf,
        }

    def sigreg_value(self, obs: np.ndarray, sample: int = 256) -> float:
        z = self.encode_np(np.asarray(obs)[:sample])
        with no_grad():
            return float(self.sigreg(_constant(z)).data)

    # -- serialization ----------------------------------------------------
    def config(self) -> Dict[str, Any]:
        return {
            "kind": "JEPAWorldModel",
            "obs_shape": list(self.obs_shape),
            "action_dim": self.action_dim,
            "dim": self.dim,
            "context_len": self.context_len,
            "freeze_encoder": self.freeze_encoder,
            "detach_targets": self.detach_targets,
            "patch_size": self.patch_size,
            "pixels": self.pixels,
            "encoder_depth": self.encoder_depth,
            "encoder_heads": self.encoder_heads,
            "predictor_dim": self.dynamics.dim,
            "predictor_depth": self.predictor_depth,
            "predictor_heads": self.predictor_heads,
            "predictor_mlp_ratio": self.predictor_mlp_ratio,
            "norm": self.norm,
            "init_std": self.init_std,
            "sigreg_slices": self.sigreg_slices,
            "residual": self.residual,
        }

    def state_dict(self) -> Dict[str, np.ndarray]:
        tensors: Dict[str, np.ndarray] = {}
        for part in ("encoder", "dynamics"):
            for name, param in named_parameters(getattr(self, part), prefix=f"{part}."):
                tensors[name] = param.data
        return tensors

    def save(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p.with_suffix(".npz"), **self.state_dict())
        p.with_suffix(".json").write_text(json.dumps(self.config(), indent=2), encoding="utf-8")
        return p.with_suffix(".npz")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "JEPAWorldModel":
        p = Path(path)
        cfg = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        cfg.pop("kind", None)
        module = cls(**_filter_kwargs(cls.__init__, cfg))
        weights = np.load(p.with_suffix(".npz"))
        for name, param in named_parameters(module):
            if name in weights:
                param.data = weights[name].astype(np.float32, copy=True)
        return module
