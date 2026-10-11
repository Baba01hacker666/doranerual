"""Zero-dependency synthetic benchmarks for the JEPA implementations.

Three generators live here, each one chosen because it isolates a property the
JEPA literature claims for latent-space prediction:

* :func:`make_shape_dataset` - images where *shape identity* is the semantic
  variable and *position* is nuisance. A masked-latent objective should expose
  shape; a pixel-reconstruction objective spends capacity on position.
* :func:`make_motion_clip_dataset` - short clips of a dot that translates at a
  constant velocity. The label is the direction of motion, which is only
  observable across time, so it probes whether V-JEPA-style tube masking learns
  temporal structure rather than static appearance.
* :class:`PointMassEnv` + :func:`sample_point_mass_trajectories` - a tiny
  action-conditioned control environment (a launched point inside a bouncing
  box) used to train and evaluate the latent world model and its CEM planner.

Everything is pure NumPy, deterministic under a seed, and small enough to run on
a laptop CPU in seconds.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


__all__ = [
    "make_shape_dataset",
    "make_motion_clip_dataset",
    "make_grating_dataset",
    "PointMassEnv",
    "sample_point_mass_trajectories",
    "render_ascii",
    "SHAPE_NAMES",
    "DIRECTION_NAMES",
]


SHAPE_NAMES = ("circle", "square", "triangle")
DIRECTION_NAMES = ("right", "left", "down", "up")


# ---------------------------------------------------------------------------
# Images: shape vs nuisance position
# ---------------------------------------------------------------------------

def _blob(mask: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Soft anti-aliased edge: blur a binary mask with a separable box filter."""
    if sigma <= 0:
        return mask.astype(np.float32)
    k = max(1, int(round(sigma * 2)))
    kernel = np.ones(2 * k + 1, dtype=np.float32) / (2 * k + 1)
    out = mask.astype(np.float32)
    pad = np.pad(out, ((0, 0), (k, k)), mode="edge")
    out = np.stack([np.convolve(row, kernel, mode="valid") for row in pad], axis=0)
    pad = np.pad(out, ((k, k), (0, 0)), mode="edge")
    out = np.stack([np.convolve(col, kernel, mode="valid") for col in pad.T], axis=1)
    return out.astype(np.float32)


def _draw_shape(kind: int, cx: float, cy: float, r: float, size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    if kind == 0:  # circle
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = dist <= r
    elif kind == 1:  # square
        mask = (np.abs(xx - cx) <= r * 0.85) & (np.abs(yy - cy) <= r * 0.85)
    else:  # triangle, apex up
        top = cy - r
        within_y = (yy >= top) & (yy <= cy + r * 0.9)
        half_width = (yy - top) * (r / max(1e-3, 2.0 * r * 0.95))
        mask = within_y & (np.abs(xx - cx) <= half_width)
    return _blob(mask, sigma=max(0.5, r / 5.0))


def make_shape_dataset(
    n: int = 256,
    img_size: int = 32,
    num_classes: int = 3,
    seed: int = 0,
    noise: float = 0.05,
    intensity_std: float = 0.15,
    jitter: float = 0.35,
) -> Dict[str, np.ndarray]:
    """Render coloured-ish shapes at random positions.

    Returns a dict with ``images`` (N, 1, H, W) float32 in [0, 1], ``labels``
    (N,) shape classes, ``positions`` (N, 2) normalised centres and ``sizes``
    (N,) normalised radii. The nuisance variables (position, size, contrast) are
    deliberately correlated with nothing, so linear-probe accuracy above chance
    can only come from the shape *identity*.
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 1, img_size, img_size), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    positions = np.zeros((n, 2), dtype=np.float32)
    sizes = np.zeros(n, dtype=np.float32)
    margin = img_size * 0.28
    for i in range(n):
        kind = int(i % num_classes) if n <= 4 * num_classes else int(rng.integers(0, num_classes))
        cx = float(rng.uniform(margin + jitter * margin, img_size - margin - jitter * margin))
        cy = float(rng.uniform(margin + jitter * margin, img_size - margin - jitter * margin))
        r = float(rng.uniform(img_size * 0.16, img_size * 0.26))
        shape = _draw_shape(kind, cx, cy, r, img_size)
        contrast = float(np.clip(rng.normal(1.0, intensity_std), 0.4, 1.4))
        img = shape * contrast + rng.normal(0.0, noise, size=(img_size, img_size)).astype(np.float32)
        images[i, 0] = np.clip(img, 0.0, 1.0)
        labels[i] = kind
        positions[i] = (cx / img_size, cy / img_size)
        sizes[i] = r / img_size
    return {"images": images, "labels": labels, "positions": positions, "sizes": sizes}


# ---------------------------------------------------------------------------
# Video: moving dot with a direction label
# ---------------------------------------------------------------------------

def make_motion_clip_dataset(
    n: int = 128,
    frames: int = 8,
    img_size: int = 32,
    seed: int = 0,
    noise: float = 0.03,
    speed: float = 0.9,
    start_jitter: float = 0.04,
    task: str = "direction",
) -> Dict[str, np.ndarray]:
    """Short clips of a dot translating in one of four directions.

    Every clip starts near the centre (``start_jitter`` of the frame), so the
    direction label cannot be recovered from where the dot *is* - only from how
    it *moves*. That makes the random-encoder baseline sit at chance, which is
    what a temporal-representation benchmark needs.

    With ``task="order"`` half of the clips are reversed in time and the label is
    the temporal order (forward / reversed). Reversal preserves the *set* of
    frames, so any order-invariant summary of appearance - including the mean
    pooled features of a randomly initialised encoder - is at chance by
    construction, and only a model that represents motion can solve it. That
    makes it the honest benchmark for a video JEPA.

    Returns ``clips`` (N, 1, T, H, W), ``labels`` (N,), ``velocities`` (N, 2)
    and the per-frame normalised positions.
    """
    rng = np.random.default_rng(seed)
    clips = np.zeros((n, 1, frames, img_size, img_size), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    velocities = np.zeros((n, 2), dtype=np.float32)
    directions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    starts = np.zeros((n, frames, 2), dtype=np.float32)
    for i in range(n):
        d = int(rng.integers(0, 4))
        v = directions[d] * speed * float(rng.uniform(0.7, 1.3))
        jitter = start_jitter * img_size
        pos = np.array(
            [
                img_size * 0.5 + rng.uniform(-jitter, jitter),
                img_size * 0.5 + rng.uniform(-jitter, jitter),
            ],
            dtype=np.float32,
        )
        for t in range(frames):
            pos = np.clip(pos, 0.34 * img_size, img_size - 0.34 * img_size)
            starts[i, t] = pos / img_size
            frame = _draw_shape(0, float(pos[0]), float(pos[1]), img_size * 0.11, img_size)
            clips[i, 0, t] = np.clip(frame + rng.normal(0.0, noise, (img_size, img_size)), 0.0, 1.0)
            pos = pos + v * img_size / frames * 0.4
        labels[i] = d
        velocities[i] = v
    if task == "order":
        order_labels = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if rng.random() < 0.5:
                clips[i] = clips[i, :, ::-1].copy()
                order_labels[i] = 1
        labels = order_labels
    return {
        "clips": clips,
        "labels": labels,
        "velocities": velocities,
        "positions": starts,
        "task": task,
    }


# ---------------------------------------------------------------------------
# Textures: global orientation of a grating
# ---------------------------------------------------------------------------

def make_grating_dataset(
    n: int = 256,
    img_size: int = 32,
    num_orientations: int = 4,
    seed: int = 0,
    noise: float = 0.05,
    min_freq: float = 0.08,
    max_freq: float = 0.16,
) -> Dict[str, np.ndarray]:
    """Sinusoidal gratings whose *global* orientation is the label.

    The orientation lives in the phase relationships between distant pixels, so
    it survives neither mean-pooling of raw patches nor a positional lookup: a
    random encoder lands at chance, while a JEPA that learns to predict masked
    blocks must encode the grating's orientation and phase to do so. Spatial
    frequency, phase and contrast are randomised as nuisance variables.

    Returns ``images`` (N, 1, H, W), ``labels`` (N,) orientation indices and
    ``angles`` (N,) in radians.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
    images = np.zeros((n, 1, img_size, img_size), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    angles = np.zeros(n, dtype=np.float32)
    for i in range(n):
        k = int(rng.integers(0, num_orientations))
        theta = math.pi * k / num_orientations
        freq = float(rng.uniform(min_freq, max_freq)) * img_size
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        contrast = float(rng.uniform(0.55, 1.0))
        pattern = np.cos(freq * (xx * math.cos(theta) + yy * math.sin(theta)) + phase)
        img = 0.5 + 0.5 * contrast * pattern
        img = img + rng.normal(0.0, noise, (img_size, img_size))
        images[i, 0] = np.clip(img, 0.0, 1.0).astype(np.float32)
        labels[i] = k
        angles[i] = theta
    return {"images": images, "labels": labels, "angles": angles}


# ---------------------------------------------------------------------------
# Control: a bouncing point mass with a goal marker
# ---------------------------------------------------------------------------

@dataclass
class PointMassConfig:
    """Physics and rendering parameters of the toy control task.

    ``damping`` is the fraction of the previous velocity that survives a step.
    Zero makes the action a pure velocity command, which keeps the *entire*
    system state visible in a single frame - the property a pixel-only world
    model needs to plan well, because velocity cannot be read off one image.
    Larger values add momentum, which turns the task into a partially observed
    control problem (useful for stress-testing, harder to plan in).
    """

    size: int = 24
    damping: float = 0.0
    dt: float = 0.25
    action_scale: float = 1.0
    radius: float = 2.2
    goal_radius: float = 2.0
    agent_value: float = 1.0
    goal_value: float = 0.45
    noise: float = 0.0


class PointMassEnv:
    """A point mass in a box with a goal marker rendered into the observation.

    The state is ``(position, velocity)``; the action is a 2D acceleration in
    [-1, 1]; walls are elastic. Observations are single-channel images, so the
    world model has to recover position, velocity and the goal location from
    pixels alone.
    """

    def __init__(self, config: Optional[PointMassConfig] = None, seed: int = 0) -> None:
        self.config = config or PointMassConfig()
        self.rng = np.random.default_rng(seed)
        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.goal = np.array([0.75, 0.25], dtype=np.float32)
        self.t = 0

    # -- geometry helpers -------------------------------------------------
    @property
    def _scale(self) -> float:
        return float(self.config.size)

    def reset(
        self,
        position: Optional[Sequence[float]] = None,
        velocity: Optional[Sequence[float]] = None,
        goal: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        c = self.config
        self.position = (
            np.asarray(position, dtype=np.float32)
            if position is not None
            else self.rng.uniform(0.15, 0.85, size=2).astype(np.float32)
        )
        self.velocity = (
            np.asarray(velocity, dtype=np.float32)
            if velocity is not None
            else self.rng.uniform(-0.6, 0.6, size=2).astype(np.float32)
        )
        self.goal = (
            np.asarray(goal, dtype=np.float32)
            if goal is not None
            else self.rng.uniform(0.1, 0.9, size=2).astype(np.float32)
        )
        self.t = 0
        return self.render()

    def step(self, action: Sequence[float]) -> Tuple[np.ndarray, Dict[str, Any]]:
        c = self.config
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        damping = float(np.clip(c.damping, 0.0, 1.0))
        self.velocity = damping * self.velocity + (1.0 - damping) * c.action_scale * a
        self.position = self.position + self.velocity * c.dt
        # elastic walls
        for axis in (0, 1):
            if self.position[axis] < 0.0:
                self.position[axis] = -self.position[axis]
                self.velocity[axis] = -0.8 * self.velocity[axis]
            elif self.position[axis] > 1.0:
                self.position[axis] = 2.0 - self.position[axis]
                self.velocity[axis] = -0.8 * self.velocity[axis]
        self.position = np.clip(self.position, 0.0, 1.0)
        self.t += 1
        obs = self.render()
        dist = float(np.linalg.norm(self.position - self.goal))
        info = {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "goal": self.goal.copy(),
            "distance": dist,
            "success": bool(dist < 0.08),
        }
        return obs, info

    def render(self) -> np.ndarray:
        c = self.config
        size = c.size
        img = np.zeros((size, size), dtype=np.float32)
        # goal marker (dimmer, behind the agent)
        img += _draw_shape(
            0,
            float(self.goal[0] * (size - 1)),
            float(self.goal[1] * (size - 1)),
            c.goal_radius,
            size,
        ) * c.goal_value
        img += _draw_shape(
            0,
            float(self.position[0] * (size - 1)),
            float(self.position[1] * (size - 1)),
            c.radius,
            size,
        ) * c.agent_value
        if c.noise:
            img = img + self.rng.normal(0.0, c.noise, img.shape).astype(np.float32)
        return np.clip(img, 0.0, 1.0)[None].astype(np.float32)


def sample_point_mass_trajectories(
    n: int = 128,
    horizon: int = 12,
    config: Optional[PointMassConfig] = None,
    seed: int = 0,
    policy: str = "mixed",
    goal_seek_prob: float = 0.6,
    action_noise: float = 0.35,
) -> Dict[str, np.ndarray]:
    """Roll out the environment with a coverage-oriented behaviour policy.

    ``policy="mixed"`` alternates between goal-seeking (so the data contains the
    effect of corrective actions) and unstructured exploration (so the model
    sees off-trajectory dynamics). Returns per-step observations, actions,
    positions, velocities and goals, plus the frame-to-frame displacement of the
    agent (a useful sanity target for the planner).
    """
    cfg = config or PointMassConfig()
    env = PointMassEnv(cfg, seed=seed)
    rng = np.random.default_rng(seed + 1234)
    obs = np.zeros((n, horizon, 1, cfg.size, cfg.size), dtype=np.float32)
    acts = np.zeros((n, horizon - 1, 2), dtype=np.float32)
    pos = np.zeros((n, horizon, 2), dtype=np.float32)
    vel = np.zeros((n, horizon, 2), dtype=np.float32)
    goals = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        obs[i, 0] = env.reset()
        pos[i, 0] = env.position
        vel[i, 0] = env.velocity
        goals[i] = env.goal
        seek = rng.random() < goal_seek_prob if policy == "mixed" else policy == "seek"
        for t in range(horizon - 1):
            if t == 0:
                seek = seek  # keep the episode's policy fixed
            if seek:
                direction = env.goal - env.position
                norm = np.linalg.norm(direction)
                base = direction / norm if norm > 1e-6 else np.zeros(2, dtype=np.float32)
            else:
                base = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            action = np.clip(base + rng.normal(0.0, action_noise, size=2), -1.0, 1.0).astype(np.float32)
            acts[i, t] = action
            frame, info = env.step(action)
            obs[i, t + 1] = frame
            pos[i, t + 1] = info["position"]
            vel[i, t + 1] = info["velocity"]
    return {"obs": obs, "actions": acts, "positions": pos, "velocities": vel, "goals": goals}


# ---------------------------------------------------------------------------
# Terminal visualisation (kept dependency-free, like the rest of the library)
# ---------------------------------------------------------------------------

def render_ascii(frame: np.ndarray, charset: str = " .:-=+*#%@") -> str:
    """Render a ``(1, H, W)`` or ``(H, W)`` float frame as ASCII art."""
    img = np.asarray(frame, dtype=np.float32)
    if img.ndim == 3:
        img = img[0]
    img = np.clip(img, 0.0, 1.0)
    idx = (img * (len(charset) - 1)).round().astype(int)
    return "\n".join("".join(charset[v] for v in row) for row in idx)


def mask_ascii(mask: np.ndarray, grid: Optional[Tuple[int, int]] = None) -> str:
    """Show a boolean token mask as a coarse grid of ``#`` (kept) and ``.`` (dropped)."""
    m = np.asarray(mask)
    if grid is not None:
        m = m.reshape(grid)
    return "\n".join("".join("#" if v else "." for v in row) for row in m)
