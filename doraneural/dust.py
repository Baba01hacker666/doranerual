"""Dust: zeroth-order pretraining by activation-space perturbation.

Reference: Samip Dahal, Bishwas Mandal, Serdar Gulbahar, Akshay Vegesna,
"Dust: Pretraining Transformers Without Backpropagation", Q Labs, October 2026.
https://qlabs.sh/research/dust

The idea, as stated in the paper: rather than perturb weights and give each
perturbation its own forward pass (weight-space ES, e.g. EGGROLL), perturb the
*output of each linear layer* at every token. One forward pass then evaluates a
population of one member per token instead of one member per forward pass, which
is where the claimed 10^3-10^4x efficiency over weight-space ES comes from.

For a linear layer y_t = W x_t, jitter the output at all tokens,

    y_t -> y_t + sigma * a_t,    a_t ~ N(0, I)

run the forward pass, and measure the per-token loss change. Credit decays over
subsequent tokens (keys, values and gates read the future, so they get gamma ~ 1;
everything else uses gamma = 0). With K draws, the estimated error at the layer
output is the reward-weighted noise averaged over draws,

    ghat_t = -(1 / (K sigma)) * sum_i r_t^(i) a_t^(i)          (paper eq. 2)

and the weight gradient is the same outer product backpropagation forms,

    Ghat_W = sum_t ghat_t x_t^T                                 (paper eq. 3)

Only the chain rule is missing; everything else is identical. Nothing in this
module differentiates the model, which is the point -- Dust can train networks
backpropagation cannot.

Scope of this implementation
----------------------------
This is the paper's simplest credit rule, not the full method:

* Sites are the two per-block linear outputs whose credit needs no attention
  internals: ``wo`` (attention output projection) and ``w2`` (FFN down
  projection). Both use gamma = 0.
* The attention internals (query, key, value, gate, value embedding), which the
  paper credits through the estimated attention-output error
  ``c_s = -<ghat_s, delta o_s>`` (eq. 4), are **not implemented**.
* The language-model head is jittered on cached logits rather than through a
  block; the paper does this too, but it is not implemented here.
* The 2L residual mixing scalars, which the paper trains with ordinary
  weight-space ES, are not implemented.

``dust_forward`` is a forward-only NumPy pass over
:class:`~doraneural.transformer.TransformerDecoderLM`. With no injection it must
match the autograd model exactly; ``tests/test_dust.py`` asserts that parity, and
it is the main correctness guard for the reimplementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .autograd import Tensor

__all__ = [
    "DustConfig",
    "DustEstimate",
    "DustTrainer",
    "DustStepStats",
    "evaluate_loss",
    "dust_forward",
    "per_token_cross_entropy",
    "estimate_site_gradient",
    "backprop_gradient",
    "cosine_to_backprop",
    "fit_cosine_ladder",
]

# Injection sites, in the order the block computes them. Each maps to the linear
# weight whose gradient that site produces.
SITES: Tuple[str, ...] = ("wo", "w2")

SITE_PARAMS: Dict[str, str] = {"wo": "wo", "w2": "w2"}

# Favour `wo` first: it sits closer to the token loss, so its credit is the
# cleanest test of the estimator.
DEFAULT_SIGMA: Dict[str, float] = {"wo": 0.01, "w2": 0.01}
DEFAULT_GAMMA: Dict[str, float] = {"wo": 0.0, "w2": 0.0}


@dataclass
class DustConfig:
    """Hyperparameters for the Dust estimator.

    Attributes:
        draws: Population size K. The paper's population ladder runs 64 to 16k.
        sigma: Per-site noise scale. Tuned in the paper by maximising cosine
            against the backprop gradient on a single batch.
        gamma: Per-site credit decay. Only keys, values, gates and value
            embeddings take gamma close to 1; every other site uses 0, meaning a
            jitter is rewarded by its own token alone.
        seed: RNG seed, so a ladder at two populations shares its noise draws.
    """

    draws: int = 64
    sigma: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SIGMA))
    gamma: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GAMMA))
    seed: int = 42

    def noise_scale(self, site: str) -> float:
        return float(self.sigma.get(site, DEFAULT_SIGMA.get(site, 0.01)))

    def credit_decay(self, site: str) -> float:
        return float(self.gamma.get(site, DEFAULT_GAMMA.get(site, 0.0)))

    def validate(self) -> None:
        if self.draws <= 0:
            raise ValueError(f"draws must be positive, got {self.draws}")
        for site in self.sigma:
            if self.sigma[site] <= 0.0:
                raise ValueError(f"sigma[{site}] must be positive, got {self.sigma[site]}")
        for site in self.gamma:
            if not 0.0 <= self.gamma[site] <= 1.0:
                raise ValueError(f"gamma[{site}] must be in [0, 1], got {self.gamma[site]}")
        unknown = (set(self.sigma) | set(self.gamma)) - set(SITES)
        if unknown:
            raise ValueError(f"unknown injection sites {sorted(unknown)}; expected {list(SITES)}")


# ---------------------------------------------------------------------------
# Forward-only NumPy pass
# ---------------------------------------------------------------------------

def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def _rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, interleaved: bool) -> np.ndarray:
    """x is (heads, time, head_size); cos/sin are (time, head_size//2)."""
    if interleaved:
        first = x[:, :, 0::2]
        second = x[:, :, 1::2]
        rot_first = first * cos - second * sin
        rot_second = first * sin + second * cos
        pieces: List[np.ndarray] = []
        for idx in range(x.shape[-1] // 2):
            pieces.append(rot_first[:, :, idx])
            pieces.append(rot_second[:, :, idx])
        return np.stack(pieces, axis=-1)
    half = x.shape[-1] // 2
    cos_d = np.repeat(cos, 2, axis=-1)
    sin_d = np.repeat(sin, 2, axis=-1)
    rot_first = x[:, :, :half] * cos_d - x[:, :, half:] * sin_d
    rot_second = x[:, :, :half] * sin_d + x[:, :, half:] * cos_d
    return np.concatenate([rot_first, rot_second], axis=-1)


def _rope_tables(dim: int, time: int, theta: float, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    half = dim // 2
    idx = np.arange(half, dtype=np.float32)
    inv_freq = 1.0 / (theta ** (2.0 * idx / dim))
    angles = np.arange(time, dtype=np.float32)[:, None] * inv_freq[None, :]
    return np.cos(angles).astype(dtype), np.sin(angles).astype(dtype)


def dust_forward(
    model,
    token_ids: Sequence[int],
    inject: Optional[Callable[[int, str, np.ndarray], Optional[np.ndarray]]] = None,
    capture: Optional[Callable[[int, str, np.ndarray, np.ndarray], None]] = None,
) -> np.ndarray:
    """Run the model forward with no autodiff tape and optional layer injection.

    Args:
        model: A :class:`TransformerDecoderLM`.
        token_ids: 1D sequence of token ids.
        inject: ``(layer, site, y) -> perturbation or None``. When it returns an
            array it is *added* to the layer output ``y`` at that site.
        capture: ``(layer, site, x, y) -> None``, receives the layer's input and
            output activations. Used to gather the ``x_t`` that eq. 3 needs.

    Returns:
        Logits of shape ``(time, vocab_size)``.
    """
    ids = np.asarray(token_ids, dtype=np.int64)
    time = ids.shape[0]
    dim = model.dim
    n_heads = model.n_heads
    n_kv_heads = model.n_kv_heads
    head_size = dim // n_heads
    interleaved = model.rope_type != "hf"

    emb = np.asarray(model.token_embedding.data)
    x = emb[ids].copy()

    cos, sin = _rope_tables(head_size, time, model.rope_theta)

    for layer_idx, layer in enumerate(model.layers):
        norm = _rmsnorm(x, np.asarray(layer.rms_att.data))
        q = (norm @ np.asarray(layer.wq.data)).reshape(time, n_heads, head_size).transpose(1, 0, 2)
        k = (norm @ np.asarray(layer.wk.data)).reshape(time, n_kv_heads, head_size).transpose(1, 0, 2)
        v = (norm @ np.asarray(layer.wv.data)).reshape(time, n_kv_heads, head_size).transpose(1, 0, 2)
        q = _rope(q, cos, sin, interleaved)
        k = _rope(k, cos, sin, interleaved)
        if n_kv_heads != n_heads:
            kv_map = np.arange(n_heads) // (n_heads // n_kv_heads)
            k = k[kv_map]
            v = v[kv_map]

        scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(head_size)
        mask = np.triu(np.full((time, time), -np.inf, dtype=np.float32), 1)
        scores = scores + mask
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        attn = (probs @ v).transpose(1, 0, 2).reshape(time, dim)

        wo_in = attn
        wo_out = wo_in @ np.asarray(layer.wo.data)
        if capture is not None:
            capture(layer_idx, "wo", wo_in, wo_out)
        if inject is not None:
            delta = inject(layer_idx, "wo", wo_out)
            if delta is not None:
                wo_out = wo_out + delta
        x = x + wo_out

        norm_ffn = _rmsnorm(x, np.asarray(layer.rms_ffn.data))
        gate = norm_ffn @ np.asarray(layer.w1.data)
        up = norm_ffn @ np.asarray(layer.w3.data)

        if layer.novel_neurons and getattr(layer, "w_dend", None) is not None:
            dend_gate = (norm_ffn @ np.asarray(layer.w_dend.data))
            dend_gate = 1.0 / (1.0 + np.exp(-dend_gate)) * 2.0
            up = up * dend_gate

        silu = (1.0 / (1.0 + np.exp(-gate))) * gate
        hidden = silu * up

        if layer.novel_neurons and getattr(layer, "c_poly", None) is not None:
            u = np.tanh(hidden)
            t1 = u
            t2 = (u * u) * 2.0 - 1.0
            t3 = (u * u * u) * 4.0 - u * 3.0
            poly = t1 * np.asarray(layer.c_poly.data)[0] + t2 * np.asarray(layer.c_poly.data)[1]
            if layer.kan_degree >= 3:
                poly = poly + t3 * np.asarray(layer.c_poly.data)[2]
            hidden = hidden + poly

        w2_in = hidden
        w2_out = w2_in @ np.asarray(layer.w2.data)
        if capture is not None:
            capture(layer_idx, "w2", w2_in, w2_out)
        if inject is not None:
            delta = inject(layer_idx, "w2", w2_out)
            if delta is not None:
                w2_out = w2_out + delta

        if layer.novel_neurons and getattr(layer, "w_ref", None) is not None:
            down_ref = np.tanh(w2_out @ np.asarray(layer.w_ref.data))
            w2_out = w2_out * 0.75 + down_ref * 0.25

        x = x + w2_out

    x = _rmsnorm(x, np.asarray(model.rms_final.data))
    return x @ np.asarray(model.lm_head.data).T


def per_token_cross_entropy(logits: np.ndarray, targets: Sequence[int]) -> np.ndarray:
    """Mean cross-entropy per predicted position, shape ``(time - 1,)``.

    ``logits`` is ``(T, V)`` and ``targets`` holds the ``T-1`` next tokens. The
    final position has no target and is dropped, matching the standard causal
    shift.
    """
    logits = np.asarray(logits, dtype=np.float64)
    tgts = np.asarray(targets, dtype=np.int64)
    n = tgts.shape[0]
    if logits.shape[0] < n + 1:
        raise ValueError(f"need at least {n + 1} positions for {n} targets, got {logits.shape[0]}")
    rows = logits[:n]
    shifted = rows - rows.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=-1))
    picked = shifted[np.arange(n), tgts]
    return -(picked - log_z)


# ---------------------------------------------------------------------------
# Credit assignment
# ---------------------------------------------------------------------------

def _decayed_credit(c: np.ndarray, gamma: float) -> np.ndarray:
    """r_t = sum_{s >= t} gamma^(s-t) c_s  (paper eq. 1).

    ``c`` is the per-token loss change for one draw. With gamma = 0 this is the
    identity; with gamma close to 1 a jitter is also credited by the loss
    changes of every later token, which is what keys, values and gates need since
    later tokens attend to them.

    Implemented as a reverse cumulative sum with geometric decay, in O(T).
    """
    if gamma <= 0.0:
        return c.copy()
    rewards = np.zeros_like(c)
    running = 0.0
    for t in range(c.shape[0] - 1, -1, -1):
        running = c[t] + gamma * running
        rewards[t] = running
    return rewards


@dataclass
class DustEstimate:
    """Result of estimating one site-layer weight gradient."""

    site: str
    layer: int
    draws: int
    sigma: float
    gamma: float
    g_out: np.ndarray          # (T, d_out) estimated error at the layer output
    grad: np.ndarray           # (d_in, d_out) Ghat_W, matching Parameter layout
    weight_grad: np.ndarray    # (d_out, d_in) the transpose, i.e. paper eq. 3
    input_act: np.ndarray      # (T, d_in) the x_t summed over in eq. 3


def estimate_site_gradient(
    model,
    token_ids: Sequence[int],
    targets: Sequence[int],
    site: str,
    layer: int,
    config: DustConfig,
    rng: Optional[np.random.Generator] = None,
    base_logits: Optional[np.ndarray] = None,
) -> DustEstimate:
    """Estimate the weight gradient at one injection site with Dust.

    Args:
        model: A :class:`TransformerDecoderLM`.
        token_ids: The sequence to evaluate.
        targets: Next-token targets for ``token_ids``.
        site: ``"wo"`` or ``"w2"``.
        layer: Block index.
        config: Population size, noise scale and credit decay.
        rng: Optional generator, so a population ladder can share noise draws
            across population sizes and isolate the variance effect.
        base_logits: Optional cached clean logits, to avoid recomputing them.

    Returns:
        A :class:`DustEstimate` whose ``weight_grad`` is eq. 3 and whose ``grad``
        is the same matrix in ``Parameter`` layout ``(d_in, d_out)``.
    """
    config.validate()
    if site not in SITES:
        raise ValueError(f"unknown site {site!r}; expected one of {list(SITES)}")
    if not 0 <= layer < len(model.layers):
        raise ValueError(f"layer {layer} out of range [0, {len(model.layers)})")

    rng = rng or np.random.default_rng(config.seed)
    sigma = config.noise_scale(site)
    gamma = config.credit_decay(site)

    captured: Dict[str, np.ndarray] = {}
    clean_logits: Dict[str, np.ndarray] = {}

    def capture(layer_idx: int, site_name: str, x_in: np.ndarray, y_out: np.ndarray) -> None:
        if layer_idx == layer and site_name == site:
            captured["x"] = np.array(x_in, dtype=np.float64)
            captured["y"] = np.array(y_out, dtype=np.float64)

    # Clean pass: the reference loss every draw is centred against, and the x_t
    # that eq. 3 reuses (the forward pass already computed it).
    if base_logits is not None:
        clean_logits["out"] = np.asarray(base_logits)
        dust_forward(model, token_ids, inject=None, capture=capture)
    else:
        clean_logits["out"] = dust_forward(model, token_ids, inject=None, capture=capture)

    clean_loss = per_token_cross_entropy(clean_logits["out"], targets)

    noises: List[np.ndarray] = []
    credits: List[np.ndarray] = []

    for _ in range(config.draws):
        noise = rng.standard_normal(captured["y"].shape)
        noises.append(noise)
        state = {"delta": sigma * noise}

        def inject(layer_idx: int, site_name: str, y: np.ndarray, _s=site, _l=layer, _st=state):
            if layer_idx == _l and site_name == _s:
                return _st["delta"]
            return None

        logits = dust_forward(model, token_ids, inject=inject, capture=None)
        perturbed_loss = per_token_cross_entropy(logits, targets)
        # Centred loss change for this draw.
        credits.append(_decayed_credit(perturbed_loss - clean_loss, gamma))

    # ghat_t = -(1/(K sigma)) sum_i r_t^(i) a_t^(i)          (paper eq. 2)
    #
    # Every token is perturbed, but only the first T-1 positions have a
    # next-token target and therefore a reward, so the final row of noise is
    # dropped here. Perturbing it still influences the forward pass (that is
    # intended -- it is population, not a target), it just earns no credit.
    stacked_noise = np.stack(noises)                      # (K, T, d_out)
    stacked_reward = np.stack(credits)                    # (K, T_loss)
    n_loss = stacked_reward.shape[1]
    # Contract over draws only.
    #
    # Sign convention: the paper's eq. 2 carries a leading minus because its r is a
    # *reward*, and flipping reward into a gradient error is what that minus does.
    # Here c is the loss *increase* (perturbed minus clean), so the directional
    # derivative along a_i is c_i / sigma and, averaging isotropic draws, the error
    # estimate is +(1/(K sigma)) sum_i r_t^(i) a_t^(i) -- the same quantity with the
    # paper's reward negated. Measured against the backprop gradient this sign gives
    # a positive cosine that rises with population; the paper's literal sign gives
    # its negation.
    g_out = np.einsum("kt,ktc->tc", stacked_reward, stacked_noise[:, :n_loss, :])
    g_out = g_out / (config.draws * sigma)

    # Token positions with no target do not contribute.
    time = len(token_ids)
    n_loss = g_out.shape[0]
    x_in = captured["x"]
    # eq. 3: sum_t ghat_t x_t^T over the token positions that had a loss.
    # The loss at position t predicts token t+1, and the activation consumed by
    # that projection is x_t, so the first n_loss positions are the right ones.
    weight_grad = g_out.T @ x_in[:n_loss]                 # (d_out, d_in)

    return DustEstimate(
        site=site,
        layer=layer,
        draws=config.draws,
        sigma=sigma,
        gamma=gamma,
        g_out=g_out,
        grad=weight_grad.T.copy(),
        weight_grad=weight_grad,
        input_act=x_in,
    )


# ---------------------------------------------------------------------------
# Ground truth and diagnostics
# ---------------------------------------------------------------------------

def backprop_gradient(model, token_ids: Sequence[int], targets: Sequence[int]) -> Dict[str, np.ndarray]:
    """True gradient at every Dust site, via the existing autograd tape.

    This is only used for the diagnostic -- Dust itself never calls it.
    """
    logits = model.forward(token_ids)
    n = len(targets)
    rows = logits[:n]
    vocab = rows.data.shape[-1]
    onehot = Tensor(
        np.eye(vocab, dtype=rows.dtype)[np.asarray(targets, dtype=np.int64)],
        dtype=rows.dtype,
    )
    # Kept on the tape with elementwise ops only (no gather is available), so the
    # backward pass actually reaches the embedding and every block.
    row_logsumexp = rows.exp().sum(axis=-1, keepdims=True).log()
    picked = (rows * onehot).sum(axis=-1)
    total = -(picked - row_logsumexp).mean()

    for param in model.parameters():
        param.zero_grad()
    total.backward()

    grads: Dict[str, np.ndarray] = {}
    for layer_idx, layer in enumerate(model.layers):
        for site in SITES:
            param = getattr(layer, SITE_PARAMS[site])
            if param.grad is None:
                continue
            grads[f"{layer_idx}.{site}"] = np.asarray(param.grad, dtype=np.float64)
    return grads


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_to_backprop(
    model,
    token_ids: Sequence[int],
    targets: Sequence[int],
    site: str,
    layer: int,
    config: DustConfig,
    truth: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[float, DustEstimate, Dict[str, np.ndarray]]:
    """Cosine between the Dust estimate and the backprop gradient.

    The paper's Eq. 5 says this rises with population as
    ``cos(K) = c_max / sqrt(1 + c/K)``. Measuring it needs no training: one
    forward pass and K noisy ones.
    """
    truth = truth if truth is not None else backprop_gradient(model, token_ids, targets)
    key = f"{layer}.{site}"
    if key not in truth:
        raise KeyError(f"no backprop gradient recorded for site {key}")

    # Share the generator seed across population sizes so the ladder isolates the
    # effect of K rather than redrawing the noise.
    rng = np.random.default_rng(config.seed)
    base = dust_forward(model, token_ids)
    estimate = estimate_site_gradient(
        model, token_ids, targets, site, layer, config,
        rng=rng, base_logits=base,
    )
    return _cosine(estimate.grad, truth[key]), estimate, truth


def fit_cosine_ladder(
    populations: Sequence[int],
    cosines: Sequence[float],
    bounds: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float]:
    """Fit the paper's Eq. 5 law ``cos(K) = c_max / sqrt(1 + c/K)``.

    Args:
        populations: The population sizes K that were measured.
        cosines: The measured cosines, same order.
        bounds: Optional ``(c_max, c)`` starting guess for the search.

    Returns:
        ``(c_max, c, rmse)``. ``c_max`` is the ceiling and ``c`` the population
        at which a site reaches ``c_max / sqrt(2)``.
    """
    k_arr = np.asarray(populations, dtype=np.float64)
    y_arr = np.asarray(cosines, dtype=np.float64)
    if k_arr.shape != y_arr.shape:
        raise ValueError("populations and cosines must have the same shape")
    if k_arr.size < 2:
        raise ValueError("need at least two population sizes to fit two parameters")
    if np.any(k_arr <= 0):
        raise ValueError("populations must be positive")

    def residual(params: np.ndarray) -> np.ndarray:
        c_max, c = params
        pred = c_max / np.sqrt(1.0 + c / k_arr)
        return pred - y_arr

    guess = np.array(bounds if bounds is not None else [1.0, float(np.min(k_arr))], dtype=np.float64)
    # Coarse grid then local refine: the law is only two-dimensional and a
    # least-squares call would drag SciPy into the dependency-free build.
    best, best_cost = guess.copy(), float(np.sum(residual(guess) ** 2))
    # A cosine is bounded by 1, so c_max -- the population-infinity limit -- is too.
    # Clamping stops a short, still-rising ladder from fitting an unphysical ceiling.
    for c_max in np.linspace(0.2, 1.0, 24):
        for c in np.geomspace(max(k_arr.min() / 100.0, 1e-3), k_arr.max() * 10.0, 36):
            trial = np.array([c_max, c])
            cost = float(np.sum(residual(trial) ** 2))
            if cost < best_cost:
                best, best_cost = trial.copy(), cost
    rmse = float(np.sqrt(best_cost / k_arr.size))
    return float(min(1.0, best[0])), float(best[1]), rmse


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def evaluate_loss(model, token_ids: Sequence[int], targets: Sequence[int]) -> float:
    """Mean cross-entropy with no jitter. This is the quantity training reduces."""
    return float(per_token_cross_entropy(dust_forward(model, token_ids), targets).mean())


@dataclass
class DustStepStats:
    """Diagnostics for one optimizer step."""

    loss_before: float
    loss_after: Optional[float]
    draws: int
    forwards: int
    updated: int


class DustTrainer:
    """Optimise ``wo`` and ``w2`` with Dust instead of backpropagation.

    Each step estimates the weight gradient at every (layer, site) from a
    population of draws, then applies an SGD or Adam update to the corresponding
    ``Parameter``. The model's autograd tape is never used -- this is the point of
    the method.

    The paper trains with SGD+momentum at a constant learning rate and finds Dust
    also works under Adam (its Sec. 3.3), so both are provided.
    """

    def __init__(
        self,
        model,
        config: Optional[DustConfig] = None,
        lr: float = 1e-3,
        optimizer: str = "sgd",
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        sites: Sequence[str] = SITES,
        split_population: bool = True,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.config = config or DustConfig()
        self.config.validate()
        if optimizer not in ("sgd", "adam"):
            raise ValueError(f"optimizer must be 'sgd' or 'adam', got {optimizer!r}")
        unknown = set(sites) - set(SITES)
        if unknown:
            raise ValueError(f"unknown sites {sorted(unknown)}")
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.sites = tuple(sites)
        # The paper splits the population across layers and layer types rather
        # than giving every site the full K: its Sec. 2.3 tunes "the share of the
        # population each layer gets". Without the split, a step costs
        # K x n_layers x n_sites forwards, which is a 10x-30x self-inflicted tax
        # and makes a wall-clock comparison meaningless.
        self.split_population = bool(split_population)
        self._n_sites = len(self.model.layers) * len(self.sites)
        self.lr = float(lr)
        self.optimizer = optimizer
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        # Momentum / Adam state, keyed by the Parameter object itself.
        self._buf: Dict[int, np.ndarray] = {}
        self.total_forwards = 0

    def _target_param(self, layer_idx: int, site: str):
        return getattr(self.model.layers[layer_idx], SITE_PARAMS[site])

    def _apply(self, param, grad: np.ndarray) -> None:
        g = np.asarray(grad, dtype=np.float64)
        key = id(param)
        buf = self._buf.get(key)
        if buf is None:
            buf = np.zeros_like(g)
            self._buf[key] = buf

        if self.optimizer == "sgd":
            buf *= self.momentum
            buf += g
            if self.weight_decay:
                param.data -= self.lr * self.weight_decay * param.data
            param.data -= (self.lr * buf).astype(param.data.dtype)
            return

        # Adam, matching the bias correction TensorAdamW uses.
        self.step_count += 1
        buf *= self.beta1
        buf += (1.0 - self.beta1) * g
        v = self._buf.get(-key)
        if v is None:
            v = np.zeros_like(g)
            self._buf[-key] = v
        v *= self.beta2
        v += (1.0 - self.beta2) * (g * g)
        b1_corr = 1.0 - self.beta1 ** self.step_count
        b2_corr = 1.0 - self.beta2 ** self.step_count
        step = self.lr * np.sqrt(b2_corr) / b1_corr
        if self.weight_decay:
            param.data -= self.lr * self.weight_decay * param.data
        param.data -= (step * buf / (np.sqrt(v) + self.eps)).astype(param.data.dtype)

    def step(self, token_ids: Sequence[int], targets: Sequence[int],
             measure_after: bool = False) -> DustStepStats:
        """One optimizer step over every (layer, site)."""
        before = evaluate_loss(self.model, token_ids, targets)
        base = dust_forward(self.model, token_ids)

        per_site = max(1, self.config.draws // self._n_sites) if self.split_population \
            else self.config.draws
        local = replace(self.config, draws=per_site)
        forwards = per_site * self._n_sites + 1
        self.total_forwards += forwards

        updated = 0
        for layer_idx in range(len(self.model.layers)):
            for site in self.sites:
                estimate = estimate_site_gradient(
                    self.model, token_ids, targets, site, layer_idx, local,
                    rng=self.rng, base_logits=base,
                )
                self._apply(self._target_param(layer_idx, site), estimate.grad)
                updated += 1

        after = evaluate_loss(self.model, token_ids, targets) if measure_after else None
        return DustStepStats(
            loss_before=before,
            loss_after=after,
            draws=per_site,
            forwards=forwards,
            updated=updated,
        )
