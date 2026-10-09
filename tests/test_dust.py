"""Tests for the Dust zeroth-order optimizer.

Reference: Dahal et al., "Dust: Pretraining Transformers Without Backpropagation",
Q Labs, 2026. https://qlabs.sh/research/dust

The load-bearing property is not that Dust matches backprop -- it does not, and
these tests do not pretend otherwise. It is that the gradient *estimate converges
toward backprop's as the population grows*, with nothing from the chain rule
built in. That is what `test_cosine_rises_with_population` pins, and everything
else here guards the machinery around it.

Populations are kept small (K <= 64) on a tiny model so the suite stays fast.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from doraneural.dust import (
    DustConfig, DustTrainer, SITES, backprop_gradient, cosine_to_backprop,
    dust_forward, estimate_site_gradient, evaluate_loss, fit_cosine_ladder,
    per_token_cross_entropy,
)
from doraneural.transformer import TransformerDecoderLM

SEQ = 24


def build(dim=32, hidden=64, layers=2, heads=4, vocab=64, seed=0):
    np.random.seed(seed)
    return TransformerDecoderLM(
        dim=dim, hidden_dim=hidden, n_layers=layers, n_heads=heads,
        n_kv_heads=heads, vocab_size=vocab, seq_len=SEQ,
    )


def window(vocab=64, n=SEQ):
    ids = [(7 * i + 3) % vocab for i in range(n)]
    return ids, ids[1:]


# --------------------------------------------------------------------------
# The forward-only reimplementation must agree with the autograd tape
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dim,layers,heads,vocab", [
    (32, 2, 4, 64),     # plain MHA
    (64, 3, 8, 128),    # more layers
])
def test_dust_forward_matches_autograd_model(dim, layers, heads, vocab):
    """dust_forward uses no autodiff tape at all; it must still be the same model.

    This is the main correctness guard for the reimplementation. Without it a typo
    in the forward would silently produce a plausible-looking but wrong estimator.
    """
    model = build(dim=dim, layers=layers, heads=heads, vocab=vocab)
    ids, _ = window(vocab)
    taped = np.asarray(model.forward(ids).data, dtype=np.float64)
    plain = dust_forward(model, ids)
    np.testing.assert_allclose(taped, plain, atol=1e-4, rtol=0)


def test_dust_forward_matches_autograd_with_grouped_query_attention():
    np.random.seed(0)
    model = TransformerDecoderLM(
        dim=64, hidden_dim=128, n_layers=2, n_heads=8, n_kv_heads=2,
        vocab_size=96, seq_len=SEQ,
    )
    ids, _ = window(96)
    taped = np.asarray(model.forward(ids).data, dtype=np.float64)
    np.testing.assert_allclose(taped, dust_forward(model, ids), atol=1e-4, rtol=0)


def test_per_token_cross_entropy_shapes_and_values():
    model = build()
    ids, targets = window()
    losses = per_token_cross_entropy(dust_forward(model, ids), targets)
    assert losses.shape == (len(targets),)
    assert np.all(np.isfinite(losses))
    # A uniform predictor over V logits scores ln(V).
    uniform = np.zeros((len(targets) + 1, 64), dtype=np.float64)
    assert per_token_cross_entropy(uniform, targets).mean() == pytest.approx(np.log(64), rel=1e-6)


def test_per_token_cross_entropy_requires_a_target_position_per_prediction():
    model = build()
    ids, targets = window()
    logits = dust_forward(model, ids)
    # Asking for a target at every position leaves the final one with no input.
    with pytest.raises(ValueError, match="positions"):
        per_token_cross_entropy(logits, ids)


# --------------------------------------------------------------------------
# Credit assignment
# --------------------------------------------------------------------------

def test_credit_is_identity_at_gamma_zero_and_decays_otherwise():
    from doraneural.dust import _decayed_credit

    c = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(_decayed_credit(c, 0.0), c)

    # gamma = 1 sums the whole suffix; gamma -> 0 recovers the identity.
    np.testing.assert_allclose(_decayed_credit(c, 1.0), [10.0, 9.0, 7.0, 4.0])
    almost_zero = _decayed_credit(c, 1e-9)
    np.testing.assert_allclose(almost_zero, c, atol=1e-6)


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------

def test_estimate_shapes_match_parameter_layout():
    model = build()
    ids, targets = window()
    truth = backprop_gradient(model, ids, targets)
    for site in SITES:
        est = estimate_site_gradient(model, ids, targets, site, 0, DustConfig(draws=4))
        assert est.grad.shape == truth[f"0.{site}"].shape
        # eq. 3 produces (d_out, d_in); Parameter is stored (d_in, d_out).
        assert est.weight_grad.shape == est.grad.shape[::-1]
        assert est.g_out.shape[0] == len(targets)


def test_estimate_rejects_unknown_sites_and_bad_config():
    model = build()
    ids, targets = window()
    with pytest.raises(ValueError, match="unknown site"):
        estimate_site_gradient(model, ids, targets, "w1", 0, DustConfig(draws=2))
    with pytest.raises(ValueError, match="draws must be positive"):
        DustConfig(draws=0).validate()
    with pytest.raises(ValueError, match="sigma"):
        DustConfig(sigma={"wo": 0.0}).validate()
    with pytest.raises(ValueError, match="gamma"):
        DustConfig(gamma={"wo": 2.0}).validate()
    with pytest.raises(ValueError, match="unknown injection sites"):
        DustConfig(sigma={"nope": 1.0}).validate()
    with pytest.raises(ValueError, match="layer"):
        estimate_site_gradient(model, ids, targets, "wo", 99, DustConfig(draws=2))


def test_gradient_estimate_has_positive_cosine_to_backprop():
    """Sign convention.

    The paper's eq. 2 carries a leading minus because its r is a *reward*. Here c
    is the loss *increase*, so the estimate needs no minus; with the literal sign
    this cosine comes out negative. Pinning it positive stops a silent sign flip.
    """
    model = build()
    ids, targets = window()
    truth = backprop_gradient(model, ids, targets)
    for site in SITES:
        cos, _, _ = cosine_to_backprop(
            model, ids, targets, site, 0, DustConfig(draws=64, seed=3), truth=truth)
        assert cos > 0.0, f"{site} estimate is anti-correlated with backprop"


def test_cosine_rises_with_population():
    """The paper's core claim, reproduced: useful gradients emerge from population.

    Eq. 5 says cos(K) = c_max / sqrt(1 + c/K), so the estimate should improve
    monotonically with K even though no chain rule is involved anywhere.
    """
    model = build()
    ids, targets = window()
    truth = backprop_gradient(model, ids, targets)
    populations = [4, 16, 64]
    cosines = [
        cosine_to_backprop(model, ids, targets, "wo", 0,
                           DustConfig(draws=k, seed=5), truth=truth)[0]
        for k in populations
    ]
    assert cosines[-1] > cosines[0], f"cosine did not improve: {cosines}"
    for earlier, later in zip(cosines, cosines[1:]):
        assert later > earlier, f"cosine not monotone in K: {cosines}"

    c_max, c, rmse = fit_cosine_ladder(populations, cosines)
    assert 0.0 < c_max <= 1.0
    assert c > 0.0
    # The paper reports RMSE below 0.06 for this two-parameter fit.
    assert rmse < 0.15, f"eq.5 fit is poor: rmse={rmse:.4f}"


def test_sigma_cancels_in_the_estimate():
    """A finding worth locking down.

    sigma is set by maximising cosine in the paper (Sec. 2.3), but the estimator
    divides by sigma and is linear in the perturbation, so the cosine is invariant
    to sigma in the small-noise regime, up to second-order terms. If a future
    change makes cosine depend on sigma properly, this test catches it.
    """
    model = build()
    ids, targets = window()
    truth = backprop_gradient(model, ids, targets)
    cosines = [
        cosine_to_backprop(model, ids, targets, "wo", 0,
                           DustConfig(draws=64, sigma={"wo": s, "w2": s}, seed=7),
                           truth=truth)[0]
        for s in (0.003, 0.03)
    ]
    # Not exact: c is the loss *change*, so sigma enters at second order once the
    # perturbation is large enough. Over this 10x range that residual is ~1e-3.
    assert abs(cosines[0] - cosines[1]) < 5e-3


def test_shared_seed_makes_population_ladder_comparable():
    """A fixed seed must give identical draws for a fixed K."""
    model = build()
    ids, targets = window()
    a = estimate_site_gradient(model, ids, targets, "wo", 0, DustConfig(draws=8, seed=11))
    b = estimate_site_gradient(model, ids, targets, "wo", 0, DustConfig(draws=8, seed=11))
    np.testing.assert_allclose(a.grad, b.grad, atol=0, rtol=0)


def test_backprop_gradient_covers_every_site():
    model = build(layers=3)
    ids, targets = window()
    truth = backprop_gradient(model, ids, targets)
    for layer in range(3):
        for site in SITES:
            assert f"{layer}.{site}" in truth


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def test_population_split_bounds_forwards_per_step():
    """Without the split a step costs K x n_layers x n_sites forwards."""
    model = build(layers=4)
    ids, targets = window()
    trainer = DustTrainer(model, DustConfig(draws=40, seed=1), lr=1e-3, seed=1)
    stats = trainer.step(ids, targets)
    n_sites = 4 * len(SITES)
    assert stats.draws == max(1, 40 // n_sites)
    assert stats.updated == n_sites
    assert stats.forwards == stats.draws * n_sites + 1
    assert trainer.total_forwards == stats.forwards

    unsplit = DustTrainer(model, DustConfig(draws=40, seed=1), lr=1e-3,
                         split_population=False, seed=1)
    assert unsplit.step(ids, targets).draws == 40


def test_trainer_reduces_loss_without_autograd():
    model = build()
    ids, targets = window()
    trainer = DustTrainer(model, DustConfig(draws=32, seed=1), lr=3e-3,
                         optimizer="adam", seed=1)
    start = evaluate_loss(model, ids, targets)
    for _ in range(8):
        trainer.step(ids, targets)
    assert evaluate_loss(model, ids, targets) < start


def test_trainer_supports_sgd_and_rejects_bad_options():
    model = build()
    ids, targets = window()
    sgd = DustTrainer(model, DustConfig(draws=8), lr=1e-3, optimizer="sgd")
    before = evaluate_loss(model, ids, targets)
    sgd.step(ids, targets)
    assert evaluate_loss(model, ids, targets) != before

    with pytest.raises(ValueError, match="optimizer"):
        DustTrainer(model, DustConfig(draws=8), lr=1e-3, optimizer="rmsprop")
    with pytest.raises(ValueError, match="lr"):
        DustTrainer(model, DustConfig(draws=8), lr=0.0)
    with pytest.raises(ValueError, match="sites"):
        DustTrainer(model, DustConfig(draws=8), lr=1e-3, sites=("w1",))