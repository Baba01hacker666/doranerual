"""Focused correctness and safety tests for the NumPy optimizers."""

import math

import numpy as np
import pytest

import doraneural as dn
from doraneural.layers import Dense
from doraneural.optimizers import Adam, AdamW, RMSprop, SGD, clip_grad_norm, clip_grad_value


def _dense_with_gradient(gradient, weight=1.0):
    layer = Dense(1, 1, weight_init="small_random")
    layer.weights.fill(weight)
    layer.biases.fill(0.0)
    layer.dweights.fill(gradient)
    layer.dbiases.fill(0.0)
    return layer


def test_global_norm_clipping_handles_large_values_and_duplicate_arrays():
    grad = np.array([1e30, 1e30], dtype=np.float32)
    norm = clip_grad_norm((grad, grad), max_norm=1.0)

    assert math.isclose(norm, math.sqrt(2.0) * 1e30, rel_tol=1e-6)
    assert math.isclose(float(np.linalg.norm(grad.astype(np.float64))), 1.0, rel_tol=1e-6)


def test_clip_helpers_accept_gradient_dictionaries_and_validate_thresholds():
    grads = {"weight": np.array([4.0, -3.0], dtype=np.float64)}
    assert clip_grad_norm(grads, max_norm=2.0) == 5.0
    np.testing.assert_allclose(grads["weight"], [1.6, -1.2])

    clip_grad_value(grads, clip_value=1.0)
    np.testing.assert_allclose(grads["weight"], [1.0, -1.0])
    with pytest.raises(ValueError):
        clip_grad_norm(grads, max_norm=float("nan"))
    with pytest.raises(ValueError):
        clip_grad_value(grads, clip_value=0.0)


def test_clipping_read_only_gradient_fails_without_partial_changes():
    writable = np.array([3.0], dtype=np.float64)
    read_only = np.array([4.0], dtype=np.float64)
    read_only.flags.writeable = False

    with pytest.raises(ValueError, match="read-only"):
        clip_grad_norm([writable, read_only], max_norm=1.0)

    np.testing.assert_array_equal(writable, [3.0])


def test_non_finite_gradients_fail_before_changing_parameters():
    layer = _dense_with_gradient(1.0, weight=2.0)
    layer.dweights[0, 0] = np.nan
    before = layer.weights.copy()

    with pytest.raises(FloatingPointError, match="NaN or infinity"):
        Adam().step([layer])

    np.testing.assert_array_equal(layer.weights, before)


def test_optimizer_rejects_mismatched_gradient_shapes_before_updates():
    layer = _dense_with_gradient(1.0, weight=2.0)
    layer._grads["weights"] = np.ones((2, 2), dtype=np.float32)
    before = layer.weights.copy()

    with pytest.raises(ValueError, match="Gradient shape mismatch"):
        SGD().step([layer])

    np.testing.assert_array_equal(layer.weights, before)


def test_extreme_finite_gradient_does_not_corrupt_adam_state():
    layer = _dense_with_gradient(1e30, weight=2.0)
    optimizer = Adam()
    before = layer.weights.copy()

    with pytest.raises(FloatingPointError):
        optimizer.step([layer])

    key = (id(layer), "weights")
    np.testing.assert_array_equal(layer.weights, before)
    assert key not in optimizer._steps
    assert not np.any(optimizer._m[key])
    assert not np.any(optimizer._v[key])


def test_adam_epsilon_is_applied_after_bias_correction():
    layer = _dense_with_gradient(0.3, weight=2.0)
    optimizer = Adam(lr=0.2, beta1=0.5, beta2=0.25, eps=0.2)

    optimizer.step([layer])

    m = 0.5 * 0.3
    v = 0.75 * (0.3 ** 2)
    expected_delta = 0.2 * (m / 0.5) / (math.sqrt(v / 0.75) + 0.2)
    assert math.isclose(float(layer.weights[0, 0]), 2.0 - expected_delta, rel_tol=1e-6)


def test_adam_uses_independent_state_and_bias_correction_per_layer():
    shared_optimizer = Adam(lr=0.1, eps=1.0)
    high_gradient = _dense_with_gradient(10.0)
    later_layer = _dense_with_gradient(1.0)
    fresh_layer = _dense_with_gradient(1.0)

    shared_optimizer.step([high_gradient])
    shared_optimizer.step([later_layer])
    Adam(lr=0.1, eps=1.0).step([fresh_layer])

    np.testing.assert_allclose(later_layer.weights, fresh_layer.weights, rtol=1e-6, atol=1e-7)
    assert shared_optimizer._steps[(id(later_layer), "weights")] == 1


def test_adam_amsgrad_keeps_running_maximum_and_is_exported():
    layer = _dense_with_gradient(2.0)
    optimizer = Adam(lr=0.01, beta1=0.0, beta2=0.5, amsgrad=True)
    key = (id(layer), "weights")

    optimizer.step([layer])
    first_max = optimizer._v_max[key].copy()
    layer.dweights.fill(0.1)
    optimizer.step([layer])

    assert np.all(optimizer._v_max[key] >= first_max)
    assert dn.AdamW is AdamW
    assert isinstance(AdamW(weight_decay=0.1), Adam)


def test_sgd_momentum_stays_correct_when_learning_rate_changes():
    layer = _dense_with_gradient(2.0, weight=10.0)
    optimizer = SGD(lr=0.1, momentum=0.5)

    optimizer.step([layer])  # velocity = 2; weight becomes 9.8
    layer.dweights.fill(2.0)
    optimizer.lr = 0.01
    optimizer.step([layer])  # velocity = 3; second update is 0.03

    assert math.isclose(float(layer.weights[0, 0]), 9.77, rel_tol=1e-6, abs_tol=1e-6)


def test_nesterov_and_rmsprop_decoupled_weight_decay():
    layer = _dense_with_gradient(2.0, weight=10.0)
    SGD(lr=0.1, momentum=0.5, nesterov=True).step([layer])
    assert math.isclose(float(layer.weights[0, 0]), 9.7, rel_tol=1e-6)

    layer = _dense_with_gradient(0.0, weight=2.0)
    RMSprop(lr=0.1, weight_decay=0.2).step([layer])
    assert math.isclose(float(layer.weights[0, 0]), 1.96, rel_tol=1e-6)


def test_optimizer_hyperparameters_are_validated_at_construction():
    for constructor, kwargs in (
        (SGD, {"lr": float("nan")}),
        (SGD, {"weight_decay": -0.1}),
        (SGD, {"nesterov": True}),
        (Adam, {"eps": 0.0}),
        (Adam, {"clip_norm": float("inf")}),
        (RMSprop, {"momentum": 1.0}),
        (RMSprop, {"eps": -1.0}),
    ):
        with pytest.raises(ValueError):
            constructor(**kwargs)


def test_amsgrad_state_survives_model_precision_conversion():
    model = dn.Sequential([Dense(1, 1)])
    optimizer = Adam(amsgrad=True)
    model.compile(loss=dn.MSELoss(), optimizer=optimizer)
    layer = model.layers[0]
    layer.dweights.fill(1.0)
    layer.dbiases.fill(1.0)
    optimizer.step(model.layers)

    assert all(value.dtype == np.float32 for value in optimizer._v_max.values())
    model.to_precision("float64")
    assert all(value.dtype == np.float64 for value in optimizer._v_max.values())
