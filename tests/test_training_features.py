"""Tests for bigger-effective-batch training and early stopping."""

import numpy as np
import pytest

import doraneural as dn


def _linear_regression_model(learning_rate=0.05, optimizer=None):
    layer = dn.Dense(1, 1, weight_init="small_random")
    layer.weights.fill(0.2)
    layer.biases.fill(-0.1)
    model = dn.Sequential([layer])
    model.compile(
        loss=dn.MSELoss(),
        optimizer=optimizer if optimizer is not None else dn.SGD(lr=learning_rate),
    )
    return model


def test_gradient_accumulation_matches_a_single_larger_batch_update():
    X = np.array([[1.0], [2.0], [-1.0], [0.5]], dtype=np.float32)
    y = np.array([[0.5], [1.0], [-0.5], [0.25]], dtype=np.float32)
    full_batch = _linear_regression_model()
    micro_batches = _linear_regression_model()

    full_batch.fit(X, y, epochs=1, batch_size=4, shuffle=False, verbose=0)
    micro_batches.fit(
        X,
        y,
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=4,
        shuffle=False,
        verbose=0,
    )

    np.testing.assert_allclose(micro_batches.layers[0].weights, full_batch.layers[0].weights, atol=1e-7)
    np.testing.assert_allclose(micro_batches.layers[0].biases, full_batch.layers[0].biases, atol=1e-7)


def test_gradient_accumulation_flushes_the_final_partial_group():
    X = np.arange(5, dtype=np.float32).reshape(-1, 1)
    y = np.ones((5, 1), dtype=np.float32)
    optimizer = dn.Adam(lr=0.01)
    model = _linear_regression_model(optimizer=optimizer)

    model.fit(
        X,
        y,
        epochs=1,
        batch_size=2,
        gradient_accumulation_steps=2,
        shuffle=False,
        verbose=0,
    )

    # Three micro-batches form one full group and one final partial group.
    assert optimizer.t == 2


def test_early_stopping_restores_weights_from_best_validation_epoch():
    X_train = np.ones((1, 1), dtype=np.float32)
    y_train = np.array([[10.0]], dtype=np.float32)
    validation_data = (X_train, np.zeros((1, 1), dtype=np.float32))

    reference = _linear_regression_model(learning_rate=0.1)
    reference.layers[0].weights.fill(0.0)
    reference.layers[0].biases.fill(0.0)
    reference.fit(X_train, y_train, epochs=1, shuffle=False, verbose=0)

    model = _linear_regression_model(learning_rate=0.1)
    model.layers[0].weights.fill(0.0)
    model.layers[0].biases.fill(0.0)
    history = model.fit(
        X_train,
        y_train,
        epochs=8,
        shuffle=False,
        validation_data=validation_data,
        early_stopping_patience=1,
        verbose=0,
    )

    assert history.stopped_early
    assert history.best_epoch == 1
    assert history.stopped_epoch == 2
    assert len(history["loss"]) == 2
    np.testing.assert_allclose(model.layers[0].weights, reference.layers[0].weights)
    np.testing.assert_allclose(model.layers[0].biases, reference.layers[0].biases)


def test_reduce_lr_on_plateau_tracks_validation_loss_in_fit():
    optimizer = dn.Adam(lr=0.1)
    model = _linear_regression_model(optimizer=optimizer)
    model.layers[0].weights.fill(0.0)
    model.layers[0].biases.fill(0.0)
    scheduler = dn.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=0,
        min_lr=0.02,
        threshold=0.0,
    )
    X = np.zeros((2, 1), dtype=np.float32)
    y_train = np.zeros((2, 1), dtype=np.float32)
    y_valid = np.ones((2, 1), dtype=np.float32)

    model.fit(
        X,
        y_train,
        epochs=4,
        shuffle=False,
        validation_data=(X, y_valid),
        scheduler=scheduler,
        verbose=0,
    )

    assert scheduler.num_reductions == 3
    assert optimizer.lr == pytest.approx(0.02)


def test_metric_scheduler_requires_validation_data():
    model = _linear_regression_model()
    X = np.ones((2, 1), dtype=np.float32)
    y = np.ones((2, 1), dtype=np.float32)
    scheduler = dn.ReduceLROnPlateau(model.optimizer)

    with pytest.raises(ValueError, match="requires validation_data"):
        model.fit(X, y, epochs=1, scheduler=scheduler, verbose=0)


def test_reduce_lr_on_plateau_supports_max_mode():
    optimizer = dn.SGD(lr=0.1)
    scheduler = dn.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=0,
        min_lr=0.01,
        mode="max",
        threshold=0.0,
    )

    assert scheduler.step(0.8) == pytest.approx(0.1)
    assert scheduler.step(0.8) == pytest.approx(0.05)
    assert scheduler.step(0.7) == pytest.approx(0.025)


def test_early_stopping_requires_validation_data_and_valid_options():
    model = _linear_regression_model()
    X = np.ones((2, 1), dtype=np.float32)
    y = np.ones((2, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="requires validation_data"):
        model.fit(X, y, epochs=2, early_stopping_patience=2, verbose=0)
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        model.fit(X, y, epochs=1, gradient_accumulation_steps=0, verbose=0)
    with pytest.raises(ValueError, match="min_delta"):
        model.fit(X, y, epochs=1, min_delta=-0.1, verbose=0)
