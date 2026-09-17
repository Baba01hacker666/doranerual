"""Automated unit test suite for neural_lib.

Tests layer math, finite-difference gradient checks, numerical stability,
shape validations, optimizer mechanics (SGD, Adam, RMSprop), normalization,
dropout, and spatial convolution operations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np

from neural_lib import (
    Sequential,
    Dense,
    Dropout,
    LayerNorm,
    Flatten,
    Conv2D,
    MaxPool2D,
    ReLU,
    Sigmoid,
    Softmax,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    SGD,
    Adam,
    RMSprop,
    Accuracy,
    set_seed,
    train_test_split,
    one_hot_encode,
    save_model,
    load_model,
)


class TestLayersAndActivations(unittest.TestCase):
    def setUp(self):
        set_seed(42)

    def test_dense_forward_backward_shapes(self):
        dense = Dense(in_features=4, out_features=3)
        x = np.random.randn(5, 4).astype(np.float32)
        out = dense.forward(x)
        self.assertEqual(out.shape, (5, 3))

        grad_out = np.ones_like(out)
        grad_in = dense.backward(grad_out)
        self.assertEqual(grad_in.shape, (5, 4))
        self.assertEqual(dense.dweights.shape, (4, 3))
        self.assertEqual(dense.dbiases.shape, (1, 3))

    def test_dense_gradient_check(self):
        """Verify Dense layer backward pass against numerical finite differences."""
        dense = Dense(in_features=3, out_features=2, weight_init="xavier")
        x = np.random.randn(4, 3).astype(np.float32)
        eps = 1e-4

        out = dense.forward(x)
        grad_out = np.random.randn(*out.shape).astype(np.float32)
        dense.backward(grad_out)
        analytical_dw = dense.dweights.copy()

        numerical_dw = np.zeros_like(dense.weights)
        for i in range(dense.weights.shape[0]):
            for j in range(dense.weights.shape[1]):
                orig = dense.weights[i, j]
                dense.weights[i, j] = orig + eps
                out_pos = dense.forward(x)
                loss_pos = np.sum(out_pos * grad_out)

                dense.weights[i, j] = orig - eps
                out_neg = dense.forward(x)
                loss_neg = np.sum(out_neg * grad_out)

                dense.weights[i, j] = orig
                numerical_dw[i, j] = (loss_pos - loss_neg) / (2 * eps)

        rel_error = np.linalg.norm(analytical_dw - numerical_dw) / (
            np.linalg.norm(analytical_dw) + np.linalg.norm(numerical_dw) + 1e-8
        )
        self.assertLess(rel_error, 1e-3)

    def test_dropout(self):
        drop = Dropout(drop_rate=0.5)
        x = np.ones((100, 100), dtype=np.float32)

        # Training mode: units dropped and scaled
        drop.train(True)
        out_train = drop.forward(x)
        zero_frac = np.mean(out_train == 0.0)
        self.assertTrue(0.40 < zero_frac < 0.60)

        # Eval mode: identity pass-through
        drop.eval()
        out_eval = drop.forward(x)
        np.testing.assert_allclose(out_eval, x)

    def test_layernorm(self):
        ln = LayerNorm(normalized_shape=10)
        x = np.random.randn(8, 10).astype(np.float32) * 5.0 + 3.0
        out = ln.forward(x)

        # Mean should be ~0, variance ~1
        mean_out = np.mean(out, axis=-1)
        var_out = np.var(out, axis=-1)
        np.testing.assert_allclose(mean_out, np.zeros_like(mean_out), atol=1e-5)
        np.testing.assert_allclose(var_out, np.ones_like(var_out), atol=1e-3)

        # Backward gradient shape
        grad_out = np.ones_like(out)
        dx = ln.backward(grad_out)
        self.assertEqual(dx.shape, x.shape)

    def test_conv2d_and_maxpool(self):
        # Conv2D: 1 in_channel, 4 out_channels, 3x3 kernel, pad=1
        conv = Conv2D(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        pool = MaxPool2D(pool_size=2, stride=2)
        flat = Flatten()

        x = np.random.randn(2, 1, 8, 8).astype(np.float32)
        out_conv = conv.forward(x)
        self.assertEqual(out_conv.shape, (2, 4, 8, 8))

        out_pool = pool.forward(out_conv)
        self.assertEqual(out_pool.shape, (2, 4, 4, 4))

        out_flat = flat.forward(out_pool)
        self.assertEqual(out_flat.shape, (2, 4 * 4 * 4))

        # Backward propagation through the mini-pipeline
        d_flat = np.ones_like(out_flat)
        d_pool = flat.backward(d_flat)
        self.assertEqual(d_pool.shape, out_pool.shape)

        d_conv = pool.backward(d_pool)
        self.assertEqual(d_conv.shape, out_conv.shape)

        dx = conv.backward(d_conv)
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(conv.dweights.shape, conv.weights.shape)

    def test_relu(self):
        relu = ReLU()
        x = np.array([[-2.0, 0.0, 3.0]], dtype=np.float32)
        out = relu.forward(x)
        np.testing.assert_allclose(out, [[0.0, 0.0, 3.0]])

        grad_out = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        grad_in = relu.backward(grad_out)
        np.testing.assert_allclose(grad_in, [[0.0, 0.0, 1.0]])

    def test_sigmoid_numerical_stability(self):
        sig = Sigmoid()
        extreme_x = np.array([[-1000.0, 0.0, 1000.0]], dtype=np.float32)
        out = sig.forward(extreme_x)
        self.assertFalse(np.isnan(out).any())
        self.assertAlmostEqual(float(out[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(out[0, 1]), 0.5, places=5)
        self.assertAlmostEqual(float(out[0, 2]), 1.0, places=5)

    def test_softmax_properties(self):
        softmax = Softmax()
        x = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, 0.0, 1000.0]], dtype=np.float32)
        out = softmax.forward(x)
        self.assertFalse(np.isnan(out).any())
        sums = np.sum(out, axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-5)


class TestOptimizers(unittest.TestCase):
    def test_adam_and_rmsprop_step(self):
        dense1 = Dense(4, 2)
        dense2 = Dense(4, 2)
        dense2.weights[:] = dense1.weights[:]
        dense2.biases[:] = dense1.biases[:]

        adam = Adam(lr=0.01)
        rmsprop = RMSprop(lr=0.01)

        x = np.random.randn(3, 4).astype(np.float32)
        grad = np.ones((3, 2), dtype=np.float32)

        # Run forward/backward to populate grads
        dense1.forward(x)
        dense1.backward(grad)

        dense2.forward(x)
        dense2.backward(grad)

        # Verify weights change upon step
        w1_before = dense1.weights.copy()
        adam.step([dense1])
        self.assertFalse(np.allclose(w1_before, dense1.weights))

        w2_before = dense2.weights.copy()
        rmsprop.step([dense2])
        self.assertFalse(np.allclose(w2_before, dense2.weights))


class TestLossesAndMetrics(unittest.TestCase):
    def test_bce_loss_and_grad(self):
        bce = BinaryCrossEntropy()
        y_true = np.array([[1.0], [0.0]], dtype=np.float32)
        y_pred = np.array([[0.9], [0.1]], dtype=np.float32)

        loss = bce.forward(y_pred, y_true)
        self.assertGreater(loss, 0.0)

        grad = bce.backward(y_pred, y_true)
        self.assertEqual(grad.shape, (2, 1))
        self.assertLess(grad[0, 0], 0.0)
        self.assertGreater(grad[1, 0], 0.0)

    def test_cce_with_indices_and_onehot(self):
        cce = CategoricalCrossEntropy()
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)

        y_idx = np.array([0, 1])
        loss1 = cce.forward(y_pred, y_idx)

        y_oh = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        loss2 = cce.forward(y_pred, y_oh)

        self.assertAlmostEqual(loss1, loss2, places=5)

    def test_accuracy_metric(self):
        acc = Accuracy()
        yt_bin = np.array([[1], [0], [1], [0]])
        yp_bin = np.array([[0.9], [0.1], [0.8], [0.7]])
        self.assertEqual(acc(yt_bin, yp_bin), 0.75)

        yt_multi = np.array([0, 1, 2, 1])
        yp_multi = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.5, 0.3, 0.2],
        ])
        self.assertEqual(acc(yt_multi, yp_multi), 0.75)


class TestModelAndSerialization(unittest.TestCase):
    def test_input_shape_validation_error(self):
        model = Sequential([
            Dense(in_features=4, out_features=2),
            ReLU(),
        ])
        x_wrong = np.random.randn(10, 3)
        with self.assertRaises(ValueError) as ctx:
            model.forward(x_wrong)
        self.assertIn("Dense layer expected 4 features", str(ctx.exception))

    def test_extended_serialization_roundtrip(self):
        import tempfile
        from pathlib import Path

        set_seed(42)
        model = Sequential([
            Dense(in_features=6, out_features=12),
            LayerNorm(normalized_shape=12),
            ReLU(),
            Dropout(drop_rate=0.2),
            Dense(in_features=12, out_features=3),
            Softmax(),
        ])

        x = np.random.randn(5, 6).astype(np.float32)
        orig_out = model.predict_proba(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_extended_model"
            model.save(model_path)

            loaded = load_model(model_path)
            loaded_out = loaded.predict_proba(x)

            np.testing.assert_allclose(orig_out, loaded_out, rtol=1e-6, atol=1e-6)
            self.assertEqual(len(model.layers), len(loaded.layers))


if __name__ == "__main__":
    unittest.main()
