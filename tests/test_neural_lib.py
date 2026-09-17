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

from doraneural import (
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
    MeanSquaredError,
    MSELoss,
    SGD,
    Adam,
    RMSprop,
    Accuracy,
    MSE,
    MAE,
    set_seed,
    train_test_split,
    one_hot_encode,
    save_model,
    load_model,
    create,
    load_csv,
    create_sample_classification_csv,
    create_sample_regression_csv,
    read_bmp,
    write_bmp,
    load_image,
    load_image_dataset,
    resize_image,
    render_image_ascii,
    create_sample_cat_dog_dataset,
    plot_ascii_curve,
    plot_history,
    export_to_standalone_python,
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


class TestRegression(unittest.TestCase):
    def test_mse_loss_and_gradient(self):
        mse = MeanSquaredError()
        y_pred = np.array([[2.0], [4.0]], dtype=np.float32)
        y_true = np.array([[1.0], [5.0]], dtype=np.float32)
        loss = mse.forward(y_pred, y_true)
        # ( (2-1)^2 + (4-5)^2 ) / 2 = (1 + 1) / 2 = 1.0
        self.assertAlmostEqual(loss, 1.0, places=5)

        # Gradient: 2 * (yp - yt) / N = 2 * [[1], [-1]] / 2 = [[1], [-1]]
        grad = mse.backward(y_pred, y_true)
        np.testing.assert_allclose(grad, [[1.0], [-1.0]], atol=1e-5)

    def test_mse_and_mae_metrics(self):
        mse_metric = MSE()
        mae_metric = MAE()
        yt = np.array([1.0, 3.0, 5.0])
        yp = np.array([2.0, 3.0, 7.0])
        self.assertAlmostEqual(mse_metric(yt, yp), (1.0 + 0.0 + 4.0) / 3.0, places=5)
        self.assertAlmostEqual(mae_metric(yt, yp), (1.0 + 0.0 + 2.0) / 3.0, places=5)

    def test_regression_model_training(self):
        # Learn y = 2*x1 + 3*x2
        X = np.random.randn(100, 2).astype(np.float32)
        y = (2.0 * X[:, 0] + 3.0 * X[:, 1]).reshape(-1, 1).astype(np.float32)

        model = create(inputs=2, hidden=16, outputs=1, task="regression", lr=0.05)
        history = model.fit(X, y, epochs=20, batch_size=16, verbose=0)
        final_loss = history["loss"][-1]
        self.assertLess(final_loss, history["loss"][0])


class TestDataAndCSVLoader(unittest.TestCase):
    def test_load_csv_classification_and_regression(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Classification
            iris_file = Path(tmpdir) / "sample_iris.csv"
            create_sample_classification_csv(iris_file)
            X, y, meta = load_csv(iris_file)
            self.assertEqual(X.shape, (6, 4))
            self.assertEqual(len(y), 6)
            self.assertEqual(len(meta["label_names"]), 3)
            self.assertEqual(meta["target_name"], "species")

            # Regression
            house_file = Path(tmpdir) / "sample_housing.csv"
            create_sample_regression_csv(house_file)
            X_reg, y_reg, meta_reg = load_csv(house_file)
            self.assertEqual(X_reg.shape, (6, 3))
            self.assertEqual(len(y_reg), 6)
            self.assertEqual(meta_reg["target_name"], "price_k")
            self.assertEqual(meta_reg["label_names"], [])


class TestASCIIPlotting(unittest.TestCase):
    def test_plot_ascii_curve(self):
        losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        plot_str = plot_ascii_curve(losses, title="Test Loss", width=30, height=5)
        self.assertIn("Test Loss", plot_str)
        self.assertIn("Epoch 1", plot_str)
        self.assertIn("Epoch 5", plot_str)

    def test_plot_history(self):
        hist = {"loss": [0.5, 0.3, 0.1], "accuracy": [0.6, 0.8, 0.95]}
        result = plot_history(hist)
        self.assertIn("Loss Progress", result)
        self.assertIn("Accuracy Progress", result)


class TestStandaloneExporter(unittest.TestCase):
    def test_standalone_export_classification(self):
        import tempfile
        import subprocess

        model = Sequential([
            Dense(in_features=3, out_features=4),
            ReLU(),
            Dense(in_features=4, out_features=2),
            Softmax(),
        ])
        x = np.random.randn(1, 3).astype(np.float32)
        expected_probas = model.predict_proba(x)[0]
        expected_class = int(np.argmax(expected_probas))

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "predict_test.py"
            export_to_standalone_python(model, script_path)
            self.assertTrue(script_path.exists())

            # Run with python subprocess
            args = [sys.executable, str(script_path)] + [str(v) for v in x[0]]
            res = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0)
            self.assertIn(f"Prediction: Class {expected_class}", res.stdout)

    def test_standalone_export_regression(self):
        import tempfile
        import subprocess

        model = Sequential([
            Dense(in_features=2, out_features=4),
            ReLU(),
            Dense(in_features=4, out_features=1),
        ])
        x = np.random.randn(1, 2).astype(np.float32)
        expected_val = float(model.forward(x)[0, 0])

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "predict_reg.py"
            export_to_standalone_python(model, script_path)
            self.assertTrue(script_path.exists())

            args = [sys.executable, str(script_path)] + [str(v) for v in x[0]]
            res = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0)
            self.assertIn("Prediction:", res.stdout)


class TestImageLoaderAndVision(unittest.TestCase):
    def test_bmp_read_write_roundtrip(self):
        import tempfile
        test_img = np.zeros((16, 20, 3), dtype=np.uint8)
        test_img[4:10, 5:15] = [255, 128, 64]

        with tempfile.TemporaryDirectory() as tmpdir:
            bmp_path = Path(tmpdir) / "test.bmp"
            write_bmp(bmp_path, test_img)
            self.assertTrue(bmp_path.exists())

            loaded = read_bmp(bmp_path)
            self.assertEqual(loaded.shape, (16, 20, 3))
            np.testing.assert_array_equal(loaded[5, 6], [255, 128, 64])

    def test_resize_image(self):
        orig = np.random.randint(0, 256, (50, 40, 3), dtype=np.uint8).astype(np.float32)
        resized = resize_image(orig, target_size=(24, 24))
        self.assertEqual(resized.shape, (24, 24, 3))

    def test_load_image_dataset_and_ascii(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = create_sample_cat_dog_dataset(Path(tmpdir) / "data", samples_per_class=4)
            X, y, classes = load_image_dataset(data_dir, target_size=(16, 16), grayscale=True)
            self.assertEqual(X.shape, (8, 1, 16, 16))
            self.assertEqual(len(y), 8)
            self.assertEqual(classes, ["cats", "dogs"])

            # Test ASCII renderer
            ascii_art = render_image_ascii(X[0, 0], width=12, height=6)
            self.assertIn("┌", ascii_art)
            self.assertIn("┘", ascii_art)


if __name__ == "__main__":
    unittest.main()
