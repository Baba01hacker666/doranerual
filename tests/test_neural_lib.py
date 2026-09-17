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
    SimpleRNN,
    RNN,
    LSTM,
    GRU,
    PositionalEncoding,
    MultiHeadAttention,
    TransformerBlock,
    LRScheduler,
    StepLR,
    CosineAnnealingLR,
    WarmupCosineLR,
    clip_grad_norm,
    clip_grad_value,
    DataLoader,
    Dataset,
    ArrayDataset,
    set_precision,
    get_precision,
    precision_scope,
    memory_summary,
    set_im2col_backend,
    get_im2col_backend,
    is_numba_available,
    Tensor,
    Parameter,
    Module,
    Linear,
    tensor,
    no_grad,
    mse_loss,
    binary_cross_entropy,
    compile_model,
    CompiledModel,
    save_dnb,
    load_dnb,
    inspect_dnb,
    LlamaLLM,
    LlamaTokenizer,
    load_pretrained_llm,
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

    def test_conv2d_dilation_and_padding_same(self):
        conv = Conv2D(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding="same", dilation=2)
        x = np.random.randn(2, 2, 8, 8).astype(np.float32)
        out = conv.forward(x)
        self.assertEqual(out.shape, (2, 3, 8, 8))

        dx = conv.backward(np.ones_like(out))
        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(conv.dweights.shape, conv.weights.shape)

    def test_dense_3d_support(self):
        dense = Dense(in_features=8, out_features=12)
        x = np.random.randn(3, 5, 8).astype(np.float32)
        out = dense.forward(x)
        self.assertEqual(out.shape, (3, 5, 12))

        dx = dense.backward(np.ones_like(out))
        self.assertEqual(dx.shape, (3, 5, 8))
        self.assertEqual(dense.dweights.shape, (8, 12))
        self.assertEqual(dense.dbiases.shape, (1, 12))

    def test_l1_l2_regularization(self):
        dense = Dense(in_features=4, out_features=2, l1_reg=0.1, l2_reg=0.2)
        x = np.random.randn(3, 4).astype(np.float32)
        out = dense.forward(x)
        grad_out = np.zeros_like(out)
        dense.backward(grad_out)

        expected_dw = 0.1 * np.sign(dense.weights) + 0.2 * dense.weights
        np.testing.assert_allclose(dense.dweights, expected_dw, atol=1e-6)


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

    def test_sgd_with_weight_decay(self):
        dense = Dense(4, 2)
        sgd = SGD(lr=0.1, weight_decay=0.05)
        x = np.random.randn(2, 4).astype(np.float32)
        dense.forward(x)
        dense.backward(np.zeros((2, 2), dtype=np.float32))

        w_before = dense.weights.copy()
        sgd.step([dense])
        expected_w = w_before - 0.1 * 0.05 * w_before
        np.testing.assert_allclose(dense.weights, expected_w, atol=1e-6)

    def test_clip_grad_norm(self):
        dense = Dense(4, 2)
        dense.dweights[:] = 100.0
        dense.dbiases[:] = 100.0
        total_norm = clip_grad_norm([dense], max_norm=1.0)
        self.assertGreater(total_norm, 1.0)

        new_norm = np.sqrt(np.sum(dense.dweights ** 2) + np.sum(dense.dbiases ** 2))
        self.assertAlmostEqual(new_norm, 1.0, places=4)

    def test_clip_grad_value(self):
        dense = Dense(4, 2)
        dense.dweights[:] = 50.0
        dense.dbiases[:] = -30.0
        clip_grad_value([dense], clip_value=5.0)
        self.assertTrue(np.all(dense.dweights <= 5.0))
        self.assertTrue(np.all(dense.dbiases >= -5.0))


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


class TestRecurrentLayers(unittest.TestCase):
    def setUp(self):
        set_seed(42)

    def test_simplernn_shapes(self):
        x = np.random.randn(2, 6, 4).astype(np.float32)

        rnn_seq = SimpleRNN(in_features=4, hidden_units=5, return_sequences=True)
        out_seq = rnn_seq.forward(x)
        self.assertEqual(out_seq.shape, (2, 6, 5))
        dx = rnn_seq.backward(np.ones_like(out_seq))
        self.assertEqual(dx.shape, (2, 6, 4))
        self.assertEqual(rnn_seq.dW_xh.shape, (4, 5))
        self.assertEqual(rnn_seq.dW_hh.shape, (5, 5))

        rnn_last = SimpleRNN(in_features=4, hidden_units=5, return_sequences=False)
        out_last = rnn_last.forward(x)
        self.assertEqual(out_last.shape, (2, 5))
        dx = rnn_last.backward(np.ones_like(out_last))
        self.assertEqual(dx.shape, (2, 6, 4))

    def test_lstm_shapes(self):
        x = np.random.randn(2, 6, 4).astype(np.float32)

        lstm_seq = LSTM(in_features=4, hidden_units=5, return_sequences=True)
        out_seq = lstm_seq.forward(x)
        self.assertEqual(out_seq.shape, (2, 6, 5))
        dx = lstm_seq.backward(np.ones_like(out_seq))
        self.assertEqual(dx.shape, (2, 6, 4))

        lstm_last = LSTM(in_features=4, hidden_units=5, return_sequences=False)
        out_last = lstm_last.forward(x)
        self.assertEqual(out_last.shape, (2, 5))
        dx = lstm_last.backward(np.ones_like(out_last))
        self.assertEqual(dx.shape, (2, 6, 4))

    def test_gru_shapes(self):
        x = np.random.randn(2, 6, 4).astype(np.float32)

        gru_seq = GRU(in_features=4, hidden_units=5, return_sequences=True)
        out_seq = gru_seq.forward(x)
        self.assertEqual(out_seq.shape, (2, 6, 5))
        dx = gru_seq.backward(np.ones_like(out_seq))
        self.assertEqual(dx.shape, (2, 6, 4))

        gru_last = GRU(in_features=4, hidden_units=5, return_sequences=False)
        out_last = gru_last.forward(x)
        self.assertEqual(out_last.shape, (2, 5))
        dx = gru_last.backward(np.ones_like(out_last))
        self.assertEqual(dx.shape, (2, 6, 4))

    def test_simplernn_numerical_gradient(self):
        rnn = SimpleRNN(in_features=3, hidden_units=3, return_sequences=True)
        x = np.random.randn(2, 4, 3).astype(np.float32)
        grad_out = np.random.randn(2, 4, 3).astype(np.float32)
        eps = 1e-4

        rnn.forward(x)
        rnn.backward(grad_out)
        analytical = rnn.dW_xh.copy()

        numerical = np.zeros_like(rnn.W_xh)
        for i in range(rnn.W_xh.shape[0]):
            for j in range(rnn.W_xh.shape[1]):
                orig = rnn.W_xh[i, j]
                rnn.W_xh[i, j] = orig + eps
                l_pos = np.sum(rnn.forward(x) * grad_out)
                rnn.W_xh[i, j] = orig - eps
                l_neg = np.sum(rnn.forward(x) * grad_out)
                rnn.W_xh[i, j] = orig
                numerical[i, j] = (l_pos - l_neg) / (2 * eps)

        rel_err = np.linalg.norm(analytical - numerical) / (np.linalg.norm(analytical) + np.linalg.norm(numerical) + 1e-8)
        self.assertLess(rel_err, 1e-3)


class TestAttentionAndTransformers(unittest.TestCase):
    def setUp(self):
        set_seed(42)

    def test_positional_encoding(self):
        pe = PositionalEncoding(d_model=8, max_len=50)
        x = np.random.randn(2, 10, 8).astype(np.float32)
        out = pe.forward(x)
        self.assertEqual(out.shape, (2, 10, 8))
        dx = pe.backward(np.ones_like(out))
        self.assertEqual(dx.shape, (2, 10, 8))
        np.testing.assert_array_equal(dx, np.ones_like(out))

    def test_multihead_attention_shapes_and_mask(self):
        x = np.random.randn(2, 6, 8).astype(np.float32)

        mha = MultiHeadAttention(d_model=8, num_heads=2, causal=False)
        out = mha.forward(x)
        self.assertEqual(out.shape, (2, 6, 8))
        dx = mha.backward(np.ones_like(out))
        self.assertEqual(dx.shape, (2, 6, 8))

        mha_causal = MultiHeadAttention(d_model=8, num_heads=2, causal=True)
        out_causal = mha_causal.forward(x)
        self.assertEqual(out_causal.shape, (2, 6, 8))
        dx_causal = mha_causal.backward(np.ones_like(out_causal))
        self.assertEqual(dx_causal.shape, (2, 6, 8))

    def test_multihead_attention_numerical_gradient(self):
        mha = MultiHeadAttention(d_model=4, num_heads=2, causal=False)
        x = np.random.randn(2, 3, 4).astype(np.float32)
        grad_out = np.random.randn(2, 3, 4).astype(np.float32)
        eps = 1e-4

        mha.forward(x)
        mha.backward(grad_out)
        analytical = mha.dW_q.copy()

        numerical = np.zeros_like(mha.W_q)
        for i in range(mha.W_q.shape[0]):
            for j in range(mha.W_q.shape[1]):
                orig = mha.W_q[i, j]
                mha.W_q[i, j] = orig + eps
                l_pos = np.sum(mha.forward(x) * grad_out)
                mha.W_q[i, j] = orig - eps
                l_neg = np.sum(mha.forward(x) * grad_out)
                mha.W_q[i, j] = orig
                numerical[i, j] = (l_pos - l_neg) / (2 * eps)

        rel_err = np.linalg.norm(analytical - numerical) / (np.linalg.norm(analytical) + np.linalg.norm(numerical) + 1e-8)
        self.assertLess(rel_err, 1e-3)

    def test_transformer_block_forward_backward(self):
        tb = TransformerBlock(d_model=8, num_heads=2, d_ff=16, dropout=0.2)
        x = np.random.randn(2, 5, 8).astype(np.float32)

        tb.train(True)
        out_train = tb.forward(x)
        self.assertEqual(out_train.shape, (2, 5, 8))
        dx_train = tb.backward(np.ones_like(out_train))
        self.assertEqual(dx_train.shape, (2, 5, 8))

        tb.eval()
        out_eval1 = tb.forward(x)
        out_eval2 = tb.forward(x)
        np.testing.assert_allclose(out_eval1, out_eval2)


class TestSchedulers(unittest.TestCase):
    def test_step_lr(self):
        dense = Dense(2, 2)
        opt = SGD(lr=0.1)
        scheduler = StepLR(opt, step_size=5, gamma=0.5)

        self.assertEqual(scheduler.step(), 0.1)
        for _ in range(4):
            scheduler.step()
        self.assertEqual(scheduler.current_lr, 0.1)
        scheduler.step()
        self.assertAlmostEqual(scheduler.current_lr, 0.05)
        self.assertAlmostEqual(opt.lr, 0.05)

    def test_cosine_annealing_lr(self):
        dense = Dense(2, 2)
        opt = SGD(lr=0.1)
        scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=0.01)

        lr0 = scheduler.step(0)
        self.assertAlmostEqual(lr0, 0.1)
        lr5 = scheduler.step(5)
        self.assertTrue(0.01 < lr5 < 0.1)
        lr10 = scheduler.step(10)
        self.assertAlmostEqual(lr10, 0.01)

    def test_warmup_cosine_lr(self):
        dense = Dense(2, 2)
        opt = SGD(lr=0.1)
        scheduler = WarmupCosineLR(opt, warmup_epochs=5, total_epochs=20, eta_min=0.001)

        lr0 = scheduler.step(0)
        self.assertAlmostEqual(lr0, 0.02)
        lr4 = scheduler.step(4)
        self.assertAlmostEqual(lr4, 0.1)

        lr10 = scheduler.step(10)
        self.assertTrue(0.001 < lr10 < 0.1)

        lr20 = scheduler.step(20)
        self.assertAlmostEqual(lr20, 0.001)

    def test_model_fit_with_scheduler_and_clipping(self):
        model = Sequential([
            Dense(in_features=4, out_features=8),
            ReLU(),
            Dense(in_features=8, out_features=1),
        ])
        opt = Adam(lr=0.05)
        model.compile(optimizer=opt, loss=MSELoss())

        scheduler = CosineAnnealingLR(opt, T_max=5)
        X = np.random.randn(20, 4).astype(np.float32)
        y = np.random.randn(20, 1).astype(np.float32)

        hist = model.fit(X, y, epochs=5, batch_size=4, verbose=0, scheduler=scheduler, clip_norm=1.0)
        self.assertEqual(len(hist.history["loss"]), 5)
        self.assertLess(opt.lr, 0.05)


class TestHardwareAcceleration(unittest.TestCase):
    def test_numba_and_numpy_im2col_equivalence(self):
        from doraneural.accel import im2col_indices, col2im_indices, set_im2col_backend

        x = np.random.randn(2, 3, 8, 8).astype(np.float32)

        # Force pure numpy
        set_im2col_backend("numpy")
        cols_np, oh_np, ow_np = im2col_indices(x, 3, 3, padding=1, stride=1, dilation=1)
        dx_np = col2im_indices(cols_np, x.shape, 3, 3, padding=1, stride=1, dilation=1)

        # Force numba if available
        if is_numba_available():
            set_im2col_backend("numba")
            cols_nb, oh_nb, ow_nb = im2col_indices(x, 3, 3, padding=1, stride=1, dilation=1)
            dx_nb = col2im_indices(cols_nb, x.shape, 3, 3, padding=1, stride=1, dilation=1)

            self.assertEqual(oh_np, oh_nb)
            self.assertEqual(ow_np, ow_nb)
            np.testing.assert_allclose(cols_np, cols_nb, atol=1e-6)
            np.testing.assert_allclose(dx_np, dx_nb, atol=1e-6)

        # Reset to auto
        set_im2col_backend("auto")

    def test_backend_toggle_error(self):
        with self.assertRaises(ValueError):
            set_im2col_backend("nonexistent_backend")


class TestPrecisionManagement(unittest.TestCase):
    def test_precision_getter_setter(self):
        orig = get_precision()
        try:
            set_precision("float64")
            self.assertEqual(get_precision(), np.float64)
            set_precision("float32")
            self.assertEqual(get_precision(), np.float32)
        finally:
            set_precision(orig)

    def test_precision_scope(self):
        orig = get_precision()
        with precision_scope("float64"):
            self.assertEqual(get_precision(), np.float64)
        self.assertEqual(get_precision(), orig)

    def test_model_and_layer_precision_casting(self):
        dense = Dense(in_features=10, out_features=5)
        self.assertEqual(dense.weights.dtype, np.float32)

        dense.to_precision("float64")
        self.assertEqual(dense.weights.dtype, np.float64)
        self.assertEqual(dense.biases.dtype, np.float64)

        dense.cast("float32")
        self.assertEqual(dense.weights.dtype, np.float32)

        model = Sequential([
            Dense(in_features=4, out_features=8),
            ReLU(),
            Dense(in_features=8, out_features=2),
        ])
        model.compile(optimizer=Adam(lr=0.01), loss=MSELoss())

        mem_f32 = model.memory_summary()
        self.assertIn("float32", mem_f32["dtype"])
        self.assertIn("50% smaller", mem_f32["formatted"])

        model.to_precision("float64")
        self.assertEqual(model.dtype, np.float64)
        self.assertEqual(model.layers[0].weights.dtype, np.float64)

        mem_f64 = model.memory_summary()
        self.assertEqual(mem_f64["total_bytes"], 2 * mem_f32["total_bytes"])

    def test_memory_summary_structure(self):
        dense = Dense(4, 2)
        summary = memory_summary(dense)
        self.assertEqual(summary["num_parameters"], 4 * 2 + 2)
        self.assertGreater(summary["total_bytes"], 0)


class TestParallelDataLoader(unittest.TestCase):
    def test_dataloader_iteration_and_shapes(self):
        X = np.arange(100).reshape(50, 2).astype(np.float32)
        y = np.arange(50).reshape(50, 1).astype(np.float32)

        loader = DataLoader((X, y), batch_size=16, shuffle=False, prefetch_factor=2)
        self.assertEqual(len(loader), 4)

        batches = list(loader)
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0][0].shape, (16, 2))
        self.assertEqual(batches[0][1].shape, (16, 1))
        self.assertEqual(batches[3][0].shape, (2, 2))

    def test_dataloader_drop_last(self):
        X = np.random.randn(50, 4).astype(np.float32)
        y = np.random.randn(50, 1).astype(np.float32)

        loader_drop = DataLoader((X, y), batch_size=16, shuffle=False, drop_last=True)
        self.assertEqual(len(loader_drop), 3)
        batches = list(loader_drop)
        self.assertEqual(len(batches), 3)

    def test_dataloader_transform(self):
        X = np.ones((20, 2), dtype=np.float32)
        y = np.zeros((20, 1), dtype=np.float32)

        def add_ten(x_batch, y_batch):
            return x_batch + 10.0, y_batch + 5.0

        loader = DataLoader((X, y), batch_size=10, transform=add_ten, shuffle=False)
        for x_b, y_b in loader:
            np.testing.assert_allclose(x_b, 11.0)
            np.testing.assert_allclose(y_b, 5.0)

    def test_dataloader_early_break_and_reiteration(self):
        X = np.random.randn(40, 2).astype(np.float32)
        y = np.random.randn(40, 1).astype(np.float32)

        loader = DataLoader((X, y), batch_size=10, prefetch_factor=2)
        for b in loader:
            break

        count = sum(1 for _ in loader)
        self.assertEqual(count, 4)

    def test_model_fit_with_dataloader(self):
        X = np.random.randn(30, 4).astype(np.float32)
        y = np.random.randn(30, 1).astype(np.float32)

        loader = DataLoader((X, y), batch_size=10, shuffle=True, prefetch_factor=2)
        model = Sequential([
            Dense(in_features=4, out_features=2),
            Dense(in_features=2, out_features=1),
        ])
        model.compile(optimizer=Adam(lr=0.01), loss=MSELoss())

        hist = model.fit(loader, epochs=2, verbose=0)
        self.assertEqual(len(hist.history["loss"]), 2)


class TestAutogradEngine(unittest.TestCase):
    def test_scalar_arithmetic_backward(self):
        a = Tensor(3.0, requires_grad=True)
        b = Tensor(4.0, requires_grad=True)
        c = (a * 2.0 + b) ** 2.0
        c.backward()
        self.assertAlmostEqual(a.grad.item(), 40.0)
        self.assertAlmostEqual(b.grad.item(), 20.0)

    def test_tensor_broadcasting_backward(self):
        A = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        B = Tensor(np.ones((4, 3)), requires_grad=True)
        C = (A + B) * 2.0
        loss = C.sum()
        loss.backward()

        np.testing.assert_allclose(A.grad, [[8.0, 8.0, 8.0]])
        np.testing.assert_allclose(B.grad, np.full((4, 3), 2.0))

    def test_matrix_multiplication_backward(self):
        X = Tensor(np.random.randn(5, 3))
        W = Tensor(np.random.randn(3, 2), requires_grad=True)
        out = X @ W
        loss = out.sum()
        loss.backward()

        expected_dw = X.data.T @ np.ones((5, 2))
        np.testing.assert_allclose(W.grad, expected_dw, atol=1e-6)

    def test_activations_backward(self):
        x = Tensor([-2.0, 0.5, 3.0], requires_grad=True)
        y = x.relu()
        y.sum().backward()
        np.testing.assert_allclose(x.grad, [0.0, 1.0, 1.0])

        x.zero_grad()
        sig = x.sigmoid()
        sig.sum().backward()
        s = 1.0 / (1.0 + np.exp(-x.data))
        np.testing.assert_allclose(x.grad, s * (1.0 - s), atol=1e-6)

    def test_finite_difference_gradient_check(self):
        X = Tensor(np.random.randn(4, 3))
        W = Tensor(np.random.randn(3, 2), requires_grad=True)
        b = Tensor(np.random.randn(1, 2), requires_grad=True)

        out = (X @ W + b).tanh()
        loss = (out ** 2).mean()
        loss.backward()
        analytical_dw = W.grad.copy()

        eps = 1e-5
        numerical_dw = np.zeros_like(W.data)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                orig = W.data[i, j]
                W.data[i, j] = orig + eps
                l_pos = np.mean(np.tanh(X.data @ W.data + b.data) ** 2)
                W.data[i, j] = orig - eps
                l_neg = np.mean(np.tanh(X.data @ W.data + b.data) ** 2)
                W.data[i, j] = orig
                numerical_dw[i, j] = (l_pos - l_neg) / (2 * eps)

        rel_err = np.linalg.norm(analytical_dw - numerical_dw) / (
            np.linalg.norm(analytical_dw) + np.linalg.norm(numerical_dw) + 1e-8
        )
        self.assertLess(rel_err, 1e-4)

    def test_linear_module_and_no_grad(self):
        fc = Linear(4, 2)
        x = Tensor(np.random.randn(3, 4))
        y = Tensor(np.random.randn(3, 2))

        out = fc(x)
        loss = mse_loss(out, y)
        loss.backward()
        self.assertIsNotNone(fc.weight.grad)

        with no_grad():
            out_eval = fc(x)
            self.assertFalse(out_eval.requires_grad)


class TestStaticCompiler(unittest.TestCase):
    def test_dense_relu_operator_fusion_and_buffers(self):
        model = Sequential([
            Dense(in_features=8, out_features=16),
            ReLU(),
            Dense(in_features=16, out_features=4),
        ])
        x = np.random.randn(5, 8).astype(np.float32)
        orig_out = model.forward(x)

        compiled = compile_model(model, sample_input=x)
        self.assertGreater(compiled.buffer_pool.total_bytes, 0)

        fused_steps = [s for s in compiled.steps if s.is_fused]
        self.assertEqual(len(fused_steps), 1)
        self.assertIn("FusedDense+RELU", fused_steps[0].name)

        comp_out = compiled.forward(x)
        np.testing.assert_allclose(orig_out, comp_out, atol=1e-6)

    def test_model_compile_graph_method(self):
        model = Sequential([
            Dense(in_features=4, out_features=8),
            ReLU(),
            Dense(in_features=8, out_features=1),
        ])
        x = np.random.randn(3, 4).astype(np.float32)
        compiled = model.compile_graph(sample_input=x)
        self.assertIsInstance(compiled, CompiledModel)
        np.testing.assert_allclose(model.forward(x), compiled(x), atol=1e-6)


class TestDNBSerialization(unittest.TestCase):
    def test_save_load_dnb_roundtrip(self):
        import tempfile
        from pathlib import Path

        model = Sequential([
            Dense(in_features=6, out_features=12),
            LayerNorm(normalized_shape=12),
            ReLU(),
            Dense(in_features=12, out_features=3),
        ])
        x = np.random.randn(4, 6).astype(np.float32)
        orig_out = model.forward(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "test_model.dnb"
            saved_path = save_dnb(model, fpath)
            self.assertTrue(saved_path.exists())

            info = inspect_dnb(saved_path)
            self.assertEqual(info["model_type"], "Sequential")
            self.assertEqual(info["tensors_count"], 6)

            loaded = load_dnb(saved_path)
            loaded_out = loaded.forward(x)
            np.testing.assert_allclose(orig_out, loaded_out, atol=1e-6)

    def test_dnb_checksum_corruption_detection(self):
        import tempfile
        from pathlib import Path

        model = Sequential([Dense(in_features=4, out_features=2)])
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "corrupt_test.dnb"
            save_dnb(model, fpath)

            with open(fpath, "r+b") as f:
                f.seek(-4, 2)
                b = f.read(1)
                f.seek(-4, 2)
                f.write(bytes([(b[0] ^ 0xFF)]))

            with self.assertRaises(ValueError) as ctx:
                load_dnb(fpath)
            self.assertIn("Checksum", str(ctx.exception))


class TestLlamaLLM(unittest.TestCase):
    def test_llama_tokenizer_encode_decode(self):
        tok_path = Path("models/stories260K/tok512.bin")
        if not tok_path.exists():
            tok_path = Path.home() / ".cache" / "doraneural" / "models" / "stories260K" / "tok512.bin"
        if not tok_path.exists():
            self.skipTest("stories260K tokenizer not downloaded")

        tok = LlamaTokenizer(tok_path, vocab_size=512)
        text = "Once upon a time"
        tokens = tok.encode(text, bos=True)
        self.assertEqual(tokens[0], 1)
        decoded = tok.decode(tokens)
        self.assertIn("Once", decoded)
        self.assertIn("time", decoded)

    def test_llama_forward_and_cache(self):
        try:
            llm = load_pretrained_llm("stories260K")
        except Exception as e:
            self.skipTest(f"Skipping Llama test: {e}")

        self.assertEqual(llm.config.dim, 64)
        self.assertEqual(llm.config.n_layers, 5)
        self.assertEqual(llm.config.vocab_size, 512)

        logits = llm.forward(token=1, pos=0)
        self.assertEqual(logits.shape, (512,))
        self.assertFalse(np.isnan(logits).any())

        # Check KV cache was populated at pos=0
        self.assertGreater(np.max(np.abs(llm.key_cache[:, 0, :])), 0.0)

    def test_llama_generation_and_streaming(self):
        try:
            llm = load_pretrained_llm("stories260K")
        except Exception as e:
            self.skipTest(f"Skipping Llama test: {e}")

        prompt = "Once upon a time"
        # Non-streaming
        story = llm.generate(prompt=prompt, max_tokens=15, temperature=0.7, stream=False)
        self.assertTrue(story.startswith(prompt))
        self.assertGreater(len(story), len(prompt))

        # Streaming
        pieces = list(llm.generate(prompt=prompt, max_tokens=10, temperature=0.7, stream=True))
        self.assertGreater(len(pieces), 0)
        self.assertIsInstance(pieces[0], str)


if __name__ == "__main__":
    unittest.main()

