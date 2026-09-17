"""Sequential container model and training orchestration.

Provides high-level Sequential model API with compile, fit, evaluate,
predict, and serialization capabilities.
"""

from collections import namedtuple
from typing import List, Optional, Union, Tuple, Dict, Any
import numpy as np

from neural_lib.base import Layer
from neural_lib.losses import Loss
from neural_lib.optimizers import Optimizer, SGD
from neural_lib.metrics import Metric, get_metric
from neural_lib.utils import batch_iterator


class History:
    """Stores epoch-by-epoch training and validation loss and metric values.

    Can be accessed as a dictionary: history['loss'] or via attribute history.history['loss'].
    """

    def __init__(self) -> None:
        self.history: Dict[str, List[float]] = {}

    def log(self, metrics: Dict[str, float]) -> None:
        """Append metric values for the completed epoch.

        Args:
            metrics (Dict[str, float]): Key-value pairs of metrics (e.g. {'loss': 0.25, 'accuracy': 0.94}).
        """
        for k, v in metrics.items():
            self.history.setdefault(k, []).append(float(v))

    def __getitem__(self, key: str) -> List[float]:
        return self.history[key]

    def __contains__(self, key: str) -> bool:
        return key in self.history

    def get(self, key: str, default: Any = None) -> Any:
        return self.history.get(key, default)

    def keys(self):
        return self.history.keys()

    def __repr__(self) -> str:
        summary_items = {k: f"[{len(v)} values, final: {v[-1]:.4f}]" if v else "[]" for k, v in self.history.items()}
        return f"History({summary_items})"


EvaluationResult = namedtuple("EvaluationResult", ["loss", "accuracy"])


class Sequential:
    """Linear stack of neural network layers.

    Example:
        >>> model = Sequential([
        ...     Dense(2, 16),
        ...     ReLU(),
        ...     Dense(16, 1),
        ...     Sigmoid(),
        ... ])
        >>> model.compile(loss=BinaryCrossEntropy(), optimizer=SGD(lr=0.1), metrics=["accuracy"])
        >>> history = model.fit(X_train, y_train, epochs=50, batch_size=32)
    """

    def __init__(self, layers: Optional[List[Layer]] = None) -> None:
        """Initialize the Sequential container with an optional list of layers.

        Args:
            layers (Optional[List[Layer]]): Ordered list of Layer instances.
        """
        self.layers: List[Layer] = []
        if layers is not None:
            for layer in layers:
                self.add(layer)

        self.loss: Optional[Loss] = None
        self.optimizer: Optional[Optimizer] = None
        self.metrics: List[Metric] = []
        self._is_compiled: bool = False

    def add(self, layer: Layer) -> "Sequential":
        """Append a layer to the network.

        Args:
            layer (Layer): Layer instance conforming to Layer interface.

        Returns:
            Sequential: Self for method chaining.
        """
        if not isinstance(layer, Layer):
            raise TypeError(f"Expected layer to inherit from Layer, got {type(layer)}")
        self.layers.append(layer)
        return self

    def compile(
        self,
        loss: Loss,
        optimizer: Optional[Optimizer] = None,
        metrics: Optional[List[Union[str, Metric]]] = None,
    ) -> None:
        """Configure model for training by specifying loss, optimizer, and metrics.

        Args:
            loss (Loss): Loss function instance (e.g. BinaryCrossEntropy).
            optimizer (Optional[Optimizer]): Optimizer instance (defaults to SGD(0.01)).
            metrics (Optional[List[Union[str, Metric]]]): Metric instances or names (e.g. ['accuracy']).
        """
        if not isinstance(loss, Loss):
            raise TypeError(f"loss must be an instance of Loss, got {type(loss)}")

        self.loss = loss
        self.optimizer = optimizer if optimizer is not None else SGD(lr=0.01)

        self.metrics = []
        if metrics is not None:
            for m in metrics:
                self.metrics.append(get_metric(m))

        self._is_compiled = True

    def train(self, mode: bool = True) -> "Sequential":
        """Set training mode across all layers (enables dropout, etc.)."""
        for layer in self.layers:
            layer.train(mode)
        return self

    def eval(self) -> "Sequential":
        """Set evaluation mode across all layers (disables dropout, etc.)."""
        for layer in self.layers:
            layer.eval()
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run input sequentially through all layers.

        Args:
            x (np.ndarray): Input data tensor with shape (batch_size, in_features).

        Returns:
            np.ndarray: Network output tensor.
        """
        current = x
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def backward(self, loss_grad: np.ndarray) -> np.ndarray:
        """Backpropagate loss gradient sequentially through all layers in reverse.

        Args:
            loss_grad (np.ndarray): Upstream gradient dL/d(output) from loss function.

        Returns:
            np.ndarray: Downstream gradient with respect to model input.
        """
        current_grad = loss_grad
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad)
        return current_grad

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
        shuffle: bool = True,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> History:
        """Train the model using mini-batch gradient descent.

        Args:
            X (np.ndarray): Training features, shape (N, features).
            y (np.ndarray): Training targets, shape (N, ...).
            epochs (int): Number of complete passes over the training dataset.
            batch_size (int): Number of samples per gradient update batch.
            verbose (int): Verbosity mode (0 = silent, 1 = print every epoch, >1 = progress step).
            shuffle (bool): Whether to shuffle training data before each epoch.
            validation_data (Optional[Tuple[np.ndarray, np.ndarray]]): Optional (X_val, y_val) tuple.

        Returns:
            History: Object containing recorded training metrics across epochs.
        """
        if not self._is_compiled or self.loss is None or self.optimizer is None:
            raise RuntimeError("Model must be compiled with loss and optimizer before calling fit().")

        X_train = np.asarray(X, dtype=np.float32)
        y_train = np.asarray(y)

        n_samples = len(X_train)
        if len(y_train) != n_samples:
            raise ValueError(
                f"Sample count mismatch: X has {n_samples} samples, y has {len(y_train)} samples."
            )

        history = History()

        for epoch in range(1, epochs + 1):
            # Set training mode for mini-batch updates
            self.train(True)
            for X_batch, y_batch in batch_iterator(X_train, y_train, batch_size=batch_size, shuffle=shuffle):
                # 1. Forward pass
                y_pred_batch = self.forward(X_batch)

                # 2. Compute loss and gradient
                self.loss.forward(y_pred_batch, y_batch)
                loss_grad = self.loss.backward(y_pred_batch, y_batch)

                # 3. Backward pass
                self.backward(loss_grad)

                # 4. Optimizer update step
                self.optimizer.step(self.layers)

            # End of epoch evaluation on full dataset (eval mode)
            self.eval()
            train_preds = self.forward(X_train)
            train_loss = self.loss.forward(train_preds, y_train)

            epoch_logs = {"loss": train_loss}
            for metric in self.metrics:
                score = metric(y_train, train_preds)
                epoch_logs[metric.name] = score

            if validation_data is not None:
                X_val, y_val = validation_data
                val_preds = self.forward(np.asarray(X_val, dtype=np.float32))
                val_loss = self.loss.forward(val_preds, y_val)
                epoch_logs["val_loss"] = val_loss
                for metric in self.metrics:
                    val_score = metric(y_val, val_preds)
                    epoch_logs[f"val_{metric.name}"] = val_score

            history.log(epoch_logs)

            # Verbose logging
            if verbose == 1 or (verbose > 1 and (epoch % verbose == 0 or epoch == epochs)):
                log_strs = [f"loss: {epoch_logs['loss']:.4f}"]
                for metric in self.metrics:
                    if metric.name in epoch_logs:
                        log_strs.append(f"{metric.name}: {epoch_logs[metric.name]:.4f}")
                if "val_loss" in epoch_logs:
                    log_strs.append(f"val_loss: {epoch_logs['val_loss']:.4f}")
                    for metric in self.metrics:
                        val_key = f"val_{metric.name}"
                        if val_key in epoch_logs:
                            log_strs.append(f"{val_key}: {epoch_logs[val_key]:.4f}")

                print(f"Epoch {epoch:3d}/{epochs} - " + " - ".join(log_strs))

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> EvaluationResult:
        """Evaluate the model on a test or validation dataset.

        Args:
            X (np.ndarray): Feature array, shape (N, features).
            y (np.ndarray): Target labels, shape (N, ...).

        Returns:
            EvaluationResult: Named tuple with fields (loss, accuracy). Can be unpacked as:
                loss, acc = model.evaluate(X_test, y_test)
        """
        if not self._is_compiled or self.loss is None:
            raise RuntimeError("Model must be compiled before calling evaluate().")

        self.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y)

        y_pred = self.forward(X_arr)
        loss_val = self.loss.forward(y_pred, y_arr)

        acc_metric = self.metrics[0] if self.metrics else get_metric("accuracy")
        acc_val = acc_metric(y_arr, y_pred)

        return EvaluationResult(loss=loss_val, accuracy=acc_val)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate raw output probabilities for input samples.

        Args:
            X (np.ndarray): Input feature array of shape (N, features).

        Returns:
            np.ndarray: Predicted probability distributions.
        """
        self.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        return self.forward(X_arr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate hard discrete class predictions.

        For binary classification (output shape (N, 1)), returns 0 or 1.
        For multi-class classification (output shape (N, C)), returns argmax class index.

        Args:
            X (np.ndarray): Input feature array of shape (N, features).

        Returns:
            np.ndarray: Discrete predicted class labels of shape (N, 1).
        """
        proba = self.predict_proba(X)
        if proba.shape[-1] == 1:
            # Binary classification threshold at 0.5
            return (proba >= 0.5).astype(np.int64)
        else:
            # Multi-class classification: pick highest probability index
            return np.argmax(proba, axis=-1, keepdims=True).astype(np.int64)

    def summary(self) -> None:
        """Print a concise summary table of the network layers and parameter count."""
        print("=" * 65)
        print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15}")
        print("=" * 65)
        total_params = 0
        for idx, layer in enumerate(self.layers):
            layer_name = f"{layer.__class__.__name__}_{idx + 1}"
            params = layer.get_params()
            count = sum(p.size for p in params.values() if p is not None)
            total_params += count

            out_shape = f"(None, {layer.out_features})" if hasattr(layer, "out_features") else "--"
            print(f"{layer_name:<25} {out_shape:<20} {count:<15,d}")

        print("=" * 65)
        print(f"Total params: {total_params:,d}")
        print("=" * 65)

    def save(self, filepath: Union[str, Any]) -> None:
        """Save model architecture and weights to disk."""
        from neural_lib.serialization import save_model

        save_model(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Any]) -> "Sequential":
        """Load model architecture and weights from disk."""
        from neural_lib.serialization import load_model

        return load_model(filepath)
