"""Sequence Modeling & Transformer Block Example in pure NumPy.

Demonstrates:
1. Recurrent sequence modeling with LSTM and GRU layers.
2. Self-Attention and TransformerBlock encoder modeling with PositionalEncoding.
3. Learning rate scheduling with CosineAnnealingLR and gradient clipping.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import doraneural as dn


def generate_synthetic_sequences(n_samples: int = 120, seq_len: int = 8, d_in: int = 4):
    """Generates synthetic sequence classification task (detect positive drift)."""
    dn.set_seed(42)
    X = np.random.randn(n_samples, seq_len, d_in).astype(np.float32)
    # Target: 1 if cumulative sum along time axis is positive, 0 otherwise
    drift = np.sum(X[:, :, 0], axis=1)
    y = (drift > 0.0).astype(np.float32)[:, np.newaxis]
    return X, y


def main():
    print("=" * 60)
    print("doraneural: Pure NumPy Sequence & Transformer Demonstration")
    print("=" * 60)

    X, y = generate_synthetic_sequences(n_samples=160, seq_len=10, d_in=8)
    X_train, X_test, y_train, y_test = dn.train_test_split(X, y, test_size=0.25, seed=42)

    print(f"Dataset shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"               X_test={X_test.shape}, y_test={y_test.shape}\n")

    # -------------------------------------------------------------
    # 1. Recurrent Model: LSTM with Gradient Clipping
    # -------------------------------------------------------------
    print("------------------------------------------------------------")
    print("1. Training LSTM Sequence Classifier (with Gradient Clipping)")
    print("------------------------------------------------------------")

    lstm_model = dn.Sequential([
        dn.LSTM(in_features=8, hidden_units=16, return_sequences=False),
        dn.Dense(in_features=16, out_features=8),
        dn.ReLU(),
        dn.Dense(in_features=8, out_features=1),
        dn.Sigmoid(),
    ])

    lstm_opt = dn.Adam(lr=0.03, weight_decay=1e-4)
    lstm_model.compile(optimizer=lstm_opt, loss=dn.BinaryCrossEntropy(), metrics=[dn.Accuracy()])

    lstm_scheduler = dn.CosineAnnealingLR(lstm_opt, T_max=12, eta_min=0.005)

    hist_lstm = lstm_model.fit(
        X_train,
        y_train,
        epochs=12,
        batch_size=16,
        verbose=1,
        validation_data=(X_test, y_test),
        scheduler=lstm_scheduler,
        clip_norm=1.0,
    )

    eval_lstm = lstm_model.evaluate(X_test, y_test)
    print(f"LSTM Test Accuracy: {eval_lstm.accuracy * 100:.1f}%\n")

    # -------------------------------------------------------------
    # 2. Transformer Model: PositionalEncoding + Multi-Head Attention Block
    # -------------------------------------------------------------
    print("------------------------------------------------------------")
    print("2. Training Pure NumPy Transformer Block (Pre-LN Multi-Head Attn)")
    print("------------------------------------------------------------")

    d_model = 16
    tf_model = dn.Sequential([
        # Project input to d_model
        dn.Dense(in_features=8, out_features=d_model),
        # Add sinusoidal positional encodings
        dn.PositionalEncoding(d_model=d_model, max_len=50),
        # Complete Transformer Encoder Block
        dn.TransformerBlock(d_model=d_model, num_heads=4, d_ff=32, dropout=0.05),
        # Flatten sequence representation and classify
        dn.Flatten(),
        dn.Dense(in_features=10 * d_model, out_features=8),
        dn.ReLU(),
        dn.Dense(in_features=8, out_features=1),
        dn.Sigmoid(),
    ])

    tf_opt = dn.Adam(lr=0.02, weight_decay=1e-4)
    tf_model.compile(optimizer=tf_opt, loss=dn.BinaryCrossEntropy(), metrics=[dn.Accuracy()])

    tf_scheduler = dn.WarmupCosineLR(tf_opt, warmup_epochs=3, total_epochs=12, eta_min=0.002)

    hist_tf = tf_model.fit(
        X_train,
        y_train,
        epochs=12,
        batch_size=16,
        verbose=1,
        validation_data=(X_test, y_test),
        scheduler=tf_scheduler,
        clip_norm=1.0,
    )

    eval_tf = tf_model.evaluate(X_test, y_test)
    print(f"Transformer Test Accuracy: {eval_tf.accuracy * 100:.1f}%\n")

    # Display ASCII learning curve
    print("Transformer Training Loss Curve:")
    print(dn.plot_history(hist_tf, width=45, height=8))

    print("\n✅ Sequence & Transformer demonstration completed successfully!")


if __name__ == "__main__":
    main()
