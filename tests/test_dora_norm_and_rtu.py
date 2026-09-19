"""Unit tests for Dora-Norm and Dora-RTU architectures."""

import numpy as np
import pytest

from doraneural.dora_norm import DoraNormBlock, DoraNormDecoderLM
from doraneural.transformer import TensorAdamW
from research.rtu_sandbox.dora_rtu import DoraRTUConfig, DoraRTULanguageModel


def test_dora_norm_forward_and_loss():
    model = DoraNormDecoderLM(
        dim=32,
        hidden_dim=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        vocab_size=128,
        seq_len=16,
    )
    toks = [10, 20, 30, 40, 50]
    tgts = [20, 30, 40, 50, 60]

    logits = model.forward(toks)
    assert logits.shape == (5, 128)
    assert not np.isnan(logits.data).any()

    opt = TensorAdamW(model.parameters(), lr=1e-3)
    l_init = model.train_batch(toks, tgts, opt)
    assert l_init > 0.0
    assert not np.isnan(l_init)

    # Gradient step check
    for _ in range(3):
        l_step = model.train_batch(toks, tgts, opt)
    assert l_step < l_init


def test_dora_rtu_forward_and_train():
    cfg = DoraRTUConfig(dim=32, n_layers=2, vocab_size=128, lr=1e-2)
    model = DoraRTULanguageModel(cfg)

    logits, stop_prob, latent = model.forward_byte(65)
    assert logits.shape == (128,)
    assert 0.0 <= stop_prob <= 1.0
    assert latent.shape == (32,)
    assert not np.isnan(logits).any()

    # Train sequence
    b_data = b"Hello, world! This is a Dora-RTU test."
    res = model.train_sequence(b_data, reset_state=True)
    assert res["loss"] > 0.0
    assert res["ce_loss"] > 0.0
    assert not np.isnan(res["loss"])

    # Generation check
    gen = model.generate("Hello", max_bytes=10, temperature=0.5, reset_state=True)
    assert len(gen) >= 5
