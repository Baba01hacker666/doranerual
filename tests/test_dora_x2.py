"""Unit tests for Dora-X2 16-layer, 768-dim, 12-head MH-RTU Chat Architecture."""

import pytest
import numpy as np

from doraneural.dora_x2 import (
    DoraX2Config,
    MultiHeadRTU,
    DoraX2Block,
    DoraX2LM,
    DoraX2ChatSession,
)


def test_dora_x2_config_dimensions():
    cfg = DoraX2Config()
    assert cfg.n_layers == 16
    assert cfg.dim == 768
    assert cfg.n_heads == 12
    assert cfg.head_dim == 64
    assert cfg.hidden_dim == 2048
    assert cfg.vocab_size == 256
    # 16 layers, 768 dim, 12 heads produces > 150M parameters
    assert cfg.parameter_count > 150_000_000
    assert abs(cfg.layer_scale - 1.0 / np.sqrt(32)) < 1e-5


def test_multi_head_rtu_forward_and_state():
    # Test scaled down for quick unit testing
    cfg = DoraX2Config(dim=128, n_heads=4, n_layers=2)
    rtu = MultiHeadRTU(cfg)
    assert rtu.head_dim == 32

    x = np.random.randn(128).astype(np.float32)
    out, state = rtu.forward_step(x)
    assert out.shape == (128,)
    assert state.shape == (4, 32)
    assert not np.isnan(out).any()
    assert not np.isnan(state).any()

    # Sequence consistency
    x_seq = np.random.randn(5, 128).astype(np.float32)
    out_seq, final_st = rtu.forward_sequence(x_seq)
    assert out_seq.shape == (5, 128)
    assert final_st.shape == (4, 32)


def test_dora_x2_block_novel_neurons():
    cfg = DoraX2Config(dim=128, hidden_dim=256, n_heads=4, n_layers=2)
    block = DoraX2Block(cfg, layer_idx=0)

    x = np.random.randn(128).astype(np.float32)
    out, st = block.forward_step(x)
    assert out.shape == (128,)
    assert st.shape == (4, 32)
    assert not np.isnan(out).any()


def test_dora_x2_lm_step_and_generation():
    cfg = DoraX2Config(dim=64, hidden_dim=128, n_heads=2, n_layers=2, vocab_size=64)
    model = DoraX2LM(cfg)

    logits, states = model.forward_step(10)
    assert logits.shape == (64,)
    assert len(states) == 2
    assert not np.isnan(logits).any()

    # Generate small string
    text = model.generate("Hello", max_tokens=6, temperature=0.7)
    assert isinstance(text, str)


def test_dora_x2_chat_session():
    cfg = DoraX2Config(dim=64, hidden_dim=128, n_heads=2, n_layers=2, vocab_size=64)
    model = DoraX2LM(cfg)
    session = DoraX2ChatSession(model)

    response = session.chat("Hello!", max_tokens=8, stream=False)
    assert isinstance(response, str)
    assert session.total_turns == 1
    assert len(session.history) == 1
    assert session.history[0]["user"] == "Hello!"

    # Reset test
    session.reset()
    assert len(session.history) == 0


def test_dora_x2_save_and_load_roundtrip(tmp_path):
    cfg = DoraX2Config(dim=64, hidden_dim=128, n_heads=2, n_layers=2, vocab_size=64)
    model = DoraX2LM(cfg)

    save_path = tmp_path / "dora_x2_test"
    npz_path = model.save(save_path)
    assert npz_path.exists()
    assert save_path.with_suffix(".json").exists()

    loaded = DoraX2LM.load(save_path)
    assert loaded.config.dim == model.config.dim
    assert loaded.config.n_layers == model.config.n_layers
    assert loaded.config.n_heads == model.config.n_heads

    # Verify identical output on probe token
    l1, _ = model.forward_step(42)
    l2, _ = loaded.forward_step(42)
    assert np.allclose(l1, l2, atol=1e-5)
