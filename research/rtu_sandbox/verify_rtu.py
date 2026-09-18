"""Unit tests for the Recurrent Trace Unit (RTU) Language Model architecture."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rtu import RTUConfig, RTULanguageModel


def test_rtu_config_and_param_count():
    cfg = RTUConfig(dim=64, n_layers=2, vocab_size=256)
    model = RTULanguageModel(cfg)

    # Expected parameter count:
    # embed: 256 * 64 = 16,384
    # per layer: 64 (decay) + 64*64 (weight) + 64 (gamma) + 64 (beta) = 4,288 * 2 = 8,576
    # decoder: 64 * 256 = 16,384
    # stop head: 64 * 1 + 1 = 65
    # total = 16,384 + 8,576 + 16,384 + 65 = 41,409
    assert cfg.parameter_count == 41409
    assert model.embed.shape == (256, 64)
    assert len(model.weights) == 2
    assert model.decoder.shape == (64, 256)


def test_rtu_forward_and_state():
    cfg = RTUConfig(dim=32, n_layers=2)
    model = RTULanguageModel(cfg)

    model.reset_state()
    initial_state_layer0 = model.layer_states[0].state.copy()
    assert np.allclose(initial_state_layer0, 0.0)

    # Forward single byte
    logits, stop_prob, latent = model.forward_byte(ord("A"), update_state=True)
    assert logits.shape == (256,)
    assert 0.0 <= stop_prob <= 1.0
    assert latent.shape == (32,)
    assert not np.isnan(logits).any()

    # State should have updated
    after_state = model.layer_states[0].state.copy()
    assert not np.allclose(initial_state_layer0, after_state)

    # Reset state should restore zero state
    model.reset_state()
    assert np.allclose(model.layer_states[0].state, 0.0)


def test_rtu_online_train_step():
    cfg = RTUConfig(dim=32, n_layers=1, lr=0.01)
    model = RTULanguageModel(cfg)

    cur_b = ord("H")
    next_b = ord("i")

    # Step before training
    logits_before, _, _ = model.forward_byte(cur_b, update_state=False)
    prob_before = np.exp(logits_before - np.max(logits_before))
    prob_before /= np.sum(prob_before)

    # Train for a few online steps on the same transition
    for _ in range(25):
        _, _, losses = model.step_online_train(cur_b, next_b, is_end=False)
        assert not np.isnan(losses["total"])

    logits_after, _, _ = model.forward_byte(cur_b, update_state=False)
    prob_after = np.exp(logits_after - np.max(logits_after))
    prob_after /= np.sum(prob_after)

    # Target byte probability should increase
    assert prob_after[next_b] > prob_before[next_b]


def test_rtu_train_text_convergence():
    cfg = RTUConfig(dim=64, n_layers=2, lr=0.005)
    model = RTULanguageModel(cfg)

    text = "Hello world! Zexo is learning recurrent traces."
    history = model.train_text(text, epochs=4, chunk_size=32, verbose=0)

    assert len(history["loss"]) == 4
    # Final loss should be lower than initial loss
    assert history["loss"][-1] < history["loss"][0]


def test_rtu_generate():
    cfg = RTUConfig(dim=32, n_layers=1)
    model = RTULanguageModel(cfg)

    output = model.generate("User: Hello\nZexo:", max_bytes=20, temperature=0.5)
    assert isinstance(output, str)


def test_rtu_save_and_load():
    cfg = RTUConfig(dim=32, n_layers=2)
    model1 = RTULanguageModel(cfg)

    # Modify state through a forward pass
    model1.forward_byte(ord("X"), update_state=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "rtu_model.npz"
        model1.save(save_path)
        assert save_path.exists()

        model2 = RTULanguageModel(cfg)
        model2.load(save_path)

        assert np.allclose(model1.embed, model2.embed)
        assert np.allclose(model1.decoder, model2.decoder)
        assert np.allclose(model1.layer_states[0].state, model2.layer_states[0].state)
