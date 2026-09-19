"""Unit tests for Jev Non-Autoregressive System-1 Decision Architecture."""

import numpy as np
import pytest

from doraneural.jev import JevDecisionModel, rlcd_loss
from doraneural.transformer import TensorAdamW


def test_jev_initialization_and_schemas():
    model = JevDecisionModel(vocab_size=256, dim=32, hidden_dim=64, n_layers=1)
    model.add_choice_head("route", options=["a", "b", "c"])
    model.add_score_head("risk", min_val=0.0, max_val=10.0)
    model.add_boolean_head("approve")

    assert "route" in model.choice_heads
    assert "risk" in model.score_heads
    assert "approve" in model.bool_heads


def test_jev_single_pass_decide():
    model = JevDecisionModel(vocab_size=256, dim=32, hidden_dim=64, n_layers=1)
    model.add_choice_head("route", options=["alpha", "beta", "gamma"])
    model.add_score_head("confidence_score", min_val=0.0, max_val=1.0)
    model.add_boolean_head("gate")

    # Test Choice decision
    dec_choice = model.decide("Route to payment gateway", "route")
    assert dec_choice.action in ["alpha", "beta", "gamma"]
    assert 0.0 <= dec_choice.confidence <= 1.0
    assert len(dec_choice.scores) == 3
    assert dec_choice.latency_ms >= 0.0

    # Test Score decision
    dec_score = model.decide("Assess priority", "confidence_score")
    assert 0.0 <= dec_score.action <= 1.0

    # Test Boolean decision
    dec_bool = model.decide("Allow transfer", "gate")
    assert isinstance(dec_bool.action, bool)
    assert 0.0 <= dec_bool.confidence <= 1.0


def test_rlcd_loss_and_training_step():
    model = JevDecisionModel(vocab_size=256, dim=32, hidden_dim=64, n_layers=1)
    model.add_choice_head("cat", options=["low", "high"])

    opt = TensorAdamW(model.parameters(), lr=0.01)
    tokens = list(b"urgent alert")

    # Forward
    h = model.encode(tokens)
    _, probs = model.choice_heads["cat"].forward(h)
    l_init = rlcd_loss(probs, target_idx=1)
    l_init.backward()
    opt.step()

    assert float(l_init.data.item()) > 0.0


def test_jev_save_and_load(tmp_path):
    model = JevDecisionModel(vocab_size=256, dim=32, hidden_dim=64, n_layers=1)
    model.add_choice_head("cat", options=["low", "high"])
    model.add_score_head("score", min_val=0.0, max_val=5.0)

    save_path = tmp_path / "test_jev_model"
    model.save(save_path)
    assert (tmp_path / "test_jev_model.npz").exists()
    assert (tmp_path / "test_jev_model.json").exists()

    loaded = JevDecisionModel.load(save_path)
    assert "cat" in loaded.choice_heads
    assert "score" in loaded.score_heads
    assert np.allclose(model.embed.data, loaded.embed.data)
