"""Tests for offline, validation-aware preference refinement."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doraneural.transformer import TransformerDecoderLM
from zexo.preferences import (
    PreferenceExample,
    PreferenceTrainer,
    format_preference_prompt,
    read_preferences,
)


class TinyTokenizer:
    """Deterministic character tokenizer for small CPU-only tests."""

    def encode(self, text, bos=True):
        tokens = [1] if bos else []
        tokens.extend(3 + (ord(char) % 13) for char in text)
        return tokens


def _tiny_model(seed=12):
    np.random.seed(seed)
    return TransformerDecoderLM(
        dim=8,
        hidden_dim=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        vocab_size=16,
        seq_len=16,
    )


def test_read_preferences_validates_and_deduplicates_ids(tmp_path):
    path = tmp_path / "prefs.jsonl"
    path.write_text(
        json.dumps({"id": "p1", "prompt": "Say hello", "chosen": "Hello!", "rejected": "Goodbye."}) + "\n",
        encoding="utf-8",
    )
    items = read_preferences(path)
    assert len(items) == 1
    assert items[0] == PreferenceExample("Say hello", "Hello!", "Goodbye.", "p1")

    path.write_text(
        '{"id":"same","prompt":"one","chosen":"yes","rejected":"no"}\n'
        '{"id":"same","prompt":"two","chosen":"yes","rejected":"no"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate preference id"):
        read_preferences(path)


def test_read_preferences_rejects_no_signal_and_malformed_records(tmp_path):
    path = tmp_path / "prefs.jsonl"
    path.write_text('{"prompt":"x","chosen":"same","rejected":"same"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="identical"):
        read_preferences(path)

    path.write_text('{broken json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        read_preferences(path)


def test_plain_and_formatted_prompts_use_chat_prefix():
    assert format_preference_prompt("Explain gravity", "Be concise") == (
        "System: Be concise\nUser: Explain gravity\nZexo:"
    )
    assert format_preference_prompt("User: Explain gravity", "Be concise") == (
        "System: Be concise\nUser: Explain gravity\nZexo:"
    )
    assert format_preference_prompt("User: hi\nZexo:", "ignored") == "User: hi\nZexo:"


def test_dpo_updates_policy_preserves_reference_and_restores_best_eval():
    policy = _tiny_model(seed=31)
    reference = copy.deepcopy(policy)
    reference_before = [p.data.copy() for p in reference.parameters()]
    example = PreferenceExample("Pick the greeting", "A", "B", "greeting-1")

    trainer = PreferenceTrainer(
        policy_model=policy,
        reference_model=reference,
        tokenizer=TinyTokenizer(),
        system_prompt="Be clear.",
        beta=0.2,
        lr=0.01,
        weight_decay=0.0,
        grad_clip=1.0,
        max_seq_len=12,
    )
    history = trainer.fit([example], eval_examples=[example], epochs=8, shuffle=False, verbose=0)

    assert len(history["loss"]) == 8
    assert len(history["eval_accuracy"]) == 8
    assert history["eval_accuracy"][-1] == 1.0
    assert history["best_epoch"] >= 1
    assert history["best_eval_loss"] < 0.6932
    for before, after in zip(reference_before, reference.parameters()):
        np.testing.assert_array_equal(before, after.data)


def test_dpo_lora_only_changes_adapter_weights():
    policy = _tiny_model(seed=73)
    reference = copy.deepcopy(policy)
    base_parameters = policy.parameters()
    base_weights = [p.data.copy() for p in base_parameters]
    policy.enable_lora(rank=2, alpha=4.0, target_modules=("q", "v"))
    example = PreferenceExample("Choose one", "A", "B", "lora-1")
    trainer = PreferenceTrainer(
        policy_model=policy,
        reference_model=reference,
        tokenizer=TinyTokenizer(),
        system_prompt="",
        beta=0.1,
        lr=0.02,
        weight_decay=0.0,
        max_seq_len=12,
    )
    trainer.fit([example], epochs=2, shuffle=False, verbose=0)

    for before, after in zip(base_weights, base_parameters):
        np.testing.assert_array_equal(before, after.data)
    assert any(np.any(adapter.B.data != 0.0) for layer in policy.layers for adapter in (layer.lora_q, layer.lora_v))


def test_dpo_rejects_incompatible_reference_and_invalid_hyperparameters():
    policy = _tiny_model(seed=1)
    incompatible = TransformerDecoderLM(
        dim=8, hidden_dim=16, n_layers=1, n_heads=2, n_kv_heads=1,
        vocab_size=17, seq_len=16,
    )
    with pytest.raises(ValueError, match="identical architectures"):
        PreferenceTrainer(policy, incompatible, TinyTokenizer(), "", beta=0.1, lr=1e-4)
    with pytest.raises(ValueError, match="independent model copies"):
        PreferenceTrainer(policy, policy, TinyTokenizer(), "", beta=0.1, lr=1e-4)
    with pytest.raises(ValueError, match="beta must be finite and positive"):
        PreferenceTrainer(policy, copy.deepcopy(policy), TinyTokenizer(), "", beta=0.0, lr=1e-4)
    with pytest.raises(ValueError, match="lr must be finite and positive"):
        PreferenceTrainer(policy, copy.deepcopy(policy), TinyTokenizer(), "", beta=0.1, lr=float("nan"))
