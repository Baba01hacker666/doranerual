"""Tests for the compact DeepSeek-Flash-inspired architecture reference."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doraneural.llm import ByteTokenizer
from zexo.flash_lite import DeepSeekFlashLiteLM, FlashLiteBlock, FlashLiteConfig
from zexo.preferences import PreferenceExample, PreferenceTrainer


def _config(**overrides):
    values = {
        "dim": 16,
        "hidden_dim": 32,
        "n_layers": 6,
        "n_heads": 4,
        "n_kv_heads": 2,
        "seq_len": 16,
        "ced_split": 3,
        "local_only_encoder_layers": 2,
        "sliding_window": 4,
        "global_top_k": 3,
        "n_experts": 3,
        "experts_per_token": 2,
    }
    values.update(overrides)
    return FlashLiteConfig(**values)


def test_flash_lite_exposes_paper_inspired_ced_swa_and_csa2_modes():
    model = DeepSeekFlashLiteLM(_config())
    assert model.attention_modes == ["swa-only", "swa-only", "full", "full", "reindex", "reuse"]
    assert model.layers[3].wk_global is not None
    assert model.layers[4].wk_global is None
    assert model.layers[5].wk_global is None

    tokens = [256, 10, 11, 12, 13, 14]
    logits = model(tokens)
    assert logits.shape == (len(tokens), model.config.vocab_size)
    assert np.isfinite(logits.data).all()
    assert np.isfinite(model.last_aux_loss.item())


def test_selected_sparse_attention_matches_dense_when_all_keys_are_selected():
    np.random.seed(8)
    block = FlashLiteBlock(_config(global_top_k=16), layer_index=3)
    q = np.random.randn(4, 5, 4).astype(np.float32)
    k = np.random.randn(4, 5, 4).astype(np.float32)
    v = np.random.randn(4, 5, 4).astype(np.float32)
    rows = np.arange(5)[:, None]
    cols = np.arange(5)[None, :]
    mask = np.where(cols <= rows, 0.0, -np.inf).astype(np.float32)
    from doraneural.autograd import Tensor

    tq, tk, tv, tmask = (Tensor(x) for x in (q, k, v, mask))
    dense = block._attend(tq, tk, tv, tmask)
    indices = np.broadcast_to(np.arange(5, dtype=np.int64), (4, 5, 5)).copy()
    sparse = block._attend(tq, tk, tv, tmask, indices)
    np.testing.assert_allclose(dense.data, sparse.data, atol=1e-6, rtol=1e-6)


def test_flash_lite_is_causal_and_full_backprop_reaches_router_and_shared_kv():
    np.random.seed(14)
    model = DeepSeekFlashLiteLM(_config())
    logits_a = model([256, 7, 8, 9, 10, 11])
    logits_b = model([256, 7, 8, 20, 21, 22])
    np.testing.assert_allclose(logits_a.data[:3], logits_b.data[:3], atol=1e-6, rtol=1e-6)

    loss = model.loss([256, 7, 8, 9, 10], [7, 8, 9, 10, 11])
    loss.backward()
    assert model.layers[3].wq.grad is not None
    assert model.layers[3].wk_global.grad is not None
    assert model.layers[4].moe.router.grad is not None
    assert model.layers[5].moe.shared_expert.w1.grad is not None


def test_flash_lite_training_checkpoint_roundtrip_and_generation(tmp_path):
    np.random.seed(22)
    model = DeepSeekFlashLiteLM(_config())
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode("tiny language model", bos=True) + [tokenizer.EOS_ID]
    before = model.token_embedding.data.copy()
    history = model.fit_tokens(tokens, epochs=2, seq_len=8, max_steps=1, verbose=0)
    assert history["windows_per_epoch"] == 1
    assert history["total_steps"] == 2
    assert len(history["loss"]) == 2
    assert not np.array_equal(before, model.token_embedding.data)
    assert not model.last_lm_loss.requires_grad
    assert not model.last_aux_loss.requires_grad
    assert all(not layer.moe.last_aux_loss.requires_grad for layer in model.layers)

    checkpoint = model.save(tmp_path / "flash_lite.npz")
    loaded = DeepSeekFlashLiteLM.load(checkpoint)
    np.testing.assert_allclose(model(tokens[:8]).data, loaded(tokens[:8]).data, atol=1e-6, rtol=1e-6)
    assert isinstance(loaded.generate("hello", max_tokens=2, temperature=0.0), str)


def test_flash_lite_can_be_refined_with_the_offline_preference_trainer():
    np.random.seed(31)
    policy = DeepSeekFlashLiteLM(_config(seq_len=12, sliding_window=4))
    reference = copy.deepcopy(policy)
    ref_weights = [p.data.copy() for p in reference.parameters()]
    trainer = PreferenceTrainer(
        policy_model=policy,
        reference_model=reference,
        tokenizer=ByteTokenizer(),
        system_prompt="",
        beta=0.1,
        lr=1e-4,
        max_seq_len=10,
    )
    example = PreferenceExample("Say hi", "Hello.", "Go away.", "flash-pref-1")
    history = trainer.fit([example], eval_examples=[example], epochs=2, shuffle=False, verbose=0)
    assert history["best_epoch"] in (1, 2)
    assert all(not layer.moe.last_aux_loss.requires_grad for layer in policy.layers)
    assert all(p.grad is None for p in reference.parameters())
    for before, after in zip(ref_weights, reference.parameters()):
        np.testing.assert_array_equal(before, after.data)


def test_flash_lite_training_and_dpo_clis_save_complete_artifacts(tmp_path):
    import json

    from zexo.train_flash import main as train_flash_main
    from zexo.refine_flash import main as refine_flash_main

    train_data = tmp_path / "train.txt"
    eval_data = tmp_path / "eval.txt"
    train_data.write_text("A small corpus for a byte-level language model. " * 2, encoding="utf-8")
    eval_data.write_text("A separate held-out corpus for evaluation. " * 2, encoding="utf-8")
    base_checkpoint = tmp_path / "flash_base.npz"
    assert train_flash_main([
        "--data", str(train_data), "--eval-data", str(eval_data),
        "--output", str(base_checkpoint), "--epochs", "1", "--seq-len", "8",
        "--max-steps", "1", "--max-eval-steps", "1", "--context", "16",
        "--dim", "16", "--hidden-dim", "32", "--layers", "4",
        "--heads", "4", "--kv-heads", "2", "--window", "4",
        "--global-top-k", "3", "--local-only-encoder-layers", "1", "--quiet",
    ]) == 0
    assert base_checkpoint.is_file()
    assert base_checkpoint.with_suffix(".train.json").is_file()

    train_preferences = tmp_path / "preferences_train.jsonl"
    eval_preferences = tmp_path / "preferences_eval.jsonl"
    train_preferences.write_text(
        '{"id":"train-1","prompt":"Say hello","chosen":"Hello there.","rejected":"Go away."}\n',
        encoding="utf-8",
    )
    eval_preferences.write_text(
        '{"id":"eval-1","prompt":"Greet me","chosen":"Good morning.","rejected":"No."}\n',
        encoding="utf-8",
    )
    refined = tmp_path / "flash_refined.npz"
    assert refine_flash_main([
        "--checkpoint", str(base_checkpoint),
        "--preferences", str(train_preferences),
        "--eval-preferences", str(eval_preferences),
        "--output", str(refined), "--epochs", "1", "--lr", "1e-5",
        "--max-seq-len", "12",
    ]) == 0
    assert refined.is_file()
    metadata_path = refined.with_suffix(".refine.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["method"] == "DPO"
    assert metadata["parent_sha256"]
    assert DeepSeekFlashLiteLM.load(refined).config == DeepSeekFlashLiteLM.load(base_checkpoint).config


def test_flash_lite_config_rejects_invalid_architecture():
    with pytest.raises(ValueError, match="ced_split"):
        _config(ced_split=6)
    with pytest.raises(ValueError, match="n_heads must be divisible"):
        _config(n_kv_heads=3)
    with pytest.raises(ValueError, match="router_aux_loss_coef must be finite"):
        _config(router_aux_loss_coef=float("nan"))
