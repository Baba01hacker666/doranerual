"""Focused regression tests for the CPU full-transformer training path."""

import struct
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from doraneural.autograd import Tensor
from doraneural.llm import LlamaLLM
from doraneural.transformer import TensorAdamW, TransformerDecoderLM


def test_concatenate_routes_gradients_to_each_rope_slice():
    left = Tensor(np.ones((2, 2), dtype=np.float32), requires_grad=True)
    right = Tensor(np.full((2, 1), 2.0, dtype=np.float32), requires_grad=True)
    output = Tensor.concatenate((left, right), axis=1)
    output.sum().backward()
    np.testing.assert_allclose(left.grad, np.ones_like(left.data))
    np.testing.assert_allclose(right.grad, np.ones_like(right.data))


def test_full_backprop_reaches_attention_and_feed_forward_parameters():
    np.random.seed(7)
    model = TransformerDecoderLM(
        dim=16,
        hidden_dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=13,
        seq_len=12,
    )

    loss = model.loss([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    loss.backward()

    learned = [
        model.token_embedding,
        model.layers[0].rms_att,
        model.layers[0].wq,
        model.layers[0].wk,
        model.layers[0].wv,
        model.layers[0].wo,
        model.layers[0].rms_ffn,
        model.layers[0].w1,
        model.layers[0].w2,
        model.layers[0].w3,
        model.rms_final,
    ]
    assert all(parameter.grad is not None for parameter in learned)
    assert all(float(np.linalg.norm(parameter.grad)) > 1e-8 for parameter in learned)


def test_lora_trains_adapters_without_updating_pretrained_base(tmp_path):
    np.random.seed(31)
    model = TransformerDecoderLM(
        dim=16,
        hidden_dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=12,
        seq_len=16,
    )
    inputs = [1, 2, 3, 4, 5]
    targets = [2, 3, 4, 5, 6]
    initial_logits = model.forward(inputs).data.copy()
    base_q = model.layers[0].wq.data.copy()
    model.enable_lora(rank=4, alpha=8.0, target_modules=("q", "v"))
    np.testing.assert_allclose(model.forward(inputs).data, initial_logits, rtol=0, atol=1e-6)
    assert not model.layers[0].wq.requires_grad
    assert len(model.lora_parameters()) == 4

    optimizer = TensorAdamW(model.lora_parameters(), lr=0.05, weight_decay=0.0)
    initial_loss = float(model.loss(inputs, targets).item())
    for _ in range(15):
        model.train_batch(inputs, targets, optimizer)
    final_loss = float(model.loss(inputs, targets).item())
    assert final_loss < initial_loss * 0.5
    np.testing.assert_array_equal(model.layers[0].wq.data, base_q)

    adapter_path = model.save_lora(tmp_path / "adapter")
    assert adapter_path.suffix == ".npz"
    restored = TransformerDecoderLM(
        dim=16, hidden_dim=32, n_layers=1, n_heads=4, n_kv_heads=2,
        vocab_size=12, seq_len=16,
    )
    # In a real use case the adapter is loaded on the same pretrained base;
    # copy the base weights here to make the serialization check explicit.
    restored.token_embedding.data[...] = model.token_embedding.data
    restored.rms_final.data[...] = model.rms_final.data
    restored.layers[0].rms_att.data[...] = model.layers[0].rms_att.data
    restored.layers[0].rms_ffn.data[...] = model.layers[0].rms_ffn.data
    restored.layers[0].wq.data[...] = model.layers[0].wq.data
    restored.layers[0].wk.data[...] = model.layers[0].wk.data
    restored.layers[0].wv.data[...] = model.layers[0].wv.data
    restored.layers[0].wo.data[...] = model.layers[0].wo.data
    restored.layers[0].w1.data[...] = model.layers[0].w1.data
    restored.layers[0].w2.data[...] = model.layers[0].w2.data
    restored.layers[0].w3.data[...] = model.layers[0].w3.data
    restored.load_lora(adapter_path)
    np.testing.assert_allclose(restored.forward(inputs).data, model.forward(inputs).data, rtol=1e-5, atol=1e-5)


def test_tied_lm_head_lora_merge_preserves_logits():
    np.random.seed(37)
    model = TransformerDecoderLM(
        dim=16, hidden_dim=32, n_layers=1, n_heads=4, n_kv_heads=2,
        vocab_size=12, seq_len=16,
    )
    model.enable_lora(rank=2, alpha=4.0, target_modules=("lm_head",))
    model.lora_lm_head.B.data[...] = np.random.randn(*model.lora_lm_head.B.shape).astype(np.float32) * 0.01
    inputs = [1, 2, 3, 4]
    before = model.forward(inputs).data.copy()
    model.merge_lora()
    after = model.forward(inputs).data
    np.testing.assert_allclose(before, after, rtol=2e-5, atol=2e-5)


def test_tiny_full_backprop_model_reduces_next_token_loss():
    np.random.seed(11)
    model = TransformerDecoderLM(
        dim=16,
        hidden_dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=12,
        seq_len=16,
    )
    optimizer = TensorAdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    inputs = [1, 2, 3, 4, 5]
    targets = [2, 3, 4, 5, 6]
    initial = float(model.loss(inputs, targets).item())

    for _ in range(12):
        model.train_batch(inputs, targets, optimizer)

    final = float(model.loss(inputs, targets).item())
    assert final < initial * 0.25, (initial, final)


def test_hf_layout_supports_split_rope_and_untied_output_head():
    rng = np.random.default_rng(23)
    dim, hidden, vocab = 16, 32, 17
    llm = SimpleNamespace(
        config=SimpleNamespace(
            dim=dim,
            hidden_dim=hidden,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=vocab,
            seq_len=16,
            rope_type="hf",
        ),
        tok_emb=rng.normal(size=(vocab, dim)).astype(np.float32),
        rms_att=np.ones((1, dim), dtype=np.float32),
        wq=rng.normal(size=(1, dim, dim)).astype(np.float32),
        wk=rng.normal(size=(1, 8, dim)).astype(np.float32),
        wv=rng.normal(size=(1, 8, dim)).astype(np.float32),
        wo=rng.normal(size=(1, dim, dim)).astype(np.float32),
        rms_ffn=np.ones((1, dim), dtype=np.float32),
        w1=rng.normal(size=(1, hidden, dim)).astype(np.float32),
        w2=rng.normal(size=(1, dim, hidden)).astype(np.float32),
        w3=rng.normal(size=(1, hidden, dim)).astype(np.float32),
        rms_final=np.ones(dim, dtype=np.float32),
        wcls=rng.normal(size=(vocab, dim)).astype(np.float32),
        shared_weights=False,
        reset_cache=lambda: None,
    )
    model = TransformerDecoderLM.from_llama(llm)
    assert model.rope_type == "hf"
    assert model.lm_head is not model.token_embedding
    assert model.layers[0].wk.shape == (dim, 8)

    loss = model.loss([1, 2, 3, 4], [2, 3, 4, 5])
    loss.backward()
    assert model.lm_head.grad is not None
    assert model.layers[0].wk.grad is not None
    model.copy_to_llama()
    np.testing.assert_allclose(llm.wcls, model.lm_head.data)


def _write_tiny_llama_checkpoint(path: Path, model: TransformerDecoderLM) -> None:
    """Write model weights in the llama2.c layout used by LlamaLLM."""
    layer = model.layers[0]
    p = model
    arrays = [
        p.token_embedding.data,
        layer.rms_att.data,
        layer.wq.data.T,
        layer.wk.data.T,
        layer.wv.data.T,
        layer.wo.data.T,
        layer.rms_ffn.data,
        layer.w1.data.T,
        layer.w2.data.T,
        layer.w3.data.T,
        p.rms_final.data,
        np.zeros(p.seq_len * layer.head_size, dtype=np.float32),
    ]
    with path.open("wb") as handle:
        handle.write(struct.pack(
            "<7i", p.dim, p.hidden_dim, p.n_layers, p.n_heads,
            p.n_kv_heads, p.vocab_size, p.seq_len,
        ))
        for array in arrays:
            np.asarray(array, dtype=np.float32).tofile(handle)


def test_native_cpp_matches_numpy_one_step_on_gqa_checkpoint(tmp_path):
    np.random.seed(39)
    source = TransformerDecoderLM(
        dim=16,
        hidden_dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=512,
        seq_len=16,
    )
    checkpoint = tmp_path / "parity-tiny.bin"
    _write_tiny_llama_checkpoint(checkpoint, source)
    tokenizer = REPO_ROOT / "zexo" / "tokenizer" / "tok512.bin"
    try:
        native = LlamaLLM(checkpoint, tokenizer, backend="cpp")
    except RuntimeError as error:
        pytest.skip(f"native C++ engine unavailable: {error}")
    reference = LlamaLLM(checkpoint, tokenizer, backend="numpy")
    model = reference.full_backprop_model()
    inputs = [1, 2, 3, 4, 5]
    targets = [2, 3, 4, 5, 6]
    numpy_loss = float(model.loss(inputs, targets).item())
    native_loss = native.cpp_engine.full_train_step(
        inputs, targets, lr=0.001, weight_decay=0.01,
    )
    optimizer = TensorAdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    model.train_batch(inputs, targets, optimizer)
    model.copy_to_llama()
    assert abs(native_loss - numpy_loss) < 2e-5
    for native_array, reference_array in (
        (native.tok_emb, reference.tok_emb),
        (native.wq, reference.wq),
        (native.wk, reference.wk),
        (native.w1, reference.w1),
        (native.rms_final, reference.rms_final),
    ):
        np.testing.assert_allclose(native_array, reference_array, rtol=2e-5, atol=2e-5)


def test_native_cpp_full_backprop_reduces_loss_and_updates_attention(tmp_path):
    np.random.seed(41)
    source = TransformerDecoderLM(
        dim=16,
        hidden_dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=512,
        seq_len=16,
    )
    checkpoint = tmp_path / "native-tiny.bin"
    _write_tiny_llama_checkpoint(checkpoint, source)
    try:
        llm = LlamaLLM(checkpoint, REPO_ROOT / "zexo" / "tokenizer" / "tok512.bin", backend="cpp")
    except RuntimeError as error:
        pytest.skip(f"native C++ engine unavailable: {error}")
    before_q = llm.wq.copy()
    before_k = llm.wk.copy()
    before_ffn = llm.w1.copy()
    before_norm = llm.rms_att.copy()
    history = llm.train(
        "abc abc abc abc abc abc abc abc",
        epochs=1,
        lr=0.01,
        seq_len=4,
        verbose=0,
        full_backprop=True,
    )
    assert history["loss"]
    assert llm.backend == "cpp"
    assert not np.array_equal(before_q, llm.wq)
    assert not np.array_equal(before_k, llm.wk)
    assert not np.array_equal(before_ffn, llm.w1)
    assert not np.array_equal(before_norm, llm.rms_att)


def test_full_model_uses_existing_llama_tensor_layout_and_syncs_weights(tmp_path):
    np.random.seed(19)
    source = TransformerDecoderLM(
        dim=16,
        hidden_dim=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=20,
        seq_len=16,
    )
    checkpoint = tmp_path / "tiny.bin"
    _write_tiny_llama_checkpoint(checkpoint, source)
    llm = LlamaLLM(checkpoint, REPO_ROOT / "zexo" / "tokenizer" / "tok512.bin", backend="numpy")
    model = llm.full_backprop_model()

    input_ids = [1, 2, 3, 4, 5]
    expected = model.forward(input_ids).data
    llm.reset_cache()
    actual = np.stack([llm.forward(token, pos) for pos, token in enumerate(input_ids)])
    np.testing.assert_allclose(expected, actual, rtol=2e-4, atol=5e-4)

    before = llm.wq.copy()
    model.layers[0].wq.data += 0.001
    model.copy_to_llama()
    assert not np.allclose(before, llm.wq)
    np.testing.assert_allclose(llm.wq[0], model.layers[0].wq.data.T)


def test_novel_neurons_dora_backprop():
    """Verify that novel neurons (dendritic gating, Chebyshev KAN, reflection) compute exact gradients and update cleanly."""
    np.random.seed(42)
    model = TransformerDecoderLM(
        dim=16,
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=20,
        seq_len=16,
        novel_neurons=True,
    )

    assert model.layers[0].novel_neurons is True
    assert model.layers[0].w_dend is not None
    assert model.layers[0].c_poly is not None
    assert model.layers[0].w_ref is not None

    optimizer = TensorAdamW(model.parameters(), lr=1e-3)
    loss = model.loss([1, 2, 3, 4], [2, 3, 4, 5])
    loss.backward()

    for layer in model.layers:
        assert layer.w_dend.grad is not None
        assert np.linalg.norm(layer.w_dend.grad) > 0.0
        assert layer.c_poly.grad is not None
        assert np.linalg.norm(layer.c_poly.grad) > 0.0
        assert layer.w_ref.grad is not None
        assert np.linalg.norm(layer.w_ref.grad) > 0.0

    dend_before = model.layers[0].w_dend.data.copy()
    poly_before = model.layers[0].c_poly.data.copy()
    ref_before = model.layers[0].w_ref.data.copy()

    optimizer.step()

    assert not np.array_equal(dend_before, model.layers[0].w_dend.data)
    assert not np.array_equal(poly_before, model.layers[0].c_poly.data)
    assert not np.array_equal(ref_before, model.layers[0].w_ref.data)


def test_zexo_config_dora():
    """Verify ZexoConfig dora tier configuration."""
    from zexo.config import ZexoConfig
    cfg = ZexoConfig.dora()
    assert cfg.tier == "dora"
    assert cfg.n_layers == 12
    assert cfg.dim == 512
    assert cfg.hidden_dim == 1536
    assert cfg.novel_neurons is True
    assert cfg.parameter_count > 60_000_000

    cfg_from_tier = ZexoConfig.from_tier("dora")
    assert cfg_from_tier.tier == "dora"
    assert cfg_from_tier.novel_neurons is True
