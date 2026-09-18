"""Focused regression tests for the CPU full-transformer training path."""

import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from doraneural.autograd import Tensor
from doraneural.llm import LlamaLLM
from doraneural.transformer import TensorAdamW, TransformerDecoderLM


REPO_ROOT = Path(__file__).resolve().parents[1]


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
