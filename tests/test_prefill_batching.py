"""Batched prefill parity and dispatch tests.

`llama_forward_chunk` (csrc/llm_engine.cpp) runs a whole prompt window in one pass
instead of one serialised llama_forward call per token, because single-token decode
is DRAM-bound on the weight matrices.

The invariant that matters is that the batched path is numerically indistinguishable
from the serialised one. AGENTS.md Rule 4 requires float32 forward passes to hold
parity with the NumPy reference at atol < 1e-4; here the stronger and more direct
check is chunked-vs-serial within the same engine, across GQA/MQA/MHA, both RoPE
layouts, and dimension counts that are not multiples of the SIMD or tile widths.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from doraneural.cpp_backend import CppLlamaEngine
from doraneural.llm import LlamaConfig

ATOL = 1e-4


def _weights(cfg: LlamaConfig, seed: int = 0):
    rng = np.random.default_rng(seed)
    dim, hidden, layers, vocab = cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.vocab_size
    kv_dim = dim * cfg.n_kv_heads // cfg.n_heads

    def rand(*shape):
        return (rng.standard_normal(shape).astype(np.float32) * 0.02).reshape(-1)

    return {
        "token_embedding_table": rand(vocab, dim),
        "rms_att_weight": np.ones((layers, dim), dtype=np.float32).reshape(-1),
        "wq": rand(layers, dim, dim),
        "wk": rand(layers, kv_dim, dim),
        "wv": rand(layers, kv_dim, dim),
        "wo": rand(layers, dim, dim),
        "rms_ffn_weight": np.ones((layers, dim), dtype=np.float32).reshape(-1),
        "w1": rand(layers, hidden, dim),
        "w2": rand(layers, dim, hidden),
        "w3": rand(layers, hidden, dim),
        "rms_final_weight": np.ones((dim,), dtype=np.float32).reshape(-1),
        "wcls": rand(vocab, dim),
        "shared_classifier": 0,
    }


def _engine(cfg, threads=1):
    engine = CppLlamaEngine(cfg, _weights(cfg))
    engine.set_threads(threads)
    return engine


# (name, config). Deliberately includes MQA, GQA, MHA, odd dims that are not
# multiples of the 16-wide SIMD tiles or the 16x64 GEMM tile, and both RoPE modes.
CASES = [
    ("gqa_interleaved", LlamaConfig(dim=384, hidden_dim=1024, n_layers=8, n_heads=8,
                                   n_kv_heads=4, vocab_size=1024, seq_len=512)),
    ("gqa_hf_split_half", LlamaConfig(dim=384, hidden_dim=1024, n_layers=8, n_heads=8,
                                     n_kv_heads=4, vocab_size=1024, seq_len=512,
                                     rope_type="hf")),
    ("mha", LlamaConfig(dim=768, hidden_dim=2048, n_layers=12, n_heads=12,
                        n_kv_heads=12, vocab_size=2048, seq_len=512)),
    ("mqa", LlamaConfig(dim=256, hidden_dim=700, n_layers=3, n_heads=8,
                        n_kv_heads=1, vocab_size=512, seq_len=256)),
    ("odd_dims", LlamaConfig(dim=130, hidden_dim=333, n_layers=2, n_heads=5,
                            n_kv_heads=5, vocab_size=300, seq_len=128)),
    ("single_layer", LlamaConfig(dim=128, hidden_dim=256, n_layers=1, n_heads=4,
                                n_kv_heads=4, vocab_size=64, seq_len=32)),
]


@pytest.mark.parametrize("name,cfg", CASES, ids=[c[0] for c in CASES])
def test_batched_prefill_matches_serialised_forward(name, cfg):
    toks = [7, 11, 42, 99, 5, 3, 1, 1, 17, 13]
    toks = [t % cfg.vocab_size for t in toks]
    engine = _engine(cfg)

    engine.reset_cache()
    serial = None
    for pos, tok in enumerate(toks):
        serial = engine.forward(tok, pos)

    engine.reset_cache()
    batched = engine.forward_chunk(toks, 0)

    assert batched is not None
    np.testing.assert_allclose(serial, batched, atol=ATOL, rtol=0)
    assert serial.argmax() == batched.argmax()


@pytest.mark.parametrize("name,cfg", CASES[:3], ids=[c[0] for c in CASES[:3]])
def test_decode_after_batched_prefill_matches_serialised_cache(name, cfg):
    """The KV cache written by the batched path must feed later decode identically."""
    toks = [7, 11, 42, 99, 5, 3, 1, 1, 17, 13]
    toks = [t % cfg.vocab_size for t in toks]
    nxt = (toks[-1] + 5) % cfg.vocab_size
    engine = _engine(cfg)

    engine.reset_cache()
    engine.forward_chunk(toks, 0)
    after_batched = engine.forward(nxt, len(toks))

    engine.reset_cache()
    for pos, tok in enumerate(toks):
        engine.forward(tok, pos)
    after_serial = engine.forward(nxt, len(toks))

    np.testing.assert_allclose(after_batched, after_serial, atol=ATOL, rtol=0)
    assert after_batched.argmax() == after_serial.argmax()


def test_batched_prefill_window_may_start_at_nonzero_position():
    cfg = LlamaConfig(dim=128, hidden_dim=256, n_layers=2, n_heads=4, n_kv_heads=2,
                      vocab_size=128, seq_len=256)
    prefix = [5, 6, 7, 8, 9]
    window = [1, 1, 2, 3, 4, 5]
    engine = _engine(cfg)

    engine.reset_cache()
    for pos, tok in enumerate(prefix):
        engine.forward(tok, pos)
    serial = None
    for offset, tok in enumerate(window):
        serial = engine.forward(tok, len(prefix) + offset)

    engine.reset_cache()
    for pos, tok in enumerate(prefix):
        engine.forward(tok, pos)
    batched = engine.forward_chunk(window, len(prefix))

    np.testing.assert_allclose(serial, batched, atol=ATOL, rtol=0)


@pytest.mark.parametrize("chunk_len", [1, 2, 3, 7, 8, 16, 64, 65, 128])
def test_prefill_chunking_is_invariant_to_split_point(chunk_len):
    """Splitting a prompt into different chunk sizes must not change the result.

    This guards the scratch-arena reuse logic in ensure_prefill_capacity().
    """
    cfg = LlamaConfig(dim=128, hidden_dim=320, n_layers=3, n_heads=4, n_kv_heads=4,
                      vocab_size=256, seq_len=512)
    toks = [(i * 37 + 11) % cfg.vocab_size for i in range(70)]
    engine = _engine(cfg)

    engine.reset_cache()
    reference = None
    for pos, tok in enumerate(toks):
        reference = engine.forward(tok, pos)

    engine.reset_cache()
    last = None
    pos = 0
    while pos < len(toks):
        piece = toks[pos:pos + chunk_len]
        last = engine.forward_chunk(piece, pos)
        pos += len(piece)

    np.testing.assert_allclose(reference, last, atol=ATOL, rtol=0)


def test_prefill_rejects_out_of_range_windows():
    cfg = LlamaConfig(dim=64, hidden_dim=128, n_layers=1, n_heads=4, n_kv_heads=4,
                      vocab_size=64, seq_len=16)
    engine = _engine(cfg)

    with pytest.raises(ValueError):
        engine.forward_chunk([1, 2, 3], 15)          # window runs past seq_len
    with pytest.raises(ValueError):
        engine.forward_chunk([1, 9999], 0)          # token outside vocabulary
    with pytest.raises(ValueError):
        engine.forward_chunk([1, 2], -1)            # negative start position
    assert engine.forward_chunk([], 0) is None      # empty window is a no-op


def test_prefill_fills_kv_cache_for_every_position():
    """Each prompt position must be independently readable from the cache."""
    cfg = LlamaConfig(dim=64, hidden_dim=128, n_layers=2, n_heads=4, n_kv_heads=2,
                      vocab_size=128, seq_len=128)
    toks = [3, 9, 27, 81, 5]
    engine = _engine(cfg)

    # Batch-prefill only the first 3, then continue serially: if positions 1 and 2
    # were left unwritten the continuation would read stale/zeros.
    engine.reset_cache()
    engine.forward_chunk(toks[:3], 0)
    continued = [engine.forward(tok, 3 + i) for i, tok in enumerate(toks[3:])]

    engine.reset_cache()
    for pos, tok in enumerate(toks):
        engine.forward(tok, pos)
    serial_tail = [engine.forward(tok, 3 + i) for i, tok in enumerate(toks[3:])]

    for got, want in zip(continued, serial_tail):
        np.testing.assert_allclose(got, want, atol=ATOL, rtol=0)


def test_greedy_generation_is_stable_across_repeated_runs():
    """Greedy decode must stay deterministic with batched prefill in the loop."""
    from zexo.model import load_zexo

    zexo = load_zexo(tier="micro-byte", from_scratch=True)
    prompt = "User: what is the sea\nZexo:"
    outputs = [zexo.generate(prompt, max_tokens=20, temperature=0.0) for _ in range(3)]
    assert len(set(outputs)) == 1