"""Regression tests for faster model loading and tokenizer paths."""

import json
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from doraneural.llm import HFTokenizer, LlamaConfig, LlamaLLM, LlamaTokenizer, load_safetensors


def test_llama_tokenizer_heap_merges_and_cache_are_stable():
    tokenizer = LlamaTokenizer(REPO_ROOT / "zexo" / "tokenizer" / "tok512.bin", vocab_size=512)
    text = "Once upon a time, a small model learns quickly."
    first = tokenizer.encode(text, bos=True)
    second = tokenizer.encode(text, bos=True)
    assert first == second
    assert first[0] == 1

    # Returning a fresh list prevents callers from corrupting the hot cache.
    first.append(999999)
    assert tokenizer.encode(text, bos=True) == second
    assert tokenizer.encode_batch([text, text], bos=False)[0] == tokenizer.encode(text, bos=False)


def test_hf_tokenizer_heap_merges_and_round_trips_batch():
    tokenizer = HFTokenizer(REPO_ROOT / "zexo" / "tokenizer" / "tokenizer.json")
    texts = ["The cat sat on the mat", "hello world", "नमस्ते"]
    encoded = tokenizer.encode_batch(texts, bos=True)
    assert all(tokens and tokens[0] == 1 for tokens in encoded)
    assert tokenizer.decode_batch(encoded) == texts


def _write_safetensor(path, name, array):
    payload = np.asarray(array, dtype=np.float32).tobytes()
    header = json.dumps({name: {
        "dtype": "F32",
        "shape": list(np.asarray(array).shape),
        "data_offsets": [0, len(payload)],
    }}).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def test_safetensors_shard_index_is_supported(tmp_path):
    _write_safetensor(tmp_path / "model-00001-of-00002.safetensors", "first", [1.0, 2.0])
    _write_safetensor(tmp_path / "model-00002-of-00002.safetensors", "second", [3.0])
    index = {
        "weight_map": {
            "first": "model-00001-of-00002.safetensors",
            "second": "model-00002-of-00002.safetensors",
        }
    }
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    loaded = load_safetensors(index_path)
    np.testing.assert_allclose(loaded["first"], [1.0, 2.0])
    np.testing.assert_allclose(loaded["second"], [3.0])


def test_safetensors_bfloat16_is_decoded_as_bfloat16(tmp_path):
    # 1.0 and -2.5 represented by the high 16 bits of float32 words.
    words = np.array([0x3F800000, 0xC0200000], dtype=np.uint32)
    payload = (words >> 16).astype(np.uint16).tobytes()
    header = json.dumps({"values": {
        "dtype": "BF16",
        "shape": [2],
        "data_offsets": [0, len(payload)],
    }}).encode("utf-8")
    path = tmp_path / "bf16.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)

    loaded = load_safetensors(path)
    np.testing.assert_allclose(loaded["values"], [1.0, -2.5], rtol=0, atol=0)
    assert loaded["values"].dtype == np.float32


def test_cpp_fused_argmax_and_profiling():
    checkpoint = REPO_ROOT / "models" / "stories260K" / "stories260K.bin"
    tokenizer = REPO_ROOT / "models" / "stories260K" / "tok512.bin"
    if not checkpoint.exists() or not tokenizer.exists():
        pytest.skip("stories260K model files not present")
    llm = LlamaLLM(checkpoint, tokenizer, backend="cpp")
    if llm.cpp_engine is None:
        pytest.skip("C++ engine not available")

    # Test fused argmax matches reference forward + argmax
    argmax_tok = llm.cpp_engine.forward_argmax(1, 0)
    logits = llm.cpp_engine.forward(1, 0, copy_logits=True)
    assert argmax_tok == int(np.argmax(logits))

    # Test profiling hooks
    llm.cpp_engine.set_profile(True)
    llm.cpp_engine.reset_profile()
    gen = llm.generate("Once", max_tokens=10, temperature=0.0)
    assert isinstance(gen, str)
    prof = llm.cpp_engine.get_profile()
    assert prof["count"] >= 10
    assert prof["total_ms"] > 0.0
    assert prof["rmsnorm_ms"] > 0.0
    assert prof["classifier_ms"] > 0.0

    # Reset
    llm.cpp_engine.reset_profile()
    prof_reset = llm.cpp_engine.get_profile()
    assert prof_reset["count"] == 0
    assert prof_reset["total_ms"] == 0.0


def test_cpp_sampling_top_k_and_custom_eos():
    checkpoint = REPO_ROOT / "models" / "stories260K" / "stories260K.bin"
    tokenizer = REPO_ROOT / "models" / "stories260K" / "tok512.bin"
    if not checkpoint.exists() or not tokenizer.exists():
        pytest.skip("stories260K model files not present")
    llm = LlamaLLM(checkpoint, tokenizer, backend="cpp")
    if llm.cpp_engine is None:
        pytest.skip("C++ engine not available")

    # Sample with top_k
    llm.reset_cache()
    llm.cpp_engine.forward(1, 0, copy_logits=False)
    tok = llm.cpp_engine.sample(temperature=0.8, top_p=0.9, top_k=10)
    assert 0 <= tok < llm.config.vocab_size

    # Custom EOS token id: set to the first token that would be generated
    first_gen = llm.cpp_engine.forward_argmax(1, 0)
    out = llm.cpp_engine.generate(prompt_tokens=[1], max_new_tokens=20, temperature=0.0, eos_token_id=first_gen)
    assert len(out) == 1
    assert out[0] == first_gen


def test_llama_config_theta_and_custom_eos():
    cfg = LlamaConfig(
        dim=64,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=512,
        seq_len=64,
        rope_theta=500000.0,
        eos_token_id=5,
    )
    assert cfg.rope_theta == 500000.0
    assert cfg.eos_token_id == 5

