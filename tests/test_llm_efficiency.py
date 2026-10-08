"""Efficiency regression tests: byte-level tiers and cross-dialogue packing.

Covers the P0/P1 changes described in research/15_llm_efficiency_audit.md:

* the vocabulary-free ByteTokenizer (V=258) and its integration into LlamaLLM
* byte-level Zexo tiers, which keep the transformer body identical to the BPE
  tier while removing ~99% of the embedding parameters
* document packing, which concatenates EOS-delimited dialogues and windows once
  instead of rounding every dialogue up to a whole number of windows
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

import doraneural as dn
from doraneural.llm import ByteTokenizer, HFTokenizer, LlamaConfig, LlamaLLM
from zexo.config import BYTE_VOCAB_SIZE, ZexoConfig

BPE_TIER_NAMES = ["micro", "mini", "chat", "base"]
BYTE_TIER_NAMES = ["micro-byte", "mini-byte", "chat-byte", "base-byte"]
CORPUS = REPO_ROOT / "zexo" / "data" / "zexo_quality_v1_train.txt"


# --------------------------------------------------------------------------
# ByteTokenizer
# --------------------------------------------------------------------------

def test_byte_tokenizer_vocabulary_layout():
    tok = ByteTokenizer()
    assert tok.vocab_size == BYTE_VOCAB_SIZE == 258
    assert tok.BOS_ID == 256
    assert tok.EOS_ID == 257
    assert 0 <= tok.BOS_ID < tok.vocab_size
    assert 0 <= tok.EOS_ID < tok.vocab_size


def test_byte_tokenizer_encode_is_raw_utf8():
    tok = ByteTokenizer()
    text = "Hello Zexo!"
    ids = tok.encode(text, bos=False)
    assert ids == list(text.encode("utf-8"))
    assert tok.encode(text, bos=True) == [tok.BOS_ID] + ids


def test_byte_tokenizer_roundtrips_multibyte_and_all_bytes():
    tok = ByteTokenizer()
    samples = [
        "café",          # 2-byte UTF-8 sequences
        "naïve — ✓ 😀",  # 3-byte and 4-byte sequences
        "User: hi\nZexo: hello\n",
        "",
    ]
    for text in samples:
        assert tok.decode(tok.encode(text, bos=False)) == text

    # Every emitted id must be a raw byte id, and every byte id must decode.
    ids = tok.encode("naïve — ✓ 😀", bos=False)
    assert all(0 <= i <= 255 for i in ids)
    for byte in range(256):
        assert 0 <= tok.encode_token_id(byte) <= 255


def test_byte_tokenizer_batch_helpers_match_scalar():
    tok = ByteTokenizer()
    texts = ["alpha", "beta", "gamma"]
    assert tok.encode_batch(texts, bos=True) == [tok.encode(t, bos=True) for t in texts]
    assert tok.decode_batch(tok.encode_batch(texts, bos=False)) == texts


def test_byte_tokenizer_is_exported_from_package():
    assert dn.ByteTokenizer is ByteTokenizer
    assert "ByteTokenizer" in dn.__all__


def test_byte_tokenizer_ignores_bos_and_eos_on_decode():
    tok = ByteTokenizer()
    ids = [tok.BOS_ID] + tok.encode("hi", bos=False) + [tok.EOS_ID]
    assert tok.decode(ids) == "hi"
    assert tok.decode_token(tok.BOS_ID) == ""
    assert tok.decode_token(tok.EOS_ID) == ""


# --------------------------------------------------------------------------
# Byte-level Zexo tiers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bpe_name,byte_name", list(zip(BPE_TIER_NAMES, BYTE_TIER_NAMES)))
def test_byte_tier_keeps_transformer_body_and_drops_embedding(bpe_name, byte_name):
    bpe = ZexoConfig.from_tier(bpe_name)
    byte_cfg = ZexoConfig.from_tier(byte_name)

    assert byte_cfg.is_byte_level and not bpe.is_byte_level
    assert byte_cfg.tokenizer == "byte"
    assert byte_cfg.vocab_size == BYTE_VOCAB_SIZE

    # The transformer body must be untouched so the comparison is apples-to-apples.
    assert byte_cfg.dim == bpe.dim
    assert byte_cfg.hidden_dim == bpe.hidden_dim
    assert byte_cfg.n_layers == bpe.n_layers
    assert byte_cfg.n_heads == bpe.n_heads
    assert byte_cfg.n_kv_heads == bpe.n_kv_heads

    assert byte_cfg.parameter_count < bpe.parameter_count
    embed_share = (byte_cfg.vocab_size * byte_cfg.dim) / byte_cfg.parameter_count
    assert embed_share < 0.10, f"byte tier still spends {embed_share:.1%} on embeddings"


def test_byte_tiers_are_the_dominant_embedding_saving():
    """The headline result: identical capacity, drastically fewer parameters."""
    # The saving shrinks as the body grows, so each tier gets its own floor.
    for bpe_name, byte_name, factor in [
        ("mini", "mini-byte", 5.0),
        ("chat", "chat-byte", 1.5),
        ("base", "base-byte", 1.2),
    ]:
        bpe = ZexoConfig.from_tier(bpe_name)
        byte_cfg = ZexoConfig.from_tier(byte_name)
        assert byte_cfg.parameter_count * factor <= bpe.parameter_count, byte_name

    # zexo-mini is the pathological case: 89.8% BPE embedding vs 6.6% byte.
    mini = ZexoConfig.from_tier("mini")
    assert (mini.vocab_size * mini.dim) / mini.parameter_count > 0.85
    assert (BYTE_VOCAB_SIZE * 192) / ZexoConfig.from_tier("mini-byte").parameter_count < 0.10


def test_from_tier_resolves_byte_aliases():
    for alias, expected in [
        ("chat-byte", "chat-byte"),
        ("chatbyte", "chat-byte"),
        ("small-byte", "chat-byte"),
        ("MINI-BYTE", "mini-byte"),
    ]:
        assert ZexoConfig.from_tier(alias).tier == expected


def test_unknown_tier_error_lists_byte_tiers():
    with pytest.raises(ValueError) as excinfo:
        ZexoConfig.from_tier("nope")
    message = str(excinfo.value)
    assert "chat-byte" in message
    assert "mini-byte" in message


def test_byte_tier_llama_config_uses_byte_eos_id():
    cfg = ZexoConfig.from_tier("chat-byte").llama_config
    assert isinstance(cfg, LlamaConfig)
    assert cfg.eos_token_id == 257
    # The BPE default of 2 must not leak into byte tiers, where id 2 is a raw byte.
    assert cfg.vocab_size == BYTE_VOCAB_SIZE


def test_byte_tier_config_roundtrips_through_json():
    cfg = ZexoConfig.from_tier("chat-byte")
    restored = ZexoConfig.from_dict(cfg.to_dict())
    assert restored.tokenizer == "byte"
    assert restored.vocab_size == BYTE_VOCAB_SIZE
    assert restored.is_byte_level


# --------------------------------------------------------------------------
# Document packing
# --------------------------------------------------------------------------

class _BatchStub(LlamaLLM):
    """Reuse _prepare_training_batches without loading engine weights."""

    def __init__(self, tokenizer, vocab_size=32000, seq_len=2048):
        self.tokenizer = tokenizer
        self.config = LlamaConfig(dim=64, hidden_dim=172, n_layers=1, n_heads=4,
                                  n_kv_heads=4, vocab_size=vocab_size, seq_len=seq_len)


@pytest.fixture(scope="module")
def bpe_batch_stub():
    tokenizer = HFTokenizer(REPO_ROOT / "zexo" / "tokenizer" / "tokenizer.json")
    return _BatchStub(tokenizer)


@pytest.fixture(scope="module")
def byte_batch_stub():
    return _BatchStub(ByteTokenizer(), vocab_size=BYTE_VOCAB_SIZE)


@pytest.mark.parametrize("stub_id", ["bpe_batch_stub", "byte_batch_stub"])
def test_packing_reduces_windows_and_raises_supervised_density(request, stub_id):
    stub = request.getfixturevalue(stub_id)
    text = CORPUS.read_text(encoding="utf-8")
    seq_len = 256

    unpacked = stub._prepare_training_batches(
        text, seq_len=seq_len, mask_prompts=True, pack_documents=False)
    packed = stub._prepare_training_batches(
        text, seq_len=seq_len, mask_prompts=True, pack_documents=True)

    assert packed and unpacked
    assert len(packed) < len(unpacked)

    def density(batches):
        slots = len(batches) * seq_len
        supervised = sum(sum(1 for t in tgt if t >= 0) for _, tgt in batches)
        return supervised / slots

    assert density(packed) > density(unpacked)
    # Packing only removes empty tail slots, so the number of supervised
    # positions may differ by at most the discarded trailing partial window.
    def supervised_count(batches):
        return sum(1 for _, tgt in batches for t in tgt if t >= 0)
    assert supervised_count(packed) >= 0.99 * supervised_count(unpacked)


def test_packing_preserves_sft_prompt_masking(bpe_batch_stub):
    text = CORPUS.read_text(encoding="utf-8")
    batches = bpe_batch_stub._prepare_training_batches(
        text, seq_len=256, mask_prompts=True, pack_documents=True)

    for inputs, targets in batches:
        assert len(inputs) == len(targets)
        assert len(inputs) == 256
        assert all(0 <= t < BYTE_VOCAB_SIZE + 32000 for t in inputs)
        assert any(t >= 0 for t in targets)
        # Masked positions must still be -1, never a padded placeholder id.
        assert all(t == -1 or t >= 0 for t in targets)


def test_packing_respects_max_batches(bpe_batch_stub):
    text = CORPUS.read_text(encoding="utf-8")
    for pack in (False, True):
        batches = bpe_batch_stub._prepare_training_batches(
            text, seq_len=256, mask_prompts=True, max_batches=5, pack_documents=pack)
        assert len(batches) == 5


def test_packing_is_inert_for_non_dialogue_text(byte_batch_stub):
    plain = "The quick brown fox jumps over the lazy dog. " * 40
    packed = byte_batch_stub._prepare_training_batches(
        plain, seq_len=128, mask_prompts=True, pack_documents=True)
    unpacked = byte_batch_stub._prepare_training_batches(
        plain, seq_len=128, mask_prompts=True, pack_documents=False)
    assert packed == unpacked


def test_packing_flag_is_accepted_by_public_train_signature():
    import inspect
    for method in (LlamaLLM.train, LlamaLLM.train_full, LlamaLLM.train_lora):
        params = inspect.signature(method).parameters
        assert "pack_documents" in params, method.__name__
        assert params["pack_documents"].default is True


# --------------------------------------------------------------------------
# End-to-end byte tier
# --------------------------------------------------------------------------

def test_micro_byte_tier_trains_and_generates():
    from zexo.model import load_zexo

    zexo = load_zexo(tier="micro-byte", from_scratch=True)
    assert isinstance(zexo.llm.tokenizer, ByteTokenizer)
    assert zexo.config.vocab_size == BYTE_VOCAB_SIZE

    text = "User: hello there\nZexo: hi there, how are you doing today?\n"
    history = zexo.llm.train(text, epochs=1, seq_len=16, verbose=0)
    assert history["loss"]
    assert all(loss == loss for loss in history["loss"])  # not NaN

    out = zexo.generate("Once upon", max_tokens=8, temperature=0.0)
    assert isinstance(out, str)


# --------------------------------------------------------------------------
# Benchmark harness (P4)
# --------------------------------------------------------------------------

def test_bench_script_reports_cpu_dispatch_signature():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bench_llm", REPO_ROOT / "scripts" / "bench_llm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "cpu_backend" in module.__doc__
    assert set(module.TIERS) >= {"mini", "mini-byte", "chat", "chat-byte"}
    for name, cfg in module.TIERS.items():
        assert cfg.dim % cfg.n_heads == 0, name

    weights = module.make_weights(module.TIERS["mini-byte"])
    assert weights["token_embedding_table"].size == 258 * 192
    assert weights["shared_classifier"] == 0