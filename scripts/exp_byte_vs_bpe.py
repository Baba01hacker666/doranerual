"""Controlled comparison: BPE tiers vs byte-level tiers, end to end.

Tests the Research Paper 15 claims on real training runs rather than synthetic
weights, using the small `stories260K` / `zexo-micro` body (dim=64, 5 layers,
hidden=172, 4 heads) because it is small enough to train repeatedly on CPU.

What is held constant:
  * transformer body (dim, hidden_dim, n_layers, n_heads, n_kv_heads)
  * config.seq_len, so both variants can be given a long enough window
  * weight-init seed, window-shuffle seed, learning rate, epochs
  * training corpus and held-out evaluation corpus

What varies:
  * vocab_size + tokenizer (SentencePiece BPE vs raw UTF-8 bytes)
  * pack_documents (per-dialogue windowing vs EOS-delimited global packing)

Because a byte model consumes one token per byte and a 512-token BPE consumes
roughly 3.8 bytes per token, equal `seq_len` does NOT mean equal text. Every run
therefore reports supervised bytes as well as supervised tokens, and the byte
variant is run at two window lengths:

  * byte @ equal-seq_len  -- same token count, ~1/4 the text
  * byte @ equal-text     -- ~3.8x the tokens, comparable text per window

Match on optimizer STEPS, not epochs. A byte model emits one token per byte, so
equal epochs hands it ~3.8x the supervised tokens and makes the comparison
meaningless. `--match-steps` gives every variant the same gradient-update budget
and derives per-variant epochs from its own window count.

Usage:
    python3 scripts/exp_byte_vs_bpe.py --body micro --match-steps 1752 --threads 4
    python3 scripts/exp_byte_vs_bpe.py --body mini  --match-steps 300  --threads 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from doraneural.llm import (  # noqa: E402
    ByteTokenizer, HFTokenizer, LlamaLLM, LlamaTokenizer,
)
from zexo.config import BYTE_VOCAB_SIZE, ZexoConfig  # noqa: E402
from zexo.init_weights import initialize_zexo_checkpoint  # noqa: E402

TRAIN_CORPUS = REPO_ROOT / "zexo" / "data" / "zexo_quality_v1_train.txt"
EVAL_CORPUS = REPO_ROOT / "zexo" / "data" / "zexo_quality_v1_eval.txt"
BPE_TOKENIZER = REPO_ROOT / "zexo" / "tokenizer" / "tok512.bin"
BPE_TOKENIZER_32K = REPO_ROOT / "zexo" / "tokenizer" / "tokenizer.json"

# Both variants get this config seq_len so a byte window can be long enough to
# cover comparable text. KV cache cost at this size is negligible.
CONFIG_SEQ_LEN = 1024


def variant_config(tier: str, byte_level: bool) -> ZexoConfig:
    """Clone a Zexo tier, changing only the vocabulary/seq_len config.

    The BPE side keeps the tier's own vocab (512 for micro, 32000 for mini); the
    byte side swaps in the 258-token byte vocabulary. Everything else -- the
    transformer body -- is held identical so the comparison isolates the
    vocabulary.
    """
    base = ZexoConfig.from_tier(tier)
    return ZexoConfig(
        tier=f"{tier}{'-byte' if byte_level else ''}",
        dim=base.dim,
        hidden_dim=base.hidden_dim,
        n_layers=base.n_layers,
        n_heads=base.n_heads,
        n_kv_heads=base.n_kv_heads,
        vocab_size=BYTE_VOCAB_SIZE if byte_level else base.vocab_size,
        seq_len=CONFIG_SEQ_LEN,
        rope_type=base.rope_type,
        tokenizer="byte" if byte_level else "bpe",
    )


def corpus_stats(text: str, tokenizer, seq_len: int, pack: bool) -> Dict[str, object]:
    """Count windows/epoch and supervised bytes for one tokenizer configuration."""
    ids = tokenizer.encode(text, bos=False)
    with redirect_stdout(StringIO()):
        from doraneural.llm import LlamaConfig  # noqa: F401
        batches = _prepare(tokenizer, text, seq_len, pack)
    supervised = sum(1 for _, tgt in batches for t in tgt if t >= 0)
    return {
        "bytes": len(text.encode("utf-8")),
        "tokens": len(ids),
        "bytes_per_token": round(len(text.encode("utf-8")) / max(1, len(ids)), 3),
        "windows_per_epoch": len(batches),
        "supervised_tokens": supervised,
    }


def bpe_tokenizer_for(tier: str):
    """The BPE tokenizer matching a tier's declared vocabulary."""
    cfg = ZexoConfig.from_tier(tier)
    if cfg.vocab_size == 512:
        return LlamaTokenizer(BPE_TOKENIZER, vocab_size=512)
    return HFTokenizer(BPE_TOKENIZER_32K)


def _prepare(tokenizer, text: str, seq_len: int, pack: bool):
    """Call the real batch builder without constructing an engine."""
    from doraneural.llm import LlamaConfig, LlamaLLM

    class _Stub(LlamaLLM):
        def __init__(self):
            self.tokenizer = tokenizer
            self.config = LlamaConfig(dim=64, hidden_dim=172, n_layers=5, n_heads=4,
                                      n_kv_heads=4, vocab_size=32000,
                                      seq_len=CONFIG_SEQ_LEN, eos_token_id=257)

    return _Stub()._prepare_training_batches(
        text, seq_len=seq_len, mask_prompts=True, pack_documents=pack)


def run_variant(
    label: str,
    cfg: ZexoConfig,
    byte_level: bool,
    seq_len: int,
    pack: bool,
    epochs: int,
    lr: float,
    threads: int,
    seed: int,
    workdir: Path,
    train_text: str,
    eval_text: str,
    decode_tokens: int,
) -> Dict[str, object]:
    workdir.mkdir(parents=True, exist_ok=True)
    tok_path = "byte" if byte_level else str(
        BPE_TOKENIZER if cfg.vocab_size <= 512 else BPE_TOKENIZER_32K)

    init_started = time.perf_counter()
    ckpt = initialize_zexo_checkpoint(cfg, workdir / "model.bin", seed=seed)
    init_seconds = time.perf_counter() - init_started

    llm = LlamaLLM(model_path=ckpt, tokenizer_path=tok_path, backend="cpp")
    llm.set_num_threads(threads)

    stats = corpus_stats(train_text, llm.tokenizer, seq_len, pack)

    train_started = time.perf_counter()
    with redirect_stdout(StringIO()):
        history = llm.train(
            train_text,
            epochs=epochs,
            lr=lr,
            seq_len=seq_len,
            full_backprop=True,
            pack_documents=pack,
            eval_text=eval_text,
            verbose=0,
            seed=seed,
        )
    train_seconds = time.perf_counter() - train_started

    final_loss = float(history["loss"][-1])
    val_loss = history.get("val_loss")
    val_loss = float(val_loss[-1]) if isinstance(val_loss, (list, tuple)) and val_loss else val_loss

    # Decode throughput (one token at a time, as a normal user experiences it).
    llm.reset_cache()
    prompt_ids = llm.tokenizer.encode("User: tell me a story\nZexo:", bos=True)[:16]
    t0 = time.perf_counter()
    llm.generate("User: tell me a story\nZexo:", max_tokens=decode_tokens,
                 temperature=0.0, top_p=1.0)
    decode_tps = decode_tokens / (time.perf_counter() - t0)

    # Prefill: serialised vs batched, as implemented by llama_forward_chunk.
    prefill = prefill_throughput(llm, prompt_ids)

    sample = llm.generate("User: tell me a story about the sea\nZexo:",
                          max_tokens=48, temperature=0.0, top_p=1.0)

    result: Dict[str, object] = {
        "label": label,
        "tier": cfg.tier,
        "tokenizer": cfg.tokenizer,
        "vocab_size": cfg.vocab_size,
        "parameters": cfg.parameter_count,
        "seq_len": seq_len,
        "pack_documents": pack,
        "init_seconds": round(init_seconds, 3),
        "train_seconds": round(train_seconds, 2),
        "epochs": epochs,
        "windows_per_epoch": stats["windows_per_epoch"],
        "optimizer_steps": epochs * stats["windows_per_epoch"],
        "supervised_tokens_per_epoch": stats["supervised_tokens"],
        "supervised_tokens_total": epochs * stats["supervised_tokens"],
        "bytes_per_token": stats["bytes_per_token"],
        "corpus_bytes": stats["bytes"],
        "final_train_loss": round(final_loss, 4),
        "val_loss": round(val_loss, 4) if isinstance(val_loss, float) else None,
        "val_perplexity": round(float(np.exp(val_loss)), 3) if isinstance(val_loss, float) else None,
        "decode_tps": round(decode_tps, 2),
        "sample": sample[-160:],
    }
    result.update(prefill)
    del llm
    return result


def prefill_throughput(llm: LlamaLLM, prompt_ids: List[int]) -> Dict[str, object]:
    """Measure serialised vs batched prefill on this engine, if the symbol exists."""
    engine = getattr(llm, "cpp_engine", None)
    if engine is None or not hasattr(getattr(engine, "lib", None), "llama_forward_chunk"):
        return {"prefill_serial_tps": None, "prefill_batched_tps": None, "prefill_speedup": None}

    toks = [int(t) for t in prompt_ids]
    serial = batched = 0.0
    for _ in range(3):
        engine.reset_cache()
        t0 = time.perf_counter()
        for pos, tok in enumerate(toks):
            engine.forward(tok, pos)
        serial = max(serial, len(toks) / (time.perf_counter() - t0))

        engine.reset_cache()
        t0 = time.perf_counter()
        engine.forward_chunk(toks, 0)
        batched = max(batched, len(toks) / (time.perf_counter() - t0))

    return {
        "prefill_serial_tps": round(serial, 2),
        "prefill_batched_tps": round(batched, 2),
        "prefill_speedup": round(batched / serial, 3) if serial else None,
    }


def run_long_prompt_prefill(threads: int, workdir: Path, seed: int) -> Dict[str, object]:
    """Prefill scaling on a longer prompt, where the batching win is largest."""
    cfg = variant_config("micro", byte_level=False)
    workdir.mkdir(parents=True, exist_ok=True)
    ckpt = initialize_zexo_checkpoint(cfg, workdir / "model.bin", seed=seed)
    llm = LlamaLLM(model_path=ckpt, tokenizer_path=str(BPE_TOKENIZER), backend="cpp")
    llm.set_num_threads(threads)
    engine = llm.cpp_engine
    rng = np.random.default_rng(seed)
    rows = []
    for plen in (16, 64, 256, 512):
        toks = [int(t) for t in rng.integers(0, cfg.vocab_size, plen)]
        serial = batched = 0.0
        for _ in range(3):
            engine.reset_cache()
            t0 = time.perf_counter()
            for pos, tok in enumerate(toks):
                engine.forward(tok, pos)
            serial = max(serial, plen / (time.perf_counter() - t0))
            engine.reset_cache()
            t0 = time.perf_counter()
            engine.forward_chunk(toks, 0)
            batched = max(batched, plen / (time.perf_counter() - t0))
        rows.append({"prompt_tokens": plen, "serial_tps": round(serial, 2),
                     "batched_tps": round(batched, 2),
                     "speedup": round(batched / serial, 3)})
    del llm
    return {"prefill_scaling": rows}


def table(results: List[Dict[str, object]]) -> str:
    header = (
        f"{'variant':22s} {'vocab':>6s} {'params':>9s} {'ep':>3s} {'steps':>6s} {'win/ep':>7s} "
        f"{'sup.tok':>9s} {'train s':>8s} {'train L':>8s} {'val L':>8s} {'ppl':>8s} {'TPS':>7s}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r['label']:22s} {r['vocab_size']:6d} {r['parameters']:9,d} "
            f"{r['epochs']:3d} {r['optimizer_steps']:6,d} "
            f"{r['windows_per_epoch']:7d} {r['supervised_tokens_per_epoch']:9,d} "
            f"{r['train_seconds']:8.1f} {r['final_train_loss']:8.4f} "
            f"{(r['val_loss'] if r['val_loss'] is not None else float('nan')):8.4f} "
            f"{(r['val_perplexity'] if r['val_perplexity'] is not None else float('nan')):8.2f} "
            f"{r['decode_tps']:8.1f}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=6,
                        help="Epochs per variant when not step-matching.")
    parser.add_argument("--match-steps", type=int, default=None,
                        help="Give every variant this many optimizer steps "
                             "(overrides --epochs). This is the fair comparison: "
                             "equal epochs favours the byte model, which emits one "
                             "token per byte and so sees ~46%% more supervised tokens.")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--body", default="micro", choices=["micro", "mini"],
                        help="Which tier's transformer body to hold fixed. 'mini' has "
                             "vocab 32000, which is where the vocabulary tax is real.")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="BPE training window length (default: 256).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/doranerual_exp"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-prefill-scaling", action="store_true")
    args = parser.parse_args(argv)

    train_text = TRAIN_CORPUS.read_text(encoding="utf-8")
    eval_text = EVAL_CORPUS.read_text(encoding="utf-8")
    print(f"corpus: {len(train_text.encode()):,} train bytes / "
          f"{len(eval_text.encode()):,} eval bytes")

    # bytes_per_token of the BPE tokenizer decides the equal-text window length.
    bpe = bpe_tokenizer_for(args.body)
    bpt = len(train_text.encode("utf-8")) / max(1, len(bpe.encode(train_text, bos=False)))
    equal_text_seq = int(round(args.seq_len * bpt))
    print(f"BPE bytes/token = {bpt:.3f} -> equal-text byte window = {equal_text_seq} bytes")

    bpe_cfg = variant_config(args.body, byte_level=False)
    byte_cfg = variant_config(args.body, byte_level=True)

    plan = [
        ("BPE  pack=off", bpe_cfg, False, args.seq_len, False),
        ("BPE  pack=on", bpe_cfg, False, args.seq_len, True),
        ("byte pack=on equal-seq", byte_cfg, True, args.seq_len, True),
        ("byte pack=on equal-text", byte_cfg, True, equal_text_seq, True),
    ]

    # Resolve per-variant epochs up front so window counts are known before any
    # training. With --match-steps every variant gets the same gradient-update
    # budget, which is the only apples-to-apples way to compare two tokenizers:
    # equal epochs hands the byte model 46% more supervised tokens simply because
    # it emits one token per byte.
    tokenizer_for = {}
    for label, cfg, byte_level, seq_len, pack in plan:
        tk = ByteTokenizer() if byte_level else bpe_tokenizer_for(args.body)
        tokenizer_for[label] = (tk, corpus_stats(train_text, tk, seq_len, pack))

    if args.match_steps:
        reference = plan[0][0]
        ref_win = tokenizer_for[reference][1]["windows_per_epoch"]
        target_steps = args.match_steps
        print(f"step matching: target={target_steps:,} steps "
              f"(from {reference!r}: {ref_win} windows/epoch)")
    else:
        target_steps = None
        print(f"equal epochs: {args.epochs} per variant")

    results: List[Dict[str, object]] = []
    for label, cfg, byte_level, seq_len, pack in plan:
        win = tokenizer_for[label][1]["windows_per_epoch"]
        if target_steps:
            variant_epochs = max(1, round(target_steps / win))
        else:
            variant_epochs = args.epochs
        print(f"\n>>> {label}: vocab={cfg.vocab_size} params={cfg.parameter_count:,} "
              f"seq_len={seq_len} pack={pack} windows/ep={win} epochs={variant_epochs} "
              f"steps={variant_epochs * win:,}")
        results.append(run_variant(
            label, cfg, byte_level, seq_len, pack, variant_epochs, args.lr,
            args.threads, args.seed,
            args.workdir / label.replace(" ", "_").replace("=", ""),
            train_text, eval_text, args.decode_tokens,
        ))
        r = results[-1]
        print(f"    params={r['parameters']:,} windows/ep={r['windows_per_epoch']} "
              f"train={r['train_seconds']}s loss={r['final_train_loss']} "
              f"val={r['val_loss']} decode={r['decode_tps']} TPS")

    report: Dict[str, object] = {
        "config": {
            "epochs": args.epochs, "match_steps": args.match_steps,
            "target_steps": target_steps, "lr": args.lr, "threads": args.threads,
            "seed": args.seed, "config_seq_len": CONFIG_SEQ_LEN,
            "bpe_bytes_per_token": round(bpt, 3),
            "equal_text_seq_len": equal_text_seq,
            "body_tier": args.body,
            "bpe_vocab": ZexoConfig.from_tier(args.body).vocab_size,
            "corpus": str(TRAIN_CORPUS),
            "eval_corpus": str(EVAL_CORPUS),
        },
        "variants": results,
    }
    if not args.skip_prefill_scaling:
        report.update(run_long_prompt_prefill(args.threads, args.workdir / "prefill", args.seed))

    print("\n" + "=" * 108)
    print(table(results))
    print("=" * 108)

    payload = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())