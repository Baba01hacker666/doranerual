"""Deterministic LLM throughput benchmark with a recorded CPU dispatch signature.

Raw tokens/sec numbers from this engine are meaningless on their own. The INT8
VNNI path in ``csrc/llm_engine.cpp`` only exists on AVX-512 VNNI hosts, so the
same binary reports ~10x different throughput on a Skylake AVX2 box than on a
Sapphire Rapids host. Research Paper 09 measured >1000 TPS on a VNNI sandbox;
that number does not transfer to an AVX2 runner.

Every result therefore records ``cpu_arch``/``cpu_backend`` alongside the
throughput, and the benchmark always pins a thread count. Shared CI runners are
noisy (Paper 09 measured 521-1108 TPS on identical runs), so throughput is
reported as best-of-N rather than mean.

Usage:
    python3 scripts/bench_llm.py                    # default tiers, JSON to stdout
    python3 scripts/bench_llm.py --tiers chat --repeats 5 --threads 2
    python3 scripts/bench_llm.py --output bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from doraneural import get_cpu_arch, get_cpu_backend  # noqa: E402
from doraneural.cpp_backend import CppLlamaEngine, build_cpp_library  # noqa: E402
from doraneural.llm import LlamaConfig  # noqa: E402

# Tuned to mirror the Zexo tier ladder (zexo/config.py). The byte tiers mirror
# the BPE bodies with a 258-token vocabulary; see research/15.
TIERS: Dict[str, LlamaConfig] = {
    "micro": LlamaConfig(dim=64, hidden_dim=172, n_layers=5, n_heads=4,
                         n_kv_heads=4, vocab_size=512, seq_len=256),
    "mini": LlamaConfig(dim=192, hidden_dim=1024, n_layers=1, n_heads=2,
                        n_kv_heads=1, vocab_size=32000, seq_len=1024),
    "mini-byte": LlamaConfig(dim=192, hidden_dim=1024, n_layers=1, n_heads=2,
                             n_kv_heads=1, vocab_size=258, seq_len=512),
    "chat": LlamaConfig(dim=384, hidden_dim=1024, n_layers=8, n_heads=8,
                        n_kv_heads=4, vocab_size=32000, seq_len=1024),
    "chat-byte": LlamaConfig(dim=384, hidden_dim=1024, n_layers=8, n_heads=8,
                             n_kv_heads=4, vocab_size=258, seq_len=1024),
    "base": LlamaConfig(dim=768, hidden_dim=2048, n_layers=12, n_heads=12,
                        n_kv_heads=12, vocab_size=32000, seq_len=2048),
}

PROMPT_TOKENS = 10


def make_weights(cfg: LlamaConfig, seed: int = 0) -> Dict[str, np.ndarray]:
    """Random-but-deterministic weights so throughput does not depend on a download."""
    rng = np.random.default_rng(seed)
    dim, hidden, layers, vocab = cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.vocab_size
    kv_dim = dim * cfg.n_kv_heads // cfg.n_heads

    def rand(*shape):
        return (rng.standard_normal(shape).astype(np.float32) * 0.02).reshape(-1)

    weights = {
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
    return weights


def bench_tier(name: str, cfg: LlamaConfig, threads: int, token_counts: List[int],
               repeats: int, library: Path) -> Dict[str, object]:
    engine = CppLlamaEngine(cfg, make_weights(cfg))
    engine.set_threads(threads)
    results: Dict[str, object] = {}
    try:
        for ntok in token_counts:
            best_tps = 0.0
            best_seconds = float("inf")
            for _ in range(repeats):
                engine.reset_cache()
                t0 = time.perf_counter()
                engine.generate([1] * PROMPT_TOKENS, max_new_tokens=ntok,
                                temperature=0.0, top_p=1.0)
                elapsed = time.perf_counter() - t0
                if ntok / elapsed > best_tps:
                    best_tps = ntok / elapsed
                    best_seconds = elapsed
            results[str(ntok)] = {
                "best_tps": round(best_tps, 2),
                "best_seconds": round(best_seconds, 6),
            }
    finally:
        del engine
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiers", nargs="*", default=["mini", "mini-byte", "chat", "chat-byte"],
                        choices=sorted(TIERS), help="Which tiers to benchmark.")
    parser.add_argument("--tokens", nargs="*", type=int, default=[32, 128],
                        help="New-token counts per measurement.")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1),
                        help="Pinned OpenMP thread count (default: min(4, cpus)).")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeats per measurement; the best is reported (default: 3).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional path to write the JSON report to.")
    args = parser.parse_args(argv)

    library = build_cpp_library()
    if library is None:
        print("Native C++ library unavailable; skipping throughput benchmark.", file=sys.stderr)
        return 1

    report = {
        "cpu_arch": get_cpu_arch(),
        "cpu_backend": get_cpu_backend(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads": args.threads,
        "repeats": args.repeats,
        "library": str(library),
        "prompt_tokens": PROMPT_TOKENS,
        "note": "best-of-N throughput; not comparable across cpu_backend values",
        "tiers": {},
    }

    for name in args.tiers:
        cfg = TIERS[name]
        report["tiers"][name] = {
            "dim": cfg.dim,
            "n_layers": cfg.n_layers,
            "vocab_size": cfg.vocab_size,
            "measurements": bench_tier(name, cfg, args.threads, args.tokens,
                                       args.repeats, library),
        }
        for ntok, data in report["tiers"][name]["measurements"].items():
            print(f"{name:10s} {ntok:>4s}tok  {data['best_tps']:9.2f} TPS")

    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())