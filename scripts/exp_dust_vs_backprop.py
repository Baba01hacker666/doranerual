"""Dust vs backprop at matched wall-clock.

The honest axis for this comparison is time, not steps. Dust's population costs
K forward passes per update against backprop's one backward pass, so comparing
"100 steps of each" flatters Dust or backprop depending on K, and comparing at
equal token counts ignores the compute. Both methods therefore get the same
wall-clock budget on the same machine, and we report how much each one managed to
do inside it.

Reference: Dahal et al., "Dust: Pretraining Transformers Without Backpropagation",
Q Labs, 2026. https://qlabs.sh/research/dust

Usage:
    python3 scripts/exp_dust_vs_backprop.py --budget-seconds 60 --draws 64
    python3 scripts/exp_dust_vs_backprop.py --budget-seconds 120 --seq-len 128 --dim 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from doraneural.dust import (  # noqa: E402
    DustConfig, DustTrainer, backprop_gradient, cosine_to_backprop, evaluate_loss,
)
from doraneural.autograd import Tensor  # noqa: E402
from doraneural.transformer import TransformerDecoderLM, TensorAdamW  # noqa: E402

TRAIN_CORPUS = REPO_ROOT / "zexo" / "data" / "zexo_quality_v1_train.txt"
EVAL_CORPUS = REPO_ROOT / "zexo" / "data" / "zexo_quality_v1_eval.txt"


def build_model(args) -> TransformerDecoderLM:
    """Same body the Zexo micro tier uses, so the comparison sits on our real model."""
    np.random.seed(args.seed)
    return TransformerDecoderLM(
        dim=args.dim,
        hidden_dim=args.hidden,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.heads,
        vocab_size=args.vocab,
        seq_len=args.seq_len,
    )


def make_batches(tokenizer_ids: np.ndarray, seq_len: int, rng: np.random.Generator) -> List[tuple]:
    """Non-overlapping windows, matching the trainer's windowing convention."""
    batches = []
    for start in range(0, len(tokenizer_ids) - seq_len, seq_len):
        window = tokenizer_ids[start : start + seq_len]
        batches.append((window.tolist(), window[1:].tolist()))
    if not batches:
        raise ValueError("corpus is shorter than one window")
    return batches


def cosine_to_budget(model, batch, site: str, layer: int, draws: int, seed: int) -> float:
    ids, targets = batch
    truth = backprop_gradient(model, ids, targets)
    cos, _, _ = cosine_to_backprop(
        model, ids, targets, site, layer,
        DustConfig(draws=draws, seed=seed), truth=truth,
    )
    return cos


def run_backprop(model, batches, eval_batch, budget: float, args) -> Dict[str, object]:
    opt = TensorAdamW(list(model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    curve: List[Dict[str, float]] = []
    steps = 0
    tokens = 0
    start = time.perf_counter()
    stop = False
    while not stop:
        order = rng.permutation(len(batches))
        for idx in order:
            ids, targets = batches[idx]
            logits = model.forward(ids)
            n = len(targets)
            rows = logits[:n]
            vocab = rows.data.shape[-1]
            onehot = np.eye(vocab, dtype=rows.dtype)[np.asarray(targets, dtype=np.int64)]
            lse = rows.exp().sum(axis=-1, keepdims=True).log()
            picked = (rows * Tensor(onehot, dtype=rows.dtype)).sum(axis=-1)
            loss = -(picked - lse).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
            tokens += n
            if steps % max(1, args.log_every) == 0 or steps == 1:
                curve.append({
                    "step": steps,
                    "seconds": round(time.perf_counter() - start, 3),
                    "train_loss": float(loss.data),
                    "eval_loss": evaluate_loss(model, eval_batch[0], eval_batch[1]),
                })
            if time.perf_counter() - start >= budget:
                stop = True
                break
    return {
        "method": "backprop",
        "steps": steps,
        "tokens": tokens,
        "seconds": round(time.perf_counter() - start, 2),
        "final_eval_loss": curve[-1]["eval_loss"] if curve else None,
        "curve": curve,
    }


def run_dust(model, batches, eval_batch, budget: float, args) -> Dict[str, object]:
    cfg = DustConfig(draws=args.draws, seed=args.seed)
    trainer = DustTrainer(
        model, cfg, lr=args.dust_lr, optimizer=args.dust_optimizer,
        weight_decay=args.weight_decay,
        split_population=not args.no_split, seed=args.seed,
    )
    rng = np.random.default_rng(args.seed + 1)
    curve: List[Dict[str, float]] = []
    steps = 0
    tokens = 0
    start = time.perf_counter()
    stop = False
    while not stop:
        order = rng.permutation(len(batches))
        for idx in order:
            ids, targets = batches[idx]
            stats = trainer.step(ids, targets)
            steps += 1
            tokens += len(targets)
            if steps % max(1, args.log_every) == 0 or steps == 1:
                curve.append({
                    "step": steps,
                    "seconds": round(time.perf_counter() - start, 3),
                    "train_loss": stats.loss_before,
                    "eval_loss": evaluate_loss(model, eval_batch[0], eval_batch[1]),
                })
            if time.perf_counter() - start >= budget:
                stop = True
                break
    return {
        "method": "dust",
        "steps": steps,
        "tokens": tokens,
        "forwards": trainer.total_forwards,
        "population_total": args.draws,
        "draws_per_site": max(1, args.draws // (args.layers * 2)),
        "seconds": round(time.perf_counter() - start, 2),
        "final_eval_loss": curve[-1]["eval_loss"] if curve else None,
        "curve": curve,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget-seconds", type=float, default=60.0,
                        help="Wall-clock budget given to EACH method.")
    parser.add_argument("--draws", type=int, default=64,
                        help="Dust population K (forward passes per update).")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=172)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="backprop lr")
    parser.add_argument("--dust-lr", type=float, default=1e-2, help="Dust lr")
    parser.add_argument("--no-split", action="store_true",
                        help="Give every (layer, site) the full population instead of "
                             "splitting K across them (10x-30x more forwards per step).")
    parser.add_argument("--dust-optimizer", default="adam", choices=["sgd", "adam"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-ladder", action="store_true",
                        help="Skip the cosine ladder (the K=256 rung costs 256 forwards).")
    args = parser.parse_args(argv)

    print(f"budget={args.budget_seconds:.0f}s per method  K={args.draws}  "
          f"dim={args.dim} L={args.layers} V={args.vocab} seq={args.seq_len}")

    # Byte-level ids keep the experiment free of any tokenizer download and are the
    # natural choice now that the repo ships ByteTokenizer.
    from doraneural.llm import ByteTokenizer
    tok = ByteTokenizer()
    train_text = TRAIN_CORPUS.read_text(encoding="utf-8")
    eval_text = EVAL_CORPUS.read_text(encoding="utf-8")

    train_ids = np.asarray(tok.encode(train_text, bos=False), dtype=np.int64) % args.vocab
    eval_ids = np.asarray(tok.encode(eval_text, bos=False), dtype=np.int64) % args.vocab
    batches = make_batches(train_ids, args.seq_len, np.random.default_rng(args.seed))
    eval_batch = make_batches(eval_ids, args.seq_len, np.random.default_rng(args.seed))[0]
    print(f"{len(batches)} train windows of {args.seq_len}, "
          f"{len(eval_ids)} eval bytes")

    results = []
    for name, runner in (("backprop", run_backprop), ("dust", run_dust)):
        model = build_model(args)
        print(f"\n>>> {name}", flush=True)
        res = runner(model, batches, eval_batch, args.budget_seconds, args)
        loss_txt = "n/a" if res['final_eval_loss'] is None else f"{res['final_eval_loss']:.4f}"
        print(f"    steps={res['steps']} tokens={res['tokens']:,} "
              f"eval_loss={loss_txt} in {res['seconds']}s")
        results.append(res)

    # Estimator quality on the freshly initialised model, for reference.
    model = build_model(args)
    ladder: Dict[str, Dict[int, float]] = {"wo": {}, "w2": {}}
    if not args.skip_ladder:
        for site in ("wo", "w2"):
            for k in (16, 64, 256):
                ladder[site][k] = cosine_to_budget(model, batches[0], site, 0, k, args.seed)

    print("\n" + "=" * 78)
    print(f"{'method':10s} {'steps':>7s} {'tokens':>9s} {'forwards':>10s} "
          f"{'sec':>7s} {'eval loss':>10s}")
    print("-" * 78)
    for r in results:
        loss_txt = "n/a" if r["final_eval_loss"] is None else f"{r['final_eval_loss']:10.4f}"
        fwd = r.get("forwards", r["steps"])
        per = fwd // max(1, r["steps"])
        print(f"{r['method']:10s} {r['steps']:7d} {r['tokens']:9,d} "
              f"{per:9,d} {r['seconds']:7.1f} {loss_txt}")
    print("=" * 78)
    if not args.skip_ladder:
        print("\nCosine to the backprop gradient (initial model):")
        for site in ("wo", "w2"):
            print(f"  {site}: " + "  ".join(f"K={k}:{v:+.4f}" for k, v in ladder[site].items()))

    report = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "results": results,
        "cosine_ladder": {site: {str(k): v for k, v in ladder[site].items()}
                          for site in ladder},
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())