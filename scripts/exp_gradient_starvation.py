"""Gradient starvation audit: reproduce the measurements in Research Paper 18.

Paper 11 attributed zexo-dora's training failure to Chebyshev KAN saturation
extinguishing upstream gradients, without measuring it. Paper 18 measured it and
found the attribution wrong: gradient norms collapse ~890x during ordinary
training in the *standard* SwiGLU model too, and Dust does not escape it.

Produces three tables:

  saturation   |tanh(u)| distribution at the KAN input across a weight scale sweep
  trained      per-site gradient norms at initialisation and after N real
               optimizer steps, standard vs novel-neuron model
  dust_vs_bp   Dust estimate vs backprop gradient at w3 -- the site the KAN sits
               between -- as saturation worsens

Usage:
    python3 scripts/exp_gradient_starvation.py --draws 128
    python3 scripts/exp_gradient_starvation.py --output research/18_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from doraneural.autograd import Tensor  # noqa: E402
from doraneural.dust import (  # noqa: E402
    DustConfig, backprop_gradient, dust_forward, estimate_site_gradient, _rmsnorm,
)
from doraneural.transformer import TransformerDecoderLM, TensorAdamW  # noqa: E402

SCALES = (1.0, 3.0, 10.0, 30.0)
TRAIN_CHECKPOINTS = (0, 20, 60, 120)
SITE_LAYER = 0


def build_body(novel: bool, scale: float, dim: int, hidden: int, layers: int,
               heads: int, kv_heads: int, vocab: int, seq_len: int, seed: int):
    """dora-tier-shaped body; `scale` multiplies the FFN weights to induce saturation."""
    np.random.seed(seed)
    model = TransformerDecoderLM(
        dim=dim, hidden_dim=hidden, n_layers=layers, n_heads=heads,
        n_kv_heads=kv_heads, vocab_size=vocab, seq_len=seq_len, novel_neurons=novel,
    )
    if scale != 1.0:
        for i in range(layers):
            for name in ("w1", "w3", "w_dend", "c_poly", "w_ref"):
                param = getattr(model.layers[i], name, None)
                if param is not None:
                    param.data = param.data * scale
    return model


def token_window(vocab: int, length: int):
    ids = [(7 * i + 3) % vocab for i in range(length)]
    return ids, ids[1:]


def kan_saturation(model, ids: List[int]) -> Dict[str, float]:
    """Replay the block arithmetic and read |tanh(u)| at the Chebyshev KAN input."""
    time = len(ids)
    dim, heads, kv_heads = model.dim, model.n_heads, model.n_kv_heads
    head_size = dim // heads
    x = np.asarray(model.token_embedding.data)[ids]
    samples: List[np.ndarray] = []

    for layer in model.layers:
        norm = _rmsnorm(x, np.asarray(layer.rms_att.data))
        q = (norm @ np.asarray(layer.wq.data)).reshape(time, heads, head_size).transpose(1, 0, 2)
        k = (norm @ np.asarray(layer.wk.data)).reshape(time, kv_heads, head_size).transpose(1, 0, 2)
        v = (norm @ np.asarray(layer.wv.data)).reshape(time, kv_heads, head_size).transpose(1, 0, 2)
        if kv_heads != heads:
            reps = heads // kv_heads
            k = np.repeat(k, reps, axis=0)
            v = np.repeat(v, reps, axis=0)
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(head_size)
        scores = scores + np.triu(np.full((time, time), -np.inf, dtype=scores.dtype), 1)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        attn = (probs @ v).transpose(1, 0, 2).reshape(time, dim)
        x = x + attn @ np.asarray(layer.wo.data)

        norm_ffn = _rmsnorm(x, np.asarray(layer.rms_ffn.data))
        gate = norm_ffn @ np.asarray(layer.w1.data)
        up = norm_ffn @ np.asarray(layer.w3.data)
        if getattr(layer, "w_dend", None) is not None:
            dend = norm_ffn @ np.asarray(layer.w_dend.data)
            up = up * (1.0 / (1.0 + np.exp(-dend)) * 2.0)
        hidden = (1.0 / (1.0 + np.exp(-gate))) * gate * up
        samples.append(np.abs(np.tanh(hidden)).ravel())
        x = x + hidden @ np.asarray(layer.w2.data)

    u = np.concatenate(samples)
    return {
        "mean": round(float(u.mean()), 4),
        "p90": round(float(np.percentile(u, 90)), 4),
        "p99": round(float(np.percentile(u, 99)), 4),
        "frac_gt_0.99": round(float(np.mean(u > 0.99)), 4),
        "frac_gt_0.999": round(float(np.mean(u > 0.999)), 5),
    }


def backprop_step(model, optimizer, ids, targets) -> float:
    logits = model.forward(ids)
    n = len(targets)
    rows = logits[:n]
    vocab = rows.data.shape[-1]
    onehot = np.eye(vocab, dtype=rows.dtype)[np.asarray(targets, dtype=np.int64)]
    lse = rows.exp().sum(axis=-1, keepdims=True).log()
    loss = -((rows * Tensor(onehot, dtype=rows.dtype)).sum(axis=-1) - lse).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.data)


def eval_loss(model, ids, targets) -> float:
    from doraneural.dust import per_token_cross_entropy
    return float(per_token_cross_entropy(dust_forward(model, ids), targets).mean())


def trained_gradient_norms(args, ids, targets) -> List[Dict[str, object]]:
    """Per-site gradient norms at real trained weights, standard vs novel."""
    rows: List[Dict[str, object]] = []
    for novel in (False, True):
        model = build_body(novel, 1.0, args.dim, args.hidden, args.layers,
                           args.heads, args.kv_heads, args.vocab, args.seq_len, args.seed)
        opt = TensorAdamW(list(model.parameters()), lr=args.lr)
        done = 0
        for checkpoint in TRAIN_CHECKPOINTS:
            for _ in range(checkpoint - done):
                backprop_step(model, opt, ids, targets)
            done = checkpoint
            grads = backprop_gradient(model, ids, targets)
            rows.append({
                "model": "dora_novel" if novel else "standard",
                "steps": checkpoint,
                "grad_norm": {k: round(float(np.linalg.norm(v)), 10)
                              for k, v in sorted(grads.items())},
                "eval_loss": round(eval_loss(model, ids, targets), 4),
            })
    return rows


def dust_vs_backprop(args, ids, targets) -> List[Dict[str, object]]:
    """Dust estimate against backprop at w3, the site the KAN sits between."""
    rows: List[Dict[str, object]] = []
    for novel in (False, True):
        for scale in SCALES:
            model = build_body(novel, scale, args.dim, args.hidden, args.layers,
                               args.heads, args.kv_heads, args.vocab, args.seq_len, args.seed)
            truth = backprop_gradient(model, ids, targets)
            est = estimate_site_gradient(
                model, ids, targets, "w3", SITE_LAYER,
                DustConfig(draws=args.draws, seed=args.seed))
            g = truth[f"{SITE_LAYER}.w3"]
            d = est.grad
            denom = np.linalg.norm(g) * np.linalg.norm(d)
            cos = float(np.dot(g.ravel(), d.ravel()) / denom) if denom > 0 else 0.0
            rows.append({
                "model": "dora_novel" if novel else "standard",
                "weight_scale": scale,
                "grad_norm_backprop": round(float(np.linalg.norm(g)), 10),
                "grad_norm_dust": round(float(np.linalg.norm(d)), 8),
                "cosine": round(cos, 4),
            })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=128, help="Dust population K")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    ids, targets = token_window(args.vocab, args.seq_len)

    print("=== saturation of the Chebyshev KAN input |tanh(u)| ===")
    saturation: Dict[str, Dict[str, float]] = {}
    for scale in SCALES:
        model = build_body(True, scale, args.dim, args.hidden, args.layers,
                           args.heads, args.kv_heads, args.vocab, args.seq_len, args.seed)
        stats = kan_saturation(model, ids)
        saturation[str(scale)] = stats
        print(f"  scale={scale:5.1f}  mean={stats['mean']:.4f} p99={stats['p99']:.4f} "
              f"frac>0.99={stats['frac_gt_0.99']:.4f} frac>0.999={stats['frac_gt_0.999']:.5f}")

    print("\n=== gradient norms at trained weights (standard vs novel) ===")
    trained = trained_gradient_norms(args, ids, targets)
    for row in trained:
        norms = row["grad_norm"]
        print(f"  {row['model']:11s} steps={row['steps']:4d}  "
              f"w3={norms['0.w3']:.3e} w2={norms['0.w2']:.3e} "
              f"loss={row['eval_loss']:.4f}")

    print("\n=== Dust vs backprop at w3 as saturation worsens ===")
    comparison = dust_vs_backprop(args, ids, targets)
    for row in comparison:
        print(f"  {row['model']:11s} scale={row['weight_scale']:5.1f}  "
              f"|g_bp|={row['grad_norm_backprop']:.3e} "
              f"|g_dust|={row['grad_norm_dust']:.3e} cos={row['cosine']:+.4f}")

    # Headline: does the novel model collapse like the standard one?
    by_model = {r["model"]: r for r in trained if r["steps"] == TRAIN_CHECKPOINTS[-1]}
    collapse = {}
    for novel in (False, True):
        first = next(r for r in trained
                     if r["model"] == ("dora_novel" if novel else "standard")
                     and r["steps"] == 0)
        last = by_model["dora_novel" if novel else "standard"]
        key = "dora_novel" if novel else "standard"
        collapse[key] = {
            site: round(first["grad_norm"][site] / last["grad_norm"][site], 1)
            for site in last["grad_norm"]
        }
    print("\n=== gradient norm collapse over "
          f"{TRAIN_CHECKPOINTS[-1]} steps (first / last) ===")
    for key, sites in collapse.items():
        print(f"  {key:11s} " + "  ".join(f"{s}:{v:.0f}x" for s, v in sites.items()))

    report = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "saturation": saturation,
        "trained_gradient_norms": trained,
        "dust_vs_backprop_at_w3": comparison,
        "collapse_factor": collapse,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())