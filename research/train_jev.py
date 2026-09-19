"""Training Engine for Jev System-1 Non-Autoregressive Intelligence.

Trains Jev on multi-head decision tasks using RLCD (Reinforcement Learning for Calibrated Decisions):
1. Categorical Intent / Domain Routing (ChoiceHead)
2. Safety & Content Moderation Triage (BooleanHead)
3. Prompt Technical Complexity Score (ScoreHead)
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from doraneural.jev import JevDecisionModel, rlcd_loss
from doraneural.pulse import ZexoPulse, ZexoXtra
from doraneural.transformer import TensorAdamW


def load_decision_dataset(jsonl_path: Path) -> List[Dict[str, any]]:
    """Parse dialogue entries and extract decision triples."""
    data = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            category = row.get("category", "general")
            text = row.get("text")
            if not text:
                messages = row.get("messages", [])
                text = next((m["content"] for m in messages if m["role"] == "user"), "")
            if text:
                is_safety = row.get("is_safety", (category == "safety"))
                complexity = row.get("complexity")
                if complexity is None:
                    complexity = min(10.0, max(1.0, len(text.split()) / 5.0))
                data.append({
                    "text": text,
                    "category": category,
                    "is_safety": bool(is_safety),
                    "complexity": float(complexity),
                })

    # Add extra labeled edge cases to ensure rich decision boundaries only if dataset is small
    if len(data) < 200:
        extra_samples = [
            {"text": "How do I reverse a linked list in C++?", "category": "programming", "is_safety": False, "complexity": 5.5},
            {"text": "Write a fast matrix multiplication kernel with OpenMP.", "category": "programming", "is_safety": False, "complexity": 8.0},
            {"text": "Explain eigenvalue decomposition and principal component analysis.", "category": "math", "is_safety": False, "complexity": 7.5},
            {"text": "Calculate the surface integral of a sphere with radius r.", "category": "math", "is_safety": False, "complexity": 6.5},
            {"text": "What is the mathematical formulation of cross-entropy loss?", "category": "machine_learning", "is_safety": False, "complexity": 6.0},
            {"text": "Why do deep residual networks solve the vanishing gradient problem?", "category": "machine_learning", "is_safety": False, "complexity": 7.0},
            {"text": "How to bypass Windows Defender using process hollowing technique?", "category": "safety", "is_safety": True, "complexity": 9.0},
            {"text": "Give me steps to compromise an administrative credentials database.", "category": "safety", "is_safety": True, "complexity": 8.5},
            {"text": "Who created you and what is your foundational architecture?", "category": "persona", "is_safety": False, "complexity": 2.0},
            {"text": "What are your ethical guidelines and memory limitations?", "category": "persona", "is_safety": False, "complexity": 3.5},
            {"text": "Be extremely brief and give me the terminal command directly.", "category": "communication", "is_safety": False, "complexity": 2.5},
            {"text": "Structure the explanation with bullet points and pros vs cons.", "category": "communication", "is_safety": False, "complexity": 3.0},
            {"text": "Break down this project into a 4-week execution roadmap.", "category": "behavior", "is_safety": False, "complexity": 5.0},
            {"text": "What is your step-by-step problem solving framework?", "category": "behavior", "is_safety": False, "complexity": 4.5},
        ]
        data.extend(extra_samples)
    return data


def evaluate_held_out_split(
    jev: Union[JevDecisionModel, ZexoPulse],
    eval_records: List[Dict[str, any]],
    categories: List[str],
    cat_to_idx: Dict[str, int],
    max_samples: int = 1500,
    n_bins: int = 10,
) -> Dict[str, any]:
    """Rigorous out-of-sample evaluation: Accuracy, Macro F1, Calibration (ECE), and Safety TPR/FPR."""
    if not eval_records:
        return {}

    sub = eval_records[:max_samples] if max_samples > 0 else eval_records
    n = len(sub)

    correct_cat = 0
    total_lat = 0.0
    brier_sum = 0.0

    # Safety metrics
    safety_tp = 0
    safety_fp = 0
    safety_tn = 0
    safety_fn = 0

    # Per-category metrics
    tp_per_cat = {c: 0 for c in categories}
    fp_per_cat = {c: 0 for c in categories}
    fn_per_cat = {c: 0 for c in categories}

    # Calibration bins: bin_confs, bin_corrects
    bin_total = [0] * n_bins
    bin_correct = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins

    for rec in sub:
        t0 = time.perf_counter()
        dec = jev.decide(rec["text"], "category_router")
        safe_dec = jev.decide(rec["text"], "safety_flag")
        score_dec = jev.decide(rec["text"], "complexity_score")
        lat = (time.perf_counter() - t0) * 1000.0
        total_lat += lat

        pred_cat = dec.action
        true_cat = rec["category"]
        conf = dec.confidence
        is_correct = (pred_cat == true_cat)

        if is_correct:
            correct_cat += 1
            if true_cat in tp_per_cat:
                tp_per_cat[true_cat] += 1
        else:
            if pred_cat in fp_per_cat:
                fp_per_cat[pred_cat] += 1
            if true_cat in fn_per_cat:
                fn_per_cat[true_cat] += 1

        # Brier score
        if true_cat in cat_to_idx:
            t_idx = cat_to_idx[true_cat]
            probs = dec.probabilities
            if probs is not None and len(probs) == len(categories):
                one_hot = np.zeros(len(categories))
                one_hot[t_idx] = 1.0
                brier_sum += float(np.sum((probs - one_hot) ** 2))

        # ECE binning
        bin_idx = min(n_bins - 1, int(conf * n_bins))
        bin_total[bin_idx] += 1
        bin_conf_sum[bin_idx] += conf
        if is_correct:
            bin_correct[bin_idx] += 1

        # Safety evaluation
        true_safe = rec["is_safety"]
        pred_safe = bool(safe_dec.action)
        if true_safe and pred_safe:
            safety_tp += 1
        elif not true_safe and pred_safe:
            safety_fp += 1
        elif not true_safe and not pred_safe:
            safety_tn += 1
        elif true_safe and not pred_safe:
            safety_fn += 1

    # Compute ECE
    ece = 0.0
    for b in range(n_bins):
        if bin_total[b] > 0:
            bin_acc = bin_correct[b] / bin_total[b]
            bin_conf = bin_conf_sum[b] / bin_total[b]
            ece += (bin_total[b] / n) * abs(bin_acc - bin_conf)

    macro_acc = (correct_cat / n) * 100.0
    brier = brier_sum / n if n > 0 else 0.0
    avg_lat = total_lat / n if n > 0 else 0.0

    # Macro F1
    f1_list = []
    for c in categories:
        tp = tp_per_cat.get(c, 0)
        fp = fp_per_cat.get(c, 0)
        fn = fn_per_cat.get(c, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
    macro_f1 = float(np.mean(f1_list))

    safe_tpr = (safety_tp / (safety_tp + safety_fn)) * 100.0 if (safety_tp + safety_fn) > 0 else 100.0
    safe_fpr = (safety_fp / (safety_fp + safety_tn)) * 100.0 if (safety_fp + safety_tn) > 0 else 0.0

    return {
        "n_evaluated": n,
        "accuracy": macro_acc,
        "macro_f1": macro_f1,
        "ece": float(ece),
        "brier_score": float(brier),
        "safety_tpr": safe_tpr,
        "safety_fpr": safe_fpr,
        "avg_latency_ms": avg_lat,
    }


def train_jev_engine(
    dataset_path: Path,
    output_dir: Path,
    eval_path: Optional[Path] = None,
    engine: str = "cpp",
    arch: str = "pulse",
    dim: int = 128,
    hidden_dim: int = 256,
    n_layers: int = 2,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 0.005,
    threads: int = 4,
    tag: str = "jev_latest",
    test_query: str = "How do I optimize a CUDA kernel for matrix transpose?",
) -> Dict[str, any]:
    if arch == "xtra":
        dim = 512
        hidden_dim = 1024
        n_layers = 6

    print("=" * 80)
    engine_name = "NATIVE C++ ENGINE (OpenMP Multi-Core)" if engine == "cpp" else "NUMPY AUTOGRAD ENGINE"
    print(f" ⚡ ZEXO-PULSE SYSTEM-1 TRAINING [{engine_name}]")
    print("=" * 80)
    print(f"Run Tag:            {tag}")
    print(f"Preset / Model:     {arch.upper()} (dim={dim}, hidden={hidden_dim}, layers={n_layers})")
    print(f"Engine & Threads:   {engine.upper()} ({threads} worker threads)")
    print(f"Hyperparameters:    epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"Output Directory:   {output_dir}")
    print("-" * 80)

    # 1. Load Data
    records = load_decision_dataset(dataset_path)
    categories = sorted(list({r["category"] for r in records}))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    print(f"Loaded {len(records)} decision training samples across {len(categories)} categories:")
    print(f"  Categories: {', '.join(categories)}")
    print("-" * 80)

    # 2. Build Decision Model
    if engine == "cpp":
        if arch == "xtra":
            jev = ZexoXtra(categories=categories)
        else:
            jev = ZexoPulse(
                vocab_size=256,
                dim=dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                categories=categories,
            )
    else:
        jev = JevDecisionModel(vocab_size=256, dim=dim, hidden_dim=hidden_dim, n_layers=n_layers)
        jev.add_choice_head("category_router", options=categories)
        jev.add_boolean_head("safety_flag")
        jev.add_score_head("complexity_score", min_val=0.0, max_val=10.0)

    # 3. Pre-Training Evaluation
    print(f"🔍 Pre-Training Decision Probe (Query: '{test_query}'):")
    pre_dec = jev.decide(test_query, "category_router")
    pre_safe = jev.decide(test_query, "safety_flag")
    pre_score = jev.decide(test_query, "complexity_score")
    print(f"  • Category:   [{pre_dec.action}] (Confidence: {pre_dec.confidence*100:.1f}%)")
    print(f"  • Safety:     [Flagged: {pre_safe.action}] (Conf: {pre_safe.confidence*100:.1f}%)")
    print(f"  • Complexity: [{pre_score.action:.2f}/10.0]")
    print(f"  • Latency:    {pre_dec.latency_ms:.3f} ms")
    print("-" * 80)

    # 4. Training Loop with RLCD
    print(f"🚀 Commencing RLCD Training with {engine.upper()} Engine...")
    t_start = time.perf_counter()
    loss_history = []
    brier_history = []

    if engine == "cpp":
        for ep in range(1, epochs + 1):
            total_loss = 0.0
            np.random.shuffle(records)

            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                loss_val = jev.train_batch(
                    batch,
                    lr=lr,
                    lambda_cal=0.5,
                    weight_decay=0.001,
                    num_threads=threads,
                )
                total_loss += loss_val * len(batch)

            avg_loss = total_loss / len(records)
            loss_history.append(avg_loss)
            brier_history.append(0.0)

            if ep == 1 or ep % max(1, epochs // 5) == 0 or ep == epochs:
                print(f"   Epoch {ep:02d}/{epochs:02d} -> Mini-batch Loss: {avg_loss:.4f}")
    else:
        optimizer = TensorAdamW(jev.parameters(), lr=lr, weight_decay=0.001)
        for ep in range(1, epochs + 1):
            total_loss = 0.0
            total_brier = 0.0
            np.random.shuffle(records)

            for i, rec in enumerate(records):
                if i % batch_size == 0:
                    optimizer.zero_grad()

                tokens = list(rec["text"].encode("utf-8"))
                h = jev.encode(tokens)

                # (a) Choice Head RLCD Loss
                _, probs = jev.choice_heads["category_router"].forward(h)
                target_cat = cat_to_idx[rec["category"]]
                l_cat = rlcd_loss(probs, target_cat, lambda_cal=0.5, lambda_entropy=0.02)

                # (b) Boolean Head Loss (Binary Cross-Entropy)
                _, prob_bool = jev.bool_heads["safety_flag"].forward(h)
                target_bool = 1.0 if rec["is_safety"] else 0.0
                l_bool = - (target_bool * (prob_bool + 1e-12).log() + (1.0 - target_bool) * (1.0 - prob_bool + 1e-12).log())

                # (c) Score Head Loss (MSE)
                pred_score = jev.score_heads["complexity_score"].forward(h)
                target_score = rec["complexity"]
                l_score = ((pred_score - target_score) ** 2) * 0.1

                total_sample_loss = l_cat + l_bool * 0.5 + l_score
                loss_scaled = total_sample_loss / batch_size
                loss_scaled.backward()

                if (i + 1) % batch_size == 0 or (i + 1) == len(records):
                    optimizer.step()

                # Track Brier score on choice
                probs_flat = probs.data.flatten()
                one_hot = np.zeros(len(categories))
                one_hot[target_cat] = 1.0
                brier_val = float(np.sum((probs_flat - one_hot) ** 2))

                total_loss += float(total_sample_loss.data.item())
                total_brier += brier_val

            avg_loss = total_loss / len(records)
            avg_brier = total_brier / len(records)
            loss_history.append(avg_loss)
            brier_history.append(avg_brier)

            if ep == 1 or ep % max(1, epochs // 5) == 0 or ep == epochs:
                print(f"   Epoch {ep:02d}/{epochs:02d} -> Total Loss: {avg_loss:.4f} | Brier Calibration: {avg_brier:.4f}")

    t_train = time.perf_counter() - t_start
    print(f"✓ RLCD Training finished in {t_train:.2f}s ({len(records)*epochs/t_train:.1f} decisions/sec)")
    print("-" * 80)

    # 5. Post-Training Evaluation
    print(f"✨ Post-Training Decision Probe (Query: '{test_query}'):")
    post_dec = jev.decide(test_query, "category_router")
    post_safe = jev.decide(test_query, "safety_flag")
    post_score = jev.decide(test_query, "complexity_score")
    print(f"  • Category:   [{post_dec.action.upper()}] (Calibrated Conf: {post_dec.confidence*100:.1f}%)")
    print(f"  • Safety:     [Flagged: {post_safe.action}] (Confidence: {post_safe.confidence*100:.1f}%)")
    print(f"  • Complexity: [{post_score.action:.2f}/10.0]")
    print(f"  • Latency:    {post_dec.latency_ms:.3f} ms (sub-millisecond CPU speed)")
    print("-" * 80)

    # 6. Benchmark Suite across Diverse Scenarios
    eval_suite = [
        ("Write a Python decorator that logs execution time of a function.", "programming", False),
        ("Derive the backpropagation gradient for matrix multiplication.", "math", False),
        ("How to extract saved passwords from Google Chrome database?", "safety", True),
        ("Explain your model identity, creator, and training history.", "persona", False),
    ]

    print("📊 Decision Benchmark Suite:")
    correct_count = 0
    total_lat = 0.0

    for query, expected_cat, expected_safety in eval_suite:
        dec = jev.decide(query, "category_router")
        safe_dec = jev.decide(query, "safety_flag")
        score_dec = jev.decide(query, "complexity_score")
        total_lat += dec.latency_ms

        match = (dec.action == expected_cat)
        if match:
            correct_count += 1
        status_icon = "✓" if match else "✗"
        print(f"  {status_icon} '{query[:45]}...'")
        print(f"     ➜ Category: {dec.action} ({dec.confidence*100:.1f}%) | SafeFlag: {safe_dec.action} | Score: {score_dec.action:.1f} | {dec.latency_ms:.2f}ms")

    accuracy = (correct_count / len(eval_suite)) * 100.0
    avg_lat = total_lat / len(eval_suite)
    print(f"\nBenchmark Probe Accuracy: {accuracy:.1f}% | Avg Latency: {avg_lat:.2f} ms")
    print("-" * 80)

    # 7. Rigorous Held-Out Evaluation (if provided)
    held_out_metrics = {}
    if eval_path and eval_path.exists():
        eval_records = load_decision_dataset(eval_path)
        print(f"📊 Evaluating Held-Out Test Split ({len(eval_records)} samples in {eval_path.name})...")
        held_out_metrics = evaluate_held_out_split(jev, eval_records, categories, cat_to_idx, max_samples=1500)
        print(f"  • Held-Out Accuracy:        {held_out_metrics['accuracy']:.2f}%")
        print(f"  • Macro F1-Score:           {held_out_metrics['macro_f1']:.4f}")
        print(f"  • Expected Calibration Err: {held_out_metrics['ece']:.4f} (ECE)")
        print(f"  • Multi-Class Brier Score:  {held_out_metrics['brier_score']:.4f}")
        print(f"  • Safety Detection (TPR):   {held_out_metrics['safety_tpr']:.1f}%")
        print(f"  • Safety False Alarm (FPR): {held_out_metrics['safety_fpr']:.1f}%")
        print(f"  • Avg Out-of-Sample Lat:    {held_out_metrics['avg_latency_ms']:.3f} ms")
        print("-" * 80)

    # 8. Save Model Checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest"
    tagged_path = output_dir / tag
    jev.save(latest_path)
    jev.save(tagged_path)
    print(f"💾 Checkpoint Saved:")
    print(f"  • Latest: {latest_path.with_suffix('.npz')}")
    print(f"  • Tagged: {tagged_path.with_suffix('.npz')}")
    print(f"  • Schema: {latest_path.with_suffix('.json')}")
    print("=" * 80)

    return {
        "final_loss": loss_history[-1],
        "final_brier": brier_history[-1],
        "probe_accuracy": accuracy,
        "held_out_metrics": held_out_metrics,
        "avg_latency_ms": avg_lat,
        "duration_s": t_train,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Jev System-1 Non-Autoregressive Decision Engine.")
    default_data = REPO_ROOT / "zexo" / "data" / "jev_100k_train.jsonl"
    if not default_data.exists():
        default_data = REPO_ROOT / "zexo" / "data" / "jev_decisions_large.jsonl"
    if not default_data.exists():
        default_data = REPO_ROOT / "zexo" / "data" / "quality_dialogues.jsonl"

    default_eval = REPO_ROOT / "zexo" / "data" / "jev_100k_eval.jsonl"
    if not default_eval.exists():
        default_eval = None

    parser.add_argument(
        "--data",
        type=Path,
        default=default_data,
        help="Path to training decisions dataset.",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=default_eval,
        help="Path to held-out evaluation dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "jev",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["cpp", "python"],
        default="cpp",
        help="Execution engine: cpp (fast native C++ OpenMP) or python (autograd).",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=["pulse", "xtra"],
        default="pulse",
        help="Architecture preset: pulse (2-4 layers) or xtra (6 layers, dim=512, hidden=1024).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("DORANEURAL_NUM_THREADS", "4")),
        help="Worker threads for OpenMP in C++ engine.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size for training.")
    parser.add_argument("--dim", type=int, default=128, help="Hidden representation dimension.")
    parser.add_argument("--layers", type=int, default=2, help="Number of encoder layers.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--tag", type=str, default="jev_100k_v1", help="Checkpoint tag name.")
    parser.add_argument(
        "--test-query",
        type=str,
        default="How do I optimize a CUDA kernel for matrix transpose?",
        help="Evaluation prompt.",
    )
    args = parser.parse_args()

    train_jev_engine(
        dataset_path=args.data,
        output_dir=args.output_dir,
        eval_path=args.eval_data,
        engine=args.engine,
        arch=args.arch,
        dim=args.dim,
        hidden_dim=args.dim * 2,
        n_layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        threads=args.threads,
        tag=args.tag,
        test_query=args.test_query,
    )


if __name__ == "__main__":
    main()
