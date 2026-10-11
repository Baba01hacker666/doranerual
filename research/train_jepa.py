"""Train and evaluate the JEPA family on CPU (I-JEPA, V-JEPA, latent world model).

Three runnable experiments, each with the baselines needed to interpret the
numbers honestly:

``--task images``
    I-JEPA masked-latent pretraining on synthetic shape/grating images, then a
    frozen-feature linear probe, collapse diagnostics, and the positional-
    shortcut audit (``masked_target_baseline``). The random-initialisation probe
    is always reported next to the trained one, because a JEPA that learns
    nothing looks identical to a JEPA that works unless you check.

``--task video``
    V-JEPA tube-masked pretraining on moving-dot clips, probed for motion
    direction.

``--task worldmodel``
    Action-conditioned latent world model trained end-to-end from pixels on a
    bouncing point-mass environment (SIGReg anti-collapse, no EMA teacher), then
    evaluated with (a) next-latent error against no-change and mean-latent
    baselines, (b) position/velocity probes of the frozen latents, and (c) a
    cross-entropy-method planner scored on the real environment against random
    actions.

Examples
--------
    python3 research/train_jepa.py --task all --quick
    python3 research/train_jepa.py --task images --dataset grating --steps 800
    python3 research/train_jepa.py --task worldmodel --steps 3000 --json results.json
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from doraneural.jepa import (  # noqa: E402
    IJEPA,
    JEPAWorldModel,
    VJEPA,
    linear_probe,
    masked_target_baseline,
    representation_diagnostics,
)
from doraneural.jepa_data import (  # noqa: E402
    PointMassConfig,
    PointMassEnv,
    make_grating_dataset,
    make_motion_clip_dataset,
    make_shape_dataset,
    sample_point_mass_trajectories,
)

CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "jepa"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _banner(title: str) -> None:
    _log("=" * 78)
    _log(f" {title}")
    _log("=" * 78)


# ---------------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------------

def diagnose(model, images: np.ndarray) -> Dict[str, Any]:
    feats = model.encode(images)
    diag = representation_diagnostics(feats)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in diag.items()}


def probe_report(model, images: np.ndarray, labels: np.ndarray, num_classes: int,
                 split: float = 0.75) -> Dict[str, Any]:
    """Linear-probe accuracy of trained vs randomly initialised features."""
    feats = model.encode(images)
    n_train = int(len(images) * split)
    trained = linear_probe(feats[:n_train], labels[:n_train], feats[n_train:], labels[n_train:],
                           num_classes=num_classes)
    return {
        "trained_accuracy": round(float(trained["accuracy"]), 4),
        "trained_macro_f1": round(float(trained["macro_f1"]), 4),
        "chance": round(1.0 / num_classes, 4),
    }


def random_init_probe(model_factory, images: np.ndarray, labels: np.ndarray, num_classes: int,
                      split: float = 0.75) -> float:
    np.random.seed(0)
    fresh = model_factory()
    feats = fresh.encode(images)
    n_train = int(len(images) * split)
    res = linear_probe(feats[:n_train], labels[:n_train], feats[n_train:], labels[n_train:],
                       num_classes=num_classes)
    return round(float(res["accuracy"]), 4)


# ---------------------------------------------------------------------------
# Experiment 1: I-JEPA on images
# ---------------------------------------------------------------------------

def run_images(args: argparse.Namespace) -> Dict[str, Any]:
    banner = "I-JEPA: MASKED LATENT PREDICTION ON SYNTHETIC IMAGES"
    _banner(banner)
    if args.dataset == "grating":
        data = make_grating_dataset(n=args.n_images, img_size=32, seed=0)
        labels, num_classes, label_name = data["labels"], 4, "grating orientation"
    else:
        data = make_shape_dataset(n=args.n_images, img_size=32, seed=0)
        labels, num_classes, label_name = data["labels"], 3, "shape identity"

    def factory():
        return IJEPA(
            img_size=32, patch_size=args.patch_size, in_chans=1, dim=args.dim, depth=args.depth,
            n_heads=args.heads, pred_dim=args.dim, pred_depth=args.pred_depth, pred_heads=args.heads,
            ema_start=args.ema_start, ema_end=args.ema_end, pos_embed=args.pos_embed,
            pos_scale=args.pos_scale, normalize_targets=not args.raw_targets,
            target_mode=args.target_mode,
        )

    model = factory()
    n_params = sum(p.data.size for p in model.parameters())
    _log(f"dataset      : {args.dataset} ({len(labels)} images, {num_classes} classes: {label_name})")
    _log(f"architecture : patch={args.patch_size} dim={args.dim} depth={args.depth} "
         f"pred_dim={args.dim} pred_depth={args.pred_depth}  ({n_params:,} params)")
    _log(f"objective    : multi-block masking, EMA teacher, smooth-L1 in latent space")
    _log(f"training     : {args.steps} steps, batch {args.batch_size}, lr {args.lr}, "
         f"sigreg={'on' if args.sigreg else 'off'}")

    baseline = random_init_probe(factory, data["images"], labels, num_classes)
    t0 = time.perf_counter()
    stats = model.fit(
        data["images"], steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, verbose_every=max(1, args.steps // 5),
        rng=np.random.default_rng(0), sigreg=args.sigreg, sigreg_weight=args.sigreg_weight,
        sigreg_level=args.sigreg_level,
    )
    train_secs = time.perf_counter() - t0

    probe = probe_report(model, data["images"], labels, num_classes)
    diag = diagnose(model, data["images"])
    shortcut = masked_target_baseline(model, data["images"][:128], rng=np.random.default_rng(1))
    shortcut = {k: round(float(v), 4) for k, v in shortcut.items()}

    _log("")
    _log(f"  train seconds        : {train_secs:.1f}")
    _log(f"  final latent loss    : {stats['final_loss']:.4f}")
    _log(f"  probe (random init)  : {baseline:.3f}   <- what 'no learning' looks like")
    _log(f"  probe (I-JEPA)       : {probe['trained_accuracy']:.3f}   (chance {probe['chance']:.3f})")
    _log(f"  collapse diagnostics : rank={diag['effective_rank']} signal_ratio={diag['signal_ratio']} "
         f"cosine={diag['mean_cosine']} collapsed={diag['collapsed']}")
    _log(f"  positional shortcut  : position_r2={shortcut['position_r2']} "
         f"(model_mse={shortcut['model_mse']} vs position_mse={shortcut['position_mse']})")
    _log("  interpretation: position_r2 near 1 means the latent target is a lookup table over")
    _log("                  positions - the loss can reach ~0 while the features stay useless.")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"ijepa_{args.dataset}"
    model.save(ckpt)
    _log(f"  checkpoint           : {ckpt.with_suffix('.npz').relative_to(REPO_ROOT)}")

    return {
        "task": "images",
        "dataset": args.dataset,
        "steps": args.steps,
        "params": n_params,
        "train_seconds": round(train_secs, 2),
        "final_loss": round(stats["final_loss"], 4),
        "probe_random_init": baseline,
        "probe_jepa": probe["trained_accuracy"],
        "probe_jepa_macro_f1": probe["trained_macro_f1"],
        "chance": probe["chance"],
        "diagnostics": diag,
        "shortcut_audit": shortcut,
    }


# ---------------------------------------------------------------------------
# Experiment 2: V-JEPA on video
# ---------------------------------------------------------------------------

def run_video(args: argparse.Namespace) -> Dict[str, Any]:
    _banner("V-JEPA: TUBE-MASKED LATENT PREDICTION ON MOVING-DOT CLIPS")
    data = make_motion_clip_dataset(
        n=args.n_clips, frames=args.frames, img_size=32, seed=0, task=args.video_task
    )
    clips, labels = data["clips"], data["labels"]
    is_order = args.video_task == "order"
    num_classes = 2 if is_order else 4
    label_name = "temporal order (forward/reversed)" if is_order else "motion direction"

    def factory():
        return VJEPA(
            frames=args.frames, img_size=32, patch_size=args.patch_size, tubelet=args.tubelet,
            in_chans=1, dim=args.dim, depth=args.depth, n_heads=args.heads, pred_dim=args.dim,
            pred_depth=args.pred_depth, pred_heads=args.heads, ema_start=args.ema_start,
            ema_end=args.ema_end, target_mode=args.target_mode,
        )

    model = factory()
    n_params = sum(p.data.size for p in model.parameters())
    _log(f"dataset      : {len(clips)} clips of {args.frames} frames, label = {label_name}")
    _log(f"architecture : tubelet={args.tubelet} patch={args.patch_size} dim={args.dim} "
         f"depth={args.depth} ({n_params:,} params)")

    baseline = random_init_probe(factory, clips, labels, num_classes)
    t0 = time.perf_counter()
    stats = model.fit(clips, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.weight_decay, verbose_every=max(1, args.steps // 5),
                      rng=np.random.default_rng(0), sigreg=args.sigreg,
                      sigreg_level=args.sigreg_level)
    secs = time.perf_counter() - t0

    probe = probe_report(model, clips, labels, num_classes)
    diag = diagnose(model, clips)
    shortcut = masked_target_baseline(model, clips[:64], rng=np.random.default_rng(1))
    shortcut = {k: round(float(v), 4) for k, v in shortcut.items()}

    _log("")
    _log(f"  train seconds        : {secs:.1f}")
    _log(f"  final latent loss    : {stats['final_loss']:.4f}")
    _log(f"  {label_name} probe, random init: {baseline:.3f}")
    _log(f"  {label_name} probe, V-JEPA    : {probe['trained_accuracy']:.3f}  "
         f"(chance {probe['chance']:.3f})")
    _log(f"  collapse diagnostics : rank={diag['effective_rank']} signal_ratio={diag['signal_ratio']} "
         f"collapsed={diag['collapsed']}")
    _log(f"  positional shortcut  : position_r2={shortcut['position_r2']}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / "vjepa_motion"
    model.save(ckpt)
    return {
        "task": "video",
        "steps": args.steps,
        "params": n_params,
        "train_seconds": round(secs, 2),
        "final_loss": round(stats["final_loss"], 4),
        "probe_random_init": baseline,
        "probe_vjepa": probe["trained_accuracy"],
        "chance": probe["chance"],
        "diagnostics": diag,
        "shortcut_audit": shortcut,
    }


# ---------------------------------------------------------------------------
# Experiment 3: action-conditioned latent world model + CEM planner
# ---------------------------------------------------------------------------

def predict_baselines(model: JEPAWorldModel, obs: np.ndarray, actions: np.ndarray) -> Dict[str, float]:
    """Next-latent MSE of the model vs. no-change and mean-latent predictors."""
    z = model.encode_np(obs.reshape(-1, *model.obs_shape)).reshape(len(obs), obs.shape[1], -1)
    model_err, nochange_err, mean_err = [], [], []
    mean_latent = z.mean(axis=(0, 1))
    for t in range(obs.shape[1] - 1):
        pred = model.rollout(z[:, t], actions[:, t : t + 1])[0]
        target = z[:, t + 1]
        model_err.append(float(np.mean((pred - target) ** 2)))
        nochange_err.append(float(np.mean((z[:, t] - target) ** 2)))
        mean_err.append(float(np.mean((mean_latent[None] - target) ** 2)))
    report = {
        "model_mse": round(float(np.mean(model_err)), 4),
        "nochange_mse": round(float(np.mean(nochange_err)), 4),
        "mean_latent_mse": round(float(np.mean(mean_err)), 4),
    }
    # MSE alone cannot tell a learned dynamics from shrinkage towards the
    # marginal: separate the two with the action-permutation ablation.
    report.update(model.action_utilization(obs, actions, seed=0))
    return report


def planner_eval(model: JEPAWorldModel, config: PointMassConfig, episodes: int = 40,
                 horizon: int = 6, seed: int = 0, success_radius: float = 0.12,
                 n_samples: int = 192, n_iters: int = 5) -> Dict[str, float]:
    """Plan in latent space with CEM, execute the actions, score against random.

    The random baseline is rolled out from the *same* start/goal pair as the
    plan, so the two distances are paired and ``improved_fraction`` is a genuine
    win-rate. Success counts alone are too coarse to trust at this budget: a
    planner that beats the random baseline on distance can still post the same
    (usually zero) success count.
    """
    env = PointMassEnv(config, seed=seed)
    rng = np.random.default_rng(seed)
    planned_success = random_success = 0
    planned_dist, random_dist, improved = [], [], []
    for i in range(episodes):
        start = rng.uniform(0.15, 0.85, 2)
        goal = rng.uniform(0.1, 0.9, 2)
        while np.linalg.norm(start - goal) < 0.35:
            goal = rng.uniform(0.1, 0.9, 2)

        env.reset(position=start, velocity=[0.0, 0.0], goal=goal)
        z_start = model.encode_np(env.render()[None])
        env.reset(position=goal, velocity=[0.0, 0.0], goal=goal)
        z_goal = model.encode_np(env.render()[None])

        plan = model.plan(z_start, z_goal, horizon=horizon, n_samples=n_samples,
                          n_elite=max(8, n_samples // 10), n_iters=n_iters,
                          rng=np.random.default_rng(seed + i))
        env.reset(position=start, velocity=[0.0, 0.0], goal=goal)
        info = {"distance": float(np.linalg.norm(start - goal))}
        for action in plan["actions"]:
            _, info = env.step(action)
        planned_dist.append(info["distance"])
        planned_success += int(info["distance"] < success_radius)

        env.reset(position=start, velocity=[0.0, 0.0], goal=goal)
        for _ in range(horizon):
            _, info = env.step(rng.uniform(-1.0, 1.0, 2))
        random_dist.append(info["distance"])
        random_success += int(info["distance"] < success_radius)
        improved.append(float(planned_dist[-1] < random_dist[-1]))

    return {
        "episodes": episodes,
        "horizon": horizon,
        "planned_success_rate": round(planned_success / episodes, 4),
        "random_success_rate": round(random_success / episodes, 4),
        "planned_mean_distance": round(float(np.mean(planned_dist)), 4),
        "random_mean_distance": round(float(np.mean(random_dist)), 4),
        "improved_fraction": round(float(np.mean(improved)), 4),
        "mean_improvement": round(float(np.mean(random_dist) - np.mean(planned_dist)), 4),
    }


def run_world_model(args: argparse.Namespace) -> Dict[str, Any]:
    _banner("ACTION-CONDITIONED JEPA WORLD MODEL (LEWM / V-JEPA 2-AC STYLE)")
    cfg = PointMassConfig(size=24)
    train = sample_point_mass_trajectories(
        n=args.n_trajectories, horizon=args.horizon, config=cfg, seed=0, goal_seek_prob=0.5
    )
    test = sample_point_mass_trajectories(
        n=max(16, args.n_trajectories // 6), horizon=args.horizon, config=cfg, seed=99,
        goal_seek_prob=0.5,
    )
    model = JEPAWorldModel(
        obs_shape=(1, 24, 24), action_dim=2, dim=args.dim, encoder_depth=args.depth,
        encoder_heads=args.heads, patch_size=args.patch_size, predictor_dim=args.dim,
        predictor_depth=args.pred_depth, predictor_heads=args.heads, context_len=4,
        freeze_encoder=args.freeze_encoder, sigreg_slices=32, residual=not args.no_residual, seed=0,
    )
    n_params = sum(p.data.size for p in model.parameters())
    _log(f"environment  : point mass in a box, {cfg.size}x{cfg.size} pixel observations")
    _log(f"data         : {len(train['obs'])} train / {len(test['obs'])} test trajectories "
         f"of {args.horizon} steps")
    _log(f"architecture : encoder dim={args.dim} depth={args.depth} patch={args.patch_size}, "
         f"predictor depth={args.pred_depth} ({n_params:,} params)")
    _log(f"objective    : {'change' if not args.no_residual else 'direct'}-of-latent smooth-L1 "
         f"+ SIGReg (weight {args.sigreg_weight}, rollout steps {args.rollout_steps})")
    _log(f"encoder      : {'frozen' if args.freeze_encoder else 'trained end-to-end from pixels'}")

    t0 = time.perf_counter()
    stats = model.fit(
        train["obs"], train["actions"], steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, sigreg_weight=args.sigreg_weight,
        rollout_steps=args.rollout_steps, verbose_every=max(1, args.steps // 5),
        rng=np.random.default_rng(0),
    )
    secs = time.perf_counter() - t0

    baselines = predict_baselines(model, test["obs"][:32], test["actions"][:32])
    z_early = model.encode_np(test["obs"][:, 0])
    z_mid = model.encode_np(test["obs"][:, args.horizon // 2])
    pos_probe = linear_probe(model.encode_np(train["obs"][:, 0]), train["positions"][:, 0],
                             z_early, test["positions"][:, 0], mode="regression")
    pos_probe_mid = linear_probe(model.encode_np(train["obs"][:, args.horizon // 2]),
                                 train["positions"][:, args.horizon // 2],
                                 z_mid, test["positions"][:, args.horizon // 2], mode="regression")
    vel_probe = linear_probe(model.encode_np(train["obs"][:, args.horizon // 2]),
                             train["velocities"][:, args.horizon // 2],
                             z_mid, test["velocities"][:, args.horizon // 2], mode="regression")
    all_latents = model.encode_np(train["obs"].reshape(-1, 1, 24, 24))
    diag = representation_diagnostics(all_latents)
    diag = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in diag.items()}
    plan = planner_eval(model, cfg, episodes=args.plan_episodes, seed=0)
    sigreg_value = model.sigreg_value(train["obs"].reshape(-1, 1, 24, 24), sample=256)

    _log("")
    _log(f"  train seconds        : {secs:.1f}")
    _log(f"  SIGReg statistic     : {sigreg_value:.3f}  (isotropic Gaussian cloud -> small)")
    _log(f"  latent prediction MSE: model {baselines['model_mse']} | no-change "
         f"{baselines['nochange_mse']} | mean-latent {baselines['mean_latent_mse']}")
    _log(f"  frozen-latent probes : position R2 {pos_probe['r2']:.3f} (t=0), "
         f"{pos_probe_mid['r2']:.3f} (t=mid); velocity R2 {vel_probe['r2']:.3f}")
    _log(f"  collapse diagnostics : rank={diag['effective_rank']} signal_ratio={diag['signal_ratio']} "
         f"collapsed={diag['collapsed']}")
    _log(f"  CEM planning         : success {plan['planned_success_rate']:.2f} vs random "
         f"{plan['random_success_rate']:.2f} | final distance {plan['planned_mean_distance']:.3f} "
         f"vs {plan['random_mean_distance']:.3f}")

    _log("  action ablation      : one-step MSE true %.4f | shuffled %.4f | zeroed %.4f (gain %.3f)"
         % (baselines['one_step_true_action_mse'], baselines['one_step_shuffled_action_mse'],
            baselines['one_step_zero_action_mse'], baselines['one_step_action_gain']))
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / "world_model"
    model.save(ckpt)
    _log(f"  checkpoint           : {ckpt.with_suffix('.npz').relative_to(REPO_ROOT)}")

    return {
        "task": "worldmodel",
        "steps": args.steps,
        "params": n_params,
        "train_seconds": round(secs, 2),
        "sigreg_statistic": round(sigreg_value, 4),
        "prediction": baselines,
        "probe_position_r2_t0": round(float(pos_probe["r2"]), 4),
        "probe_position_r2_mid": round(float(pos_probe_mid["r2"]), 4),
        "probe_velocity_r2_mid": round(float(vel_probe["r2"]), 4),
        "diagnostics": diag,
        "planning": plan,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate JEPA models (I-JEPA, V-JEPA, latent world model) on CPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=["images", "video", "worldmodel", "all"], default="all")
    parser.add_argument("--dataset", choices=["shape", "grating"], default="grating",
                        help="synthetic image dataset for --task images")
    parser.add_argument("--steps", type=int, default=600, help="optimizer steps for masked pretraining")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--pred-depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--tubelet", type=int, default=2)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--video-task", choices=["order", "direction"], default="order",
                        help="temporal order is the metric a static encoder cannot solve")
    parser.add_argument("--n-images", type=int, default=256)
    parser.add_argument("--n-clips", type=int, default=192)
    parser.add_argument("--n-trajectories", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--plan-episodes", type=int, default=24)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--no-residual", action="store_true",
                        help="predict the next latent directly instead of predicting its change "
                             "(the absolute objective has a shrinkage optimum that can ignore actions)")
    parser.add_argument("--ema-start", type=float, default=0.9,
                        help="EMA momentum at step 0; small-scale runs need a fast teacher")
    parser.add_argument("--ema-end", type=float, default=0.99)
    parser.add_argument("--pos-embed", default="sincos", choices=["sincos", "rope2d", "rope3d", "learned", "none"])
    parser.add_argument("--pos-scale", type=float, default=1.0)
    parser.add_argument("--raw-targets", action="store_true",
                        help="disable per-token target normalisation")
    parser.add_argument("--sigreg", action="store_true",
                        help="use the LeJEPA objective (SIGReg regulariser, no EMA teacher)")
    parser.add_argument("--sigreg-weight", type=float, default=0.1)
    parser.add_argument("--sigreg-level", choices=["token", "image"], default="token",
                        help="regularise the token cloud or the pooled per-image latents "
                             "(image level is what rules out a position-only encoder)")
    parser.add_argument("--target-mode", choices=["tokens", "global"], default="tokens",
                        help="predict masked token latents or the whole-image latent")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="keep the world-model encoder frozen (V-JEPA 2-AC style fine-tuning)")
    parser.add_argument("--quick", action="store_true", help="tiny fast run used by CI")
    parser.add_argument("--json", type=str, default=None, help="write metrics to this JSON file")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def apply_quick(args: argparse.Namespace) -> argparse.Namespace:
    args.steps = min(args.steps, 120)
    args.n_images = min(args.n_images, 96)
    args.n_clips = min(args.n_clips, 64)
    args.n_trajectories = min(args.n_trajectories, 64)
    args.horizon = min(args.horizon, 8)
    args.plan_episodes = min(args.plan_episodes, 8)
    args.batch_size = min(args.batch_size, 8)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args = apply_quick(args)
    np.random.seed(args.seed)

    results: Dict[str, Any] = {"config": vars(args), "results": []}
    tasks = ["images", "video", "worldmodel"] if args.task == "all" else [args.task]
    for task in tasks:
        if task == "images":
            results["results"].append(run_images(args))
        elif task == "video":
            results["results"].append(run_video(args))
        elif task == "worldmodel":
            results["results"].append(run_world_model(args))
        _log("")

    _banner("SUMMARY")
    for entry in results["results"]:
        _log(json.dumps(entry))

    out_path = Path(args.json) if args.json else REPO_ROOT / "research" / "19_results_jepa.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
