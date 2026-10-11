"""Explore JEPA: joint-embedding predictive architectures on a laptop CPU.

Runs three self-contained experiments with pure NumPy and the doraneural
autograd engine:

1. **I-JEPA** on synthetic images - multi-block masking, EMA target encoder,
   smooth-L1 in latent space - plus the positional-shortcut audit that tells you
   whether the objective actually learned anything semantic.
2. **V-JEPA + SIGReg (LeJEPA)** on moving-dot clips, probed for *temporal order*
   (forward vs reversed). Reversal keeps the set of frames identical, so a
   randomly initialised encoder is at chance by construction.
3. **JEPA world model** trained from pixels on a bouncing point mass, then used
   for cross-entropy-method planning towards a goal image, scored against random
   actions on the real environment.

Usage:
    python3 examples/explore_jepa.py            # full demo (~3-5 minutes)
    python3 examples/explore_jepa.py --quick    # tiny smoke run (~30 seconds)
"""

import argparse
from pathlib import Path
import sys
import time

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
    sample_ijepa_masks,
)
from doraneural.jepa_data import (  # noqa: E402
    PointMassConfig,
    PointMassEnv,
    make_grating_dataset,
    make_motion_clip_dataset,
    make_shape_dataset,
    mask_ascii,
    render_ascii,
    sample_point_mass_trajectories,
)


def header(text: str) -> None:
    print("=" * 78)
    print(f" {text}")
    print("=" * 78)


def explore_masking(grid: int = 8) -> None:
    """Show what multi-block masking actually hides."""
    header("1. THE MASK IS THE METHOD: I-JEPA MULTI-BLOCK SAMPLING")
    rng = np.random.default_rng(0)
    context, targets = sample_ijepa_masks(grid, grid, n_targets=4, rng=rng)
    grid_mask = np.zeros(grid * grid, dtype=bool)
    grid_mask[context] = True
    print(mask_ascii(grid_mask, grid=(grid, grid)))
    print("  '#' = visible context tokens, '.' = masked targets (4 blocks of 15-20%)")
    print(f"  context: {len(context)} tokens | targets: "
          f"{[len(t) for t in targets]} tokens | disjoint: "
          f"{not (set(context) & set(np.concatenate(targets)))}")


def explore_ijepa(steps: int) -> dict:
    header("2. I-JEPA ON SYNTHETIC GRATINGS (MASKED PREDICTION IN LATENT SPACE)")
    data = make_grating_dataset(n=192, img_size=32, seed=0)
    images, labels = data["images"], data["labels"]
    print(render_ascii(images[1]))
    print("  a grating: the label (orientation) lives in relations between distant patches\n")

    model = IJEPA(img_size=32, patch_size=8, in_chans=1, dim=64, depth=2, n_heads=4,
                  pred_dim=64, pred_depth=2, pred_heads=4, ema_start=0.9, ema_end=0.99)
    split = int(0.75 * len(images))

    def probe(encoder):
        feats = encoder.encode(images)
        return linear_probe(feats[:split], labels[:split], feats[split:], labels[split:], num_classes=4)

    np.random.seed(0)
    random_model = IJEPA(img_size=32, patch_size=8, in_chans=1, dim=64, depth=2, n_heads=4,
                         pred_dim=64, pred_depth=2, pred_heads=4, ema_start=0.9, ema_end=0.99)
    before = probe(random_model)
    t0 = time.perf_counter()
    stats = model.fit(images, steps=steps, batch_size=16, lr=1e-3, verbose_every=max(1, steps // 4),
                      rng=np.random.default_rng(0))
    secs = time.perf_counter() - t0
    after = probe(model)
    audit = masked_target_baseline(model, images[:128], rng=np.random.default_rng(1))
    diag = representation_diagnostics(model.encode(images))

    print(f"\n  latent loss        : {stats['final_loss']:.4f}  ({secs:.1f}s for {steps} steps)")
    print(f"  orientation probe  : random init {before['accuracy']:.3f} -> JEPA {after['accuracy']:.3f} "
          f"(chance {1/4:.2f})")
    print(f"  collapse check     : effective rank {diag['effective_rank']:.1f}/64, "
          f"signal ratio {diag['signal_ratio']:.4f}, collapsed={diag['collapsed']}")
    print(f"  shortcut audit     : position alone explains {audit['position_r2']*100:.1f}% of the target "
          f"(model MSE {audit['model_mse']:.4f} vs position-lookup MSE {audit['position_mse']:.4f})")
    if audit["position_r2"] > 0.9:
        print("  -> at this scale the EMA objective solves the task with a positional lookup table:")
        print("     the loss looks perfect while the features stay at chance. Use SIGReg (experiment 3).")
    return {"probe_random": before["accuracy"], "probe_after": after["accuracy"], "audit": audit}


def explore_vjepa(steps: int) -> dict:
    header("3. V-JEPA + SIGREG (LEJEPA): TEMPORAL ORDER FROM TUBE MASKING")
    data = make_motion_clip_dataset(n=160, frames=8, img_size=32, seed=0, task="order")
    clips, labels = data["clips"], data["labels"]
    split = int(0.75 * len(clips))
    print("  task: is this clip playing forward or reversed?")
    print("  reversal keeps the set of frames identical, so mean-pooled appearance")
    print("  features of a random encoder are at chance - only motion models can solve it.\n")

    def build():
        return VJEPA(frames=8, img_size=32, patch_size=8, tubelet=2, in_chans=1, dim=64, depth=2,
                     n_heads=4, pred_dim=64, pred_depth=2, pred_heads=4, ema_start=0.9, ema_end=0.99)

    def probe(encoder):
        feats = encoder.encode(clips)
        return linear_probe(feats[:split], labels[:split], feats[split:], labels[split:], num_classes=2)

    np.random.seed(0)
    before = probe(build())
    model = build()
    t0 = time.perf_counter()
    stats = model.fit(clips, steps=steps, batch_size=16, lr=1e-3, verbose_every=max(1, steps // 4),
                      rng=np.random.default_rng(0), sigreg=True, sigreg_weight=0.1)
    secs = time.perf_counter() - t0
    after = probe(model)
    diag = representation_diagnostics(model.encode(clips))

    print(f"\n  latent loss        : {stats['final_loss']:.4f}  ({secs:.1f}s for {steps} steps)")
    print(f"  temporal-order probe: random init {before['accuracy']:.3f} -> V-JEPA+SIGReg "
          f"{after['accuracy']:.3f} (chance 0.50)")
    print(f"  collapse check     : effective rank {diag['effective_rank']:.1f}, "
          f"signal ratio {diag['signal_ratio']:.4f}, collapsed={diag['collapsed']}")
    return {"probe_random": before["accuracy"], "probe_after": after["accuracy"]}


def explore_world_model(steps: int, episodes: int) -> dict:
    header("4. LATENT WORLD MODEL: LEARNED DYNAMICS + CEM PLANNING FROM PIXELS")
    cfg = PointMassConfig(size=24)
    train = sample_point_mass_trajectories(n=192, horizon=10, config=cfg, seed=0, goal_seek_prob=0.4)
    test = sample_point_mass_trajectories(n=32, horizon=10, config=cfg, seed=99, goal_seek_prob=0.4)
    print(render_ascii(train["obs"][0, 0]))
    print("  the agent is the bright dot, the dimmer disc is the goal\n")

    model = JEPAWorldModel(obs_shape=(1, 24, 24), action_dim=2, dim=64, encoder_depth=2, encoder_heads=4,
                           patch_size=8, predictor_dim=64, predictor_depth=2, predictor_heads=4,
                           context_len=4, freeze_encoder=False, sigreg_slices=32, seed=0)
    t0 = time.perf_counter()
    model.fit(train["obs"], train["actions"], steps=steps, batch_size=16, lr=1e-3, sigreg_weight=0.005,
              rollout_steps=2, verbose_every=max(1, steps // 4), rng=np.random.default_rng(0))
    secs = time.perf_counter() - t0

    z = model.encode_np(test["obs"].reshape(-1, 1, 24, 24))
    z_seq = z.reshape(len(test["obs"]), test["obs"].shape[1], -1)
    model_err, nochange = [], []
    for t in range(test["obs"].shape[1] - 1):
        pred = model.rollout(z_seq[:, t], test["actions"][:, t : t + 1])[0]
        model_err.append(np.mean((pred - z_seq[:, t + 1]) ** 2))
        nochange.append(np.mean((z_seq[:, t] - z_seq[:, t + 1]) ** 2))
    diag = representation_diagnostics(z)
    pos_probe = linear_probe(model.encode_np(train["obs"][:, 5]), train["positions"][:, 5],
                             model.encode_np(test["obs"][:, 5]), test["positions"][:, 5], mode="regression")

    audit = model.action_utilization(test["obs"], test["actions"], seed=0)
    print(f"\n  next-latent MSE    : model {np.mean(model_err):.4f} vs no-change {np.mean(nochange):.4f}")
    print(f"  position probe R2  : {pos_probe['r2']:.3f}")
    print(f"  collapse check     : rank {diag['effective_rank']:.1f}, "
          f"signal ratio {diag['signal_ratio']:.3f}, collapsed={diag['collapsed']}")
    print(f"  action ablation    : true {audit['one_step_true_action_mse']:.4f} | "
          f"shuffled {audit['one_step_shuffled_action_mse']:.4f} | "
          f"zeroed {audit['one_step_zero_action_mse']:.4f}  ->  gain "
          f"{audit['one_step_action_gain']:+.3f}")
    print("  (A gain near zero means the model shrank towards the latent marginal and is")
    print("   ignoring its action input, whatever its MSE looks like.)")

    # Plan towards a goal image and execute the plan in the real environment.
    env = PointMassEnv(cfg, seed=7)
    rng = np.random.default_rng(7)
    planned_success = random_success = 0
    planned_distance, random_distance, improved = [], [], []
    for i in range(episodes):
        start = rng.uniform(0.15, 0.85, 2)
        goal = rng.uniform(0.1, 0.9, 2)
        while np.linalg.norm(start - goal) < 0.35:
            goal = rng.uniform(0.1, 0.9, 2)
        env.reset(position=start, velocity=[0, 0], goal=goal)
        z_start = model.encode_np(env.render()[None])
        env.reset(position=goal, velocity=[0, 0], goal=goal)
        z_goal = model.encode_np(env.render()[None])

        plan = model.plan(z_start, z_goal, horizon=6, n_samples=160, n_elite=16, n_iters=4,
                          rng=np.random.default_rng(100 + i))
        if i == 0:
            print("\n  first episode (start " f"{np.round(start, 2)} -> goal {np.round(goal, 2)}):")
            for step_idx, frame in enumerate(_rollout_frames(cfg, start, goal, plan["actions"])):
                print(render_ascii(frame, charset=" .:-=+*#%@"), flush=True)
                if step_idx >= 3:
                    break
        env.reset(position=start, velocity=[0, 0], goal=goal)
        info = {"distance": float(np.linalg.norm(start - goal))}
        for action in plan["actions"]:
            _, info = env.step(action)
        planned_distance.append(info["distance"])
        planned_success += int(info["distance"] < 0.12)

        env.reset(position=start, velocity=[0, 0], goal=goal)
        for _ in range(6):
            _, info = env.step(rng.uniform(-1, 1, 2))
        random_distance.append(info["distance"])
        random_success += int(info["distance"] < 0.12)
        improved.append(float(planned_distance[-1] < random_distance[-1]))

    print(f"\n  CEM planning       : success {planned_success/episodes:.2f} vs random "
          f"{random_success/episodes:.2f}  |  final distance "
          f"{np.mean(planned_distance):.3f} vs {np.mean(random_distance):.3f}  ({secs:.1f}s training)")
    print(f"  paired win-rate    : the plan finished closer to the goal than the random "
          f"rollout in {np.mean(improved)*100:.0f}% of {episodes} episodes")
    return {
        "model_mse": float(np.mean(model_err)),
        "nochange_mse": float(np.mean(nochange)),
        "position_r2": float(pos_probe["r2"]),
        "planned_success": planned_success / episodes,
        "random_success": random_success / episodes,
        "improved_fraction": float(np.mean(improved)),
        "action_gain": float(audit["one_step_action_gain"]),
    }


def _rollout_frames(cfg: PointMassConfig, start, goal, actions):
    env = PointMassEnv(cfg, seed=0)
    env.reset(position=start, velocity=[0, 0], goal=goal)
    frames = [env.render()]
    for action in actions:
        frame, _ = env.step(action)
        frames.append(frame)
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Explore JEPA implementations on CPU.")
    parser.add_argument("--quick", action="store_true", help="tiny budgets for a fast smoke run")
    parser.add_argument("--image-steps", type=int, default=300)
    parser.add_argument("--video-steps", type=int, default=600)
    parser.add_argument("--world-steps", type=int, default=600)
    parser.add_argument("--plan-episodes", type=int, default=16)
    args = parser.parse_args()
    if args.quick:
        # 60 world-model steps is not enough for the dynamics to use its actions at
        # all, which makes the planning section look broken rather than untrained.
        args.image_steps, args.video_steps, args.world_steps, args.plan_episodes = 40, 60, 240, 6

    header("JEPA: LEARN BY PREDICTING REPRESENTATIONS, NOT PIXELS")
    print("  E(x) -> encoder, P(., z) -> predictor, and a loss that only ever compares")
    print("  latent vectors:  L = || P(E(context), z) - sg(E(target)) ||")
    print("  I-JEPA (images) | V-JEPA (video) | SIGReg (LeJEPA) | action-conditioned world model\n")

    explore_masking()
    print()
    ijepa = explore_ijepa(args.image_steps)
    print()
    vjepa = explore_vjepa(args.video_steps)
    print()
    world = explore_world_model(args.world_steps, args.plan_episodes)

    header("WHAT WE LEARNED")
    print(f"  I-JEPA orientation probe     : {ijepa['probe_random']:.2f} (random) -> "
          f"{ijepa['probe_after']:.2f} (trained); position alone explains "
          f"{ijepa['audit']['position_r2']*100:.0f}% of the target")
    print(f"  V-JEPA temporal-order probe  : {vjepa['probe_random']:.2f} (random) -> "
          f"{vjepa['probe_after']:.2f} (trained with SIGReg)")
    print(f"  world model latent MSE       : {world['model_mse']:.4f} vs no-change "
          f"{world['nochange_mse']:.4f}; position R2 {world['position_r2']:.2f}; "
          f"action gain {world['action_gain']:+.3f}")
    print(f"  latent planner vs random     : {world['planned_success']:.2f} vs {world['random_success']:.2f} "
          f"success rate; plan closer than random in {world['improved_fraction']*100:.0f}% of episodes")
    print("\n  The takeaway from this playground: the *mask* and the *anti-collapse mechanism*")
    print("  carry the method. Latent-space loss is a sharp instrument that will happily")
    print("  report ~0 while learning a positional shortcut - always pair it with a random")
    print("  baseline, a probe, and the shortcut audit shipped in this module. And when a")
    print("  world model does predict, check the permutation ablation before crediting the")
    print("  dynamics: shrinking towards the marginal is a cheap way to look accurate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
