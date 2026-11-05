"""Seeded task regression benchmark with explicit, machine-readable comparisons."""

import argparse
from importlib import import_module
import json
from pathlib import Path
import statistics

import numpy as np


TASKS = ("ar", "mobile-typing-5c-words", "public-display-4-buttons")
REWARD_COMPONENTS = (
    "distance_reward", "phase_bonus_reward", "done_reward",
    "neural_effort_reward", "jac_effort_reward",
)


def summarize_rollouts(rollouts):
    """Summarize episodes, retaining zero-time reset samples only for trajectories."""
    episodes = []
    for rollout in rollouts:
        states = [s for s in rollout if float(s.data.time) > 0]
        if not states:
            raise ValueError("A benchmark episode must contain at least one simulation step.")
        reward = np.asarray([float(s.reward) for s in states])
        components = {key: sum(float(s.metrics.get(key, 0)) for s in states) for key in REWARD_COMPONENTS}
        max_error = max(abs(float(s.reward) - sum(float(s.metrics.get(key, 0)) for key in REWARD_COMPONENTS)) for s in states)
        if not np.isfinite(reward).all() or not all(np.isfinite(v) for v in components.values()):
            raise ValueError("Non-finite reward in benchmark rollout.")
        if max_error > 1e-4:
            raise ValueError(f"Reward components do not sum to the optimized reward (max error {max_error:g}).")
        episodes.append({
            "success": bool(states[-1].info.get("task_completed", False)),
            "duration_seconds": float(states[-1].data.time),
            "steps": len(states),
            "reward": float(reward.sum()),
            "reward_components": components,
            "max_reward_decomposition_error": max_error,
        })
    if not episodes:
        raise ValueError("At least one evaluation episode is required.")
    return {
        "episodes": episodes,
        "success_rate": statistics.mean(float(e["success"]) for e in episodes),
        "mean_duration_seconds": statistics.mean(e["duration_seconds"] for e in episodes),
        "mean_reward": statistics.mean(e["reward"] for e in episodes),
        "mean_reward_components": {key: statistics.mean(e["reward_components"][key] for e in episodes) for key in REWARD_COMPONENTS},
    }


def compare_reports(actual, reference, *, reward_atol=1e-4, reward_rtol=1e-4, success_atol=0.0, duration_atol=1e-6):
    """Compare only matching benchmark cases/protocols, with declared tolerances."""
    protocol_keys = ("policy", "seeds", "episodes_per_seed", "max_steps")
    if any(actual["protocol"].get(key) != reference["protocol"].get(key) for key in protocol_keys):
        raise ValueError("Reference benchmark protocol differs from this run.")
    actual_cases = {case["task"]: case for case in actual["cases"]}
    reference_cases = {case["task"]: case for case in reference["cases"]}
    if actual_cases.keys() != reference_cases.keys():
        raise ValueError("Reference benchmark tasks differ from this run.")
    failures = []
    for name, case in actual_cases.items():
        before = reference_cases[name]
        if case["config_sha256"] != before["config_sha256"]:
            raise ValueError(f"Reference task configuration differs for {name}.")
        if case.get("checkpoint") != before.get("checkpoint"):
            raise ValueError(f"Reference policy checkpoint differs for {name}.")
        if case.get("checkpoint_sha256") != before.get("checkpoint_sha256"):
            raise ValueError(f"Reference policy checkpoint content differs for {name}.")
        if abs(case["summary"]["success_rate"] - before["summary"]["success_rate"]) > success_atol:
            failures.append(f"{name}: success rate changed")
        if abs(case["summary"]["mean_duration_seconds"] - before["summary"]["mean_duration_seconds"]) > duration_atol:
            failures.append(f"{name}: mean duration changed")
        values = {"reward": case["summary"]["mean_reward"], **case["summary"]["mean_reward_components"]}
        expected = {"reward": before["summary"]["mean_reward"], **before["summary"]["mean_reward_components"]}
        for key, value in values.items():
            if not np.isclose(value, expected[key], atol=reward_atol, rtol=reward_rtol):
                failures.append(f"{name}: {key} changed from {expected[key]:g} to {value:g}")
    return failures


def _optional_movement_metrics(rollouts, env):
    module_name = "myosuite.envs.myo.myouser.pointing_metrics"
    try:
        metrics = import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        return {"status": "unavailable", "reason": "Movement metrics are supplied by upstream PR #3."}
    raw = metrics.rollout_movement_metrics(rollouts, env=env)
    # Undefined scientific metrics are represented explicitly as JSON null.
    return {"status": "available", "values": {key: float(value) if np.isfinite(value) else None for key, value in raw.items()}}


def evaluate_task(task, *, seeds, episodes, max_steps, policy, checkpoints_dir=None):
    import jax
    import jax.numpy as jp
    from ml_collections import ConfigDict

    from myosuite.envs.myo.myouser.evaluate import evaluate_policy
    from myosuite.envs.myo.myouser.myouser_universal import MyoUserUniversal
    from myosuite.envs.myo.myouser.paths import get_bundled_checkpoint_dir
    from myosuite.train.utils.train import resolve_checkpoint_path, train_or_load_checkpoint
    from myosuite.train.utils.wrapper import _maybe_wrap_env_for_evaluation
    from myosuite.utils.run_manifest import create_run_manifest

    config_path = Path(__file__).parent / "configs" / f"{task}.json"
    config = ConfigDict(json.loads(config_path.read_text(encoding="utf-8")))
    config.env.vision = config.vision
    config.wandb = {"enabled": False}
    config.env.eval_mode = True
    config.rl.num_timesteps = 0
    checkpoint = None
    if policy == "checkpoint":
        root = Path(checkpoints_dir) if checkpoints_dir else get_bundled_checkpoint_dir()
        checkpoint = resolve_checkpoint_path(root / task / "checkpoints")
        env, make_policy, params = train_or_load_checkpoint(
            config.env.env_name, config, eval_mode=True, checkpoint_path=checkpoint,
        )
        inference = jax.jit(make_policy(params, deterministic=True))
    else:
        env = MyoUserUniversal(config.env)
        env.enable_eval_mode()
        def zero_policy(obs, rng):
            del rng
            batch = jax.tree.leaves(obs)[0].shape[0]
            return jp.zeros((batch, env.action_size)), {}
        inference = jax.jit(zero_policy)
    # The environment migrates historical checkpoint task defaults on its private
    # copy. Record those effective defaults, including omitted zero-weight terms.
    config.env = env._config.to_dict()
    env, _ = _maybe_wrap_env_for_evaluation(env, seeds[0])
    reset, step = jax.jit(env.eval_reset), jax.jit(env.step)
    rollouts = []
    for seed in seeds:
        trajectories, _ = evaluate_policy(
            eval_env=env, jit_inference_fn=inference, jit_reset=reset, jit_step=step,
            seed=seed, n_episodes=episodes, ep_length=max_steps, include_reset_state=True,
        )
        rollouts.extend(trajectories)
    manifest = create_run_manifest(config, checkpoint_path=checkpoint, backend=jax.default_backend(), devices=jax.devices())
    manifest["status"] = "completed"
    result = {
        "task": task,
        "checkpoint": checkpoint.name if checkpoint else None,
        "checkpoint_sha256": manifest["restore"]["checkpoint_sha256"],
        "config_sha256": manifest["config_sha256"],
        "manifest": manifest,
        "summary": summarize_rollouts(rollouts),
        "movement_metrics": _optional_movement_metrics(rollouts, env),
    }
    # Different target counts compile different MJX shapes; release cached executables
    # between benchmark cases so the three-task CPU run has bounded memory use.
    jax.clear_caches()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None, help="Omit for full task episodes; use 2 for a fast smoke run.")
    parser.add_argument("--policy", choices=("zero", "checkpoint"), default="checkpoint")
    parser.add_argument("--checkpoints-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--reward-atol", type=float, default=1e-4)
    parser.add_argument("--reward-rtol", type=float, default=1e-4)
    parser.add_argument("--success-atol", type=float, default=0.0)
    parser.add_argument("--duration-atol", type=float, default=1e-6)
    args = parser.parse_args(argv)
    if args.episodes < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("episodes and max-steps must be positive")
    if any(seed < 0 or seed > 2**32 - 1 for seed in args.seeds):
        parser.error("seeds must be uint32 integers")
    if any(not np.isfinite(v) or v < 0 for v in (args.reward_atol, args.reward_rtol, args.success_atol, args.duration_atol)):
        parser.error("comparison tolerances must be finite and nonnegative")
    report = {
        "schema_version": 1,
        "protocol": {"policy": args.policy, "seeds": args.seeds, "episodes_per_seed": args.episodes, "max_steps": args.max_steps},
        "cases": [],
    }
    for task in args.tasks:
        print(f"Evaluating {task} ({args.policy}, seeds={args.seeds})", flush=True)
        report["cases"].append(evaluate_task(
            task, seeds=args.seeds, episodes=args.episodes, max_steps=args.max_steps,
            policy=args.policy, checkpoints_dir=args.checkpoints_dir,
        ))
    failures = []
    if args.compare:
        failures = compare_reports(report, json.loads(args.compare.read_text(encoding="utf-8")),
                                   reward_atol=args.reward_atol, reward_rtol=args.reward_rtol,
                                   success_atol=args.success_atol, duration_atol=args.duration_atol)
        report["comparison"] = {"passed": not failures, "failures": failures,
                                "tolerances": {"reward_atol": args.reward_atol, "reward_rtol": args.reward_rtol,
                                               "success_atol": args.success_atol, "duration_atol": args.duration_atol}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
