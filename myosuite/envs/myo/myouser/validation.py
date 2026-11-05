"""Shared, dependency-light preflight checks for GUI and CLI training."""

import math
import re
from numbers import Real


def finite_number(value, name, *, minimum=None, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number.")
    if strictly_positive and value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return float(value)


def integer(value, name, *, minimum=0, maximum=None):
    number = finite_number(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer.")
    result = int(number)
    if result < minimum or (maximum is not None and result > maximum):
        bounds = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{name} must be {bounds}.")
    return result


def boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def parse_color(value):
    """Convert supported CSS hex/rgb/rgba colors to normalized RGB channels."""
    message = "Color must be a valid hex color or rgb/rgba color with RGB channels between 0 and 255 and alpha between 0 and 1."
    if not isinstance(value, str):
        raise ValueError(message)
    value = value.strip()
    hex_match = re.fullmatch(r"#?([0-9a-fA-F]{6})", value)
    if hex_match:
        digits = hex_match.group(1)
        return tuple(int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4))
    hex_match = re.fullmatch(r"#([0-9a-fA-F]{3,4}|[0-9a-fA-F]{8})", value)
    if hex_match:
        digits = hex_match.group(1)
        if len(digits) in (3, 4):
            digits = "".join(character * 2 for character in digits)
        return tuple(int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4))
    rgb_match = re.fullmatch(r"(rgb|rgba)\(([^()]*)\)", value, flags=re.IGNORECASE)
    if not rgb_match:
        raise ValueError(message)
    try:
        channels = [float(channel.strip()) for channel in rgb_match.group(2).split(",")]
    except ValueError as error:
        raise ValueError(message) from error
    expected = 4 if rgb_match.group(1).lower() == "rgba" else 3
    if len(channels) != expected or any(not math.isfinite(channel) or not 0 <= channel <= 255 for channel in channels[:3]):
        raise ValueError(message)
    if expected == 4 and (not math.isfinite(channels[3]) or not 0 <= channels[3] <= 1):
        raise ValueError(message)
    return tuple(channel / 255 for channel in channels[:3])


def validate_rl_parameters(rl):
    for key in ("num_envs", "batch_size", "num_minibatches", "unroll_length", "action_repeat", "num_updates_per_batch", "num_eval_envs", "episode_length"):
        if key in rl:
            integer(rl[key], f"rl.{key}", minimum=1)
    for key in ("num_timesteps", "num_evals", "num_checkpoints", "num_resets_per_eval"):
        if key in rl:
            integer(rl[key], f"rl.{key}", minimum=0)
    if all(key in rl for key in ("batch_size", "num_minibatches", "num_envs")):
        if rl["batch_size"] * rl["num_minibatches"] % rl["num_envs"]:
            raise ValueError("rl.batch_size * rl.num_minibatches must be divisible by rl.num_envs.")
    for key in ("log_training_metrics", "normalize_observations"):
        if key in rl:
            boolean(rl[key], f"rl.{key}")


def validate_run_parameters(run):
    for key in ("seed", "eval_seed"):
        if key in run:
            integer(run[key], f"run.{key}", maximum=2**32 - 1)
    if "eval_episodes" in run:
        integer(run["eval_episodes"], "run.eval_episodes", minimum=1)
    if run.get("rscope_envs") is not None:
        integer(run["rscope_envs"], "run.rscope_envs")
    for key in ("play_only", "use_tb", "deterministic_rscope", "domain_randomization", "local_plotting", "log_wandb_videos", "using_gradio"):
        if key in run:
            boolean(run[key], f"run.{key}")


def validate_vision_parameters(vision):
    if "enabled" in vision:
        boolean(vision["enabled"], "vision.enabled")
    for key in ("num_worlds", "render_batch_size", "render_width", "render_height"):
        if key in vision:
            integer(vision[key], f"vision.{key}", minimum=1)
    if "gpu_id" in vision:
        integer(vision["gpu_id"], "vision.gpu_id")
    if "use_rasterizer" in vision:
        boolean(vision["use_rasterizer"], "vision.use_rasterizer")


def validate_restore_modes(rl, run=None, env=None, *, vision_enabled=None):
    """Keep exact training continuation distinct from parameter warm starts."""
    run, env = run or {}, env or {}
    resume = rl.get("resume_training_path")
    if resume in (None, "None"):
        return
    if not isinstance(resume, str) and not hasattr(resume, "__fspath__"):
        raise ValueError("rl.resume_training_path must name a full training-state checkpoint.")
    if not str(resume).strip():
        raise ValueError("rl.resume_training_path cannot be empty.")
    if rl.get("load_checkpoint_path") not in (None, "None"):
        raise ValueError("Choose either rl.resume_training_path for exact continuation or rl.load_checkpoint_path for a parameter warm start, not both.")
    if run.get("play_only", False) or env.get("eval_mode", False):
        raise ValueError("Exact training resume cannot be combined with play_only or environment evaluation mode; use load_checkpoint_path for evaluation.")
    if integer(rl.get("num_timesteps", 0), "rl.num_timesteps") == 0:
        raise ValueError("Exact training resume requires the original positive rl.num_timesteps total horizon.")
    vision = env.get("vision", {}).get("enabled", False) if vision_enabled is None else vision_enabled
    if vision:
        raise ValueError("Exact training resume is currently supported only without vision; Madrona renderer state cannot be restored exactly.")


def validate_task_parameters(env, vision_enabled=None):
    if "eval_mode" in env:
        boolean(env["eval_mode"], "env.eval_mode")
    ctrl_dt = finite_number(env["ctrl_dt"], "env.ctrl_dt", strictly_positive=True)
    if "sim_dt" in env:
        sim_dt = finite_number(env["sim_dt"], "env.sim_dt", strictly_positive=True)
        ratio = ctrl_dt / sim_dt
        if ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("env.ctrl_dt must be a positive integer multiple of env.sim_dt.")
    task = env.get("task_config", {})
    if "max_duration" in task:
        duration = finite_number(task["max_duration"], "env.task_config.max_duration", strictly_positive=True)
        if duration < ctrl_dt:
            raise ValueError("Maximum episode duration must include at least one control timestep.")
    if "target_init_seed" in task:
        integer(task["target_init_seed"], "target_init_seed", maximum=2**32 - 1)
    obs = task.get("obs_keys", [])
    omni = task.get("omni_keys", [])
    vision = env.get("vision", {}).get("enabled", False) if vision_enabled is None else vision_enabled
    if "obs_keys" in task and not obs and (vision or not omni):
        raise ValueError("Select at least one observation for the policy.")
    for key, weight in task.get("weighted_reward_keys", {}).items():
        finite_number(weight, f"reward weight {key}")
    if "targets" in task:
        from .target_validation import validate_target_sampling
        targets = task["targets"]
        if hasattr(targets, "keys"):
            count = integer(targets.get("num_targets", len(targets)), "number of targets", minimum=1, maximum=10)
            try:
                values = [targets[f"target_{i}"] for i in range(count)]
            except KeyError as error:
                raise ValueError(f"Missing target configuration: {error}") from error
        else:
            values = targets
        origin = task.get("reach_settings", {}).get("target_origin_rel", (0, 0, 0))
        validate_target_sampling(values, ctrl_dt, origin)


def validate_training_config(config):
    """Raise ValueError before allocating an environment or starting a run."""
    validate_rl_parameters(config["rl"])
    validate_run_parameters(config.get("run", {}))
    validate_vision_parameters(config.get("vision", {}))
    validate_vision_parameters(config["env"].get("vision", {}))
    if "enabled" in config.get("wandb", {}):
        boolean(config["wandb"]["enabled"], "wandb.enabled")
    validate_restore_modes(config["rl"], config.get("run", {}), config["env"], vision_enabled=config.get("vision", {}).get("enabled", False))
    validate_task_parameters(config["env"], config.get("vision", {}).get("enabled", False))
    return config
