"""Hydra CLI machinery for myointeract configs.

All dataclass definitions live in myointeract_configs.py (pure Python, no Hydra
dependency).  This module adds the OmegaConf resolver registration, the
ConfigStore setup, and the interactive helper used by notebooks / scripts.
"""

import re
import warnings
import math
from pathlib import Path

import hydra
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from ml_collections import ConfigDict
from omegaconf import OmegaConf, open_dict

from myoInteract.myointeract_configs import (
    BaseEnvConfig,
    ButtonTarget,
    Config,
    LIST_CONFIGS,
    NetworkConfig,
    PointingEnvConfig,
    PointingTarget,
    RLConfig,
    RunConfig,
    UniversalConfig,
    UniversalEnvConfig,
    VisionDisabledConfig,
    VisionEnabledConfig,
    VisionNetworkConfig,
    WANDBDisabledConfig,
    WANDBEnabledConfig,
)

# Re-export so callers that did `from myointeract_hydra_cli import XConfig` still work.
__all__ = [
    "BaseEnvConfig",
    "ButtonTarget",
    "Config",
    "LIST_CONFIGS",
    "NetworkConfig",
    "PointingEnvConfig",
    "PointingTarget",
    "RLConfig",
    "RunConfig",
    "UniversalConfig",
    "UniversalEnvConfig",
    "VisionDisabledConfig",
    "VisionEnabledConfig",
    "VisionNetworkConfig",
    "WANDBDisabledConfig",
    "WANDBEnabledConfig",
    "load_config_interactive",
    "load_config_from_checkpoint_dir",
    "gradio_json_to_overrides",
]


# ---------------------------------------------------------------------------
# OmegaConf resolvers
# ---------------------------------------------------------------------------


def _select_network(vision_enabled: str) -> str:
    return "vision" if vision_enabled == "enabled" else "no_vision"


def _select_targets(num_targets: int):
    from myoInteract.myointeract_configs import (
        EightTargetConfig,
        ElevenTargetConfig,
        FifteenTargetConfig,
        FiveTargetConfig,
        FourTargetConfig,
        FourteenTargetConfig,
        NineTargetConfig,
        OneTargetConfig,
        SevenTargetConfig,
        SixTargetConfig,
        TenTargetConfig,
        ThirteenTargetConfig,
        ThreeTargetConfig,
        TwelveTargetConfig,
        TwoTargetConfig,
    )

    configs = [
        None,
        OneTargetConfig(),
        TwoTargetConfig(),
        ThreeTargetConfig(),
        FourTargetConfig(),
        FiveTargetConfig(),
        SixTargetConfig(),
        SevenTargetConfig(),
        EightTargetConfig(),
        NineTargetConfig(),
        TenTargetConfig(),
        ElevenTargetConfig(),
        TwelveTargetConfig(),
        ThirteenTargetConfig(),
        FourteenTargetConfig(),
        FifteenTargetConfig(),
    ]
    if 1 <= num_targets <= 15:
        return configs[num_targets]
    if num_targets > 15:
        from myoInteract.myointeract_configs import create_target_config_class  # noqa: PLC0415

        return create_target_config_class(num_targets)()
    raise ValueError(f"num_targets must be at least 1, got {num_targets}")


OmegaConf.register_new_resolver(
    "check_string", lambda x: "" if x is None else "-" + str(x)
)
OmegaConf.register_new_resolver("select_network", _select_network)
OmegaConf.register_new_resolver("select_targets", _select_targets)
OmegaConf.register_new_resolver("int_divide", lambda x, y: int(float(x) // float(y)))


# ---------------------------------------------------------------------------
# ConfigStore registration
# ---------------------------------------------------------------------------


def register_universal_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_universal", node=UniversalConfig)
    for i, name, config in LIST_CONFIGS:
        cs.store(group="env/task_config/targets", name=name, node=config)
        for j in range(i + 1):
            cs.store(
                group=f"env/task_config/targets/target_{j}",
                name="sphere",
                node=PointingTarget,
            )
            cs.store(
                group=f"env/task_config/targets/target_{j}",
                name="box",
                node=ButtonTarget,
            )
    cs.store(name="config", node=Config)
    cs.store(group="wandb", name="enabled", node=WANDBEnabledConfig)
    cs.store(group="wandb", name="disabled", node=WANDBDisabledConfig)
    cs.store(group="vision", name="enabled", node=VisionEnabledConfig)
    cs.store(group="vision", name="disabled", node=VisionDisabledConfig)
    cs.store(group="env", name="base_env_config", node=BaseEnvConfig)
    cs.store(group="env", name="pointing", node=PointingEnvConfig)
    cs.store(group="env", name="universal", node=UniversalEnvConfig)
    cs.store(group="rl", name="rl_config", node=RLConfig)
    cs.store(group="run", name="run", node=RunConfig)
    cs.store(group="rl/network_factory", name="vision", node=VisionNetworkConfig)
    cs.store(group="rl/network_factory", name="no_vision", node=NetworkConfig)


# ---------------------------------------------------------------------------
# Interactive config loader (used from notebooks and training scripts)
# ---------------------------------------------------------------------------


def load_config_interactive(overrides: list[str] = (), cfg_only: bool = False):
    """Compose a myointeract config from a list of Hydra override strings.

    Example::

        config = load_config_interactive([
            "env=universal",
            "env/task_config/targets=four",
            "env.ctrl_dt=0.05",
        ])

    Returns a ``ml_collections.ConfigDict`` (or the raw OmegaConf DictConfig
    when *cfg_only* is True).
    """
    register_universal_configs()
    GlobalHydra.instance().clear()
    register_target_groups(list(overrides))
    config_name = "config_universal" if "env=universal" in overrides else "config"
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    if cfg_only:
        return cfg
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    container["env"]["vision"] = container["vision"]
    return ConfigDict(container)


CONFIG_SIDECAR_NAME = "myointeract_config.json"
_LEGACY_CONFIG_SIDECAR_NAME = (
    "myouser_config.json"  # written by checkpoints from before the rename
)


def checkpoint_config_path(checkpoint_dir: str | Path) -> Path:
    """Return the config sidecar of a checkpoint dir, falling back to the legacy file name."""
    directory = Path(checkpoint_dir)
    legacy = directory / _LEGACY_CONFIG_SIDECAR_NAME
    if not (directory / CONFIG_SIDECAR_NAME).exists() and legacy.exists():
        return legacy
    return directory / CONFIG_SIDECAR_NAME


def checkpoint_task_overrides(overrides: list[str]) -> list[str]:
    """Retain the actual fixed vision recipe used by unversioned checkpoints."""
    values = [_normalize_vision_recipe_override(value) for value in overrides]
    if any(value.lstrip("+").startswith("vision.recipe_version=") for value in values):
        return values
    config = load_config_interactive(values)
    if config.vision.enabled:
        values.append("vision.recipe_version=1")
    return values


def _normalize_vision_recipe_override(value: str) -> str:
    key, separator, setting = value.partition("=")
    if key.lstrip("+") in ("vision.recipe_version", "env.vision.recipe_version"):
        return f"vision.recipe_version={setting}" if separator else value
    return value


def training_task_overrides(preset: dict, argv: list[str]) -> list[str]:
    """Resolve continuation's sensing recipe before the native CLI builds its defaults."""
    import json

    values = gradio_json_to_overrides(preset)
    explicit_recipe = any(
        value.lstrip("+").split("=", 1)[0]
        in ("vision.recipe_version", "env.vision.recipe_version")
        for value in preset.get("extra_overrides", [])
    )
    checkpoint = None
    automatic = False
    for index, argument in enumerate(argv):
        key, separator, setting = argument.partition("=")
        if key in ("--resume-checkpoint", "--agent.resume"):
            if not separator:
                setting = argv[index + 1] if index + 1 < len(argv) else ""
            if key == "--resume-checkpoint":
                checkpoint = setting or None
            else:
                automatic = setting.lower() == "true"
    if explicit_recipe or not (checkpoint or automatic):
        return values
    requested = load_config_interactive(values).vision
    if checkpoint is not None:
        sidecar = checkpoint_config_path(Path(checkpoint).expanduser().parent)
        if sidecar.is_file():
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            source = saved.get("env", {}).get("vision", saved.get("vision", {}))
            if bool(source.get("enabled", False)) != bool(requested.enabled):
                raise ValueError(
                    "The resume checkpoint and preset must agree on whether vision is enabled."
                )
            return values + _vision_config_overrides(saved)
        if requested.enabled:
            raise ValueError(
                "Vision continuation needs the checkpoint's myointeract_config.json sidecar. "
                "Restore it or set vision.recipe_version explicitly in the preset's extra_overrides."
            )
    elif requested.enabled:
        raise ValueError(
            "Vision auto-resume needs --resume-checkpoint to inspect its saved recipe, "
            "or an explicit vision.recipe_version in the preset's extra_overrides."
        )
    return values


def _vision_config_overrides(config_dict: dict) -> list[str]:
    env = config_dict.get("env", {})
    vision = env.get("vision", config_dict.get("vision", {}))
    if not vision.get("enabled", False):
        return []
    values = [f"vision.recipe_version={vision.get('recipe_version', 1)}"]
    for key in (
        "vision_mode",
        "render_width",
        "render_height",
        "depth_min",
        "depth_cutoff",
        "enabled_geom_groups",
    ):
        if key in vision:
            values.append(f"vision.{key}={vision[key]}")
    return values


def load_config_from_checkpoint_dir(
    checkpoint_dir: str, config_path: str | None = None
) -> ConfigDict:
    """Load the fully-resolved config saved alongside a training checkpoint.

    Factors out the ``ConfigDict(json.loads((ckpt_dir / "myointeract_config.json").read_text()))``
    pattern otherwise copy-pasted at every checkpoint-loading call site (notebooks,
    ``ui.py``, training scripts).

    Args:
        checkpoint_dir: Directory containing a training run's saved checkpoints and
                        its ``myointeract_config.json`` sidecar.
        config_path: Optional explicit path to a config JSON file, overriding the
                     default ``<checkpoint_dir>/myointeract_config.json``.

    Returns:
        The loaded config as a ``ml_collections.ConfigDict``.

    Raises:
        FileNotFoundError: If neither ``config_path`` nor the default sidecar exists.
    """
    import json
    from pathlib import Path

    path = Path(config_path) if config_path else checkpoint_config_path(checkpoint_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"No config found at {path}. Pass config_path explicitly, or ensure "
            f"'{checkpoint_dir}/myointeract_config.json' exists."
        )
    return ConfigDict(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# gradio JSON → Hydra overrides converter
# ---------------------------------------------------------------------------

_POSSIBLE_OBS_KEYS = ["qpos", "qvel", "qacc", "ee_pos", "act"]
_POSSIBLE_FATIGUE_KEYS = ["MA", "MR", "MF"]
_POSSIBLE_VISION_KEYS = ["rgb", "depth"]
_POSSIBLE_OMNI_KEYS = ["target_pos", "target_size", "phase_progress", "dwell_fraction"]
_NUM_TARGETS_TEXT = [
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
]
_MANY_TARGETS = re.compile(r"^env/task_config/targets=n(\d+)$")


def targets_group(num_targets: int) -> str:
    """Name of the Hydra config group option for ``num_targets`` targets (registered on demand above fifteen)."""
    if num_targets < len(_NUM_TARGETS_TEXT):
        return _NUM_TARGETS_TEXT[num_targets]
    from myoInteract.myointeract_configs import create_target_config_class  # noqa: PLC0415

    warnings.warn(
        f"{num_targets} targets: the Gradio task builder takes at most 15, so this configuration and its "
        "checkpoints will not open there (Studio and the command line take any number).",
        stacklevel=2,
    )
    cs, name = ConfigStore.instance(), f"n{num_targets}"
    cs.store(
        group="env/task_config/targets",
        name=name,
        node=create_target_config_class(num_targets),
    )
    for j in range(num_targets):
        cs.store(
            group=f"env/task_config/targets/target_{j}",
            name="sphere",
            node=PointingTarget,
        )
        cs.store(
            group=f"env/task_config/targets/target_{j}", name="box", node=ButtonTarget
        )
    return name


def register_target_groups(overrides: list[str]) -> None:
    """Make sure the target groups a list of overrides names exist (a run's saved overrides, in a new process)."""
    for override in overrides:
        match = _MANY_TARGETS.match(override)
        if match:
            targets_group(int(match[1]))


BM_MODEL_MAP = {
    "MoBL_Arms_Index": (
        "myo_user/envs/myo/assets/arm/mobl_arms_index_universal_myointeract.xml",
        "fingertip",
        "humphant",
    ),
    "MoBL_Arms_Hand": (
        "myo_user/envs/myo/assets/arm/mobl_arms_hand_universal_myointeract.xml",
        "IFtip",
        "humphant",
    ),
    # The MyoArm models are MyoSuite's myoarm_r: sentinel paths, see _MYOARM_VARIANT_BY_MODEL_PATH.
    "MyoArm_nohand": (
        "myosuite/envs/myo/assets/arm/myoarm_nohand_universal_myointeract.xml",
        "IFtip",
        "R.Shoulder_marker",
    ),
    "MyoArm": (
        "myosuite/envs/myo/assets/arm/myoarm_universal_myointeract.xml",
        "IFtip",
        "R.Shoulder_marker",
    ),
    "MyoArm_reach": (
        "myosuite/envs/myo/assets/arm/myoarm_reach_universal_myointeract.xml",
        "IFtip",
        "R.Shoulder_marker",
    ),
    "MyoArm_pointing": (
        "myosuite/envs/myo/assets/arm/myoarm_pointing_universal_myointeract.xml",
        "IFtip",
        "R.Shoulder_marker",
    ),
}


def _parse_rgba(rgba_str: str) -> list[float]:
    """'rgba(R,G,B,1)' or '#rrggbb' → [r, g, b] normalised to 0–1."""
    if "rgba(" in rgba_str:
        inner = rgba_str.split("rgba(")[1].rstrip(")")
        r, g, b, *_ = (float(x.strip()) for x in inner.split(","))
        return [r / 255.0, g / 255.0, b / 255.0]
    hex_color = rgba_str.lstrip("#")
    return [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


def restore_button_geometry(box: dict, stored_box: dict) -> dict:
    """Keep geometry without Gradio widgets while applying the editable Y angle."""
    result = dict(box)
    if stored_box.get("euler") is not None:
        euler = list(stored_box["euler"])
        euler[1] = -float(box.get("orientation_angle", 0)) * math.pi / 180
        result["euler"] = euler
    if "box_thickness" in stored_box:
        result["box_thickness"] = stored_box["box_thickness"]
    return result


def button_geometry(box: dict) -> tuple[list[float], list[float]]:
    """Read full Euler angles and the two half-depth bounds from a labelled box."""
    angle = float(box.get("orientation_angle", 0))
    euler = box.get("euler") or [0, -angle * math.pi / 180, 0]
    thickness = box.get("box_thickness", 0.02)
    if not isinstance(thickness, (list, tuple)):
        thickness = [thickness, thickness]
    return list(euler), [0.5 * float(value) for value in thickness]


def gradio_json_to_overrides(cfg: dict) -> list[str]:
    """Convert a saved gradio JSON config dict to Hydra override strings.

    The JSON format is produced by ``ConfigSaver.to_labelled_dict`` in
    ``gradio_ui.py``.  This function mirrors the logic of
    ``args_to_cfg_overrides`` without any dependency on gradio or the
    form-value flat-list encoding.

    The returned list can be passed directly to ``load_config_interactive``.
    """
    overrides: list[str] = ["env=universal", "run.using_gradio=True"]

    # ── RL parameters ────────────────────────────────────────────────────────
    rl = cfg.get("rl", {})
    num_targets = int(cfg.get("num_targets", 1))
    overrides += [
        f"env/task_config/targets={targets_group(num_targets)}",
        # Presets store "Number of Total Training Steps"; very old ones "Number of Training Steps".
        f"rl.num_timesteps={int(rl.get('Number of Total Training Steps', rl.get('Number of Training Steps', 1_000_000)))}",
        f"rl.save_interval={int(rl.get('Checkpoint Interval (in Number of Training Iterations)', 30))}",
        f"rl.num_envs={int(rl.get('Number of Parallel Environments', 4096))}",
        f"rl.unroll_length={int(rl.get('Unroll Length', 10))}",
        f"rl.num_minibatches={int(rl.get('Number of Minibatches', 8))}",
        f"rl.num_epochs_per_update={int(rl.get('Number of Epochs Per Update', 8))}",
        f"env.task_config.target_init_seed={int(rl.get('Target Initial Seed', 0))}",
    ]

    # ── Biomechanical model ───────────────────────────────────────────────────
    bm = cfg.get("bm", {})
    bm_model = bm.get("bm_model", "MoBL_Arms_Index")
    model_path, ee_site, ref_site = BM_MODEL_MAP[bm_model]
    sigdepnoise = "white" if bm.get("sigdepnoise_enabled", False) else "None"
    constnoise = "white" if bm.get("constantnoise_enabled", False) else "None"
    overrides += [
        f"env.ctrl_dt={float(bm.get('ctrl_dt', 0.05))}",
        f"env.task_config.reset_type={bm.get('reset_type', 'epsilon_uniform')}",
        f"env.muscle_config.noise_params.sigdepnoise_type={sigdepnoise}",
        f"env.muscle_config.noise_params.constantnoise_type={constnoise}",
        f"env.muscle_config.fatigue_enabled={bm.get('fatigue_enabled', False)}",
        f"env.mj_impl={bm.get('mj_impl_type', 'warp')}",
        f"env.model_path={model_path}",
        f"env.task_config.reach_settings.ref_site={ref_site}",
        f"env.task_config.reach_settings.ee_site={ee_site}",
    ]

    # ── Task parameters ───────────────────────────────────────────────────────
    task = cfg.get("task", {})
    overrides.append(
        f"env.task_config.max_duration={float(task.get('max_duration', 4.0))}"
    )
    _reject_oor = bool(task.get("reject_oor", False))

    # ── Observation space ─────────────────────────────────────────────────────
    # JSON: {'<field_0>': [bool, ...], '<field_1>': [...], ...} where the values
    # are [obs_keys bools, fatigue_keys bools, vision_keys bools, omni_keys bools]
    # stored in insertion order (the key names themselves are irrelevant artifacts
    # of ConfigSaver.to_labelled_dict's zip-based encoding).
    obs_groups = list(cfg.get("obs", {}).values())
    obs_bools = (
        obs_groups[0] if len(obs_groups) > 0 else [True] * len(_POSSIBLE_OBS_KEYS)
    )
    fatigue_bools = obs_groups[1] if len(obs_groups) > 1 else []
    vision_bools = obs_groups[2] if len(obs_groups) > 2 else []
    omni_bools = obs_groups[3] if len(obs_groups) > 3 else []

    obs_selected = [k for k, v in zip(_POSSIBLE_OBS_KEYS, obs_bools) if v]
    fatigue_selected = [k for k, v in zip(_POSSIBLE_FATIGUE_KEYS, fatigue_bools) if v]
    omni_selected = [k for k, v in zip(_POSSIBLE_OMNI_KEYS, omni_bools) if v]
    vision_selected = [k for k, v in zip(_POSSIBLE_VISION_KEYS, vision_bools) if v]
    vision_enabled = bool(vision_selected)
    overrides += [
        f"env.task_config.obs_keys={obs_selected}",
        f"env.muscle_config.fatigue_obs_keys={fatigue_selected}",
        f"vision={'enabled' if vision_enabled else 'disabled'}",
        f"env.task_config.omni_keys={omni_selected}",
    ]
    if vision_enabled:
        overrides.append("vision.recipe_version=2")
        if "rgb" in vision_selected and "depth" in vision_selected:
            mode = "rgbd"
        elif "rgb" in vision_selected:
            mode = "rgb"
        else:
            mode = "depth"
        overrides.append(f"vision.vision_mode={mode}")

    # ── Reward weights ────────────────────────────────────────────────────────
    _rename = {
        "subtask_bonus": "phase_bonus",
        "completion_bonus": "done",
        "body_contact_penalty": "torso_contact_penalty",
    }
    reward = cfg.get("reward", {})
    _reward_defaults: dict = {}
    for key in [
        "distance",
        "subtask_bonus",
        "completion_bonus",
        "neural_effort",
        "jac_effort",
        "body_contact_penalty",
    ]:
        hydra_key = _rename.get(key, key)
        overrides.append(
            f"env.task_config.weighted_reward_keys.{hydra_key}={reward.get(key, _reward_defaults.get(key, 0))}"
        )

    # ── Per-target geometry ───────────────────────────────────────────────────
    for i in range(num_targets):
        t = cfg.get(f"target_{i + 1}", {})
        target_type = t.get("radio", "Sphere")
        overrides.append(f"+env/task_config/targets/target_{i}={target_type.lower()}")

        if target_type == "Box":
            b = t.get("box", {})
            px = b.get("box_position_x", [0.3, 0.3])
            py = b.get("box_position_y", [0.0, 0.0])
            pz = b.get("box_position_z", [0.0, 0.0])
            # Older configs stored scalars; newer ones store [min, max] pairs.
            if not isinstance(px, list):
                px = [px, px]
            if not isinstance(py, list):
                py = [py, py]
            if not isinstance(pz, list):
                pz = [pz, pz]
            sx = b.get("box_size_x", [0.05, 0.05])
            sy = b.get("box_size_y", [0.05, 0.05])
            if not isinstance(sx, list):
                sx = [sx, sx]
            if not isinstance(sy, list):
                sy = [sy, sy]
            # Studio: a button on a screen carries its full orientation (MuJoCo euler, rad) and thickness (m).
            euler, half_t = button_geometry(b)
            rgb = _parse_rgba(b.get("rgb", "rgba(204.0,25.5,25.5,1)"))
            overrides += [
                f"env.task_config.targets.target_{i}.position="
                f"[[{px[0]},{py[0]},{pz[0]}],[{px[1]},{py[1]},{pz[1]}]]",
                f"env.task_config.targets.target_{i}.size="
                f"[[{0.5*sx[0]},{0.5*sy[0]},{half_t[0]}],[{0.5*sx[1]},{0.5*sy[1]},{half_t[1]}]]",
                f"env.task_config.targets.target_{i}.min_touch_force={b.get('min_touch_force', 1.0)}",
                f"env.task_config.targets.target_{i}.euler=[{euler[0]},{euler[1]},{euler[2]}]",
                f"env.task_config.targets.target_{i}.rgb=[{rgb[0]},{rgb[1]},{rgb[2]}]",
            ]
        else:  # Sphere
            s = t.get("sphere", {})
            xr = s.get("x_range", [0.3, 0.3])
            yr = s.get("y_range", [-0.1, 0.1])
            zr = s.get("z_range", [-0.3, 0.3])
            sr = s.get("size_range", [0.05, 0.15])
            rgb = _parse_rgba(s.get("rgb", "rgba(204.0,25.5,25.5,1)"))
            overrides += [
                f"env.task_config.targets.target_{i}.position="
                f"[[{xr[0]},{yr[0]},{zr[0]}],[{xr[1]},{yr[1]},{zr[1]}]]",
                f"env.task_config.targets.target_{i}.size=[{sr[0]},{sr[1]}]",
                f"env.task_config.targets.target_{i}.dwell_duration={s.get('dwell_duration', 0.25)}",
                f"env.task_config.targets.target_{i}.rgb=[{rgb[0]},{rgb[1]},{rgb[2]}]",
            ]
        overrides.append(
            f"+env.task_config.targets.target_{i}.reject_oor={_reject_oor}"
        )

    # Settings the UI has no control for (e.g. the *_v2 task examples' reward terms): Hydra
    # overrides listed in the preset itself, applied last.
    overrides += [
        _normalize_vision_recipe_override(value)
        for value in cfg.get("extra_overrides", [])
    ]
    return overrides


def training_config_to_labelled_dict(config_dict: dict) -> dict:
    """Convert a full hydra training config dict to the ConfigSaver labelled dict format.

    Accepts the nested dict produced by ``wandb.Api().run(...).config`` or parsed
    from a training JSON (e.g. myointeract_test_configs/pointing_anthro_high.json).
    Returns a dict compatible with ``ConfigSaver.from_labelled_dict()``.
    """
    import math

    env = config_dict.get("env", {})
    rl = config_dict.get("rl", {})
    task = env.get("task_config", {})
    muscle = env.get("muscle_config", {})
    noise = muscle.get("noise_params", {})
    vision = env.get("vision", config_dict.get("vision", {}))
    targets_cfg = task.get("targets", {})

    # ── BM model (reverse lookup from model_path) ─────────────────────────
    _model_to_bm = {v[0]: k for k, v in BM_MODEL_MAP.items()}
    bm_model = _model_to_bm.get(env.get("model_path", ""), "MoBL_Arms_Index")

    # ── RL parameters ─────────────────────────────────────────────────────
    rl_dict = {
        "Number of Total Training Steps": int(rl.get("num_timesteps", 5_000_000)),
        "Checkpoint Interval (in Number of Training Iterations)": int(
            rl.get("save_interval", 30)
        ),
        "Number of Parallel Environments": int(rl.get("num_envs", 1024)),
        "Unroll Length": int(rl.get("unroll_length", 10)),
        "Number of Minibatches": int(rl.get("num_minibatches", 8)),
        "Number of Epochs Per Update": int(rl.get("num_epochs_per_update", 8)),
        "Select Experiment from Which to Load Checkpoints": "None",
        "Select Checkpoint Number": "None",
        "Show Checkpoints from Own Training Runs": False,
        "Target Initial Seed": int(task.get("target_init_seed", 0)),
    }

    # ── BM fields ─────────────────────────────────────────────────────────
    bm_dict = {
        "bm_model": bm_model,
        "ctrl_dt": float(env.get("ctrl_dt", 0.05)),
        "reset_type": task.get("reset_type", "epsilon_uniform"),
        "sigdepnoise_enabled": noise.get("sigdepnoise_type", "None")
        not in (None, "None"),
        "constantnoise_enabled": noise.get("constantnoise_type", "None")
        not in (None, "None"),
        "fatigue_enabled": bool(muscle.get("fatigue_enabled", False)),
        "mj_impl_type": env.get("mj_impl", "mjlab"),
    }

    # ── Task ──────────────────────────────────────────────────────────────
    _t0 = targets_cfg.get("target_0", {})
    task_dict = {
        "max_duration": float(task.get("max_duration", 4.0)),
        "reject_oor": bool(_t0.get("reject_oor", False)),
    }

    # ── Observation space ─────────────────────────────────────────────────
    obs_active = task.get("obs_keys", list(_POSSIBLE_OBS_KEYS))
    fatigue_active = muscle.get("fatigue_obs_keys", [])
    omni_active = task.get("omni_keys", [])
    vis_enabled = bool(vision.get("enabled", False))
    vis_mode = vision.get("vision_mode", "rgb")

    obs_bools = [k in obs_active for k in _POSSIBLE_OBS_KEYS]
    fatigue_bools = [k in fatigue_active for k in _POSSIBLE_FATIGUE_KEYS]
    omni_bools = [k in omni_active for k in _POSSIBLE_OMNI_KEYS]
    if vis_enabled:
        vision_bools = [
            vis_mode in ("rgb", "rgbd")
            if k == "rgb"
            else vis_mode in ("depth", "rgbd", "depth_w_aux_task")
            for k in _POSSIBLE_VISION_KEYS
        ]
    else:
        vision_bools = [False] * len(_POSSIBLE_VISION_KEYS)

    obs_dict = {
        "obs": obs_bools,
        "fatigue": fatigue_bools,
        "vision": vision_bools,
        "omni": omni_bools,
    }

    # ── Reward weights ────────────────────────────────────────────────────
    rw = task.get("weighted_reward_keys", {})
    reward_dict = {
        "distance": float(rw.get("distance", 1.0)),
        "subtask_bonus": float(rw.get("phase_bonus", 0.0)),
        "completion_bonus": float(rw.get("done", 10.0)),
        "neural_effort": float(rw.get("neural_effort", 0.0)),
        "jac_effort": float(rw.get("jac_effort", 0.0)),
        "body_contact_penalty": float(rw.get("torso_contact_penalty", 0.0)),
    }

    # ── Targets ───────────────────────────────────────────────────────────
    num_targets = min(int(targets_cfg.get("num_targets", 1)), 10)

    _DEF_SPHERE = {
        "x_range": [0.3, 0.6],
        "y_range": [-0.3, 0.3],
        "z_range": [-0.3, 0.15],
        "size_range": [0.02, 0.05],
        "dwell_duration": 0.5,
        "rgb": "rgba(204.0,25.5,25.5,1)",
    }
    _DEF_BOX = {
        "box_position_x": [0.4, 0.4],
        "box_position_y": [0.0, 0.0],
        "box_position_z": [0.0, 0.0],
        "box_size_x": [0.05, 0.05],
        "box_size_y": [0.05, 0.05],
        "min_touch_force": 1.0,
        "orientation_angle": 0.0,
        "rgb": "rgba(204.0,25.5,25.5,1)",
    }

    def _rgb_str(rgb):
        return f"rgba({rgb[0]*255:.1f},{rgb[1]*255:.1f},{rgb[2]*255:.1f},1)"

    def _parse_target(t: dict):
        # Saved training configs identify target type via ``name``
        # ("button_target" vs "pointing_target" — see ButtonTarget/PointingTarget
        # in myointeract_configs.py and the name == "button_target" check in
        # register_mjlab_myointeract_tasks.py).
        is_box = t.get("name", "pointing_target") == "button_target"
        pos = t.get("position", [[0.3, -0.3, 0.0], [0.6, 0.3, 0.3]])
        size = t.get("size", [[0.02, 0.02, 0.02], [0.05, 0.05, 0.05]])
        rgb_raw = t.get("rgb", [0.8, 0.1, 0.1])
        rgb_str = _rgb_str(rgb_raw)

        if is_box:
            euler = t.get("euler", [0, 0, 0])
            angle_deg = -euler[1] * 180.0 / math.pi if len(euler) > 1 else 0.0
            if isinstance(pos[0], (int, float)):
                xmin = xmax = float(pos[0])
                ymin = ymax = float(pos[1])
                zmin = zmax = float(pos[2])
            else:
                xmin, ymin, zmin = pos[0]
                xmax, ymax, zmax = pos[1]
            if isinstance(size[0], (int, float)):
                hx_min = hx_max = float(size[0])
                hy_min = hy_max = float(size[1])
                hz_min = hz_max = float(size[2])
            else:
                hx_min, hy_min = size[0][0], size[0][1]
                hx_max, hy_max = size[1][0], size[1][1]
                hz_min, hz_max = size[0][2], size[1][2]
            box_p = {
                "box_position_x": [xmin, xmax],
                "box_position_y": [ymin, ymax],
                "box_position_z": [zmin, zmax],
                "box_size_x": [2 * hx_min, 2 * hx_max],
                "box_size_y": [2 * hy_min, 2 * hy_max],
                "min_touch_force": float(t.get("min_touch_force", 1.0)),
                "orientation_angle": float(angle_deg),
                "rgb": rgb_str,
                "euler": list(euler),
                "box_thickness": 2 * hz_min
                if hz_min == hz_max
                else [2 * hz_min, 2 * hz_max],
            }
            return "Box", box_p, dict(_DEF_SPHERE)
        else:
            if isinstance(pos[0], (int, float)):
                x_range = [float(pos[0]), float(pos[0])]
                y_range = [float(pos[1]), float(pos[1])]
                z_range = [float(pos[2]), float(pos[2])]
            else:
                x_range = [pos[0][0], pos[1][0]]
                y_range = [pos[0][1], pos[1][1]]
                z_range = [pos[0][2], pos[1][2]]
            if isinstance(size[0], (int, float)):
                size_range = [float(size[0]), float(size[1])]
            else:
                size_range = [float(size[0][0]), float(size[1][0])]
            sph_p = {
                "x_range": x_range,
                "y_range": y_range,
                "z_range": z_range,
                "size_range": size_range,
                "dwell_duration": float(t.get("dwell_duration", 0.5)),
                "rgb": rgb_str,
            }
            return "Sphere", dict(_DEF_BOX), sph_p

    target_dicts = {}
    for i in range(10):
        if i < num_targets:
            # Configs commonly define only "target_0" and let every target_i (i >= 1)
            # reuse it as a template -- register_mjlab_myointeract_task()/_update_model_geoms()
            # already fall back this way (`getattr(targets, f"target_{i}", None) or
            # _t0_cfg`). We mirror that here.
            t = targets_cfg.get(f"target_{i}", targets_cfg.get("target_0", {}))
            radio, box_p, sph_p = _parse_target(t)
        else:
            radio, box_p, sph_p = "Sphere", dict(_DEF_BOX), dict(_DEF_SPHERE)
        target_dicts[f"target_{i+1}"] = {"radio": radio, "box": box_p, "sphere": sph_p}

    return {
        "rl": rl_dict,
        "num_targets": num_targets,
        **target_dicts,
        "bm": bm_dict,
        "task": task_dict,
        "obs": obs_dict,
        "reward": reward_dict,
        "extra_overrides": _vision_config_overrides(config_dict),
    }


def extra_hydra_overrides(config_dict: dict) -> list:
    """Generate Hydra-style overrides for fields NOT covered by gradio_json_to_overrides.

    Complements ``gradio_json_to_overrides(labelled)`` to ensure that a loaded
    checkpoint's ``run_state.loaded_policy_extra_overrides`` captures every
    training-time setting, including those absent from the UI (e.g.
    ``fatigue_reset_random``, ``rl.learning_rate``, ``rl.discounting``).

    Accepts the same nested hydra config dict as ``training_config_to_labelled_dict``
    (top-level keys ``"env"`` and ``"rl"``).  None / null values are skipped.
    """
    overrides: list = []
    env = config_dict.get("env", {})
    rl = config_dict.get("rl", {})

    # ── muscle_config: full sub-tree ────────────────────────────────────────
    # gradio_json_to_overrides already emits fatigue_enabled, noise.*_type,
    # fatigue_obs_keys — but duplicates are fine (last wins in Hydra).
    mc = env.get("muscle_config", {})
    noise = mc.get("noise_params", {})
    for nk, nv in noise.items():
        if nv is not None:
            overrides.append(f"env.muscle_config.noise_params.{nk}={nv}")
    _mc_skip = {"noise_params", "fatigue_obs_keys"}
    for key, val in mc.items():
        if key in _mc_skip or val is None:
            continue
        overrides.append(f"env.muscle_config.{key}={val}")

    # ── task_config: extra scalars not in gradio_json_to_overrides ──────────
    tc = env.get("task_config", {})
    for key in (
        "distance_metric",
        "dwell_duration",
        "enable_extra_dist",
        "exponential_distance_reward",
        "max_trials",
        "show_all_targets",
    ):
        val = tc.get(key)
        if val is not None:
            overrides.append(f"env.task_config.{key}={val}")

    # ── env-level extras ─────────────────────────────────────────────────────
    for key in ("sim_dt", "eval_mode"):
        val = env.get(key)
        if val is not None:
            overrides.append(f"env.{key}={val}")

    # ── reward weights: keys not covered by the RewardFunction UI ───────────
    # gradio_json_to_overrides() only round-trips distance/phase_bonus/done/
    # neural_effort through UI sliders. Any other weighted_reward_keys entry
    # (e.g. jac_effort, fatigue_mean) has no widget and would otherwise be
    # silently dropped on checkpoint load, falling back to the env's default
    # (typically 0.0) instead of the value it was actually trained with.
    # ``++`` (rather than a plain override) is required because keys like
    # ``fatigue_mean`` are not present in every task config's default
    # weighted_reward_keys dict (e.g. UniversalTaskConfig in
    # myointeract_configs.py) -- Hydra's struct mode rejects a plain override
    # for a dict key that isn't already defined, but ``++`` overrides it
    # if present and appends it otherwise.
    _reward_ui_keys = {
        "distance",
        "phase_bonus",
        "done",
        "neural_effort",
        "torso_contact_penalty",
    }
    rw = tc.get("weighted_reward_keys", {})
    for key, val in rw.items():
        if key in _reward_ui_keys or val is None:
            continue
        overrides.append(f"++env.task_config.weighted_reward_keys.{key}={val}")

    # ── rl: everything not already in gradio_json_to_overrides ───────────────
    _rl_ui_keys = {
        "num_timesteps",
        "save_interval",
        "num_envs",
        "unroll_length",
        "num_minibatches",
        "num_epochs_per_update",
        "load_checkpoint_path",
        # network_factory is a Hydra config group (rl/network_factory=vision|no_vision);
        # its sub-keys cannot be overridden individually with dot notation.
        "network_factory",
        # num_checkpoints is a myointeract-only field not present in the Hydra rl schema.
        "num_checkpoints",
    }
    for key, val in rl.items():
        if key in _rl_ui_keys or val is None:
            continue
        if isinstance(val, dict):
            for subk, subv in val.items():
                if subv is not None:
                    overrides.append(f"rl.{key}.{subk}={subv}")
        else:
            overrides.append(f"rl.{key}={val}")

    return overrides + _vision_config_overrides(config_dict)


# ---------------------------------------------------------------------------
# Hydra entry-point (for direct CLI use)
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_name="config")
def my_app(cfg: Config) -> None:
    if getattr(cfg.env, "env_name", None) == "MyoInteractUniversal":
        with open_dict(cfg):
            cfg.env.task_config.targets = "default"
    container = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    print(OmegaConf.to_yaml(container))


if __name__ == "__main__":
    register_universal_configs()
    my_app()
