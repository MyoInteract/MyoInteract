from dataclasses import dataclass, field, make_dataclass
from enum import Enum
from typing import Any
from omegaconf import MISSING


@dataclass
class NoiseParams:
    sigdepnoise_type: str | None = None
    sigdepnoise_level: float = 0.103
    constantnoise_type: str | None = None
    constantnoise_level: float = 0.185


@dataclass
class MuscleConfig:
    muscle_condition: str | None = None  # TODO: remove as deprecated
    sex: str | None = None
    control_type: str = "default"
    fatigue_enabled: bool = False
    fatigue_reset_vec: list[float] | None = None
    fatigue_reset_random: bool = False
    fatigue_obs_keys: list[str] = field(default_factory=lambda: [])
    noise_params: NoiseParams = field(default_factory=lambda: NoiseParams())
    dep_enabled: bool = False  # TODO: move into RL config? (requires passing ppo_params or even entire config to registry.load()...)
    dep_stochastic_switch: bool = True


@dataclass
class BaseEnvConfig:
    env_name: str = MISSING
    model_path: str = MISSING
    mj_impl: str = "mjlab"
    ctrl_dt: float = 0.002 * 25
    sim_dt: float = 0.002
    muscle_config: MuscleConfig = field(default_factory=lambda: MuscleConfig())
    eval_mode: bool = False


# OmegaConf.register_new_resolver(
#     "check_string", lambda x: "" if x is None else "-" + str(x)
# )


def select_network(vision_enabled):
    if vision_enabled == "enabled":
        return "vision"
    else:
        return "no_vision"


def select_targets(num_targets):
    target_configs = [
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
        return target_configs[num_targets]
    if num_targets > 15:
        return create_target_config_class(num_targets)()
    raise ValueError(f"num_targets must be at least 1, got {num_targets}")


# OmegaConf.register_new_resolver("select_network", select_network)
# OmegaConf.register_new_resolver("select_targets", select_targets)


@dataclass
class WANDBEnabledConfig:
    enabled: bool = True
    entity: str | None = None  # None: the default entity of the W&B login
    name: str | None = (
        "${env.env_name}-${now:%Y%m%d}-${now:%H%M%S}${check_string:${run.suffix}}"
    )
    project: str = "myointeract"
    tags: list[str] | None = None
    group: str | None = None


@dataclass
class WANDBDisabledConfig:
    enabled: bool = False
    entity: str | None = None  # None: the default entity of the W&B login
    name: str | None = (
        "${env.env_name}-${now:%Y%m%d}-${now:%H%M%S}${check_string:${run.suffix}}"
    )
    project: str = "myointeract"
    tags: list[str] | None = None
    group: str | None = None


@dataclass
class NetworkConfig:
    policy_hidden_layer_sizes: list[int] = field(
        default_factory=lambda: [128, 128, 128, 128]
    )
    value_hidden_layer_sizes: list[int] = field(
        default_factory=lambda: [256, 256, 256, 256, 256]
    )


@dataclass
class VisionNetworkConfig(NetworkConfig):
    policy_hidden_layer_sizes: list[int] = field(
        default_factory=lambda: [32, 32, 32, 32]
    )
    value_hidden_layer_sizes: list[int] = field(
        default_factory=lambda: [256, 256, 256, 256, 256]
    )
    encoder_out_size: int = 4
    cheat_vision_aux_output: bool = False
    has_vision_aux_output: bool = False
    vision_aux_output_mlp: bool = True
    vision_aux_output_decoder: bool = False
    vision_aux_output_mlp_output_size: int = 4
    vision_encoder_normalize_output: bool = True
    stop_vision_gradient: bool = False


@dataclass
class RLConfig:
    num_timesteps: int = 15_000_000
    log_training_metrics: bool = True  # unused
    training_metrics_steps: int = 100000  # unused
    num_evals: int = 0  # unused
    save_interval: int = 30  # replaces num_checkpoints
    reward_scaling: float = 0.1  # unused
    episode_length: int = "${int_divide:${env.task_config.max_duration},${env.ctrl_dt}}"  # TODO: check and fix this dependency!  # deprecated (env.task_config.max_duration is used directly)
    clipping_epsilon: float = 0.3
    normalize_observations: bool = True  # unused (could be used though)
    action_repeat: int = 1  # unused (could be linked with "decimation" though)
    unroll_length: int = 10
    num_minibatches: int = 8
    num_epochs_per_update: int = 8
    num_resets_per_eval: int = 1  # unused
    discounting: float = 0.97
    learning_rate: float = 3e-4
    entropy_cost: float = 0.001
    num_envs: int = 4096
    batch_size: int = 128  # deprecated
    num_eval_envs: int = 5  # unused
    max_grad_norm: float = 1.0
    network_factory: NetworkConfig = field(
        default_factory=lambda: NetworkConfig()
    )  # unused
    load_checkpoint_path: str | None = None


class VisionModes(str, Enum):
    rgbd = "rgbd"
    rgb = "rgb"
    depth = "depth"
    depth_w_aux_task = "depth_w_aux_task"


@dataclass
class VisionEnabledConfig:
    enabled: bool = True
    # Version 1 checkpoints used a fixed RGBD/128px recipe regardless of settings.
    recipe_version: int = 2
    vision_mode: VisionModes = field(default_factory=lambda: VisionModes.rgb)
    gpu_id: int = 0
    render_width: int = 96  # 640  #960
    render_height: int = 96  # 480  #720
    depth_min: float = 0.01
    depth_cutoff: float = 1.0
    enabled_geom_groups: list[int] = field(default_factory=lambda: [0, 1, 2])
    observation_cameras: list[int] = field(default_factory=lambda: [0])
    evaluation_cameras: list[int] = field(default_factory=lambda: [0, 1])
    use_rasterizer: bool = False
    num_train_envs: int = 1024  # do not override, use RLConfig.num_envs instead!
    num_eval_envs: int = 8  # do not override, use RLConfig.num_eval_envs instead!


@dataclass
class VisionDisabledConfig:
    enabled: bool = False


defaults = [
    {"wandb": "enabled"},
    {"vision": "disabled"},
    {"env": "pointing"},
    {"rl": "rl_config"},
    {"run": "run"},
    {"rl/network_factory": "${select_network:${vision}}"},
]


@dataclass
class RunConfig:
    seed: int = 0
    play_only: bool = False
    use_tb: bool = False
    rscope_envs: int | None = None
    deterministic_rscope: bool = True
    domain_randomization: bool = False
    suffix: str | None = None
    local_plotting: bool = False
    log_wandb_videos: bool = True
    eval_episodes: int = 10
    eval_seed: int = 123
    using_gradio: bool = False


@dataclass
class Config:
    defaults: list[Any] = field(default_factory=lambda: defaults)
    wandb: Any = MISSING
    vision: Any = MISSING
    env: BaseEnvConfig = MISSING
    rl: RLConfig = MISSING
    run: RunConfig = MISSING


@dataclass
class UniversalConfig(Config):
    defaults: list[Any] = field(
        default_factory=lambda: [
            {"wandb": "enabled"},
            {"vision": "disabled"},
            {"env": "universal"},
            {"rl": "rl_config"},
            {"run": "run"},
            {"rl/network_factory": "${select_network:${vision}}"},
            {"env/task_config/targets": "default"},
        ]
    )


########################
### POINTING CONFIGS ###
########################


@dataclass
class ReachSettings:
    ref_site: str = "humphant"
    ee_site: str = "fingertip"
    target_origin_rel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class PointingTaskConfig:
    """
    Default config for pointing task.
    """

    reach_settings: ReachSettings = field(default_factory=lambda: ReachSettings())
    obs_keys: list[str] = field(
        default_factory=lambda: [
            "qpos",
            "qvel",
            "qacc",
            "ee_pos",
            "act",
        ]
    )
    omni_keys: list[str] = field(
        default_factory=lambda: ["target_pos", "target_radius"]
    )
    weighted_reward_keys: dict[str, float] = field(
        default_factory=lambda: {
            "distance": 1,
            "bonus": 8,
        }
    )
    distance_metric: float = 10.0
    max_duration: float = 4.0
    dwell_duration: float = 0.25
    max_trials: int = 1
    reset_type: str = "range_uniform"
    using_vision_domain_randomisation: bool = "${run.domain_randomization}"


@dataclass
class PointingEnvConfig(BaseEnvConfig):
    env_name: str = "MyoInteractPointing"
    model_path: str = (
        "myo_user/envs/myo/assets/arm/mobl_arms_index_universal_myointeract.xml"
    )
    task_config: PointingTaskConfig = field(
        default_factory=lambda: PointingTaskConfig()
    )


#########################
### TRACKING CONFIGS ###
#########################


######################
### TARGET CONFIGS ###
######################


@dataclass
class IndividualTargetConfig:
    name: str = MISSING
    rgb: list[float] = MISSING


@dataclass
class TargetsConfig:
    @property
    def targets(self):
        return [getattr(self, f"target_{i}") for i in range(self.num_targets)]


@dataclass
class PointingTarget(IndividualTargetConfig):
    # penetrable: bool = False
    name: str = "pointing_target"
    # Position can either be a 3d vector or a 2 x list of 3d vectors specifying the min and max values for each dimension
    position: list[list[float]] = field(
        default_factory=lambda: [[0.225, -0.1, -0.3], [0.35, 0.1, 0.3]]
    )
    shape: str = "sphere"
    # Size can either be a single value or a list of 2 values specifying the min and max values
    size: list[float] = field(default_factory=lambda: [0.05, 0.15])
    site_pos: list[float] = field(
        default_factory=lambda: [0, 0, 0.01]
    )  # deprecated, should be removed
    # Any rewards received when inside the target
    reward_incentive: float = 0.0
    completion_bonus: float = 0.0
    dwell_duration: float = 0.25
    rgb: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])


@dataclass
class ButtonTarget(IndividualTargetConfig):
    position: list[list[float]] = MISSING
    name: str = "button_target"
    size: list[list[float]] = field(
        default_factory=lambda: [[0.025, 0.025, 0.01], [0.025, 0.025, 0.01]]
    )
    site_pos: list[float] = field(default_factory=lambda: [0, 0, 0.01])
    geom_margin: float = 0.0
    completion_bonus: float = 0.0
    min_touch_force: float = 1.0
    rgb: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    euler: list[float] = field(default_factory=lambda: [0, -0.79, 0])


@dataclass
class SingleTargetConfig(TargetsConfig):
    target_0: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]]
        )
    )
    num_targets: int = 1


@dataclass
class DefaultTargetsConfig(TargetsConfig):
    target_0: PointingTarget = field(default_factory=lambda: PointingTarget())
    target_1: ButtonTarget = field(
        default_factory=lambda: ButtonTarget(
            position=[[0.41, -0.05, -0.16], [0.41, -0.05, -0.16]], rgb=[0.8, 0.1, 0.8]
        )
    )
    target_2: PointingTarget = field(
        default_factory=lambda: PointingTarget(rgb=[0.4, 0.0, 1.0])
    )
    target_3: ButtonTarget = field(
        default_factory=lambda: ButtonTarget(
            position=[[0.41, 0.07, -0.16], [0.41, 0.07, -0.16]], rgb=[0.1, 0.8, 0.8]
        )
    )
    target_4: PointingTarget = field(
        default_factory=lambda: PointingTarget(rgb=[0.0, 1.0, 0.4])
    )
    target_5: ButtonTarget = field(
        default_factory=lambda: ButtonTarget(
            position=[[0.50, -0.05, -0.06], [0.50, -0.05, -0.06]], rgb=[0.4, 1.0, 0.2]
        )
    )
    target_6: PointingTarget = field(
        default_factory=lambda: PointingTarget(rgb=[1.0, 1.0, 0.0])
    )
    target_7: ButtonTarget = field(
        default_factory=lambda: ButtonTarget(
            position=[[0.50, 0.07, -0.06], [0.50, 0.07, -0.06]], rgb=[0.1, 0.1, 0.1]
        )
    )
    num_targets: int = 8


@dataclass
class TenPointingTargetsConfig(TargetsConfig):
    num_targets: int = 10
    target_0: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.8, 0.1, 0.1]
        )
    )
    target_1: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.1, 0.8, 0.1]
        )
    )
    target_2: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.1, 0.1, 0.8]
        )
    )
    target_3: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.8, 0.8, 0.1]
        )
    )
    target_4: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.4, 0.4, 0.4]
        )
    )
    target_5: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.4, 0.1, 0.4]
        )
    )
    target_6: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.1, 0.4, 0.4]
        )
    )
    target_7: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.4, 0.4, 0.1]
        )
    )
    target_8: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.1, 0.1, 0.1]
        )
    )
    target_9: PointingTarget = field(
        default_factory=lambda: PointingTarget(
            position=[[0.55, -0.2, -0.3], [0.55, 0.4, 0.3]], rgb=[0.8, 0.8, 0.8]
        )
    )


########################
### UNIVERSAL CONFIG ###
########################


@dataclass
class UniversalTaskConfig:
    reach_settings: ReachSettings = field(default_factory=lambda: ReachSettings())
    obs_keys: list[str] = field(
        default_factory=lambda: [
            "qpos",
            "qvel",
            "qacc",
            "ee_pos",
            "act",
        ]
    )
    omni_keys: list[str] = field(
        default_factory=lambda: [
            "target_pos",
            "target_size",
            "phase_progress",
            "dwell_fraction",
        ]
    )
    weighted_reward_keys: dict[str, float] = field(
        default_factory=lambda: {
            "distance": 1,
            "phase_bonus": 0,
            "done": 10,
            "neural_effort": 0.000,
            "jac_effort": 0,
            "elbow_velocity_penalty": 0,
            "fatigue_mean": 0,
            "fatigue_max": 0,
            "torso_contact_penalty": 0,
            "pain_penalty": 0,
        }
    )
    distance_metric: float = 10.0
    max_duration: float = 6.0
    dwell_duration: float = 0.25
    max_trials: int = 1
    reset_type: str = "range_uniform"
    targets: TargetsConfig = MISSING
    show_all_targets: bool = True
    target_init_seed: int = 1
    exponential_distance_reward: bool = False
    enable_extra_dist: bool = True


@dataclass
class UniversalEnvConfig(BaseEnvConfig):
    env_name: str = "MyoInteractUniversal"
    model_path: str = (
        "myo_user/envs/myo/assets/arm/mobl_arms_index_universal_myointeract.xml"
    )
    task_config: UniversalTaskConfig = field(
        default_factory=lambda: UniversalTaskConfig()
    )


def create_target_config_class(num_targets: int) -> type:
    """Create a dataclass with the specified number of target fields."""
    fields = [
        (f"target_{i}", IndividualTargetConfig, MISSING) for i in range(num_targets)
    ]
    fields.append(("num_targets", int, num_targets))

    num2str = [
        "",
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
        "Six",
        "Seven",
        "Eight",
        "Nine",
        "Ten",
        "Eleven",
        "Twelve",
        "Thirteen",
        "Fourteen",
        "Fifteen",
    ]

    # Named classes up to fifteen (as presets and checkpoints know them); beyond, by number.
    class_name = (
        f"{num2str[num_targets]}TargetConfig"
        if num_targets < len(num2str)
        else f"Targets{num_targets}Config"
    )

    return make_dataclass(
        class_name,
        fields,
        frozen=False,
        bases=(TargetsConfig,),
    )


# # Usage
OneTargetConfig = create_target_config_class(1)
TwoTargetConfig = create_target_config_class(2)
ThreeTargetConfig = create_target_config_class(3)
FourTargetConfig = create_target_config_class(4)
FiveTargetConfig = create_target_config_class(5)
SixTargetConfig = create_target_config_class(6)
SevenTargetConfig = create_target_config_class(7)
EightTargetConfig = create_target_config_class(8)
NineTargetConfig = create_target_config_class(9)
TenTargetConfig = create_target_config_class(10)
ElevenTargetConfig = create_target_config_class(11)
TwelveTargetConfig = create_target_config_class(12)
ThirteenTargetConfig = create_target_config_class(13)
FourteenTargetConfig = create_target_config_class(14)
FifteenTargetConfig = create_target_config_class(15)


LIST_CONFIGS = [
    (1, "one", OneTargetConfig),
    (2, "two", TwoTargetConfig),
    (3, "three", ThreeTargetConfig),
    (4, "four", FourTargetConfig),
    (5, "five", FiveTargetConfig),
    (6, "six", SixTargetConfig),
    (7, "seven", SevenTargetConfig),
    (8, "eight", EightTargetConfig),
    (9, "nine", NineTargetConfig),
    (10, "ten", TenTargetConfig),
    (11, "eleven", ElevenTargetConfig),
    (12, "twelve", TwelveTargetConfig),
    (13, "thirteen", ThirteenTargetConfig),
    (14, "fourteen", FourteenTargetConfig),
    (15, "fifteen", FifteenTargetConfig),
    (7, "default", DefaultTargetsConfig),
    (10, "ten_pointing", TenPointingTargetsConfig),
    (1, "single", SingleTargetConfig),
]
