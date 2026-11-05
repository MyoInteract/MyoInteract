# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from dataclasses import dataclass, field
from typing import Any
from omegaconf import MISSING, OmegaConf
from hydra.core.config_store import ConfigStore
from mjlab.rl import RslRlPpoAlgorithmCfg

from .task import TaskConfig
from .vision import VisionConfig


OmegaConf.register_new_resolver(
    "myo_suffix_append", lambda x: "" if x is None else "-" + str(x)
)


@dataclass
class EnvConfig:
    seed: int | None = 0
    num_envs: int = 4096
    num_play_envs: int = 1
    max_episode_length: int = 10
    decimation: int = 25
    sim_timestep: float = 0.002


@dataclass
class RlConfig(RslRlPpoAlgorithmCfg):
    num_steps_per_env: int = 10
    max_iterations: int = 250
    save_interval: int = 250

    activation: str = "elu"
    obs_normalization: bool = True
    actor_hidden_dims: tuple[int, ...] = field(
        default_factory=lambda: (128, 128, 128, 128)
    )
    critic_hidden_dims: tuple[int, ...] = field(
        default_factory=lambda: (256, 256, 256, 256)
    )

    num_learning_epochs: int = 8
    num_mini_batches: int = 8
    learning_rate: float = 3e-4
    schedule: str = "adaptive"
    gamma: float = 0.97
    lam: float = 0.95
    entropy_coef: float = 0.001
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.3
    optimizer: str = "adam"


@dataclass
class WanDbConfig:
    enabled: bool = True
    project: str = "mjlab"
    name: str = "${hydra:runtime.choices.task}-${now:%Y%m%d}-${now:%H%M%S}${myo_suffix_append:${wandb.suffix}}"
    tags: tuple[str, ...] = ()
    suffix: str | None = None


@dataclass
class MyoConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"task": "universal"},
            {"vision": "disabled"},
        ]
    )

    env: EnvConfig = field(default_factory=EnvConfig)
    rl: RlConfig = field(default_factory=RlConfig)
    wandb: WanDbConfig = field(default_factory=WanDbConfig)

    task: TaskConfig = MISSING
    vision: VisionConfig = MISSING


ConfigStore.instance().store(name="myo_config", node=MyoConfig)
