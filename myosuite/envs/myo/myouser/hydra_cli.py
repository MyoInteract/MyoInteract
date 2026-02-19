from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import Any, List, Union, Dict, Callable

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
from enum import Enum
from hydra.utils import instantiate, call
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
from ml_collections import ConfigDict
from myosuite.envs.myo.myouser.base import BaseEnvConfig
from myosuite.envs.myo.myouser.myouser_universal import UniversalEnvConfig, LIST_CONFIGS, ButtonTarget, PointingTarget

OmegaConf.register_new_resolver("check_string", lambda x: "" if x is None else "-" + str(x))

def select_network(vision_enabled):
    if vision_enabled == "enabled":
        return "vision"
    else:
        return "no_vision"

def select_targets(num_targets):
    from myosuite.envs.myo.myouser.myouser_universal import (
        OneTargetConfig, TwoTargetConfig, ThreeTargetConfig, FourTargetConfig,
        FiveTargetConfig, SixTargetConfig, SevenTargetConfig, EightTargetConfig,
        NineTargetConfig, TenTargetConfig
    )

    target_configs = [
        None, OneTargetConfig(), TwoTargetConfig(), ThreeTargetConfig(),
        FourTargetConfig(), FiveTargetConfig(), SixTargetConfig(),
        SevenTargetConfig(), EightTargetConfig(), NineTargetConfig(), TenTargetConfig()
    ]

    if 1 <= num_targets <= 10:
        return target_configs[num_targets]
    else:
        raise ValueError(f"num_targets must be between 1 and 10, got {num_targets}")

OmegaConf.register_new_resolver("select_network", select_network)
OmegaConf.register_new_resolver("select_targets", select_targets)

@dataclass
class WANDBEnabledConfig:
    enabled: bool = True
    entity: Union[str, None] = 'hci-biomechsims' # Set to this by default, choose a different entity for personal projects
    name: Union[str, None] = '${env.env_name}-${now:%Y%m%d}-${now:%H%M%S}${check_string:${run.suffix}}'
    project: str = "MJXRL"
    tags: Union[List[str], None] = None
    group: Union[str, None] = None

@dataclass
class WANDBDisabledConfig:
    enabled: bool = False
    
@dataclass
class NetworkConfig:
    policy_hidden_layer_sizes: List[int] = field(
        default_factory=lambda: [128, 128, 128, 128]
    )
    value_hidden_layer_sizes: List[int] = field(
        default_factory=lambda: [256, 256, 256, 256, 256]
    )

@dataclass
class VisionNetworkConfig(NetworkConfig):
    policy_hidden_layer_sizes: List[int] = field(
        default_factory=lambda: [32, 32, 32, 32]
    )
    value_hidden_layer_sizes: List[int] = field(
        default_factory=lambda: [256, 256, 256, 256, 256]
    )
    encoder_out_size: int = 4
    cheat_vision_aux_output: bool = False
    has_vision_aux_output: bool = True
    vision_aux_output_mlp: bool = True
    vision_aux_output_mlp_output_size: int = 4
    vision_encoder_normalize_output: bool = True
    stop_vision_gradient: bool = False


@dataclass
class RLConfig:
    num_timesteps: int = 15_000_000
    log_training_metrics: bool = True
    training_metrics_steps: int = 100000
    num_evals: int = 0
    num_checkpoints: int = 1
    reward_scaling: float = 0.1
    episode_length: int = "${int_divide:${env.task_config.max_duration},${env.ctrl_dt}}" #TODO: check and fix this dependency!
    clipping_epsilon: float = 0.3
    normalize_observations: bool = True
    action_repeat: int = 1
    unroll_length: int = 10
    num_minibatches: int = 8
    num_updates_per_batch: int = 8
    num_resets_per_eval: int = 1
    discounting: float = 0.97
    learning_rate: float = 3e-4
    entropy_cost: float = 0.001
    num_envs: int = 1024
    batch_size: int = 128
    max_grad_norm: float = 1.0
    network_factory: NetworkConfig = field(default_factory=lambda: NetworkConfig())
    load_checkpoint_path: Union[str, None] = None

class VisionModes(str, Enum):
    rgbd = 'rgbd'
    depth = 'depth'
    depth_w_aux_task = 'depth_w_aux_task'

@dataclass
class VisionEnabledConfig:
    enabled: bool = True
    vision_mode: VisionModes = field(default_factory=lambda: VisionModes.rgbd)
    gpu_id: int = 0
    num_worlds: int = '${rl.num_envs}'
    render_width: int = 120
    render_height: int = 120
    enabled_geom_groups: List[int] = field(default_factory=lambda: [0, 1, 2])
    enabled_cameras: List[int] = field(default_factory=lambda: [0])
    use_rasterizer: bool = False

@dataclass
class VisionDisabledConfig:
    enabled: bool = False

defaults = [
    {'wandb': 'enabled'},
    {'vision': 'disabled'},
    {'env': 'pointing'},
    {'rl': 'rl_config'},
    {'run': 'run'},
    {'rl/network_factory': '${select_network:${vision}}'},
    {'env/task_config/targets': 'default'},

]

@dataclass
class RunConfig:
    seed: int = 0
    play_only: bool = False
    use_tb: bool = False
    rscope_envs: Union[int, None] = None
    deterministic_rscope: bool = True
    domain_randomization: bool = False
    suffix: Union[str, None] = None
    local_plotting: bool = False
    log_wandb_videos: bool = True
    eval_episodes: int = 10
    eval_seed: int = 123
    using_gradio: bool = False

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    wandb: Any = MISSING
    vision: Any = MISSING
    env: BaseEnvConfig = MISSING
    rl: RLConfig = MISSING
    run: RunConfig = MISSING

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="wandb", name="enabled", node=WANDBEnabledConfig)
cs.store(group="wandb", name="disabled", node=WANDBDisabledConfig)
cs.store(group="vision", name="enabled", node=VisionEnabledConfig)
cs.store(group="vision", name="disabled", node=VisionDisabledConfig)
cs.store(group="env", name="base_env_config", node=BaseEnvConfig)
cs.store(group="env", name="universal", node=UniversalEnvConfig)
cs.store(group="rl", name="rl_config", node=RLConfig)
cs.store(group="run", name="run", node=RunConfig)
cs.store(group="rl/network_factory", name="vision", node=VisionNetworkConfig)
cs.store(group="rl/network_factory", name="no_vision", node=NetworkConfig)

for (i, name, config) in LIST_CONFIGS:
    cs.store(group="env/task_config/targets", name=name, node=config)
    for j in range(0, i+1):
        cs.store(group=f"env/task_config/targets/target_{j}", name="sphere", node=PointingTarget)
        cs.store(group=f"env/task_config/targets/target_{j}", name="box", node=ButtonTarget)
    


def load_config_interactive(overrides=[], cfg_only=False):
    """
    Use this function to load the config interactively from a jupyer notebook.

    Example Usage:
    ----
    >>> from myosuite.envs.myo.myouser.hydra_cli import load_config_interactive
    >>> overrides=["env=pointing", "env.task_config.reach_settings.target_radius_range.fingertip=[0.01,0.15]"]
    >>> config_dict=load_config_interactive(overrides)
    """
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    with initialize(version_base=None, config_path=None):
        cfg = compose(config_name="config", overrides=overrides)

    if cfg_only:
        return cfg
    
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    container['env']['vision'] = container['vision']
    config = ConfigDict(container)
    return config

@hydra.main(version_base=None, config_name="config")
def my_app(cfg: Config) -> None:
    container = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    print(OmegaConf.to_yaml(container))

if __name__ == "__main__":
    my_app()