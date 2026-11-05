# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task

from .common import dataclass_as_base
from .myo_config import MyoConfig
from .task import myo_create_task
from .vision import myo_create_vision


def _myo_default_env_cfg(cfg: MyoConfig, play: bool) -> ManagerBasedRlEnvCfg:
    return ManagerBasedRlEnvCfg(
        decimation=cfg.env.decimation,
        scene=SceneCfg(num_envs=cfg.env.num_play_envs if play else cfg.env.num_envs),
        seed=cfg.env.seed,
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=cfg.env.sim_timestep),
        ),
        episode_length_s=cfg.env.max_episode_length,
    )


def _myo_default_rsl_runner(cfg: MyoConfig) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=cfg.rl.actor_hidden_dims,
            activation=cfg.rl.activation,
            obs_normalization=cfg.rl.obs_normalization,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
            class_name="MLPModel",
        ),
        critic=RslRlModelCfg(
            hidden_dims=cfg.rl.critic_hidden_dims,
            activation=cfg.rl.activation,
            obs_normalization=cfg.rl.obs_normalization,
            class_name="MLPModel",
        ),
        algorithm=dataclass_as_base(cfg.rl, RslRlPpoAlgorithmCfg),
        num_steps_per_env=cfg.rl.num_steps_per_env,
        max_iterations=cfg.rl.max_iterations,
        obs_groups={},
        save_interval=cfg.rl.save_interval,
        wandb_project=cfg.wandb.project,
        experiment_name=cfg.wandb.name,
        wandb_tags=cfg.wandb.tags,
    )


def myo_mjlab_register(cfg: MyoConfig) -> None:
    env_cfg = _myo_default_env_cfg(cfg, play=False)
    play_cfg = _myo_default_env_cfg(cfg, play=True)
    rl_cfg = _myo_default_rsl_runner(cfg)

    components = [
        myo_create_task(cfg.task),
        myo_create_vision(cfg.vision),
    ]

    for component in components:
        component.modify_env_cfg(env_cfg, False)
        component.modify_env_cfg(play_cfg, True)
        component.modify_rl_cfg(rl_cfg)

    register_mjlab_task(
        task_id="MyoInteract",
        env_cfg=env_cfg,
        play_env_cfg=play_cfg,
        rl_cfg=rl_cfg,
        runner_cls=None,
    )
