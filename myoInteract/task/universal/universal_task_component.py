# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
# Full task logic should be ported from:
#   myosuite/envs/myo/backends/mjlab/register_mjlab_myointeract_tasks.py
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from myoInteract.common import MyoComponent
from .universal_task_config import UniversalTaskConfig
from ..task_registry import myo_register_task


@myo_register_task("universal")
class UniversalTaskComponent(MyoComponent):
    """Universal pointing/interaction task component.

    Registers target geometry and configures observations, rewards, events,
    and terminations for the universal myointeract task.

    TODO: Port the full implementation from
        myosuite/envs/myo/backends/mjlab/register_mjlab_myointeract_tasks.py
    """

    def __init__(self, cfg: UniversalTaskConfig):
        self.cfg = cfg

    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        # TODO: configure observations, actions, rewards, events, terminations
        # based on self.cfg (pointing targets, reach config, etc.)
        raise NotImplementedError(
            "UniversalTaskComponent.modify_env_cfg must be ported from "
            "myosuite/envs/myo/backends/mjlab/register_mjlab_myointeract_tasks.py"
        )

    def modify_rl_cfg(self, cfg: RslRlOnPolicyRunnerCfg) -> None:
        pass
