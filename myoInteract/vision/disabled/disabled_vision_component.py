# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from mjlab.rl import RslRlOnPolicyRunnerCfg

from myoInteract.common import MyoComponent
from .disabled_vision_config import DisabledVisionConfig
from ..vision_registry import myo_register_vision


@myo_register_vision("disabled")
class DisabledVisionComponent(MyoComponent):
    def __init__(self, cfg: DisabledVisionConfig):
        self.cfg = cfg

    def modify_rl_cfg(self, cfg: RslRlOnPolicyRunnerCfg) -> None:
        cfg.actor.class_name = "MLPModel"
        cfg.obs_groups = {
            "actor": ("agent_state", "task_state"),
            "critic": ("agent_state", "task_state"),
        }
