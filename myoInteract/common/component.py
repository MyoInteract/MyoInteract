# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg


@dataclass
class MyoComponentConfig:
    pass


class MyoComponent:
    def modify_env_cfg(self, cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
        """Mutate the environment config in place."""
        pass

    def modify_rl_cfg(self, cfg: RslRlOnPolicyRunnerCfg) -> None:
        """Mutate the RL runner config in place."""
        pass
