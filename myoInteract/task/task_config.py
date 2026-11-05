# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from dataclasses import dataclass
from omegaconf import MISSING

from ..common import MyoComponentConfig


@dataclass
class TaskConfig(MyoComponentConfig):
    model_path: str = MISSING
