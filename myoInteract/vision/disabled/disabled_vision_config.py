# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from dataclasses import dataclass
from ..vision_config import VisionConfig


@dataclass
class DisabledVisionConfig(VisionConfig):
    pass
