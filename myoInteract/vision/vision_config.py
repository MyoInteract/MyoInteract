# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from dataclasses import dataclass, field
from omegaconf import MISSING

from ..common import MyoComponentConfig


@dataclass
class VisionConfig(MyoComponentConfig):
    use_textures: bool = True
    use_shadows: bool = False
    enabled_geom_groups: list[int] = field(default_factory=lambda: [0, 3])

    def __post_init__(self) -> None:
        if len(self.enabled_geom_groups) == 0:
            raise ValueError("enabled_geom_groups must contain at least one geom group")
        if any(g < 0 for g in self.enabled_geom_groups):
            raise ValueError("enabled_geom_groups must not contain negative values")
        if len(set(self.enabled_geom_groups)) != len(self.enabled_geom_groups):
            raise ValueError("enabled_geom_groups must not contain duplicates")


@dataclass
class MonoscopicVisionConfig(VisionConfig):
    camera_name: str = MISSING
    width: int = 128
    height: int = 128

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.camera_name:
            raise ValueError("camera_name must not be empty")
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.height <= 0:
            raise ValueError("height must be positive")


@dataclass
class MonoscopicDepthVisionConfig(MonoscopicVisionConfig):
    depth_min: float = 0.01
    depth_cutoff: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.depth_min < 0.0:
            raise ValueError("depth_min must be non-negative")
        if self.depth_cutoff <= 0.0:
            raise ValueError("depth_cutoff must be positive")
        if self.depth_min >= self.depth_cutoff:
            raise ValueError("depth_min must be smaller than depth_cutoff")
