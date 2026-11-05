# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, Any
from omegaconf import OmegaConf

Vec3: TypeAlias = list[float]


@dataclass
class ScalarRange:
    value: float | None = None
    min: float | None = None
    max: float | None = None

    def __post_init__(self) -> None:
        has_value = self.value is not None
        has_range = self.min is not None or self.max is not None
        if has_value and has_range:
            raise ValueError("ScalarRange must use either value or min/max, not both")
        if not has_value and (self.min is None or self.max is None):
            raise ValueError("ScalarRange requires either value or both min and max")

    @staticmethod
    def of(value: Any) -> ScalarRange:
        if isinstance(value, ScalarRange):
            return value
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, dict):
            return ScalarRange(**value)
        raise TypeError(f"Expected ScalarRange or dict, got {type(value).__name__}")


@dataclass
class Vec3Range:
    value: Vec3 | None = None
    min: Vec3 | None = None
    max: Vec3 | None = None

    def __post_init__(self) -> None:
        has_value = self.value is not None
        has_range = self.min is not None or self.max is not None
        if has_value and has_range:
            raise ValueError("Vec3Range must use either value or min/max, not both")
        if not has_value and (self.min is None or self.max is None):
            raise ValueError("Vec3Range requires either value or both min and max")
        if self.value is not None and len(self.value) != 3:
            raise ValueError(f"Vec3Range.value must have length 3, got {self.value}")
        if self.min is not None and len(self.min) != 3:
            raise ValueError(f"Vec3Range.min must have length 3, got {self.min}")
        if self.max is not None and len(self.max) != 3:
            raise ValueError(f"Vec3Range.max must have length 3, got {self.max}")

    @staticmethod
    def of(value: Any) -> Vec3Range:
        if isinstance(value, Vec3Range):
            return value
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, dict):
            return Vec3Range(**value)
        raise TypeError(f"Expected Vec3Range or dict, got {type(value).__name__}")
