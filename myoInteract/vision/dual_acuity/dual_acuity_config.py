from dataclasses import dataclass

from ..vision_config import VisionConfig


@dataclass
class DualAcuityVisionConfig(VisionConfig):
    periphery_camera_name: str = "head_periphery"
    fovea_camera_name: str | None = "head_fovea"
    width: int = 96
    height: int = 96
    stream_width: int = 48
    stream_height: int = 48
    fov_ratio: float = 0.35
    use_rgb: bool = True
    use_depth: bool = False
    depth_min: float = 0.01
    depth_cutoff: float = 1.5
    spatial_softmax: bool = True
    default_fixation_uv: tuple[float, float] = (0.5, 0.5)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.periphery_camera_name:
            raise ValueError("periphery_camera_name must not be empty")
        if self.fovea_camera_name is not None and not self.fovea_camera_name:
            raise ValueError("fovea_camera_name must not be an empty string")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.stream_width <= 0 or self.stream_height <= 0:
            raise ValueError("stream_width and stream_height must be positive")
        if not (0.0 < float(self.fov_ratio) <= 1.0):
            raise ValueError("fov_ratio must lie in (0, 1]")
        if not self.use_rgb and not self.use_depth:
            raise ValueError(
                "DualAcuityVisionConfig requires at least one of use_rgb or use_depth"
            )
        if self.depth_min < 0.0 or self.depth_cutoff <= self.depth_min:
            raise ValueError("depth_min must be >= 0 and smaller than depth_cutoff")
        if len(self.default_fixation_uv) != 2:
            raise ValueError("default_fixation_uv must have 2 elements")
        u, v = float(self.default_fixation_uv[0]), float(self.default_fixation_uv[1])
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise ValueError("default_fixation_uv entries must lie in [0, 1]")
        self.default_fixation_uv = (u, v)
