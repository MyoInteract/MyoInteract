from dataclasses import dataclass

from myoInteract.foveation import FoveationParams
from ..vision_config import MonoscopicDepthVisionConfig


@dataclass
class CnnVisionConfig(MonoscopicDepthVisionConfig):
    use_rgb: bool = True
    use_depth: bool = True
    spatial_softmax: bool = True
    foveated_acuity: bool = False
    fovea_sigma0_deg: float = 0.04
    half_resolution_deg: float = 2.3
    foveation_levels: int = 5
    camera_fovy_deg: float = 90.0
    fixation_uv: tuple[float, float] = (0.5, 0.5)

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.use_rgb and not self.use_depth:
            raise ValueError(
                "CnnVisionConfig requires at least one of use_rgb or use_depth"
            )
        if not isinstance(self.foveated_acuity, bool):
            raise ValueError("foveated_acuity must be a boolean")
        params = FoveationParams(
            fovea_sigma0_deg=self.fovea_sigma0_deg,
            half_resolution_deg=self.half_resolution_deg,
            levels=self.foveation_levels,
            camera_fovy_deg=self.camera_fovy_deg,
            fixation_uv=self.fixation_uv,
        )
        self.fixation_uv = params.fixation_uv

    def foveation_params(self) -> FoveationParams | None:
        if not self.foveated_acuity:
            return None
        return FoveationParams(
            fovea_sigma0_deg=self.fovea_sigma0_deg,
            half_resolution_deg=self.half_resolution_deg,
            levels=self.foveation_levels,
            camera_fovy_deg=self.camera_fovy_deg,
            fixation_uv=self.fixation_uv,
        )
