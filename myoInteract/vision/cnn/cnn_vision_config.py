from dataclasses import dataclass
from ..vision_config import MonoscopicDepthVisionConfig


@dataclass
class CnnVisionConfig(MonoscopicDepthVisionConfig):
    use_rgb: bool = True
    use_depth: bool = True
    spatial_softmax: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.use_rgb and not self.use_depth:
            raise ValueError(
                "CnnVisionConfig requires at least one of use_rgb or use_depth"
            )
