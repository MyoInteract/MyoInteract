# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from dataclasses import dataclass, field
from typing import Literal, Any
from hydra.utils import instantiate

from myoInteract.common.config import Vec3, ScalarRange, Vec3Range
from ..task_config import TaskConfig

ShapeType = Literal["sphere", "box"]


@dataclass
class ReachConfig:
    reference_site: str = "humphant"
    reference_offset: Vec3 = field(default_factory=lambda: (0.0, 0.0, 0.0))
    end_effector_site: str = "fingertip"
    dwell_continuous: bool = False


@dataclass
class TargetConfig:
    position: Vec3Range = field(
        default_factory=lambda: Vec3Range(value=(0.3, 0.0, 0.0))
    )
    rgb: Vec3Range = field(
        default_factory=lambda: Vec3Range(min=(0, 0, 0), max=(1, 1, 1))
    )

    def __post_init__(self) -> None:
        self.position = Vec3Range.of(self.position)
        self.rgb = Vec3Range.of(self.rgb)


@dataclass
class PointingTargetConfig(TargetConfig):
    shape: ShapeType = "sphere"
    size: ScalarRange = field(default_factory=lambda: ScalarRange(value=0.05))
    dwell_duration: float = 0.25

    def __post_init__(self) -> None:
        super().__post_init__()
        self.size = ScalarRange.of(self.size)


@dataclass
class ButtonTargetConfig(TargetConfig):
    size: Vec3Range = field(
        default_factory=lambda: Vec3Range(value=(0.025, 0.025, 0.01))
    )
    site_pos: Vec3Range = field(
        default_factory=lambda: Vec3Range(value=(0.0, 0.0, 0.01))
    )
    euler: Vec3 = field(default_factory=lambda: (0, -0.79, 0))
    geom_margin: float = 0.001
    min_touch_force: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.size = Vec3Range.of(self.size)
        self.site_pos = Vec3Range.of(self.site_pos)


@dataclass
class DistractorConfig:
    count: int = 0
    similar_color_prob: float = 0.5
    similar_shape_prob: float = 0.5


@dataclass
class SequenceConfig:
    show_future_targets: bool = True
    color_code_progress: bool = True


@dataclass
class RewardConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "distance": 1,
            "phase_bonus": 10,
            "done": 10,
        }
    )
    distance_exponential: bool = False
    distance_metric: float = 10.0


@dataclass
class UniversalTaskConfig(TaskConfig):
    model_path: str = (
        "myo_user/envs/myo/assets/arm/mobl_arms_index_universal_myointeract.xml"
    )

    targets: list[Any] = field(default_factory=list)

    reach: ReachConfig = field(default_factory=ReachConfig)
    distractor: DistractorConfig = field(default_factory=DistractorConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    agent_state_keys: list[str] = field(
        default_factory=lambda: [
            "qpos",
            "qvel",
            "qacc",
            "act",
            "ee_pos",
        ]
    )
    task_state_keys: list[str] = field(
        default_factory=lambda: [
            "target_pos",
            "target_size",
            "phase_progress",
            "dwell_fraction",
        ]
    )

    def __post_init__(self) -> None:
        self.targets = instantiate(self.targets, _convert_="object")
