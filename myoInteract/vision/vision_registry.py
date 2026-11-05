# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from myoInteract.common import registry, MyoComponent
from .vision_config import VisionConfig

_myo_vision_registry = registry.MyoComponentRegistry(
    kind="vision",
    base_config_cls=VisionConfig,
    base_component_cls=MyoComponent,
    hydra_group="vision",
)


def myo_register_vision(name: str):
    return _myo_vision_registry.component(name)


def myo_create_vision(cfg: VisionConfig) -> MyoComponent:
    return _myo_vision_registry.build(cfg)
