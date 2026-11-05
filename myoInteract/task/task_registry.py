# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from myoInteract.common import registry, MyoComponent
from .task_config import TaskConfig

_myo_task_registry = registry.MyoComponentRegistry(
    kind="task",
    base_config_cls=TaskConfig,
    base_component_cls=MyoComponent,
    hydra_group="task",
)


def myo_register_task(name: str):
    return _myo_task_registry.component(name)


def myo_create_task(cfg: TaskConfig) -> MyoComponent:
    return _myo_task_registry.build(cfg)
