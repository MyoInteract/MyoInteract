# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
"""MyoInteract, with training dependencies loaded only when requested."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .myo_config import MyoConfig
    from .myo_task_init import myo_mjlab_register

__all__ = ["MyoConfig", "myo_mjlab_register"]


def __getattr__(name: str) -> Any:
    if name == "MyoConfig":
        from .myo_config import MyoConfig

        return MyoConfig
    if name == "myo_mjlab_register":
        from .myo_task_init import myo_mjlab_register

        return myo_mjlab_register
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
