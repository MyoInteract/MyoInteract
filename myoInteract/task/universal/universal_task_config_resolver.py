# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from __future__ import annotations

from omegaconf import OmegaConf

_TARGETS: dict[str, str] = {
    "pointing": "myoInteract.task.universal.universal_task_config.PointingTargetConfig",
    "button": "myoInteract.task.universal.universal_task_config.ButtonTargetConfig",
}


def _resolve_target(name: str) -> str:
    try:
        return _TARGETS[name]
    except KeyError:
        known = ", ".join(sorted(_TARGETS))
        raise ValueError(f"Unknown target type {name!r}. Known: {known}") from None


def register_target_resolver() -> None:
    if OmegaConf.has_resolver("target"):
        return
    OmegaConf.register_new_resolver("target", _resolve_target, use_cache=True)


register_target_resolver()
