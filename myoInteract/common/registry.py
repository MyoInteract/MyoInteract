# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Annotated,
    Generic,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

from .component import MyoComponent, MyoComponentConfig


ConfigT = TypeVar("ConfigT", bound=MyoComponentConfig)
ComponentT = TypeVar("ComponentT", bound=MyoComponent)


@dataclass(frozen=True)
class MyoComponentRegistration(Generic[ConfigT, ComponentT]):
    name: str
    config_cls: type[ConfigT]
    component_cls: type[ComponentT]


class MyoComponentRegistry(Generic[ConfigT, ComponentT]):
    def __init__(
        self,
        *,
        kind: str,
        base_config_cls: type[ConfigT],
        base_component_cls: type[ComponentT],
        hydra_group: str,
    ):
        self.kind = kind
        self.base_config_cls = base_config_cls
        self.base_component_cls = base_component_cls
        self.hydra_group = hydra_group

        self.by_name: dict[str, MyoComponentRegistration[ConfigT, ComponentT]] = {}
        self.by_component: dict[
            type[ComponentT], MyoComponentRegistration[ConfigT, ComponentT]
        ] = {}

    def component(self, name: str):
        if not name:
            raise ValueError(f"{self.kind} registration name must not be empty")

        def decorator(component_cls: type[ComponentT]) -> type[ComponentT]:
            self.register(name=name, component_cls=component_cls)
            return component_cls

        return decorator

    def register(
        self, *, name: str, component_cls: type[ComponentT]
    ) -> MyoComponentRegistration[ConfigT, ComponentT]:
        if name in self.by_name:
            existing = self.by_name[name]
            raise ValueError(
                f"Duplicate {self.kind} component name {name!r}. "
                f"Already registered by {existing.component_cls.__qualname__}"
            )
        if component_cls in self.by_component:
            existing = self.by_component[component_cls]
            raise ValueError(
                f"Duplicate {self.kind} component class {component_cls.__qualname__}. "
                f"Already registered as {existing.name!r}"
            )
        if not isinstance(component_cls, type):
            raise TypeError(
                f"@{self.kind}_component can only decorate classes, got {component_cls!r}"
            )
        if not issubclass(component_cls, self.base_component_cls):
            raise TypeError(
                f"{component_cls.__qualname__} must inherit from {self.base_component_cls.__qualname__}"
            )

        config_cls = self._infer_single_config_ctor_arg(component_cls)
        ConfigStore.instance().store(group=self.hydra_group, name=name, node=config_cls)

        registration = MyoComponentRegistration(
            name=name, config_cls=config_cls, component_cls=component_cls
        )
        self.by_name[name] = registration
        self.by_component[component_cls] = registration
        return registration

    def build(self, cfg: ConfigT, *, name: str | None = None) -> ComponentT:
        selected_name = name or self._get_selected_hydra_option()
        try:
            registration = self.by_name[selected_name]
        except KeyError:
            known = ", ".join(sorted(self.by_name))
            raise TypeError(
                f"No {self.kind} component registered under name "
                f"{selected_name!r}. Known: {known or '<none>'}"
            ) from None
        if not isinstance(cfg, registration.config_cls):
            raise TypeError(
                f"{self.kind} component {selected_name!r} expects config "
                f"{registration.config_cls.__qualname__}, got {type(cfg).__qualname__}"
            )
        return registration.component_cls(cfg)

    def _get_selected_hydra_option(self) -> str:
        if not HydraConfig.initialized():
            raise RuntimeError(
                f"Cannot infer selected {self.kind} component because HydraConfig "
                f"is not initialized. Pass name=... explicitly to build()."
            )
        choices = HydraConfig.get().runtime.choices
        try:
            selected = choices[self.hydra_group]
        except KeyError:
            known = ", ".join(sorted(choices.keys()))
            raise RuntimeError(
                f"Hydra runtime choices do not contain group {self.hydra_group!r}. "
                f"Known groups: {known or '<none>'}"
            ) from None
        if selected is None:
            raise RuntimeError(
                f"Hydra group {self.hydra_group!r} has no selected option"
            )
        return selected

    def _infer_single_config_ctor_arg(
        self, component_cls: type[ComponentT]
    ) -> type[ConfigT]:
        try:
            sig = inspect.signature(component_cls.__init__)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Could not inspect {component_cls.__qualname__}.__init__"
            ) from exc

        params = [p for p in sig.parameters.values() if p.name != "self"]
        if len(params) != 1:
            names = ", ".join(p.name for p in params) or "<none>"
            raise TypeError(
                f"{component_cls.__qualname__}.__init__ must have exactly one "
                f"constructor argument besides self. Got: {names}"
            )
        param = params[0]
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(
                f"{component_cls.__qualname__}.__init__ parameter "
                f"{param.name!r} must be a normal positional/keyword argument"
            )
        hints = get_type_hints(component_cls.__init__, include_extras=True)
        config_cls = hints.get(param.name)
        if config_cls is None:
            raise TypeError(
                f"{component_cls.__qualname__}.__init__ parameter "
                f"{param.name!r} must be type-annotated with a "
                f"{self.base_config_cls.__qualname__} subclass"
            )
        config_cls = self._unwrap_annotated(config_cls)
        if not isinstance(config_cls, type):
            raise TypeError(
                f"{component_cls.__qualname__}.__init__ parameter "
                f"{param.name!r} must be annotated with a concrete config class, got {config_cls!r}"
            )
        if not issubclass(config_cls, self.base_config_cls):
            raise TypeError(
                f"{component_cls.__qualname__}.__init__ parameter "
                f"{param.name!r} must be assignable to {self.base_config_cls.__qualname__}. "
                f"Got: {config_cls.__qualname__}"
            )
        return config_cls

    @staticmethod
    def _unwrap_annotated(tp: Any) -> Any:
        if get_origin(tp) is Annotated:
            return get_args(tp)[0]
        return tp
