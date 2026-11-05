# Derived from https://github.com/Ingrim4/myo_user (MIT License, © Ingrim4)
from dataclasses import fields, is_dataclass
from typing import TypeVar, Any

T = TypeVar("T")


def dataclass_as_base(instance: object, base_cls: type[T]) -> T:
    if not is_dataclass(instance):
        raise TypeError("instance must be a dataclass instance")
    if not is_dataclass(base_cls):
        raise TypeError("base_cls must be a dataclass type")
    valid_fields = {f.name for f in fields(base_cls)}
    kwargs = {
        name: getattr(instance, name)
        for name in valid_fields
        if hasattr(instance, name)
    }
    return base_cls(**kwargs)


def dataclass_as_derived(instance: object, derived_cls: type[T], **overrides: Any) -> T:
    if not is_dataclass(instance):
        raise TypeError("instance must be a dataclass instance")
    if not is_dataclass(derived_cls):
        raise TypeError("derived_cls must be a dataclass type")
    valid_fields = {f.name for f in fields(derived_cls) if f.init}
    kwargs = {
        name: getattr(instance, name)
        for name in valid_fields
        if hasattr(instance, name)
    }
    kwargs.update(overrides)
    return derived_cls(**kwargs)
