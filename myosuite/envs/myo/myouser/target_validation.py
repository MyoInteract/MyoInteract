"""Target configuration checks that do not require the simulator runtime."""

import math
from collections.abc import Mapping

import numpy as np


def _field(target, name):
    return target[name] if isinstance(target, Mapping) else getattr(target, name)


def normalize_pointing_ranges(position, size):
    """Return ordered position/radius bounds, accepting fixed values too."""
    position = np.asarray(position, dtype=float)
    if position.shape == (3,):
        position = np.stack([position, position])
    radius = np.asarray(size, dtype=float)
    if radius.shape == ():
        radius = np.repeat(radius, 2)
    if position.shape != (2, 3) or not np.isfinite(position).all():
        raise ValueError("Sphere position must be a finite 3D position or two 3D bounds.")
    if np.any(position[0] > position[1]):
        raise ValueError("Sphere position lower bounds must not exceed upper bounds.")
    if radius.shape != (2,) or not np.isfinite(radius).all():
        raise ValueError("Sphere radius must be finite, either a value or two bounds.")
    if radius[0] <= 0 or radius[0] > radius[1]:
        raise ValueError("Sphere radius bounds must be positive and ordered.")
    return position, radius


def validate_target_sampling(target_configs, ctrl_dt, pointing_origin_rel=(0, 0, 0)):
    """Reject configurations that cannot always reset with separated targets.

    For each sphere, every possible previous target position must admit a point
    in its box at least twice its maximum radius away. This conservative check
    rules out seed-dependent dead ends before entering JIT-compiled sampling.
    A farthest box corner then provides a valid, finite sampling fallback.
    """
    if not math.isfinite(float(ctrl_dt)) or ctrl_dt <= 0:
        raise ValueError("Control timestep must be finite and positive.")
    origin = np.asarray(pointing_origin_rel, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Target origin offset must be a finite 3D vector.")
    configs = list(target_configs)
    if not configs:
        raise ValueError("At least one target is required.")
    previous_bounds = None
    for index, target in enumerate(configs):
        name = _field(target, "name")
        if name == "pointing_target":
            bounds, radius = normalize_pointing_ranges(
                _field(target, "position"), _field(target, "size")
            )
            bounds = bounds + origin
            duration = float(_field(target, "dwell_duration"))
            if not math.isfinite(duration) or duration + 1e-12 < ctrl_dt:
                raise ValueError(
                    f"Target {index + 1} dwell duration must be at least one control timestep ({ctrl_dt:g}s)."
                )
            if previous_bounds is not None:
                midpoint = bounds.mean(axis=0)
                halfwidth = (bounds[1] - bounds[0]) / 2
                # Distance of the midpoint from the previous interval, per axis.
                midpoint_distance = np.maximum(
                    np.maximum(previous_bounds[0] - midpoint, midpoint - previous_bounds[1]), 0
                )
                guaranteed_distance = np.linalg.norm(halfwidth + midpoint_distance)
                if guaranteed_distance < 2 * radius[1]:
                    raise ValueError(
                        f"Target {index + 1} cannot always be placed at least {2 * radius[1]:g}m "
                        f"from target {index}. Widen or move its position range, or reduce its radius."
                    )
        elif name == "button_target":
            position = np.asarray(_field(target, "position"), dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(f"Target {index + 1} button position must be a finite 3D vector.")
            bounds = np.stack([position, position])
        else:
            raise ValueError(f"Unsupported target type: {name}")
        previous_bounds = bounds
