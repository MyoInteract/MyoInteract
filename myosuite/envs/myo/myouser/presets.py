"""Versioned GUI presets and safe persistence; no simulation dependencies."""

import json
import os
import re
from pathlib import Path

from .validation import boolean, finite_number, integer, parse_color, validate_rl_parameters


SCHEMA_VERSION = 1
RL_FIELDS = ("num_timesteps", "num_checkpoints", "num_evals", "batch_size", "num_envs", "num_minibatches", "checkpoint_run", "checkpoint_number", "target_init_seed")
BOX_FIELDS = ("position_x", "position_y", "position_z", "size_x", "size_y", "min_touch_force", "orientation_angle", "rgb")
SPHERE_FIELDS = ("x_range", "y_range", "z_range", "size_range", "dwell_duration", "rgb")
OBS_FIELDS = ("qpos", "qvel", "qacc", "ee_pos", "act")
OMNI_FIELDS = ("target_pos", "target_size", "phase", "dwell_fraction")
REWARD_FIELDS = ("distance", "subtask_bonus", "completion_bonus", "neural_effort")
FIELD_NAMES = (
    tuple("rl." + field for field in RL_FIELDS)
    + ("task.num_targets",)
    + tuple(f"targets.{i}.type" for i in range(10))
    + tuple(f"boxes.{i}.{field}" for i in range(10) for field in BOX_FIELDS)
    + tuple(f"spheres.{i}.{field}" for i in range(10) for field in SPHERE_FIELDS)
    + ("biomechanics.ctrl_dt", "biomechanics.reset_type", "biomechanics.sigdepnoise_enabled", "biomechanics.constantnoise_enabled", "task.max_duration")
    + tuple("observations." + field for field in OBS_FIELDS)
    + tuple("observations." + field for field in OMNI_FIELDS)
    + tuple("rewards." + field for field in REWARD_FIELDS)
)


def validate_preset_name(name):
    """Permit readable names (including dots), but never filesystem syntax."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Enter a configuration name.")
    if name != name.strip() or name in (".", "..") or name.endswith("."):
        raise ValueError("Configuration names cannot begin/end with whitespace or end with a dot.")
    if len(name) > 100 or re.search(r'[<>:"/\\|?*\x00-\x1f]', name):
        raise ValueError("Configuration names must be at most 100 characters and cannot contain path separators or reserved filename characters.")
    if name.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("Choose a name other than a reserved device filename.")
    return name


def encode_preset(values):
    if len(values) != len(FIELD_NAMES):
        raise ValueError(f"Expected {len(FIELD_NAMES)} preset fields, received {len(values)}.")
    parameters = dict(zip(FIELD_NAMES, values))
    _validate_parameters(parameters)
    return {"schema_version": SCHEMA_VERSION, "parameters": parameters}


def _validate_parameters(parameters):
    rl = {field: parameters["rl." + field] for field in RL_FIELDS if field not in ("checkpoint_run", "checkpoint_number", "target_init_seed")}
    validate_rl_parameters(rl)
    integer(parameters["rl.target_init_seed"], "target seed", maximum=2**32 - 1)
    integer(parameters["task.num_targets"], "number of targets", minimum=1, maximum=10)
    for field in FIELD_NAMES:
        value = parameters[field]
        if field.endswith(".type"):
            if value not in ("Box", "Sphere"):
                raise ValueError(f"{field} must be Box or Sphere.")
        elif field.endswith(".rgb"):
            try:
                parse_color(value)
            except ValueError as error:
                raise ValueError(f"Invalid {field}: {error}") from error
        elif field.endswith(".checkpoint_run") or field.endswith(".checkpoint_number"):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a nonempty string.")
        elif field.startswith("observations.") or field.endswith("noise_enabled"):
            boolean(value, field)
        elif field.endswith(".reset_type"):
            if value not in ("zero", "epsilon_uniform", "range_uniform", "None"):
                raise ValueError(f"Unsupported reset type {value!r}.")
        elif field.endswith("_range"):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{field} must contain a lower and upper bound.")
            lower = finite_number(value[0], field, strictly_positive=field.endswith(".size_range"))
            upper = finite_number(value[1], field)
            if lower > upper:
                raise ValueError(f"{field} lower bound must not exceed its upper bound.")
        elif not field.startswith("rl.") and field != "task.num_targets":
            finite_number(value, field, strictly_positive=field.endswith((".ctrl_dt", ".max_duration", ".size_x", ".size_y")))


def decode_preset(data):
    """Read version 1 and migrate the original positional format in memory."""
    if isinstance(data, list):
        data = encode_preset(data)
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported preset schema version; expected version 1 or a legacy preset array.")
    parameters = data.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Preset parameters must be an object of named fields.")
    missing = set(FIELD_NAMES) - parameters.keys()
    unknown = parameters.keys() - set(FIELD_NAMES)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing fields: " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown fields: " + ", ".join(sorted(unknown)))
        raise ValueError("Invalid preset: " + "; ".join(details))
    _validate_parameters(parameters)
    values = tuple(parameters[field] for field in FIELD_NAMES)
    # Reject non-finite JSON numbers before they enter Gradio or Hydra.
    json.dumps(values, allow_nan=False)
    return values


class PresetStore:
    def __init__(self, user_dir, bundled_dir=None):
        self.user_dir = Path(user_dir).expanduser().resolve()
        self.bundled_dir = Path(bundled_dir).expanduser().resolve() if bundled_dir else None

    def names(self):
        names = set()
        for folder in (self.bundled_dir, self.user_dir):
            if folder and folder.is_dir():
                for path in folder.glob("*.json"):
                    try:
                        validate_preset_name(path.stem)
                    except ValueError:
                        continue
                    if path.is_file() and path.resolve().parent == folder:
                        names.add(path.stem)
        return sorted(names, key=str.casefold)

    @staticmethod
    def _path(folder, name):
        validate_preset_name(name)
        result = folder / f"{name}.json"
        if result.resolve().parent != folder:
            raise ValueError("Configuration path must stay inside its configuration directory.")
        return result

    def save(self, name, values):
        path = self._path(self.user_dir, name)
        if name in self.names():
            raise FileExistsError("There is already a configuration with this name. Choose a different name.")
        payload = json.dumps(encode_preset(values), indent=2, allow_nan=False) + "\n"
        self.user_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)

    def load(self, name):
        for folder in (self.user_dir, self.bundled_dir):
            if folder:
                path = self._path(folder, name)
                if path.is_file():
                    try:
                        with path.open(encoding="utf-8") as stream:
                            return decode_preset(json.load(stream))
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise ValueError(f"Cannot read preset {name!r}: {exc}") from exc
        raise FileNotFoundError(f"Configuration {name!r} was not found.")
