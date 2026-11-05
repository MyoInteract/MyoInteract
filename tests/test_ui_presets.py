import json
from pathlib import Path

import pytest

from myosuite.envs.myo.myouser.presets import (
    FIELD_NAMES, PresetStore, decode_preset, encode_preset, validate_preset_name,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def values():
    return json.loads((ROOT / "gradio_configs" / "public-display-4-buttons.json").read_text())


@pytest.mark.parametrize("name", ["../escaped", "..\\escaped", "/absolute", "C:\\absolute", "folder/name", "", ".", "..", "trailing.", "NUL", "CON.json", "a\x00b"])
def test_preset_names_cannot_escape_or_use_device_names(name):
    with pytest.raises(ValueError):
        validate_preset_name(name)


def test_save_is_exclusive_and_dotted_name_round_trips(tmp_path, values):
    store = PresetStore(tmp_path / "presets")
    store.save("study.v2", values)
    original = (tmp_path / "presets" / "study.v2.json").read_bytes()
    restarted = PresetStore(tmp_path / "presets")
    assert restarted.names() == ["study.v2"]
    assert restarted.load("study.v2") == tuple(values)
    with pytest.raises(FileExistsError):
        restarted.save("study.v2", [None] * len(FIELD_NAMES))
    assert (tmp_path / "presets" / "study.v2.json").read_bytes() == original


def test_invalid_names_do_not_create_files(tmp_path, values):
    store = PresetStore(tmp_path / "presets")
    with pytest.raises(ValueError):
        store.save("../outside", values)
    assert list(tmp_path.iterdir()) == []


def test_bundle_cannot_be_overwritten_and_loads_from_any_cwd(tmp_path, monkeypatch, values):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "example.json").write_text(json.dumps(values))
    user = tmp_path / "user"
    other = tmp_path / "other"
    other.mkdir()
    store = PresetStore(user, bundled)
    monkeypatch.chdir(other)
    assert store.load("example") == tuple(values)
    with pytest.raises(FileExistsError):
        store.save("example", values)
    assert not user.exists()


def test_schema_uses_field_names_instead_of_object_order(values):
    encoded = encode_preset(values)
    encoded["parameters"] = dict(reversed(list(encoded["parameters"].items())))
    assert decode_preset(encoded) == tuple(values)
    encoded["parameters"].pop("biomechanics.ctrl_dt")
    with pytest.raises(ValueError, match="missing fields.*biomechanics.ctrl_dt"):
        decode_preset(encoded)


def test_schema_rejects_bad_field_types_before_updating_controls(values):
    encoded = encode_preset(values)
    encoded["parameters"]["biomechanics.ctrl_dt"] = "bad"
    with pytest.raises(ValueError, match="biomechanics.ctrl_dt.*finite"):
        decode_preset(encoded)
    encoded = encode_preset(values)
    encoded["parameters"]["spheres.0.x_range"] = [1, 0]
    with pytest.raises(ValueError, match="lower bound"):
        decode_preset(encoded)
    encoded = encode_preset(values)
    encoded["parameters"]["spheres.0.rgb"] = "rgba"
    with pytest.raises(ValueError, match="Invalid spheres.0.rgb.*Color must"):
        decode_preset(encoded)


@pytest.mark.parametrize("filename", ["ar.json", "mobile-typing-5c-words.json", "public-display-4-buttons.json"])
def test_all_shipped_presets_migrate_without_changing_values(filename):
    original = json.loads((ROOT / "gradio_configs" / filename).read_text())
    bundled = json.loads((ROOT / "myosuite" / "data" / "gradio_configs" / filename).read_text())
    assert len(original) == len(FIELD_NAMES) == 178
    assert decode_preset(original) == decode_preset(bundled)
    assert bundled["schema_version"] == 1


@pytest.mark.parametrize("invalid", [{"schema_version": 99}, [1, 2], {"schema_version": 1, "parameters": []}])
def test_malformed_or_unknown_presets_fail_cleanly(invalid):
    with pytest.raises(ValueError):
        decode_preset(invalid)
