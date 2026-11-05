"""Native MuJoCo reference semantics and the actual JavaScript actor contract."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from myoInteract.studio.browser import Reference  # noqa: E402
from myoInteract.studio.examples import example_dir, list_examples  # noqa: E402
from studio_reference_profiles import (  # noqa: E402
    BROWSER_VERSION,
    FIXTURES,
    load_profile,
    provenance,
)  # noqa: E402


def _zero_dwell_reference() -> Reference:
    model = mujoco.MjModel.from_xml_string("""<mujoco><option gravity="0 0 0"/>
        <worldbody><body><joint name="finger" type="slide" axis="1 0 0" range="-2 2"/>
        <geom type="sphere" size="0.01" mass="0.1" contype="0" conaffinity="0"/>
        <site name="tip"/></body><body name="target"><geom name="target_geom"
        type="sphere" size="0.1" contype="0" conaffinity="0"/></body></worldbody>
        <actuator><motor joint="finger" gear="0"/></actuator></mujoco>""")
    task = {
        "joints": ["finger"],
        "ee_site": "tip",
        "obs_terms": ["phase_progress", "dwell_fraction"],
        "start": {"qadr": [0], "vadr": [0], "limits": [[-2, 2]]},
        "targets": [
            {
                "body": "target",
                "geom": "target_geom",
                "sensor": None,
                "dwell_steps": 0,
                "min_force": 0,
            }
        ],
        "substeps": 1,
        "ctrl_dt": 0.002,
    }
    weights = np.array([0, 0, 1, 1, 0, 0, 0], dtype="<f4").tobytes()
    return Reference(model, task, {"obs": 2, "layers": [[2, 1]]}, weights)


def test_zero_dwell_sphere_requires_entry_on_native_mujoco() -> None:
    ref = _zero_dwell_reference()
    ref.reset({"qpos": [1], "qvel": [0]}, [[0, 0, 0]], [[0.1] * 3])
    assert not ref.step()
    assert ref.current == 0
    ref.reset({"qpos": [0], "qvel": [0]}, [[0, 0, 0]], [[0.1] * 3])
    assert ref.step()
    assert ref.current == 1


def test_noise_is_rounded_to_float32_before_native_control() -> None:
    ref = _zero_dwell_reference()
    ref.task["noise"] = {"signal_dependent": 0.1, "constant": 0.1}
    ref.rng = np.random.default_rng(7)
    ref.reset({"qpos": [1], "qvel": [0]}, [[0, 0, 0]], [[0.1] * 3])
    excitation = ref.excitation(ref.act(ref.observe()))
    rng = np.random.default_rng(7)
    expected = np.clip(
        excitation
        + 0.1 * excitation * rng.standard_normal(1)
        + 0.1 * rng.standard_normal(1),
        0,
        1,
    ).astype(np.float32)
    ref.step()
    np.testing.assert_array_equal(ref.d.ctrl, expected.astype(np.float64))


def test_sigmoid_has_javascript_saturation_and_float32_storage() -> None:
    result = Reference.excitation(np.array([-1000, 1000], np.float32))
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [0, 1])


def _policy_contract_examples() -> list[dict]:
    """Export same-input actor cases and fixed controls for separate physics checks."""
    examples = []
    rng = np.random.default_rng(37)
    for entry in list_examples():
        folder = example_dir(entry["name"])
        info = json.loads((folder / "sim.json").read_text())
        model = mujoco.MjModel.from_xml_path(str(folder / info["model"]))
        ref = Reference(
            model, info["task"], info["network"], (folder / info["policy"]).read_bytes()
        )
        initial = info["reference"]
        ref.reset(
            initial["start"],
            initial["positions"],
            initial["sizes"],
            initial.get("fatigue_state"),
        )
        initial_observation = ref.observe()
        observations = [
            np.zeros(info["network"]["obs"], np.float32),
            initial_observation,
        ]
        for i in range(96):
            scale = (0.01, 0.1, 1.0, 10.0)[i % 4]
            observations.append(
                (ref.mean + ref.std * rng.normal(0, scale, ref.mean.shape)).astype(
                    np.float32
                )
            )
        control_replay = []
        ref.noisy = False
        state_signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        for _ in initial["qpos"]:
            observations.append(ref.observe())
            integration_state = np.empty(
                mujoco.mj_stateSize(model, state_signature), np.float64
            )
            mujoco.mj_getState(model, ref.d, integration_state, state_signature)
            ref.step()
            control_replay.append(
                {
                    "control": ref.d.ctrl.tolist(),
                    "qpos": ref.d.qpos.tolist(),
                    "qvel": ref.d.qvel.tolist(),
                    "act": ref.d.act.tolist(),
                    "integration_before": integration_state.tolist(),
                }
            )
        cases = []
        for obs in observations:
            action = ref.act(obs)
            cases.append(
                {
                    "observation": obs.tolist(),
                    "normalized": ref.normalize(obs).tolist(),
                    "action": action.tolist(),
                    "excitation": ref.excitation(action).tolist(),
                }
            )
        examples.append(
            {
                "name": entry["name"],
                "initial_observation": initial_observation.tolist(),
                "cases": cases,
                "control_replay": control_replay,
                "state_signature": state_signature,
                "provenance": provenance(folder, info),
            }
        )
    return examples


def test_native_full_integration_state_replays_each_control_step_exactly() -> None:
    """The component test restores all integration carry, including solver warm starts."""
    for example in _policy_contract_examples():
        folder = example_dir(example["name"])
        info = json.loads((folder / "sim.json").read_text())
        model = mujoco.MjModel.from_xml_path(str(folder / info["model"]))
        ref = Reference(
            model, info["task"], info["network"], (folder / info["policy"]).read_bytes()
        )
        initial = info["reference"]
        ref.reset(
            initial["start"],
            initial["positions"],
            initial["sizes"],
            initial.get("fatigue_state"),
        )
        data = mujoco.MjData(model)
        for sample in example["control_replay"]:
            mujoco.mj_setState(
                model,
                data,
                np.asarray(sample["integration_before"]),
                example["state_signature"],
            )
            data.ctrl[:] = sample["control"]
            for _ in range(info["task"]["substeps"]):
                mujoco.mj_step(model, data)
            np.testing.assert_array_equal(data.qpos, sample["qpos"])
            np.testing.assert_array_equal(data.qvel, sample["qvel"])
            np.testing.assert_array_equal(data.act, sample["act"])


def test_native_actor_and_browser_engine_components_match_actual_javascript(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the actual TaskSim actor comparison")
    root = Path(__file__).resolve().parents[1]
    # The installed engine may have intentionally different dynamics (e.g. MuJoCo3.16
    # tendon wrapping). Browser physics is compared to its own3.15 engine profile;
    # actor inputs remain sampled from the actual installed native engine.
    load_profile("browser", BROWSER_VERSION)
    artifact = tmp_path / "actor-contract.json"
    artifact.write_text(
        json.dumps(
            {
                "native_mujoco_version": mujoco.mj_versionString(),
                "examples": _policy_contract_examples(),
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            node,
            str(root / "tests" / "studio_policy_contract.mjs"),
            str(artifact),
            str(FIXTURES / f"browser-{BROWSER_VERSION}.json"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_unreviewed_engine_profile_has_an_actionable_error() -> None:
    with pytest.raises(
        ValueError, match=r"No reviewed native physics profile.*--native"
    ):
        load_profile("native", "3.999.0")


def test_all_versioned_profiles_match_the_shipped_assets() -> None:
    paths = list(FIXTURES.glob("*.json"))
    assert paths, "the browser/native physics baselines must be packaged with the tests"
    for path in paths:
        kind, version = path.stem.split("-", 1)
        load_profile(kind, version)


def test_changed_assets_cannot_reuse_an_old_physics_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import studio_reference_profiles

    profile = load_profile("browser", BROWSER_VERSION)
    profile["examples"][0]["provenance"]["model_sha256"] = "0" * 64
    (tmp_path / f"browser-{BROWSER_VERSION}.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    monkeypatch.setattr(studio_reference_profiles, "FIXTURES", tmp_path)
    with pytest.raises(ValueError, match="model/policy/task changed"):
        load_profile("browser", BROWSER_VERSION)
