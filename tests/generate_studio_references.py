"""Generate reviewed CPU physics profiles with the engine installed in this environment.

Run each native engine in isolation, for example:
    uv run --no-project --with mujoco==3.17.0 --with numpy python tests/generate_studio_references.py --native
The browser profile must be generated with the exact vendored WASM version.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from myoInteract.studio.browser import Reference, reference_rollout  # noqa: E402
from myoInteract.studio.examples import example_dir, list_examples  # noqa: E402
from studio_reference_profiles import BROWSER_VERSION, FIXTURES, provenance  # noqa: E402


def generate(*, browser: bool) -> dict:
    version = mujoco.mj_versionString()
    if browser and version != BROWSER_VERSION:
        raise ValueError(
            f"Browser physics profile requires native mujoco=={BROWSER_VERSION}; found {version}"
        )
    profile = {"format_version": 1, "native_mujoco_version": version, "examples": []}
    if browser:
        profile["wasm_mujoco_version"] = BROWSER_VERSION
    for entry in list_examples():
        folder = example_dir(entry["name"])
        info = json.loads((folder / "sim.json").read_text())
        model = mujoco.MjModel.from_xml_path(str(folder / info["model"]))
        ref = Reference(
            model, info["task"], info["network"], (folder / info["policy"]).read_bytes()
        )
        record = {
            "name": entry["name"],
            "provenance": provenance(folder, info),
            "reference": reference_rollout(ref),
        }
        if browser:
            initial = record["reference"]
            ref.reset(
                initial["start"],
                initial["positions"],
                initial["sizes"],
                initial.get("fatigue_state"),
            )
            ref.noisy = False
            record["initial_observation"] = ref.observe().tolist()
            signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
            record["state_signature"] = signature
            record["control_replay"] = []
            for _ in initial["qpos"]:
                state = np.empty(mujoco.mj_stateSize(model, signature), np.float64)
                mujoco.mj_getState(model, ref.d, state, signature)
                ref.step()
                record["control_replay"].append(
                    {
                        "integration_before": state.tolist(),
                        "control": ref.d.ctrl.tolist(),
                        "qpos": ref.d.qpos.tolist(),
                        "qvel": ref.d.qvel.tolist(),
                        "act": ref.d.act.tolist(),
                    }
                )
        profile["examples"].append(record)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--native", action="store_true")
    choice.add_argument("--browser", action="store_true")
    args = parser.parse_args()
    profile = generate(browser=args.browser)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    kind = "browser" if args.browser else "native"
    path = FIXTURES / f"{kind}-{profile['native_mujoco_version']}.json"
    path.write_text(
        json.dumps(profile, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
