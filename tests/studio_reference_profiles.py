"""Versioned physics baselines; installed native and bundled WASM are independent engines."""

import hashlib
import json
from pathlib import Path

from myoInteract.studio.examples import example_dir, list_examples

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "studio"
BROWSER_VERSION = "3.15.0"


def provenance(folder: Path, info: dict) -> dict:
    """Bind each baseline to the exact model, policy and task/network definitions."""

    def digest(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    def canonical(value: object) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    return {
        "model_sha256": digest(
            (folder / info["model"]).read_bytes().replace(b"\r\n", b"\n")
        ),
        "policy_sha256": digest((folder / info["policy"]).read_bytes()),
        "task_sha256": digest(canonical(info["task"])),
        "network_sha256": digest(canonical(info["network"])),
    }


def load_profile(kind: str, version: str) -> dict:
    if kind not in ("native", "browser") or not version.replace(".", "").isdigit():
        raise ValueError("profile kind and engine version must be explicit")
    path = FIXTURES / f"{kind}-{version}.json"
    if not path.is_file():
        raise ValueError(
            f"No reviewed {kind} physics profile for MuJoCo {version}. "
            f"Generate it in an isolated environment with mujoco=={version}: "
            f"python tests/generate_studio_references.py --{kind}. "
            "Review engine release changes and parity before committing the new baseline."
        )
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile["format_version"] != 1 or profile["native_mujoco_version"] != version:
        raise ValueError(f"Invalid engine provenance in {path}")
    if kind == "browser" and profile["wasm_mujoco_version"] != BROWSER_VERSION:
        raise ValueError("Browser physics profile must match the vendored WASM engine")
    expected_names = {entry["name"] for entry in list_examples()}
    names = [example["name"] for example in profile["examples"]]
    if len(names) != len(expected_names) or set(names) != expected_names:
        raise ValueError(f"{path} must cover every shipped simulation exactly once")
    for example in profile["examples"]:
        folder = example_dir(example["name"])
        info = json.loads((folder / "sim.json").read_text())
        if example["provenance"] != provenance(folder, info):
            raise ValueError(
                f"{example['name']}: model/policy/task changed; regenerate and review {path}"
            )
        if example["reference"]["mujoco_version"] != version:
            raise ValueError(
                f"{example['name']}: reference has the wrong native engine version"
            )
    return profile
