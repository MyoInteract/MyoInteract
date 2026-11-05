"""Prepare Playground's import-time dependency without downloading robot meshes.

MyoInteract tests use the arm assets shipped in this repository. Playground 0.0.5
nevertheless checks for Menagerie on import. A real, pinned sparse checkout keeps
that import offline and avoids a large, unrelated clone during test collection.
Run this after installing the test environment, before importing Playground.
"""

import ast
from importlib.util import find_spec
from pathlib import Path
import subprocess


def main():
    package = find_spec("mujoco_playground")
    if package is None:
        raise SystemExit("Install MyoInteract's test dependencies first.")
    root = Path(package.origin).parent
    source = ast.parse((root / "_src" / "mjx_env.py").read_text(encoding="utf-8"))
    revision = next(
        ast.literal_eval(node.value)
        for node in source.body if isinstance(node, ast.Assign)
        if any(isinstance(target, ast.Name) and target.id == "MENAGERIE_COMMIT_SHA" for target in node.targets)
    )
    target = root / "external_deps" / "mujoco_menagerie"
    if target.exists():
        print(f"Using existing Playground assets: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "git", "-c", "core.longpaths=true", "clone", "--filter=blob:none", "--no-checkout", "--depth=1",
        "https://github.com/google-deepmind/mujoco_menagerie.git", str(target),
    ], check=True)
    for args in (
        ("config", "core.longpaths", "true"),
        ("sparse-checkout", "set", "--no-cone", "/README.md", "/LICENSE"),
        ("fetch", "--filter=blob:none", "--depth=1", "origin", revision),
        ("checkout", "--detach", "FETCH_HEAD"),
    ):
        subprocess.run(["git", "-C", str(target), *args], check=True)
    print(f"Prepared Menagerie metadata at {revision} for MyoInteract tests.")


if __name__ == "__main__":
    main()
