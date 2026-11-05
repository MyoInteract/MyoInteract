"""Check packaged resources and real model loading from outside the checkout."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import zipfile


SMOKE = r'''
from pathlib import Path
import sys
import mujoco
import myosuite
from myosuite.envs.myo.myouser.paths import bundled_path, get_bundled_config_dir
from myosuite.envs.myo.myouser.presets import PresetStore
from myosuite.envs.myo.myouser.hydra_cli import load_config_interactive
from myosuite.envs.myo.myouser.myouser_universal import MyoUserUniversal

installed = Path(sys.argv[1]).resolve()
assert Path(myosuite.__file__).resolve().is_relative_to(installed), myosuite.__file__
presets = PresetStore(Path.cwd() / "personal", get_bundled_config_dir())
assert presets.names() == ["ar", "mobile-typing-5c-words", "public-display-4-buttons"]
for name in presets.names():
    assert len(presets.load(name)) == 178
    assert (installed / "myosuite" / "benchmarks" / "configs" / (name + ".json")).is_file()
cfg = load_config_interactive([
    "wandb=disabled", "env/task_config/targets=one",
    "+env/task_config/targets/target_0=sphere",
])
env = MyoUserUniversal(cfg.env)
assert env.action_size > 0
assert Path(env.xml_path).is_relative_to(installed)
print("Installed wheel: bundled presets, benchmark configs, and real arm model load successfully.")
'''


def main():
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    wheels = sorted(directory.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"No wheel found in {directory}")
    with tempfile.TemporaryDirectory(prefix="myointeract-wheel-") as temporary:
        target = Path(temporary) / "installed"
        with zipfile.ZipFile(wheels[-1]) as wheel:
            if any(name.endswith((".pyc", ".pyo")) for name in wheel.namelist()):
                raise SystemExit("Wheel contains bytecode generated in the source checkout")
            wheel.extractall(target)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(target)
        subprocess.run([sys.executable, "-c", SMOKE, str(target)], cwd=temporary, env=environment, check=True)


if __name__ == "__main__":
    main()
