"""Execute Studio's actual JavaScript, including its vendored MuJoCo WASM runtime."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_studio_browser_javascript() -> None:
    """Run the dependency-free Node regression suite when Node is available."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for Studio's browser JavaScript tests")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [node, "--test", str(root / "tests" / "studio_browser.test.mjs")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
