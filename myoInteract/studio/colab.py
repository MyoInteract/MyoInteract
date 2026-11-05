"""Studio on Google Colab: start the backend in the notebook's VM and open it through Colab's port proxy.

    from myoInteract.studio.colab import launch
    launch()

Colab's proxy only lets the notebook's own user in, so the server runs with ``--behind-proxy`` (no
pairing code). Outside Colab ``launch`` starts the server and prints its local address.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def serve_command(port: int = 8000, root: Path | None = None) -> list[str]:
    """Command line of the Studio server ``launch`` starts."""
    cmd = [
        sys.executable,
        "-m",
        "myoInteract.studio.server",
        "serve",
        "--port",
        str(port),
        "--no-browser",
        "--behind-proxy",
    ]
    return [*cmd, "--root", str(root)] if root is not None else cmd


def launch(
    port: int = 8000, root: Path | None = None, timeout_s: float = 120.0
) -> subprocess.Popen:
    """Start Studio in the background and show its link (Colab: a new browser tab through the port proxy).

    Args:
        port: Port of the server inside the VM.
        root: Data directory for runs and presets (e.g. a mounted Drive folder, so runs outlive the session).
        timeout_s: How long to wait for the server to answer.

    Returns:
        The server process; ``.terminate()`` stops it.
    """
    log = Path(f"studio_{port}.log")
    proc = subprocess.Popen(
        serve_command(port, root), stdout=log.open("ab"), stderr=subprocess.STDOUT
    )
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/health", timeout=2
            ).close()
            break
        except OSError as exc:
            if proc.poll() is not None or time.monotonic() > deadline:
                proc.terminate()
                raise RuntimeError(f"Studio did not start; see {log}") from exc
            time.sleep(1.0)
    try:
        from google.colab import output  # noqa: PLC0415
    except ImportError:
        print(f"MyoInteract Studio: http://localhost:{port}  (log: {log})")
    else:
        output.serve_kernel_port_as_window(port, anchor_text="Open MyoInteract Studio")
    return proc
