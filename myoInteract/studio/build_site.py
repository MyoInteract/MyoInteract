"""Build the hosted Studio page: a static site that runs without a backend and connects to a local one.

    python -m myoInteract.studio.build_site --out site/

The site holds the page, its scripts, the body models' scenes (``scenes/``) and the example policies'
stored results and recorded rollouts (``examples/``, without the checkpoints);
``.github/workflows/studio-site.yml`` deploys it to GitHub Pages. ``myo_studio serve`` allows that
origin (``server.HOSTED_ORIGINS``) once the page sends the pairing code the server prints.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from myoInteract.studio.examples import EXAMPLES, PAGE_FILES, SIM_FILES, list_examples
from myoInteract.studio.scene import STUDIO_MODELS, write_static_scenes
from myoInteract.studio.server import STATIC


def build_site(out: Path, models: list[str] = STUDIO_MODELS) -> None:
    """Write the static site to ``out`` (replaced if it exists)."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(STATIC, out / "static", ignore=shutil.ignore_patterns("index.html"))
    shutil.copy(STATIC / "index.html", out / "index.html")
    (out / ".nojekyll").write_text("")  # serve the files as they are
    write_static_scenes(out / "scenes", models)
    examples = [e for e in list_examples() if e["replay"]["hello"]["model"] in models]
    (out / "examples").mkdir()
    for e in examples:
        e["sim"] = all(
            (EXAMPLES / e["name"] / f).is_file() for f in SIM_FILES
        )  # the page can run this policy itself
        if e["sim"]:
            sensors = [
                t["sensor"]
                for t in json.loads((EXAMPLES / e["name"] / "sim.json").read_text())[
                    "task"
                ]["targets"]
            ]
            e["sim_kind"] = "sphere" if sensors[0] is None else "button"
        (out / "examples" / e["name"]).mkdir()
        for f in (*PAGE_FILES, *(SIM_FILES if e["sim"] else ())):
            shutil.copy(EXAMPLES / e["name"] / f, out / "examples" / e["name"] / f)
    (out / "examples" / "index.json").write_text(json.dumps(examples))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models", nargs="*", default=STUDIO_MODELS)
    args = p.parse_args()
    build_site(args.out, args.models)


if __name__ == "__main__":
    main()
