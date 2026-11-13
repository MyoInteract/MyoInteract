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

# Cookie-free visit counts for the hosted page only (GoatCounter: path, referrer, browser, country).
ANALYTICS = '<script data-goatcounter="https://myointeract.goatcounter.com/count" async src="https://gc.zgo.at/count.js"></script>'


def build_site(
    out: Path, models: list[str] = STUDIO_MODELS, analytics: bool = True
) -> None:
    """Write the static site to ``out`` (replaced if it exists).

    Args:
        out: Site directory.
        models: Body models whose scenes the site ships.
        analytics: Add the GoatCounter visit counter to the page.
    """
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(STATIC, out / "static", ignore=shutil.ignore_patterns("index.html"))
    page = (STATIC / "index.html").read_text()
    if analytics:
        page = page.replace("</head>", f"{ANALYTICS}\n</head>", 1)
    (out / "index.html").write_text(page)
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
    p.add_argument(
        "--no-analytics", action="store_true", help="leave out the visit counter"
    )
    args = p.parse_args()
    build_site(args.out, args.models, analytics=not args.no_analytics)


if __name__ == "__main__":
    main()
