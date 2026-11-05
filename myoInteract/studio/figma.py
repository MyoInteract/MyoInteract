"""Import a Figma frame as a screen in Studio's 3D scene; its interactive elements become targets.

Uses Figma's REST API with a personal access token (sent only to api.figma.com, never stored):
the frame's node tree gives every element's box, the image endpoint a PNG of the frame.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request

FIGMA_API = "https://api.figma.com/v1"
MAX_ELEMENTS = 200  # a guard against runaway frames; Studio takes any number of targets
_FILE_KEY = re.compile(r"figma\.com/(?:file|design|proto|board)/([A-Za-z0-9]+)")
_INTERACTIVE = re.compile(
    r"button|btn|key|icon|link|tab|toggle|check|switch|slider|input|field|menu|item",
    re.I,
)
_LEAF_TYPES = {"INSTANCE", "COMPONENT"}


class FigmaError(ValueError):
    """The link or the token does not lead to a frame."""


def parse_figma_url(url: str) -> tuple[str, str]:
    """File key and node id (``12:34``) of a Figma frame link.

    Raises:
        FigmaError: Not a Figma file link, or no frame selected in it.
    """
    match = _FILE_KEY.search(url)
    if match is None:
        raise FigmaError("This is not a link to a Figma file.")
    node = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("node-id", [""])[
        0
    ]
    if not node:
        raise FigmaError(
            'The link has no frame: in Figma, select the frame and use "Copy link to selection".'
        )
    return match[1], node.replace("-", ":")


def _painted(node: dict) -> bool:
    """Whether the node draws a background or an outline of its own."""
    return any(
        p.get("visible", True) and p.get("opacity", 1) > 0
        for p in (node.get("fills") or []) + (node.get("strokes") or [])
    )


def frame_elements(node: dict, limit: int = MAX_ELEMENTS) -> dict:
    """Frame size and its clickable elements, in pixels relative to the frame's top-left corner.

    An element is the outermost node that is clickable as a whole: one with a prototype interaction,
    a component instance, a node named like a control (button, key, icon, ...), or a small container
    with its own background (a chip, a pill, a call-to-action), so a button counts with its full
    area and not as the icon inside it. A picked element's children are not searched. Elements are
    cut to the frame (what the exported image shows); those mostly outside it are dropped. Without
    any element, the frame's direct children are used.
    """
    box = node["absoluteBoundingBox"]
    width, height = box["width"], box["height"]

    def rel(n: dict) -> dict | None:
        b = n["absoluteBoundingBox"]
        x0, y0 = max(b["x"] - box["x"], 0), max(b["y"] - box["y"], 0)
        x1, y1 = (
            min(b["x"] - box["x"] + b["width"], width),
            min(b["y"] - box["y"] + b["height"], height),
        )
        if (
            x1 <= x0
            or y1 <= y0
            or (x1 - x0) * (y1 - y0) < 0.5 * b["width"] * b["height"]
        ):
            return None  # scrolled out of the frame
        return {"name": n.get("name", ""), "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}

    def usable(n: dict) -> bool:
        b = n.get("absoluteBoundingBox")
        return (
            n.get("visible", True)
            and b is not None
            and b["width"] > 0
            and b["height"] > 0
        )

    def control(n: dict) -> bool:
        if n.get("interactions") or n.get("reactions"):
            return True
        if n.get("type") in _LEAF_TYPES or _INTERACTIVE.search(n.get("name", "")):
            return True
        b = n["absoluteBoundingBox"]
        small = b["width"] <= 0.9 * width and b["height"] <= 0.25 * height
        return (
            bool(n.get("children"))
            and small
            and (_painted(n) or bool(n.get("cornerRadius")))
        )

    picked: list[dict] = []

    def walk(n: dict) -> None:
        for child in n.get("children", []):
            if not usable(child):
                continue
            if control(child):
                element = rel(child)
                if element is not None:
                    picked.append(element)
            else:
                walk(child)

    walk(node)
    if not picked:
        picked = [
            e
            for e in (rel(c) for c in node.get("children", []) if usable(c))
            if e is not None
        ]
    picked.sort(key=lambda e: (round(e["y"]), e["x"]))
    return {
        "name": node.get("name", "frame"),
        "width": width,
        "height": height,
        "elements": picked[:limit],
        "more": max(0, len(picked) - limit),
    }


def _get(url: str, token: str | None = None, timeout: float = 20.0) -> bytes:
    req = urllib.request.Request(url, headers={"X-Figma-Token": token} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed https hosts)
            return r.read()
    except urllib.error.HTTPError as err:
        if err.code in (401, 403):
            raise FigmaError(
                "Figma refused the token: create a personal access token with file read access."
            ) from err
        if err.code == 404:
            raise FigmaError(
                "Figma does not know this file or frame (or the token cannot see it)."
            ) from err
        raise FigmaError(f"Figma answered HTTP {err.code}.") from err
    except urllib.error.URLError as err:
        raise FigmaError(
            f"Figma is not reachable from the backend: {err.reason}"
        ) from err


def fetch_frame(url: str, token: str) -> dict:
    """Frame, elements and a PNG (data URL) of the frame a Figma link points to."""
    key, node_id = parse_figma_url(url)
    ids = urllib.parse.quote(node_id)
    nodes = json.loads(_get(f"{FIGMA_API}/files/{key}/nodes?ids={ids}", token))
    entry = (nodes.get("nodes") or {}).get(node_id)
    if not entry or "document" not in entry:
        raise FigmaError("Figma does not know this frame.")
    frame = frame_elements(entry["document"])
    # use_absolute_bounds: the image is exactly the frame's box (no overflow, no shadow margin), so pixels and elements agree
    images = json.loads(
        _get(
            f"{FIGMA_API}/images/{key}?ids={ids}&format=png&scale=2&use_absolute_bounds=true",
            token,
        )
    )
    image_url = (images.get("images") or {}).get(node_id)
    if not image_url:
        raise FigmaError("Figma could not render this frame as an image.")
    frame["image"] = (
        "data:image/png;base64," + base64.b64encode(_get(image_url)).decode()
    )
    return frame
