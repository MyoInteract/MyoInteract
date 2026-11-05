"""The gradio 6 range slider: value handling, and a real-browser round trip when playwright and a browser exist."""

import socket

import pytest

gr = pytest.importorskip("gradio")

from myoInteract.scripts.range_slider import RangeSlider  # noqa: E402


def test_value_round_trip() -> None:
    slider = RangeSlider(
        label="x", minimum=0.0, maximum=1.0, value=(0.2, 0.7), step=0.01
    )
    assert slider.value == [0.2, 0.7]
    assert slider.preprocess([0.25, 0.6]) == (0.25, 0.6)
    assert slider.postprocess((0.3, 0.9)) == [0.3, 0.9]


def test_default_value_spans_the_range() -> None:
    assert RangeSlider(minimum=-0.5, maximum=0.75).value == [-0.5, 0.75]


def test_builds_inside_blocks() -> None:
    with gr.Blocks() as demo:
        RangeSlider(label="Depth", minimum=0.0, maximum=0.75, value=(0.1, 0.4))
    assert demo.get_config_file()["components"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_browser_round_trip() -> None:
    sync_api = pytest.importorskip("playwright.sync_api")
    with gr.Blocks() as demo:
        slider = RangeSlider(
            label="Depth", minimum=0.0, maximum=0.75, value=(0.1, 0.4), step=0.001
        )
        out = gr.Textbox(elem_id="out")
        read = gr.Button("read", elem_id="read")
        reset = gr.Button("set", elem_id="set")
        read.click(lambda v: str(v), inputs=[slider], outputs=out)
        reset.click(lambda: [0.2, 0.7], outputs=slider)
    port = _free_port()
    demo.launch(server_port=port, prevent_thread_lock=True, quiet=True)
    try:
        with sync_api.sync_playwright() as p:
            try:
                browser = p.chromium.launch(args=["--no-sandbox"])
            except Exception:
                try:
                    browser = p.chromium.launch(
                        executable_path="/usr/bin/google-chrome", args=["--no-sandbox"]
                    )
                except Exception as err:
                    pytest.skip(f"no browser available: {err}")
            page = browser.new_page()
            page.route(
                "**/*",
                lambda r: r.abort() if "fonts.g" in r.request.url else r.continue_(),
            )
            page.set_default_timeout(15000)
            page.goto(f"http://127.0.0.1:{port}")
            page.wait_for_selector(".rs")
            assert page.inner_text(".rs-vals") == "0.100 – 0.400"
            page.evaluate(
                """() => { for (const [cls, v] of [['.rs-a', 0.25], ['.rs-b', 0.6]]) {
                    const el = document.querySelector(cls); el.value = v;
                    el.dispatchEvent(new Event('input', {bubbles: true}));
                    el.dispatchEvent(new Event('change', {bubbles: true})); } }"""
            )
            page.wait_for_timeout(500)
            page.click("#read", force=True)
            page.wait_for_function(
                "document.querySelector('#out textarea').value !== ''"
            )
            assert page.input_value("#out textarea") == "(0.25, 0.6)"
            page.click("#set", force=True)
            page.wait_for_function(
                "document.querySelector('.rs-vals').textContent.includes('0.700')"
            )
            browser.close()
    finally:
        demo.close()
