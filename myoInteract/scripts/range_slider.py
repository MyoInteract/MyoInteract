"""Dual-thumb range slider for gradio >= 6, as a gr.HTML component (value: [low, high])."""

from __future__ import annotations

from typing import Any

import gradio as gr

_TEMPLATE = """
<div class="rs">
  <div class="rs-head"><span class="rs-label">${label}</span>
    <span class="rs-vals"><b class="rs-lo"></b> &ndash; <b class="rs-hi"></b></span></div>
  <div class="rs-track"><div class="rs-fill"></div>
    <input class="rs-in rs-a" type="range" min="@MIN@" max="@MAX@" step="@STEP@" @DISABLED@>
    <input class="rs-in rs-b" type="range" min="@MIN@" max="@MAX@" step="@STEP@" @DISABLED@>
  </div>
</div>
"""

_CSS = """
.rs { width: 100%; padding: 4px 2px; }
.rs-head { display: flex; justify-content: space-between; font-size: var(--text-sm); color: var(--body-text-color); margin-bottom: 8px; }
.rs-vals { color: var(--body-text-color-subdued); font-variant-numeric: tabular-nums; }
.rs-track { position: relative; height: 18px; }
.rs-track::before { content: ""; position: absolute; left: 0; right: 0; top: 7px; height: 4px; border-radius: 2px; background: var(--neutral-300, #ccc); }
.rs-fill { position: absolute; top: 7px; height: 4px; border-radius: 2px; background: var(--color-accent, #f97316); }
.rs-in { position: absolute; left: 0; width: 100%; top: 0; height: 18px; margin: 0; background: none; pointer-events: none; -webkit-appearance: none; appearance: none; }
.rs-in::-webkit-slider-thumb { -webkit-appearance: none; pointer-events: auto; width: 16px; height: 16px; border-radius: 50%; background: white; border: 2px solid var(--color-accent, #f97316); cursor: pointer; }
.rs-in::-moz-range-thumb { pointer-events: auto; width: 12px; height: 12px; border-radius: 50%; background: white; border: 2px solid var(--color-accent, #f97316); cursor: pointer; }
.rs-in::-webkit-slider-runnable-track { background: none; }
.rs-in::-moz-range-track { background: none; }
.rs-in:disabled::-webkit-slider-thumb { cursor: not-allowed; opacity: 0.5; }
"""

_JS = """
const a = element.querySelector('.rs-a'), b = element.querySelector('.rs-b');
const fill = element.querySelector('.rs-fill');
const lo_el = element.querySelector('.rs-lo'), hi_el = element.querySelector('.rs-hi');
const mn = Number(a.min), mx = Number(a.max);
const dec = (String(a.step).split('.')[1] || '').length;
function bounds() { const x = Number(a.value), y = Number(b.value); return [Math.min(x, y), Math.max(x, y)]; }
function paint() {
  const [lo, hi] = bounds();
  fill.style.left = (100 * (lo - mn) / (mx - mn)) + '%';
  fill.style.width = (100 * (hi - lo) / (mx - mn)) + '%';
  lo_el.textContent = lo.toFixed(dec); hi_el.textContent = hi.toFixed(dec);
}
function load() {
  const v = props.value || [mn, mx];
  a.value = v[0]; b.value = v[1]; paint();
}
function commit() {
  const [lo, hi] = bounds();
  props.value = [lo, hi];
  trigger('change');
}
a.addEventListener('input', paint); b.addEventListener('input', paint);
a.addEventListener('change', commit); b.addEventListener('change', commit);
load();
watch('value', load);
"""


class RangeSlider(gr.HTML):
    """Two-thumb slider; as an input it yields ``(low, high)``, as an output it takes a pair."""

    def __init__(
        self,
        label: str | None = None,
        minimum: float = 0.0,
        maximum: float = 1.0,
        value: tuple[float, float] | list[float] | None = None,
        step: float = 0.01,
        interactive: bool = True,
        **kwargs: Any,
    ) -> None:
        value = (
            [minimum, maximum] if value is None else [float(value[0]), float(value[1])]
        )
        kwargs.setdefault("show_label", False)
        super().__init__(
            value=value,
            label=label or "",
            # The numbers are written into the template, not passed as props: a prop that is
            # missing on a re-render would make the template fail ("minimum is not defined").
            html_template=(
                _TEMPLATE.replace("@MIN@", str(minimum))
                .replace("@MAX@", str(maximum))
                .replace("@STEP@", str(step))
                .replace("@DISABLED@", "" if interactive else "disabled")
            ),
            css_template=_CSS,
            js_on_load=_JS,
            **kwargs,
        )

    def preprocess(self, payload: Any) -> tuple[float, float]:
        low, high = (float(payload[0]), float(payload[1])) if payload else (0.0, 0.0)
        return (low, high)

    def postprocess(self, value: Any) -> list[float]:
        return [float(value[0]), float(value[1])]
