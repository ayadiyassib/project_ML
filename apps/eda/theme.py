"""
Shared Plotly theme and serialisation helpers for the EDA and diagnostics modules.

Every chart in the studio goes through :func:`make_chart`, which applies the
"Omakase" dark theme (lacquer surface, rice-paper ink, recessive grid) and
serialises the figure to a JSON payload that is safe to embed inside a
``<script type="application/json">`` tag.  The browser side
(``static/js/ml_charts.js``) only calls ``Plotly.newPlot`` on that payload, so
no image is ever rendered server-side.

Colour roles
------------
* ``CATEGORICAL`` — identity (classes, clusters).  Fixed order, never cycled:
  the order itself was validated for colour-vision deficiencies on the studio
  surface ``#1b1918`` (worst adjacent CVD ΔE 8.4, normal-vision ΔE 19.3).
  Scatter plots add a marker symbol per class as a secondary encoding.
* ``SEQUENTIAL_BLUE`` — magnitude (confusion matrix), one hue dark → light.
* ``DIVERGING`` — polarity (correlations): blue ↔ red with a neutral midpoint.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Iterable

import plotly.graph_objects as go
import plotly.io as pio
from django.utils.safestring import SafeString, mark_safe

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

SURFACE = "#1b1918"
INK = "#F3EDE2"
INK_SECONDARY = "#c9c1b4"
INK_MUTED = "#9B9388"
GRID = "#2c2a28"
BASELINE = "#3a3633"
HOVER_BG = "#141312"
HOVER_BORDER = "#4f4a46"
FONT_FAMILY = '"Noto Sans JP", Inter, system-ui, sans-serif'
MONO_FAMILY = '"JetBrains Mono", ui-monospace, monospace'

CATEGORICAL = (
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#0ca30c",  # 6 green
    "#9085e9",  # 7 violet
    "#e66767",  # 8 red
)
OVERFLOW_COLOR = "#756e66"  # slots 9+ fold into one neutral, told apart by symbol
SYMBOLS = (
    "circle",
    "diamond",
    "square",
    "triangle-up",
    "cross",
    "star",
    "triangle-down",
    "hexagon",
    "pentagon",
    "x",
)

PRIMARY = CATEGORICAL[0]
ACCENT = CATEGORICAL[1]
REFERENCE_LINE = "#9B9388"

SEQUENTIAL_BLUE = [
    [0.0, "#16202c"],
    [0.15, "#0d366b"],
    [0.35, "#184f95"],
    [0.55, "#256abf"],
    [0.75, "#3987e5"],
    [1.0, "#6da7ec"],
]

DIVERGING = [
    [0.0, "#1c5cab"],
    [0.25, "#4f7fbf"],
    [0.5, "#383835"],
    [0.75, "#c95b5b"],
    [1.0, "#e66767"],
]


def series_color(index: int) -> str:
    """Colour of the ``index``-th categorical series (fixed order, no cycling)."""
    return CATEGORICAL[index] if 0 <= index < len(CATEGORICAL) else OVERFLOW_COLOR


def series_symbol(index: int) -> str:
    return SYMBOLS[index % len(SYMBOLS)]


def rgba(hex_color: str, alpha: float) -> str:
    """Return ``hex_color`` as an ``rgba()`` string with the given opacity."""
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_AXIS = dict(
    gridcolor=GRID,
    gridwidth=1,
    zerolinecolor=BASELINE,
    zerolinewidth=1,
    linecolor=BASELINE,
    showline=True,
    ticks="outside",
    tickcolor=BASELINE,
    ticklen=4,
    tickfont=dict(color=INK_MUTED, size=10, family=MONO_FAMILY),
    title=dict(font=dict(color=INK_SECONDARY, size=11)),
    automargin=True,
)

BASE_LAYOUT: dict[str, Any] = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=INK_SECONDARY, family=FONT_FAMILY, size=11),
    margin=dict(l=56, r=20, t=24, b=48),
    hoverlabel=dict(
        bgcolor=HOVER_BG,
        bordercolor=HOVER_BORDER,
        font=dict(color=INK, family=FONT_FAMILY, size=11),
        align="left",
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(color=INK_SECONDARY, size=10),
        bgcolor="rgba(0,0,0,0)",
    ),
    colorway=list(CATEGORICAL),
    bargap=0.18,
    hovermode="closest",
)


def apply_theme(fig: go.Figure, **overrides: Any) -> go.Figure:
    """Apply the studio theme to ``fig`` (all axes, including subplots)."""
    fig.update_layout(template="none", **BASE_LAYOUT)
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    if overrides:
        fig.update_layout(**overrides)
    return fig


def style_colorbar(title: str = "") -> dict[str, Any]:
    return dict(
        title=dict(text=title, font=dict(color=INK_SECONDARY, size=10)),
        tickfont=dict(color=INK_MUTED, size=9, family=MONO_FAMILY),
        outlinewidth=0,
        thickness=10,
        len=0.9,
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def to_payload(fig: go.Figure) -> SafeString:
    """Serialise ``fig`` as JSON that is safe to embed in a ``<script>`` tag.

    ``<``, ``>`` and ``&`` can only appear inside JSON strings (column names are
    user-controlled), so escaping them as ``\\u00XX`` keeps the JSON valid while
    making ``</script>`` injection impossible — the same rule as Django's
    ``json_script`` filter.
    """
    raw = pio.to_json(fig, validate=False, pretty=False)
    escaped = raw.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return mark_safe(escaped)  # noqa: S308 - escaped above


@dataclass
class Chart:
    """A rendered chart ready for ``templates/components/plotly_chart.html``."""

    id: str
    title: str
    payload: SafeString
    subtitle: str = ""
    height: int = 360
    note: str = ""
    wide: bool = False


def make_chart(
    fig: go.Figure,
    key: str,
    title: str,
    subtitle: str = "",
    height: int = 360,
    note: str = "",
    wide: bool = False,
    **layout: Any,
) -> Chart:
    """Theme ``fig`` and wrap it in a :class:`Chart` with a unique DOM id."""
    apply_theme(fig, **layout)
    fig.update_layout(height=height)
    safe_key = "".join(ch if ch.isalnum() else "-" for ch in key)[:40]
    return Chart(
        id=f"fig-{safe_key}-{uuid.uuid4().hex[:8]}",
        title=title,
        subtitle=subtitle,
        payload=to_payload(fig),
        height=height,
        note=note,
        wide=wide,
    )


def short_label(value: Any, limit: int = 28) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def labels_for(values: Iterable[Any]) -> list[str]:
    return [str(value) for value in values]
