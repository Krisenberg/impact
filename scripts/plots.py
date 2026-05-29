"""Module for creating plots that will be presented in the thesis."""

from pathlib import Path
from typing import NamedTuple

import altair as alt
import structlog

logger = structlog.stdlib.get_logger(__name__)


class OkabeItoPalette(NamedTuple):
    """Okabe-Ito palette as a named tuple of hex color strings.

    This color palette is colorblind-safe and prints well in greyscale. Encapsulating
    it in a NamedTuple makes colors accessible by name rather than by index.
    """

    orange: str
    sky_blue: str
    bluish_green: str
    yellow: str
    blue: str
    vermillion: str
    reddish_purple: str
    black: str


# General constants
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Chart theme styling
FONT: str = "Libertinus Serif"  # Match the serif font used in the thesis
FONT_SIZE_BASE = 12  # Match the 12pt font used in the thesis
FONT_SIZE_SMALL = 10
# Okabe-Ito palette that is  colorblind-safe and prints well in greyscale
OKABE_ITO_PALETTE = OkabeItoPalette(
    orange="#E69F00",
    sky_blue="#56B4E9",
    bluish_green="#009E73",
    yellow="#F0E442",
    blue="#0072B2",
    vermillion="#D55E00",
    reddish_purple="#CC79A7",
    black="#000000",
)
COLOR_INK = "#1A1A1A"  # Labels, ticks, axis domains
COLOR_GRID = "#E0E0E0"  # Passive guides - gridlines, legend borders


@alt.theme.register("thesis_theme", enable=True)
def thesis_theme() -> alt.theme.ThemeConfig:
    """Register and enable a ..."""
    return {
        "config": {
            # -- Canvas --
            "view": {
                "strokeWidth": 0,  # No border around plot area
                "continuousWidth": 400,
                "continuousHeight": 250,  # Keep the aspect ratio of 16:10 (approx.)
            },
            "background": "white",
            # -- Typography --
            "title": {
                "font": FONT,
                "fontSize": FONT_SIZE_BASE,
                "fontWeight": "normal",  # Let the thesis captions catch the eye
                "color": COLOR_INK,
                "anchor": "start",
            },
            # -- Axes --
            "axis": {
                "labelFont": FONT,
                "titleFont": FONT,
                "labelFontSize": FONT_SIZE_SMALL,
                "titleFontSize": FONT_SIZE_BASE,
                "titleFontWeight": "normal",  # Let the thesis captions catch the eye
                "labelColor": COLOR_INK,
                "titleColor": COLOR_INK,
                "domainColor": COLOR_INK,
                "domainWidth": 1,
                "tickColor": COLOR_INK,
                "tickWidth": 1,
                "gridColor": COLOR_GRID,
                "gridWidth": 0.5,
                "gridDash": [4, 4],
            },
            "axisX": {"grid": False},  # Disable vertical gridlines
            "axisY": {"grid": True},  # Keep the horizontal though
            # -- Legend --
            "legend": {
                "labelFont": FONT,
                "titleFont": FONT,
                "labelFontSize": FONT_SIZE_SMALL,
                "titleFontSize": FONT_SIZE_SMALL,
                "titleFontWeight": "normal",  # Let the thesis captions catch the eye
                "orient": "top-right",
                "fillColor": "white",
                "strokeColor": COLOR_GRID,
                "padding": 5,
                "cornerRadius": 2,
            },
            # -- Header --
            "header": {
                "labelFont": FONT,
                "titleFont": FONT,
            },
            # -- Marks --
            "line": {"strokeWidth": 1.8},
            "point": {"size": 35, "filled": True},
            "bar": {"fill": OKABE_ITO_PALETTE.blue},
            # -- Color scale default --
            "range": {
                "category": OKABE_ITO_PALETTE,
                "ordinal": OKABE_ITO_PALETTE,
            },
        }
    }
