from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

import altair as alt
import polars as pl
import structlog

from .plots_data import AlphaCurvePlotSchema

if TYPE_CHECKING:
    from pathlib import Path

    from pandera.typing.polars import DataFrame

    from .types import Language, PlotFormat


logger = structlog.stdlib.get_logger(__name__)


class OkabeItoPalette(StrEnum):
    """Okabe-Ito palette which is colorblind-safe and prints well in grayscale."""

    ORANGE = "#E69F00"
    SKY_BLUE = "#56B4E9"
    BLUISH_GREEN = "#009E73"
    YELLOW = "#F0E442"
    BLUE = "#0072B2"
    VERMILLION = "#D55E00"
    REDDISH_PURPLE = "#CC79A7"
    BLACK = "#000000"


# Chart theme styling
FONT: str = "Libertinus Serif"  # Match the serif font used in the thesis
FONT_SIZE_BASE = 12  # Match the 12pt font used in the thesis
FONT_SIZE_SMALL = 10
FONT_SIZE_TINY = 8
COLOR_INK = "#1A1A1A"  # Labels, ticks, axis domains
COLOR_GRID = "#E0E0E0"  # Passive guides - gridlines, legend borders, etc.

# PNG rendering settings
PNG_SCALE = 3.0


@alt.theme.register("thesis_theme", enable=True)
def thesis_theme() -> alt.theme.ThemeConfig:
    """Register and enable an Altair theme for publication-quality visualizations."""
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
                "fontWeight": "bold",
                "color": COLOR_INK,
                "anchor": "middle",
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
            "line": {"strokeWidth": 1.6},
            "point": {"size": 35, "filled": True},
            "bar": {"fill": OkabeItoPalette.BLUE.value},
            # -- Color scale default --
            "range": {
                "category": [e.value for e in OkabeItoPalette],
                "ordinal": [e.value for e in OkabeItoPalette],
            },
        }
    }


def render_alpha_curve_plot(
    alpha_curve_df: DataFrame[AlphaCurvePlotSchema],
    alpha_start: float,
    alpha_stop: float,
    out_dir: Path,
    out_filename: str,
    formats: tuple[PlotFormat, ...],
    lang: Language = "en",
    no_title: bool = False,
) -> tuple[Path, ...]:
    """Plot IMPACT evaluation metrics as a function of alpha.

    Alpha controls the relative contribution of SASRec and ACT-R in the final
    late-fusion score (trade-off).
    """
    # -- Labels per language --
    labels: dict[Language, dict[str, str]] = {
        "en": {
            "x_title": "α (0 = pure ACT-R, 1 = pure SASRec)",  # noqa: RUF001
            "y_left_title": "NDCG@10 (Overall & Repetition)",
            "y_right_title": "NDCG@10 (Exploration)",
            "title": "Effect of α on IMPACT Metrics",  # noqa: RUF001
            "ndcg_at_10": "NDCG@10 (Overall)",
            "ndcg_local_rep_at_10": "Local Repetition",
            "ndcg_global_rep_at_10": "Global Repetition",
            "ndcg_local_exp_at_10": "Local Exploration",
            "ndcg_global_exp_at_10": "Global Exploration",
        },
        "pl": {
            "x_title": "α (0 = wyłącznie ACT-R, 1 = wyłącznie SASRec)",  # noqa: RUF001
            "y_left_title": "NDCG@10 (ogólne i powtarzalność)",
            "y_right_title": "NDCG@10 (eksploracja)",
            "title": "Wpływ parametru α na metryki IMPACT",  # noqa: RUF001
            "ndcg_at_10": "NDCG@10 (ogólne)",
            "ndcg_local_rep_at_10": "Powtarzalność lokalna",
            "ndcg_global_rep_at_10": "Powtarzalność globalna",
            "ndcg_local_exp_at_10": "Eksploracja lokalna",
            "ndcg_global_exp_at_10": "Eksploracja globalna",
        },
    }
    lbl = labels[lang]

    df_plot = alpha_curve_df.with_columns(
        pl.col(AlphaCurvePlotSchema.metric).replace(labels[lang])
    )

    # Subplots specs
    left_subplot_metrics_to_colors: dict[str, str] = {
        lbl[metric]: color
        for metric, color in {
            "ndcg_at_10": OkabeItoPalette.BLACK.value,
            "ndcg_local_rep_at_10": OkabeItoPalette.BLUE.value,
            "ndcg_global_rep_at_10": OkabeItoPalette.VERMILLION.value,
        }.items()
    }
    right_subplot_metrics_to_colors: dict[str, str] = {
        lbl[metric]: color
        for metric, color in {
            "ndcg_local_exp_at_10": OkabeItoPalette.ORANGE.value,
            "ndcg_global_exp_at_10": OkabeItoPalette.BLUISH_GREEN.value,
        }.items()
    }

    # Shared x-axis
    base_x = alt.X(
        "alpha:Q",
        title=lbl["x_title"],
        scale=alt.Scale(domain=[alpha_start, alpha_stop]),
        axis=alt.Axis(tickCount=5, grid=False),
    )

    def _create_subplot(
        subplot_df: pl.DataFrame,
        y_title: str,
        colors: list[str],
        domain: list[str],
    ) -> alt.LayerChart:
        base = alt.Chart(subplot_df)

        lines = base.mark_line().encode(
            x=base_x,
            y=alt.Y("score:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(domain=domain, range=colors),
                legend=alt.Legend(title=None),
            ),
            tooltip=["alpha:Q", "metric:N", alt.Tooltip("score:Q", format=".4f")],
        )

        points = base.mark_point(filled=True).encode(
            x=base_x,
            y=alt.Y("score:Q"),
            color=alt.Color("metric:N"),
        )

        text_labels = (
            base.mark_text(
                align="left",
                baseline="middle",
                dx=6,  # offset to the right
                dy=-6,  # offset up
                fontSize=FONT_SIZE_TINY,
                opacity=0.6,
            )
            .encode(
                x=base_x,
                y=alt.Y("score:Q"),
                text=alt.Text("score:Q", format=".4f"),
                color=alt.Color("metric:N"),
            )
            .transform_filter(
                " || ".join([f"datum.alpha == {a}" for a in [0.0, 0.5, 1.0]])
            )
        )

        return cast(alt.LayerChart, alt.layer(lines, points, text_labels))

    # Left subplot
    left_subplot = _create_subplot(
        subplot_df=(
            df_plot.filter(
                pl.col("metric").is_in(left_subplot_metrics_to_colors.keys())
            )
        ),
        y_title=lbl["y_left_title"],
        colors=list(left_subplot_metrics_to_colors.values()),
        domain=list(left_subplot_metrics_to_colors.keys()),
    )

    # Right subplot
    right_subplot = _create_subplot(
        subplot_df=(
            df_plot.filter(
                pl.col("metric").is_in(right_subplot_metrics_to_colors.keys())
            )
        ),
        y_title=lbl["y_right_title"],
        colors=list(right_subplot_metrics_to_colors.values()),
        domain=list(right_subplot_metrics_to_colors.keys()),
    )

    # Final chart
    chart = (left_subplot | right_subplot).resolve_scale(color="independent")
    if not no_title:
        chart = chart.properties(title=lbl["title"])

    out_files: list[Path] = []
    for fmt in formats:
        kwargs: dict[str, Any] = {"scale_factor": PNG_SCALE} if fmt == "png" else {}
        out_file = out_dir / f"{out_filename}.{fmt}"
        chart.save(out_file, **kwargs)
        out_files.append(out_file)

    return tuple(out_files)
