"""Module for creating plots that will be presented in the thesis."""

import logging
from pathlib import Path
from typing import Any, Literal, NamedTuple

import altair as alt
import click
import polars as pl
import structlog

from impact.logger import configure_logging

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
COLOR_INK = "#1A1A1A"  # Labels, ticks, axis domains
COLOR_GRID = "#E0E0E0"  # Passive guides - gridlines, legend borders
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


@alt.theme.register("thesis_theme", enable=True)
def thesis_theme() -> alt.theme.ThemeConfig:
    """Register and enable an Altair theme for publication-quality visualisations."""
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
            "line": {"strokeWidth": 1.6},
            "point": {"size": 35, "filled": True},
            "bar": {"fill": OKABE_ITO_PALETTE.blue},  # pyright: ignore[reportReturnType]
            # -- Color scale default --
            "range": {
                "category": OKABE_ITO_PALETTE,
                "ordinal": OKABE_ITO_PALETTE,
            },
        }
    }


@click.group()
def cli() -> None:
    """CLI entrypoint for the functions defined in this module."""
    pass


@cli.command(name="sasrec-training")
@click.option(
    "--log-file-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the JSON log file to parse.",
)
@click.option(
    "--log-level",
    type=click.Choice(list(logging.getLevelNamesMapping()), case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging verbosity level.",
)
@click.option(
    "--lang",
    type=click.Choice(["pl", "en"], case_sensitive=False),
    default="pl",
    show_default=True,
    help="Language for axis labels and annotations.",
)
@click.option(
    "--out-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Output directory for the generated plots and other artifacts. (e.g. ImpactConfig
        YAML dumps). Defaults to <project_root>/reports/plots/sasrec-training.
    """,
)
@click.option(
    "--format",
    "-f",
    "formats",
    type=click.Choice(["svg", "pdf", "png"], case_sensitive=False),
    multiple=True,
    default=["svg", "pdf"],
    show_default=True,
    help="Output format(s). Can be specified multiple times: -f svg -f pdf.",
)
@click.option(
    "--png-scale",
    type=float,
    default=3.0,
    show_default=True,
    help="Scale factor for PNG export. 3.0 is approx. 216 PPI - suitable for print.",
)
def plot_sasrec_training(
    log_file_path: Path,
    log_level: str,
    lang: str,
    out_dir: Path | None,
    formats: list[Literal["svg", "pdf", "png"]],
    png_scale: float,
) -> None:
    """Plot SASRec training curves from a structured NDJSON log file.

    Renders training loss (BCE) and validation NDCG@10 over epochs as a dual-axis chart,
    styled for direct inclusion in a LaTeX thesis.
    """
    configure_logging(
        level=logging.getLevelNamesMapping()[log_level],
    )  # Configure logging without piping logs to files

    out_dir = out_dir or PROJECT_ROOT / "reports" / "plots" / "sasrec-training"
    out_dir.mkdir(parents=True, exist_ok=True)

    logs_df = pl.read_ndjson(log_file_path)
    epochs_df = logs_df.filter(pl.col("epoch").is_not_null()).select(
        pl.col("epoch"), pl.col("train_loss"), pl.col("val_ndcg")
    )

    # -- Labels per language --
    labels = {
        "pl": {
            "epoch": "Epoka",
            "loss": "Strata treningowa (BCE)",
            "ndcg": "NDCG@10 (walidacja)",
        },
        "en": {
            "epoch": "Epoch",
            "loss": "Training Loss (BCE)",
            "ndcg": "NDCG@10 (validation)",
        },
    }[lang]

    # -- Shared x-axis --
    base = alt.Chart(epochs_df).encode(
        x=alt.X(
            shorthand="epoch:Q",
            axis=alt.Axis(title=labels["epoch"], tickMinStep=1),
            scale=alt.Scale(nice=False),
        )
    )

    color_loss = OKABE_ITO_PALETTE.blue
    color_ndcg = OKABE_ITO_PALETTE.vermillion

    # -- Loss (left axis) --
    loss_line = base.mark_line(color=color_loss).encode(
        y=alt.Y(
            shorthand="train_loss:Q",
            axis=alt.Axis(
                title=labels["loss"],
                titleColor=color_loss,
                labelColor=color_loss,
                tickColor=color_loss,
                domainColor=color_loss,
                gridOpacity=0.5,
            ),
        ),
    )
    loss_points = base.mark_point(filled=True, color=color_loss).encode(
        y=alt.Y(shorthand="train_loss:Q", axis=None)
    )

    # -- NDCG (right axis) --
    ndcg_line = base.mark_line(color=color_ndcg, strokeDash=[8, 4]).encode(
        y=alt.Y(
            shorthand="val_ndcg:Q",
            axis=alt.Axis(
                title=labels["ndcg"],
                titleColor=color_ndcg,
                labelColor=color_ndcg,
                tickColor=color_ndcg,
                domainColor=color_ndcg,
                grid=False,
                format=".4f",
            ),
        ),
    )
    ndcg_points = base.mark_point(filled=True, color=color_ndcg).encode(
        y=alt.Y("val_ndcg:Q", axis=None)
    )

    # -- Compose --
    chart = (
        alt.layer(loss_line, loss_points, ndcg_line, ndcg_points).resolve_scale(
            y="independent"
        )
        # .properties(width=380, height=240)
    )

    # -- Export --
    stem = f"sasrec_training_{log_file_path.stem}"
    for fmt in formats:
        kwargs: dict[str, Any] = {"scale_factor": png_scale} if fmt == "png" else {}
        path = out_dir / f"{stem}.{fmt}"
        chart.save(path, **kwargs)
        logger.info(f"Saved chart in {fmt.upper()} format to {path}")


if __name__ == "__main__":
    cli()
