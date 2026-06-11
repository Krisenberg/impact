"""Module for creating plots that will be presented in the thesis."""

import dataclasses
import datetime
import logging
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import altair as alt
import click
import dotenv
import numpy as np
import polars as pl
import structlog

from impact.cli import inject_impact_config
from impact.config import ImpactConfig
from impact.logger import configure_logging
from impact.pipelines import DataPipeline, IMPACTEvaluationPipeline
from impact.utils import seed_everything

logger = structlog.stdlib.get_logger(__name__)


class OkabeItoPalette(NamedTuple):
    """Okabe-Ito palette as a named tuple of hex color strings.

    This color palette is colorblind-safe and prints well in grayscale. Encapsulating
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
FONT_SIZE_TINY = 8
COLOR_INK = "#1A1A1A"  # Labels, ticks, axis domains
COLOR_GRID = "#E0E0E0"  # Passive guides - gridlines, legend borders, etc.
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


@cli.command(name="plot-alpha-curve")
@click.option(
    "--sasrec-checkpoint-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to a trained SASRecNBR checkpoint that will be evaluated.",
)
@click.option(
    "--log-level",
    type=click.Choice(list(logging.getLevelNamesMapping()), case_sensitive=False),
    default="WARNING",
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
    "--n-partitions",
    type=int,
    default=1,
    show_default=True,
    help="""
        Number of partitions for sequential processing. Use >1 for large datasets that
        exceed available RAM (data is partitioned by user, i.e. uid column).
    """,
)
@click.option(
    "--raw-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Base directory for raw data (the original, immutable data dump).
        Defaults to <project_root>/data/raw.
    """,
)
@click.option(
    "--interim-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Base directory for interim data (intermediate data that has been transformed).
        Defaults to <project_root>/data/interim.
    """,
)
@click.option(
    "--processed-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    help="""
        Base directory for processed data (the final, canonical datasets for modeling).
        Defaults to <project_root>/data/processed.
    """,
)
@click.option(
    "--alpha-start",
    type=float,
    default=0.0,
    show_default=True,
    help="Lower bound of the alpha sweep range (inclusive).",
)
@click.option(
    "--alpha-stop",
    type=float,
    default=1.0,
    show_default=True,
    help="Upper bound of the alpha sweep range (inclusive).",
)
@click.option(
    "--alpha-step",
    type=float,
    default=0.01,
    show_default=True,
    help="Step size between consecutive alpha values in the sweep.",
)
@click.option(
    "--metrics-at-k",
    type=int,
    default=10,
    show_default=True,
    help="Cutoff level k for the metrics collected during the alpha sweep.",
)
@click.option(
    "--out-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Output directory for the generated plots.
        Defaults to <project_root>/reports/plots/alpha-curve.
    """,
)
@click.option(
    "--out-filename",
    type=str,
    default=None,
    help="""
        Base filename (without extension) for the saved plots.
        Defaults to a UTC timestamp: alpha-curve_YYYY-MM-DD_HH-MM-SS.
    """,
)
@click.option(
    "--no-title",
    is_flag=True,
    default=False,
    help="Suppress the chart title (e.g. when the figure will be captioned in LaTeX).",
)
@click.option(
    "--format",
    "-f",
    "formats",
    type=click.Choice(("svg", "pdf", "png"), case_sensitive=False),
    multiple=True,
    default=("svg", "pdf"),
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
@inject_impact_config
def plot_alpha_curve(
    impact_config: ImpactConfig,
    sasrec_checkpoint_path: Path,
    log_level: str,
    lang: str,
    n_partitions: int,
    raw_data_dir: Path | None = None,
    interim_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    alpha_start: float = 0.0,
    alpha_stop: float = 1.0,
    alpha_step: float = 0.01,
    metrics_at_k: int = 10,
    out_dir: Path | None = None,
    out_filename: str | None = None,
    no_title: bool = False,
    formats: tuple[Literal["svg", "pdf", "png"], ...] = ("svg", "pdf"),
    png_scale: float = 3.0,
) -> None:
    """Plot IMPACT evaluation metrics as a function of alpha.

    Alpha controls the relative contribution of SASRec and ACT-R in the final
    late-fusion score (trade-off).
    """
    raw_data_dir = raw_data_dir or PROJECT_ROOT / "data" / "raw"
    interim_data_dir = interim_data_dir or PROJECT_ROOT / "data" / "interim"
    processed_data_dir = processed_data_dir or PROJECT_ROOT / "data" / "processed"
    out_dir = out_dir or PROJECT_ROOT / "reports" / "plots" / "alpha-curve"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.UTC)
    out_filename = (
        out_filename or f"alpha-curve_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    # Configure logging without piping logs to files
    configure_logging(
        level=logging.getLevelNamesMapping()[log_level],
    )

    # Load env configuration (e.g. HuggingFace API token)
    logger.info("Attempting to load env variables from .env file")
    _ = dotenv.load_dotenv()

    # Set seed for reproducibility
    seed_everything(impact_config.system.seed)

    data_pipeline = DataPipeline(
        raw_dir=raw_data_dir,
        interim_dir=interim_data_dir,
        processed_dir=processed_data_dir,
        config=impact_config,
    )

    logger.info("Fetching raw events data...")
    events_lf = data_pipeline.get_raw_events()

    logger.info("Fetching train/val/test splits...")
    splits, item_index = data_pipeline.get_splits(n_partitions=n_partitions)

    logger.info("Tensorizing splits...")
    tensorized_splits = data_pipeline.tensorize(splits)

    alphas = np.round(
        a=np.arange(alpha_start, alpha_stop + alpha_step / 2, alpha_step),
        decimals=2,
    )
    results: list[dict[str, float]] = []

    for alpha in alphas:
        eval_config = dataclasses.replace(
            impact_config,
            eval=dataclasses.replace(
                impact_config.eval,
                alpha=float(alpha),
                k_cutoffs=(metrics_at_k,),
            ),
        )

        logger.info(f"Running IMPACT evaluation for {alpha=}...")
        eval_pipeline = IMPACTEvaluationPipeline(eval_config, item_index, events_lf)
        metrics: dict[str, float] = eval_pipeline.evaluate(
            split_partition=tensorized_splits.val,
            sasrec_state_dict_path=sasrec_checkpoint_path,
        )

        row = {"alpha": float(alpha), **metrics}
        results.append(row)

    # -- Labels per language --
    labels: dict[str, dict[str, str]] = {
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

    # -- Plot data --
    df_plot = (
        pl.DataFrame(results)
        .unpivot(
            on=[r for r in results[0] if r != "alpha"],
            index="alpha",
            variable_name="metric",
            value_name="score",
        )
        .filter(pl.col("metric").str.contains("(?i)ndcg"))  # only NDCG
        .with_columns(pl.col("metric").replace(labels[lang]))
    )

    # -- Subplots specification --
    left_subplot_metrics_to_colors: dict[str, str] = {
        lbl[metric]: color
        for metric, color in {
            "ndcg_at_10": OKABE_ITO_PALETTE.black,
            "ndcg_local_rep_at_10": OKABE_ITO_PALETTE.blue,
            "ndcg_global_rep_at_10": OKABE_ITO_PALETTE.vermillion,
        }.items()
    }
    right_subplot_metrics_to_colors: dict[str, str] = {
        lbl[metric]: color
        for metric, color in {
            "ndcg_local_exp_at_10": OKABE_ITO_PALETTE.orange,
            "ndcg_global_exp_at_10": OKABE_ITO_PALETTE.bluish_green,
        }.items()
    }

    # -- Shared x-axis --
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

    # -- Left subplot --
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

    # -- Right subplot --
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

    # -- Final chart (horizontal concat) --
    chart = (
        (left_subplot | right_subplot)
        .resolve_scale(color="independent")
        .properties(title=lbl["title"] if not no_title else None)
    )

    for fmt in formats:
        kwargs: dict[str, Any] = {"scale_factor": png_scale} if fmt == "png" else {}
        path = out_dir / f"{out_filename}.{fmt}"
        chart.save(path, **kwargs)
        click.echo(f"Saved chart in {fmt.upper()} format to {path}")


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

    # -- Parse log file --
    logs_df = pl.read_ndjson(
        source=log_file_path,
        infer_schema_length=None,
        schema_overrides={
            "epoch": pl.UInt32,
            "train_loss": pl.Float64,
            "val_ndcg": pl.Float64,
        },
    )
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
