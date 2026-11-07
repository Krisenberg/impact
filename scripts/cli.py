from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

import click

from impact.cli import (
    inject_impact_config,
    inject_impact_config_from_checkpoint,
    inject_impact_config_overrides,
)
from impact.logger import configure_logging

from . import plots_render, tables_render_latex, tables_render_markdown
from .plots_data import compute_alpha_curve_plot_data
from .tables_data import compute_data_funnel_stats_for_variants

if TYPE_CHECKING:
    from impact.config import ImpactConfig

    from .types import Language, PlotFormat, TableFormat

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]


@click.group()
def cli() -> None:
    """CLI entry point for the scripts."""
    pass


@cli.group(name="tables")
def tables() -> None:
    """Commands for generating tables (used in thesis/documentation)."""
    pass


@cli.group(name="plots")
def plots() -> None:
    """Commands for generating plots (used in thesis/documentation)."""
    pass


@tables.command(name="data-funnel-stats")
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
    default="en",
    show_default=True,
    help="Language for axis labels and annotations.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["latex", "markdown"], case_sensitive=False),
    default="latex",
    show_default=True,
    help="Table format.",
)
@click.option(
    "--out-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Output directory for the rendered table code and corresponding ImpactConfig YAML
        dumps. Defaults to project root / 'reports' / 'tables' / 'data-funnel-stats'.
    """,
)
@click.option(
    "--raw-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Root directory for raw datasets (the original, immutable data dump).
        Defaults to project root / 'data' / 'raw'.
    """,
)
@click.option(
    "--interim-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Root directory for interim data (intermediate data that has been transformed).
        Defaults to project root / 'data' / 'interim'.
    """,
)
@click.option(
    "--processed-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Root directory for processed data (the final datasets for model training and
        evaluation). Defaults to project root / 'data' / 'processed'.
    """,
)
@inject_impact_config
def generate_data_funnel_stats_table(
    impact_config: ImpactConfig,
    log_level: str = "WARNING",
    lang: Language = "en",
    fmt: TableFormat = "latex",
    out_dir: Path | None = None,
    raw_data_dir: Path | None = None,
    interim_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
) -> None:
    """Generate a table of record counts across data preparation stages."""
    configure_logging(level=logging.getLevelNamesMapping()[log_level])

    out_dir = out_dir or PROJECT_ROOT / "reports" / "tables" / "data-funnel-stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_df, impact_config_by_display_id = compute_data_funnel_stats_for_variants(
        base_impact_config=impact_config,
        raw_data_dir=raw_data_dir or PROJECT_ROOT / "data" / "raw",
        interim_data_dir=interim_data_dir or PROJECT_ROOT / "data" / "interim",
        processed_data_dir=processed_data_dir or PROJECT_ROOT / "data" / "processed",
    )

    match fmt:
        case "latex":
            table = tables_render_latex.render_data_funnel_stats(data_df, lang)
            ext = "tex"
        case "markdown":
            table = tables_render_markdown.render_data_funnel_stats(data_df, lang)
            ext = "md"

    for display_id, config in impact_config_by_display_id.items():
        config_dump_path = out_dir / f"{display_id}.yaml"
        config.write_yaml(config_dump_path)
        click.echo(f"Config dumped to {config_dump_path.as_posix()}")

    out_file = out_dir / f"data_funnel_stats-{lang}.{ext}"
    _ = out_file.write_text(table)
    click.echo(f"Table saved at {out_file.as_posix()}")


@plots.command(name="alpha-curve")
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
    default="en",
    show_default=True,
    help="Language for axis labels and annotations.",
)
@click.option(
    "--out-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Output directory for the generated plots and corresponding ImpactConfig YAML
        dumps. Defaults to project root / 'reports' / 'plots' / 'alpha-curve'.
    """,
)
@click.option(
    "--raw-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Root directory for raw datasets (the original, immutable data dump).
        Defaults to project root / 'data' / 'raw'.
    """,
)
@click.option(
    "--interim-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Root directory for interim data (intermediate data that has been transformed).
        Defaults to project root / 'data' / 'interim'.
    """,
)
@click.option(
    "--processed-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Root directory for processed data (the final datasets for model training and
        evaluation). Defaults to project root / 'data' / 'processed'.
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
@inject_impact_config_from_checkpoint
@inject_impact_config_overrides("actr", "eval", "system")
def plot_alpha_curve(
    impact_config: ImpactConfig,
    sasrec_checkpoint_path: Path,
    log_level: str = "WARNING",
    lang: Language = "en",
    out_dir: Path | None = None,
    raw_data_dir: Path | None = None,
    interim_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    alpha_start: float = 0.0,
    alpha_stop: float = 1.0,
    alpha_step: float = 0.01,
    metrics_at_k: int = 10,
    no_title: bool = False,
    formats: tuple[PlotFormat, ...] = ("svg", "pdf"),
) -> None:
    """Plot IMPACT evaluation metrics as a function of alpha."""
    configure_logging(level=logging.getLevelNamesMapping()[log_level])

    out_dir = out_dir or PROJECT_ROOT / "reports" / "plots" / "alpha-curve"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_identifier_suffix = "-".join(
        [
            str(alpha_start).replace(".", "_"),
            str(alpha_stop).replace(".", "_"),
            str(alpha_step).replace(".", "_"),
            str(metrics_at_k).replace(".", "_"),
        ]
    )
    config_identifier = f"{impact_config.display_id}-{config_identifier_suffix}"

    alpha_curve_df = compute_alpha_curve_plot_data(
        impact_config=impact_config,
        sasrec_checkpoint_path=sasrec_checkpoint_path,
        raw_data_dir=raw_data_dir or PROJECT_ROOT / "data" / "raw",
        interim_data_dir=interim_data_dir or PROJECT_ROOT / "data" / "interim",
        processed_data_dir=processed_data_dir or PROJECT_ROOT / "data" / "processed",
        alpha_start=alpha_start,
        alpha_stop=alpha_stop,
        alpha_step=alpha_step,
        metrics_at_k=metrics_at_k,
    )
    out_files = plots_render.render_alpha_curve_plot(
        alpha_curve_df=alpha_curve_df,
        alpha_start=alpha_start,
        alpha_stop=alpha_stop,
        out_dir=out_dir,
        out_filename=f"alpha-curve-{lang}-{config_identifier}",
        formats=formats,
        lang=lang,
        no_title=no_title,
    )
    for out_file in out_files:
        click.echo(f"Saved chart at {out_file.as_posix()}")

    config_dump_path = out_dir / f"{config_identifier}.yaml"
    impact_config.write_yaml(config_dump_path)
    click.echo(f"Config dumped to {config_dump_path.as_posix()}")


if __name__ == "__main__":
    cli()
