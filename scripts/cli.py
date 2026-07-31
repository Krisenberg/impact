from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

import click

from impact.cli import inject_impact_config
from impact.logger import configure_logging

from . import tables_render_latex
from .tables_data import compute_data_funnel_stats_for_variants

if TYPE_CHECKING:
    from collections.abc import Callable

    from impact.config import ImpactConfig

    from .types import Language, TableFormat

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]


@click.group()
def cli() -> None:
    """Main script's CLI entry point."""
    pass


@cli.group(name="tables")
def tables() -> None:
    """Commands for generating tables (used in thesis/documentation)."""
    pass


@tables.command(name="data-funnel-stats")
@click.option(
    "--log-level",
    type=click.Choice(choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="WARNING",
    show_default=True,
    help="Logging level.",
)
@click.option(
    "--lang",
    type=click.Choice(["pl", "en"], case_sensitive=False),
    default="en",
    show_default=True,
    help="Language for the table headers and textual content.",
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
        Output directory for the LaTeX code and corresponding ImpactConfig YAML dumps.
        Defaults to project root / 'reports' / 'tables' / 'data-funnel-stats'.
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
    log_level: str = "INFO",
    lang: Language = "en",
    fmt: TableFormat = "latex",
    out_dir: Path | None = None,
    raw_data_dir: Path | None = None,
    interim_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
) -> None:
    """Generate a table of record counts across data preparation stages."""
    configure_logging(
        level=logging.getLevelNamesMapping()[log_level],
    )  # Configure logging without piping logs to files

    out_dir = out_dir or PROJECT_ROOT / "reports" / "tables" / "data-funnel-stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_df, impact_config_by_display_id = compute_data_funnel_stats_for_variants(
        base_impact_config=impact_config,
        raw_data_dir=raw_data_dir or PROJECT_ROOT / "data" / "raw",
        interim_data_dir=interim_data_dir or PROJECT_ROOT / "data" / "interim",
        processed_data_dir=processed_data_dir or PROJECT_ROOT / "data" / "processed",
    )

    renderer_ext_by_fmt: dict[TableFormat, tuple[Callable[..., str], str]] = {
        "latex": (tables_render_latex.render_data_funnel_stats, "tex"),
        # "markdown": tables_render_markdown.render_data_funnel_stats,
    }

    rendered_table = renderer_ext_by_fmt[fmt][0](data_df, lang.lower())
    ext = renderer_ext_by_fmt[fmt][1]

    for display_id, config in impact_config_by_display_id.items():
        config_dump_path = out_dir / f"{display_id}.yaml"
        config.write_yaml(config_dump_path)
        click.echo(f"Config dumped to {config_dump_path.as_posix()}")

    out_file = out_dir / f"data_funnel_stats.{ext}"
    _ = out_file.write_text(rendered_table)
    click.echo(f"Table saved at {out_file.as_posix()}")


if __name__ == "__main__":
    cli()
