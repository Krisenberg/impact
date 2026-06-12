"""Module for creating tables in LaTeX format that are presented in the thesis."""

import dataclasses
import logging
import string
from pathlib import Path
from typing import NamedTuple

import click
import polars as pl
import structlog

from impact.cli import inject_impact_config
from impact.config import ImpactConfig
from impact.dataset import DatasetSize
from impact.logger import configure_logging
from impact.pipelines import DataPipeline

logger = structlog.stdlib.get_logger(__name__)


@click.group()
def cli() -> None:
    """CLI entrypoint for the functions defined in this module."""
    pass


@cli.command(name="data-funnel-stats")
@click.option(
    "--log-level",
    type=click.Choice(choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="""
        Logging level.
    """,
)
@click.option(
    "--lang",
    type=click.Choice(["pl", "en"], case_sensitive=False),
    default="en",
    help="Language for the table headers and textual content (pl or en).",
)
@click.option(
    "--out-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Output directory for the LaTeX code and corresponding PipelineConfig YAML dumps.
        Defaults to project root / 'reports' / 'tables' / 'data-funnel-stats'.
    """,
)
@click.option(
    "--raw-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Root directory for raw data (the original, immutable data dump).
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
        Root directory for processed data (the final, canonical datasets for modeling).
        Defaults to project root / 'data' / 'processed'.
    """,
)
@inject_impact_config
def generate_data_funnel_stats_table(
    impact_config: ImpactConfig,
    log_level: str = "INFO",
    lang: str = "en",
    out_dir: Path | None = None,
    raw_data_dir: Path | None = None,
    interim_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
) -> None:
    """Generate a LaTeX table of record counts across data preparation stages.

    Compares Yambda-50M and Yambda-5B dataset variants.
    """
    configure_logging(
        level=logging.getLevelNamesMapping()[log_level],
    )  # Configure logging without piping logs to files

    translations: dict[str, dict[str, str]] = {
        "pl": {
            "stage": "Etap przygotowania danych",
            "raw": "Dane wejściowe ($E$)",
            "sessionized": "Sekwencja odsłuchów z identyfikatorem sesji ($S_{ses}$)",
            "sequenced": "Sekwencje sesji odsłuchowych ($S_{sek}$)",
            "caption": "Liczba rekordów po kolejnych etapach przygotowania zbioru eksperymentalnego.",  # noqa: E501
            "reproducibility_note": (
                "Konfiguracja IMPACT Framework: \\texttt{$display_id}~"
                "(\\cref{$appendix_label})."
            ),
        },
        "en": {
            "stage": "Data preparation stage",
            "raw": "Raw data ($E$)",
            "sessionized": "Listen events with session identifiers ($S_{ses}$)",
            "sequenced": "Listening session sequences ($S_{sek}$)",
            "caption": "Record counts across data preparation stages for the experimental dataset.",  # noqa: E501
            "reproducibility_note": (
                "The IMPACT Framework configuration: \\texttt{$display_id}~"
                "(\\cref{$appendix_label})."
            ),
        },
    }
    t = translations[lang.lower()]

    project_root = Path(__file__).resolve().parents[1]
    pipeline = DataPipeline(
        raw_dir=raw_data_dir or project_root / "data" / "raw",
        interim_dir=interim_data_dir or project_root / "data" / "interim",
        processed_dir=processed_data_dir or project_root / "data" / "processed",
        config=impact_config,
    )

    out_dir = out_dir or project_root / "reports" / "tables" / "data-funnel-stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    class _VariantConfig(NamedTuple):
        step: int
        dataset_name: str  # Name of the dataset for the "reporting" (in the thesis)
        marker: str  # Marker for the footnotes

    variants: dict[DatasetSize, _VariantConfig] = {
        DatasetSize.SIZE_50M: _VariantConfig(
            step=5,
            dataset_name="Yambda-50M",
            marker="a",
        ),
        DatasetSize.SIZE_5B: _VariantConfig(
            step=20,
            dataset_name="Yambda-5B",
            marker="b",
        ),
    }

    rows: list[dict[str, str | int]] = []
    footnotes: list[str] = []

    for ds_size, variant in variants.items():
        logger.info(f"Processing {ds_size.name} dataset variant")
        variant_config = dataclasses.replace(
            impact_config,
            dataset=dataclasses.replace(impact_config.dataset, size=ds_size),
            sequence=dataclasses.replace(impact_config.sequence, step=variant.step),
        )
        pipeline.config = variant_config  # Switch pipeline config

        events_lf = pipeline.get_raw_events()
        sessions_lf = pipeline.get_sessions()
        sequences_lf = pipeline.get_sequences()

        tnote = f"\\tnote{{\\hyperlink{{note:{variant.marker}}}{{{variant.marker}}}}}"
        rows.append({
            "dataset": f"{variant.dataset_name}{tnote}",
            # TODO: report also the number of valid listening events
            "raw": _format_num(events_lf.select(pl.len()).collect().item()),
            "sessionized": _format_num(sessions_lf.select(pl.len()).collect().item()),
            "sequenced": _format_num(sequences_lf.select(pl.len()).collect().item()),
        })  # fmt: skip

        hypertarget = f"\\hypertarget{{note:{variant.marker}}}{{{variant.marker}}}"
        note_text = string.Template(t["reproducibility_note"]).substitute(
            display_id=variant_config.display_id,
            appendix_label=f"appendix:impactconfig:{variant_config.display_id}",
        )
        footnotes.append(f"\\item[{hypertarget}]: {note_text}")

        config_dump_path = out_dir / f"{variant_config.display_id}.yaml"
        variant_config.write_yaml(config_dump_path)
        click.echo(f"Config dumped to {config_dump_path.as_posix()}")

    df = pl.DataFrame({
        t["stage"]: [t["raw"], t["sessionized"], t["sequenced"]],
        **{r["dataset"]: [r["raw"], r["sessionized"], r["sequenced"]] for r in rows}
    })  # fmt: skip

    tablenotes_latex = (
        "\\begin{tablenotes}[flushleft]\n"
        "\\footnotesize\n"
        f"{'\n'.join(footnotes)}\n"
        "\\end{tablenotes}\n"
    )
    latex_output = (
        df.to_pandas()
        .style.hide(axis="index")
        # Generate LaTeX table using Pandas Styler
        .to_latex(
            hrules=True,
            environment="threeparttable",  # To allow table footnotes
            position="H",
            position_float="centering",
            label="tab:data_funnel_stats",
            caption=" ".join(t["caption"].split()),
            column_format="lrr",  # 'l' for the text, 'r' for the numbers
        )
        .replace("\\end{tabular}\n", f"\\end{{tabular}}\n{tablenotes_latex}")
    )

    out_file = out_dir / "tab:data_funnel_stats.tex"
    with click.open_file(out_file, "w") as f:
        click.echo(latex_output, file=f)

    click.echo(f"LaTeX table saved at {out_file.as_posix()}")


def _format_num(n: int) -> str:
    """Format integer with space as thousands separator."""
    return f"{n:,}".replace(",", " ")


if __name__ == "__main__":
    cli()
