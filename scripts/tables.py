"""Module for creating tables in LaTeX format that are presented in the thesis."""

import dataclasses
import logging
from pathlib import Path
from typing import NamedTuple

import click
import polars as pl
import structlog

from impact.cli import inject_pipeline_config
from impact.dataset import DatasetSize
from impact.logger import configure_logging
from impact.pipeline import DataPipeline, PipelineConfig

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
    default="pl",
    help="Language for the table headers and textual content (pl or en).",
)
@click.option(
    "--raw-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    help="""
        Root directory for raw data (the original, immutable data dump).
        Defaults to project root / 'data' / 'raw'.
    """,
)
@click.option(
    "--interim-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    help="""
        Root directory for interim data (intermediate data that has been transformed).
        Defaults to project root / 'data' / 'interim'.
    """,
)
@click.option(
    "--processed-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    help="""
        Root directory for processed data (the final, canonical datasets for modeling).
        Defaults to project root / 'data' / 'processed'.
    """,
)
@inject_pipeline_config
def data_funnel_stats_table(
    pipeline_config: PipelineConfig,
    log_level: str,
    lang: str = "pl",
    raw_data_dir: Path | None = None,
    interim_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
) -> None:
    """Generate LaTeX table showing data flow statistics across 2 dataset variants."""
    configure_logging(
        level=logging.getLevelNamesMapping()[log_level],
    )  # Configure logging without piping logs to files

    translations: dict[str, dict[str, str]] = {
        "pl": {
            "stage": "Etap przygotowania danych",
            "raw": "Dane wejściowe ($E$)",
            "sessionized": "Sekwencja odsłuchów z identyfikatorem sesji ($S_{ses}$)",
            "sequenced": "Sekwencje sesji odsłuchowych ($S_{sek}$)",
            "caption": "Liczebność danych po kolejnych etapach przygotowania zbioru eksperymentalnego.",  # noqa: E501
        },
        # "en": {
        #     "stage": "Processing Stage",
        #     "raw": "Raw events",
        #     "sessionized": "After sessionization ($K=10$)",
        #     "sequenced": "Generated sequences ($L=20$)",
        #     "caption": "Data processing funnel and temporal split statistics across Yambda dataset variants.",  # noqa: E501
        # },
    }
    t = translations[lang.lower()]

    project_root = Path(__file__).resolve().parents[1]
    pipeline = DataPipeline(
        raw_dir=raw_data_dir or project_root / "data" / "raw",
        interim_dir=interim_data_dir or project_root / "data" / "interim",
        processed_dir=processed_data_dir or project_root / "data" / "processed",
        config=pipeline_config,
    )

    class VariantConfig(NamedTuple):
        step: int
        n_partitions: int
        dataset_name: str  # Name of the dataset for the "reporting" (in the thesis)

    variants: dict[DatasetSize, VariantConfig] = {
        DatasetSize.SIZE_50M: VariantConfig(
            step=5,
            n_partitions=1,
            dataset_name="Yambda-50M",
        ),
        DatasetSize.SIZE_5B: VariantConfig(
            step=20,
            n_partitions=32,
            dataset_name="Yambda-5B",
        ),
    }

    rows: list[dict[str, str | int]] = []
    for ds_size, variant in variants.items():
        logger.info(f"Processing {ds_size.name} dataset variant")
        variant_config = dataclasses.replace(
            pipeline_config,
            dataset_size=ds_size,
            step=variant.step,
        )
        pipeline.config = variant_config  # Switch pipeline config

        events_lf = pipeline.get_raw_events()
        sessions_lf = pipeline.get_sessions(n_partitions=variant.n_partitions)
        sequences_lf = pipeline.get_sequences(n_partitions=variant.n_partitions)

        rows.append({
            "dataset": variant.dataset_name,
            "raw": _format_num(events_lf.select(pl.len()).collect().item()),
            "sessionized": _format_num(sessions_lf.select(pl.len()).collect().item()),
            "sequenced": _format_num(sequences_lf.select(pl.len()).collect().item()),
        })  # fmt: skip

    df = pl.DataFrame({
        t["stage"]: [t["raw"], t["sessionized"], t["sequenced"]],
        **{r["dataset"]: [r["raw"], r["sessionized"], r["sequenced"]] for r in rows}
    })  # fmt: skip

    # Generate LaTeX table using Pandas Styler
    latex_output = (
        df.to_pandas()
        .style.hide(axis="index")
        .to_latex(
            hrules=True,
            environment="table",
            position="H",
            position_float="centering",
            label="tab:data_funnel_variants",
            caption=" ".join(t["caption"].split()),
            column_format="lrr",  # 'l' for the text, 'r' for the numbers
        )
    )

    click.echo(latex_output)


def _format_num(n: int) -> str:
    """Format integer with space as thousands separator."""
    return f"{n:,}".replace(",", " ")


if __name__ == "__main__":
    cli()
