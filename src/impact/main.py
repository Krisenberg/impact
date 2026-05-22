"""Main entrypoint for the IMPACT application and pipelines execution."""

import datetime
import logging
from pathlib import Path

import click
import polars as pl
import structlog

from impact.cli import inject_pipeline_config
from impact.configuration import load_config
from impact.logger import configure_logging
from impact.pipeline import DataPipeline, PipelineConfig

logger = structlog.stdlib.get_logger(__name__)


@click.group()
def cli() -> None:
    """IMPACT - Intention-Modulated Personalized ACT-R."""
    pass


@cli.command(name="main-pipeline")
@click.option(
    "--log-level",
    type=click.Choice(choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="""
        Logging level.
    """,
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
@click.option(
    "--base-logs-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    help="""
        Root directory for log files. Defaults to project root / 'logs'.
    """,
)
@inject_pipeline_config
def main_pipeline(
    pipeline_config: PipelineConfig,
    log_level: str,
    raw_data_dir: Path | None = None,
    interim_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    base_logs_dir: Path | None = None,
    exec_time: datetime.datetime | None = None,
) -> None:
    """IMPACT main pipeline execution entrypoint."""
    timestamp = exec_time or datetime.datetime.now(datetime.UTC)
    project_root = Path(__file__).resolve().parents[2]

    raw_data_dir = raw_data_dir or project_root / "data" / "raw"
    interim_data_dir = interim_data_dir or project_root / "data" / "interim"
    processed_data_dir = processed_data_dir or project_root / "data" / "processed"
    base_logs_dir = base_logs_dir or project_root / "logs"

    # Configure logging
    logs_dir = _get_logs_dir(base_dir=base_logs_dir, timestamp=timestamp)
    log_filename = f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_impact"
    configure_logging(
        level=logging.getLevelNamesMapping()[log_level],
        log_file=logs_dir / f"{log_filename}.log",
        json_log_file=logs_dir / f"{log_filename}.json",
        details=True,
    )

    logger.info("IMPACT application initialized", run_timestamp=timestamp.isoformat())

    _ = load_config()  # Load env configuration (e.g. HuggingFace API token)

    pipeline = DataPipeline(
        raw_dir=raw_data_dir,
        interim_dir=interim_data_dir,
        processed_dir=processed_data_dir,
        config=pipeline_config,
    )
    ds_size = pipeline_config.dataset_size

    events_lf = pipeline.get_raw_events()
    print(f"{ds_size} events_lf: {events_lf.select(pl.len()).collect().item()}")

    sessions_lf = pipeline.get_sessions()
    print(f"{ds_size} sessions_lf: {sessions_lf.select(pl.len()).collect().item()}")

    sequences_lf = pipeline.get_sequences()
    print(f"{ds_size} sequences_lf: {sequences_lf.select(pl.len()).collect().item()}")

    train, val, test = pipeline.get_splits()
    print(f"{ds_size} train: {train.select(pl.len()).collect().item()}")
    print(f"{ds_size} val: {val.select(pl.len()).collect().item()}")
    print(f"{ds_size} test: {test.select(pl.len()).collect().item()}")

    logger.info("IMPACT application completed")


def _get_logs_dir(base_dir: Path, timestamp: datetime.datetime) -> Path:
    """Return path to the hierarchical logs directory after ensuring it exists.

    Structure: base_dir/YYYY/YYYY-MM/YYYY-MM-DD.
    """
    year = timestamp.strftime("%Y")
    month = timestamp.strftime("%Y-%m")
    day = timestamp.strftime("%Y-%m-%d")

    dir_path = base_dir / year / month / day
    dir_path.mkdir(parents=True, exist_ok=True)

    return dir_path


if __name__ == "__main__":
    cli()
