"""Main entrypoint for the IMPACT application and pipelines execution."""

import datetime
import logging
from pathlib import Path

import click
import dotenv
import polars as pl
import structlog

from impact.cli import inject_impact_config
from impact.config import ImpactConfig
from impact.logger import configure_logging
from impact.pipelines import DataPipeline, SASRecTrainingPipeline

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
    "--n-partitions",
    type=int,
    default=1,
    help="""
        Number of partitions for sequential processing. Use >1 for large datasets that
        exceed available RAM (data is partitioned by user, i.e. uid column).
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
@click.option(
    "--models-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    help="""
        Root directory for trained and serialized models, model predictions, or model
        summaries. Defaults to project root / 'models'.
    """,
)
@inject_impact_config
def main_pipeline(
    impact_config: ImpactConfig,
    log_level: str,
    n_partitions: int,
    raw_data_dir: Path | None = None,
    interim_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    base_logs_dir: Path | None = None,
    models_dir: Path | None = None,
    exec_time: datetime.datetime | None = None,
) -> None:
    """IMPACT main pipeline execution entrypoint."""
    timestamp = exec_time or datetime.datetime.now(datetime.UTC)
    project_root = Path(__file__).resolve().parents[2]

    raw_data_dir = raw_data_dir or project_root / "data" / "raw"
    interim_data_dir = interim_data_dir or project_root / "data" / "interim"
    processed_data_dir = processed_data_dir or project_root / "data" / "processed"
    base_logs_dir = base_logs_dir or project_root / "logs"
    models_dir = models_dir or project_root / "models"

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

    # Load env configuration (e.g. HuggingFace API token)
    logger.info("Attempting to load env variables from .env file")
    _ = dotenv.load_dotenv()

    pipeline = DataPipeline(
        raw_dir=raw_data_dir,
        interim_dir=interim_data_dir,
        processed_dir=processed_data_dir,
        config=impact_config,
    )
    ds_size = impact_config.dataset_size

    events_lf = pipeline.get_raw_events()
    print(f"{ds_size} events_lf: {events_lf.select(pl.len()).collect().item()}")

    sessions_lf = pipeline.get_sessions(n_partitions=n_partitions)
    print(f"{ds_size} sessions_lf: {sessions_lf.select(pl.len()).collect().item()}")

    sequences_lf = pipeline.get_sequences(n_partitions=n_partitions)
    print(f"{ds_size} sequences_lf: {sequences_lf.select(pl.len()).collect().item()}")

    splits = pipeline.get_splits(n_partitions=n_partitions)
    print(f"{ds_size} train: {splits.train.lf.select(pl.len()).collect().item()}")
    print(f"{ds_size} val: {splits.val.lf.select(pl.len()).collect().item()}")
    print(f"{ds_size} test: {splits.test.lf.select(pl.len()).collect().item()}")
    print(
        f"{ds_size} item-idx: {splits.item_to_idx.lf.select(pl.len()).collect().item()}"
    )

    tensors = pipeline.tensorize(splits)
    n, l, k = tensors.train.meta  # noqa: E741
    print(f"{ds_size} train tensors: N={n}, L={l}, K={k}")
    print(f"{ds_size} val tensors:   N={tensors.val.meta[0]}")
    print(f"{ds_size} test tensors:  N={tensors.test.meta[0]}")

    sasrec_training_pipeline = SASRecTrainingPipeline(
        models_dir=models_dir,
        config=impact_config,
    )
    sasrec_training_pipeline.train(tensors)

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
