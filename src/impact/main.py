"""Main entrypoint for the IMPACT application and pipelines execution."""

from __future__ import annotations

import dataclasses
import datetime
import itertools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import dotenv
import polars as pl
import structlog

from impact.cli import inject_impact_config
from impact.logger import configure_logging
from impact.pipelines import DataPipeline, EvaluationPipeline, SASRecTrainingPipeline
from impact.utils import seed_everything

if TYPE_CHECKING:
    from collections.abc import Iterator

    from impact.config import ImpactConfig

logger = structlog.stdlib.get_logger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@click.group()
def cli() -> None:
    """IMPACT - Intention-Modulated Personalized ACT-R."""
    pass


@cli.command(name="main-pipeline")
@click.option(
    "--log-level",
    type=click.Choice(list(logging.getLevelNamesMapping()), case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging verbosity level.",
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
        Root directory for raw data (the original, immutable data dump).
        Defaults to <project_root>/data/raw.
    """,
)
@click.option(
    "--interim-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Root directory for interim data (intermediate data that has been transformed).
        Defaults to <project_root>/data/interim.
    """,
)
@click.option(
    "--processed-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    help="""
        Root directory for processed data (the final, canonical datasets for modeling).
        Defaults to <project_root>/data/processed.
    """,
)
@click.option(
    "--base-logs-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    help="Root directory for log files. Defaults to <project_root>/logs.",
)
@click.option(
    "--models-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    help="""
        Root directory for trained and serialized models, model predictions, or model
        summaries. Defaults to <project_root>/models.
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

    raw_data_dir = raw_data_dir or PROJECT_ROOT / "data" / "raw"
    interim_data_dir = interim_data_dir or PROJECT_ROOT / "data" / "interim"
    processed_data_dir = processed_data_dir or PROJECT_ROOT / "data" / "processed"
    base_logs_dir = base_logs_dir or PROJECT_ROOT / "logs"
    models_dir = models_dir or PROJECT_ROOT / "models"

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

    # Set seed for reproducibility
    seed_everything(impact_config.seed)

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

    sasrec_training_pipeline = SASRecTrainingPipeline(config=impact_config)
    state_dict_filename = f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_best_model.pt"
    sasrec_state_dict_path = models_dir / state_dict_filename
    _ = sasrec_training_pipeline.train(split=tensors, save_path=sasrec_state_dict_path)

    evaluation_pipeline = EvaluationPipeline(impact_config)
    metrics = evaluation_pipeline.evaluate(tensors, sasrec_state_dict_path)

    logger.info("Evaluation pipeline completed", **metrics)
    logger.info("IMPACT Framework configuration", **impact_config.to_dict())
    logger.info("IMPACT application completed")


@cli.command(name="tune-baseline")
@click.option(
    "--log-level",
    type=click.Choice(list(logging.getLevelNamesMapping()), case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging verbosity level.",
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
        Root directory for raw data (the original, immutable data dump).
        Defaults to <project_root>/data/raw.
    """,
)
@click.option(
    "--interim-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Root directory for interim data (intermediate data that has been transformed).
        Defaults to <project_root>/data/interim.
    """,
)
@click.option(
    "--processed-data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    help="""
        Root directory for processed data (the final, canonical datasets for modeling).
        Defaults to <project_root>/data/processed.
    """,
)
@click.option(
    "--base-logs-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    help="Root directory for log files. Defaults to <project_root>/logs.",
)
@click.option(
    "--models-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=False,
    help="""
        Root directory for trained and serialized models and grid search results.
        Defaults to <project_root>/models.
    """,
)
def tune_baseline(
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
    """Run a grid search on Yambda-50M to find optimal SASRecNBR hyperparameters."""
    timestamp = exec_time or datetime.datetime.now(datetime.UTC)

    raw_data_dir = raw_data_dir or PROJECT_ROOT / "data" / "raw"
    interim_data_dir = interim_data_dir or PROJECT_ROOT / "data" / "interim"
    processed_data_dir = processed_data_dir or PROJECT_ROOT / "data" / "processed"
    base_logs_dir = base_logs_dir or PROJECT_ROOT / "logs"
    models_dir = models_dir or PROJECT_ROOT / "models"
    out_dir = models_dir / f"grid_search_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir.mkdir(exist_ok=True, parents=True)

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

    # Set seed for reproducibility
    seed_everything(impact_config.seed)

    pipeline = DataPipeline(
        raw_dir=raw_data_dir,
        interim_dir=interim_data_dir,
        processed_dir=processed_data_dir,
        config=impact_config,
    )
    splits = pipeline.get_splits(n_partitions=n_partitions)
    tensors = pipeline.tensorize(splits)

    logger.info(
        "Dataset preprocessing and tensorization completed",
        train_size=tensors.train.meta[0],
        val_size=tensors.val.meta[0],
        test_size=tensors.test.meta[0],
    )

    best_ndcg = 0.0
    best_config: ImpactConfig | None = None
    tuning_results: list[dict[str, Any]] = []

    best_model_path: Path = out_dir / "best_model.pt"
    current_model_path: Path = out_dir / "current_model.pt"

    logger.info("Starting hyperparameter tuning (Grid Search)...")

    param_grid: dict[str, list[Any]] = {
        # "learning_rate": [0.0002, 0.0005, 0.001, 0.002],
        "learning_rate": [0.0005, 0.001, 0.002],
        "embedding_dim": [64, 128],
        "dropout_rate": [0.2, 0.5],
        "num_layers": [2, 3],
    }
    param_combinations = list(_dict_product(param_grid))

    try:
        for i, params in enumerate(param_combinations, start=1):
            _ = structlog.contextvars.bind_contextvars(
                config_id=i,
                learning_rate=params["learning_rate"],
                embedding_dim=params["embedding_dim"],
                dropout_rate=params["dropout_rate"],
                num_layers=params["num_layers"],
            )
            logger.info(f"Testing configuration {i}/{len(param_combinations)}")

            sasrec_config = dataclasses.replace(
                impact_config.sasrec,
                learning_rate=params["learning_rate"],
                embedding_dim=params["embedding_dim"],
                dropout_rate=params["dropout_rate"],
                num_layers=params["num_layers"],
            )
            run_config = dataclasses.replace(impact_config, sasrec=sasrec_config)
            sasrec_training_pipeline = SASRecTrainingPipeline(config=run_config)
            val_ndcg = sasrec_training_pipeline.train(
                split=tensors,
                save_path=current_model_path,
            )

            tuning_results.append({"config_id": i, "val_ndcg": val_ndcg, **params})

            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                best_config = run_config
                # Remove the so-far best model checkpoint
                if best_model_path.exists():
                    best_model_path.unlink()
                # Make the current model a new best one
                _ = current_model_path.rename(best_model_path)
                logger.info(
                    "New best model configuration found and saved",
                    val_ndcg=round(val_ndcg, 4),
                )

            # Remove the current model checkpoint (if it exists)
            current_model_path.unlink(missing_ok=True)
    finally:
        structlog.contextvars.unbind_contextvars(
            "config_id", "learning_rate", "embedding_dim", "dropout_rate", "num_layers"
        )

    if best_config is None:
        raise RuntimeError("No model found with NDCG > 0")

    results_path = out_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(tuning_results, f, indent=4)

    best_config_path = out_dir / "sasrec_best_config.yaml"
    best_config.write_yaml(best_config_path)

    logger.info(
        "Grid search completed",
        best_ndcg=best_ndcg,
        best_config=best_config.display_id,
        results_path=results_path,
        best_config_path=best_config_path,
        best_model_path=best_model_path,
    )


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


def _dict_product(options: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Yield all combinations from a cartesian product of option lists.

    Args:
        options: Mapping of parameter names to lists of candidate values.

    Yields:
        Dicts representing each combination, one per cartesian product entry.

    Example:
        >>> list(_dict_product({'number': [1, 2], 'character': ['a', 'b']}))
        [{'number': 1, 'character': 'a'},
         {'number': 1, 'character': 'b'},
         {'number': 2, 'character': 'a'},
         {'number': 2, 'character': 'b'}]
    """
    return (
        dict(zip(options.keys(), x, strict=True))
        for x in itertools.product(*options.values())
    )


if __name__ == "__main__":
    cli()
