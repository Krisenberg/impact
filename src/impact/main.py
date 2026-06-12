from __future__ import annotations

import dataclasses
import datetime
import itertools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import click
import dotenv
import structlog

from impact.cli import (
    inject_impact_config,
    inject_impact_config_from_checkpoint,
    inject_impact_config_overrides,
)
from impact.dataset import DatasetSize
from impact.logger import configure_logging
from impact.pipelines import (
    DataPipeline,
    ImpactEvaluationPipeline,
    SASRecTrainingPipeline,
)
from impact.schemas import TensorizedDataset
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


@cli.command(name="train-sasrec")
@click.option(
    "--log-level",
    type=click.Choice(list(logging.getLevelNamesMapping()), case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging verbosity level.",
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
    default=None,
    help="""
        Base directory for processed data (the final, canonical datasets for modeling).
        Defaults to <project_root>/data/processed.
    """,
)
@click.option(
    "--base-logs-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Base directory for log files. Defaults to <project_root>/logs.",
)
@click.option(
    "--models-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Base directory where a serialized checkpoint of the trained SASRecNBR model
        will be saved. Defaults to <project_root>/models.
    """,
)
@inject_impact_config
def train_sasrec(
    impact_config: ImpactConfig,
    log_level: str = "INFO",
    raw_data_dir: Path | None = None,
    interim_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    base_logs_dir: Path | None = None,
    models_dir: Path | None = None,
    exec_time: datetime.datetime | None = None,
) -> None:
    """Run SASRecNBR training pipeline."""
    timestamp = exec_time or datetime.datetime.now(datetime.UTC)

    raw_data_dir = raw_data_dir or PROJECT_ROOT / "data" / "raw"
    interim_data_dir = interim_data_dir or PROJECT_ROOT / "data" / "interim"
    processed_data_dir = processed_data_dir or PROJECT_ROOT / "data" / "processed"
    base_logs_dir = base_logs_dir or PROJECT_ROOT / "logs"
    models_dir = (
        (models_dir or PROJECT_ROOT / "models")
        / "train-sasrec"
        / impact_config.display_id
    )
    models_dir.mkdir(exist_ok=True, parents=True)

    # Configure logging
    logs_dir = _get_logs_dir(base_dir=base_logs_dir, timestamp=timestamp)
    log_filename = f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_train-sasrec"
    configure_logging(
        level=logging.getLevelNamesMapping()[log_level],
        log_file=logs_dir / f"{log_filename}.log",
        json_log_file=logs_dir / f"{log_filename}.json",
        details=True,
    )

    logger.info(
        "IMPACT application initialized",
        run_timestamp=timestamp.isoformat(),
        **impact_config.to_dict(),
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

    logger.info("Fetching train/val/test splits...")
    splits, item_index = data_pipeline.get_splits()

    logger.info("Tensorizing splits...")
    tensorized_splits = data_pipeline.tensorize(splits)

    sasrec_training_pipeline = SASRecTrainingPipeline(config=impact_config)
    sasrec_state_dict_path = models_dir / "sasrecnbr.pt"

    if sasrec_state_dict_path.exists():
        msg = (
            "SASRecNBR checkpoint already exists, skipping training. Please note that "
            "results may differ across machines even with identical config."
        )
        logger.warning(msg, path=sasrec_state_dict_path.as_posix())
        return

    logger.info("Training SASRecNBR...")
    _ = sasrec_training_pipeline.train(
        split=tensorized_splits,
        num_items=item_index.num_items,
        checkpoint_path=sasrec_state_dict_path,
        config_path=sasrec_state_dict_path.parent / "config.yaml",
    )

    logger.info("SASRecNBR training completed", path=sasrec_state_dict_path.as_posix())


@cli.command(name="evaluate-impact")
@click.option(
    "--partition",
    type=click.Choice(["val", "test"], case_sensitive=False),
    required=True,
    help="""
        Dataset partition to evaluate on. Use 'val' during hyperparameter search,
        'test' for final evaluation on the held-out data.
    """,
)
@click.option(
    "--log-level",
    type=click.Choice(list(logging.getLevelNamesMapping()), case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging verbosity level.",
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
    default=None,
    help="""
        Base directory for processed data (the final, canonical datasets for modeling).
        Defaults to <project_root>/data/processed.
    """,
)
@click.option(
    "--base-logs-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Base directory for log files. Defaults to <project_root>/logs.",
)
@inject_impact_config_from_checkpoint
@inject_impact_config_overrides("actr", "eval", "system")
def evaluate_impact(
    impact_config: ImpactConfig,
    sasrec_checkpoint_path: Path,
    partition: Literal["val", "test"],
    log_level: str = "INFO",
    raw_data_dir: Path | None = None,
    interim_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    base_logs_dir: Path | None = None,
    exec_time: datetime.datetime | None = None,
) -> None:
    """Run IMPACT late-fusion evaluation on the specified split.

    Loads a trained SASRecNBR checkpoint, builds the ACT-R history index for test users,
    runs batched inference, and computes sample-weighted metrics at all configured
    cutoff levels: global, repetition-focused, and exploration-focused NDCG@K and
    Recall@K.
    """
    timestamp = exec_time or datetime.datetime.now(datetime.UTC)

    raw_data_dir = raw_data_dir or PROJECT_ROOT / "data" / "raw"
    interim_data_dir = interim_data_dir or PROJECT_ROOT / "data" / "interim"
    processed_data_dir = processed_data_dir or PROJECT_ROOT / "data" / "processed"
    base_logs_dir = base_logs_dir or PROJECT_ROOT / "logs"

    # Configure logging
    logs_dir = _get_logs_dir(base_dir=base_logs_dir, timestamp=timestamp)
    log_filename = f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_train-sasrec"
    configure_logging(
        level=logging.getLevelNamesMapping()[log_level],
        log_file=logs_dir / f"{log_filename}.log",
        json_log_file=logs_dir / f"{log_filename}.json",
        details=True,
    )

    logger.info(
        "IMPACT application initialized",
        run_timestamp=timestamp.isoformat(),
        **impact_config.to_dict(),
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

    logger.info("Fetching listen events...")
    listens_lf = data_pipeline.get_listen_events()

    logger.info("Fetching train/val/test splits...")
    splits, item_index = data_pipeline.get_splits()

    logger.info("Tensorizing splits...")
    tensorized_splits = data_pipeline.tensorize(splits)

    evaluation_pipeline = ImpactEvaluationPipeline(
        config=impact_config,
        item_index=item_index,
        listens_lf=listens_lf,
    )
    logger.info("Running IMPACT evaluation...")

    split_partition = cast(TensorizedDataset, getattr(tensorized_splits, partition))
    metrics = evaluation_pipeline.evaluate(split_partition, sasrec_checkpoint_path)
    logger.info("IMPACT evaluation complete", **metrics)


@cli.command(name="sasrec-grid-search")
@click.option(
    "--log-level",
    type=click.Choice(list(logging.getLevelNamesMapping()), case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging verbosity level.",
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
    default=None,
    help="""
        Base directory for processed data (the final, canonical datasets for modeling).
        Defaults to <project_root>/data/processed.
    """,
)
@click.option(
    "--base-logs-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Base directory for log files. Defaults to <project_root>/logs.",
)
@click.option(
    "--models-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="""
        Base directory where the grid search artifacts will be saved:
        the best-performing SASRecNBR model checkpoint, optimal ImpactConfig snapshot,
        and full tuning results in JSON format. Defaults to <project_root>/models.
    """,
)
@inject_impact_config
def grid_search(
    impact_config: ImpactConfig,
    log_level: str = "INFO",
    raw_data_dir: Path | None = None,
    interim_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    base_logs_dir: Path | None = None,
    models_dir: Path | None = None,
    exec_time: datetime.datetime | None = None,
) -> None:
    """Run grid search to find optimal SASRecNBR hyperparameters."""
    timestamp = exec_time or datetime.datetime.now(datetime.UTC)

    raw_data_dir = raw_data_dir or PROJECT_ROOT / "data" / "raw"
    interim_data_dir = interim_data_dir or PROJECT_ROOT / "data" / "interim"
    processed_data_dir = processed_data_dir or PROJECT_ROOT / "data" / "processed"
    base_logs_dir = base_logs_dir or PROJECT_ROOT / "logs"

    out_dir = (
        (models_dir or PROJECT_ROOT / "models")
        / "sasrec-grid-search"
        / timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    )
    out_dir.mkdir(exist_ok=True, parents=True)

    # Configure logging
    logs_dir = _get_logs_dir(base_dir=base_logs_dir, timestamp=timestamp)
    log_filename = f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_sasrec-grid-search"
    configure_logging(
        level=logging.getLevelNamesMapping()[log_level],
        log_file=logs_dir / f"{log_filename}.log",
        json_log_file=logs_dir / f"{log_filename}.json",
        details=True,
    )

    logger.info(
        "IMPACT application initialized",
        run_timestamp=timestamp.isoformat(),
        **impact_config.to_dict(),
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

    logger.info("Fetching train/val/test splits...")
    splits, item_index = data_pipeline.get_splits()

    logger.info("Tensorizing splits...")
    tensorized_splits = data_pipeline.tensorize(splits)

    best_ndcg = 0.0
    best_config: ImpactConfig | None = None
    tuning_results: list[dict[str, Any]] = []

    best_model_path: Path = out_dir / "best_model.pt"
    best_config_path: Path = out_dir / "best_config.yaml"
    current_model_path: Path = out_dir / "current_model.pt"
    current_config_path: Path = out_dir / "current_config.yaml"

    logger.info("Starting hyperparameter tuning (Grid Search)...")
    if impact_config.dataset.size != DatasetSize.SIZE_50M:
        msg = (
            "Grid search on variant larger than Yambda-50M (Yambda-"
            f"{impact_config.dataset.size.value.upper()}). This might take a long time"
        )
        logger.warning(msg)

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
            _ = structlog.contextvars.bind_contextvars(config_id=i, **params)
            logger.info(f"Testing configuration {i}/{len(param_combinations)}")

            sasrec_config = dataclasses.replace(impact_config.sasrec, **params)
            run_config = dataclasses.replace(impact_config, sasrec=sasrec_config)

            sasrec_training_pipeline = SASRecTrainingPipeline(config=run_config)
            val_ndcg = sasrec_training_pipeline.train(
                split=tensorized_splits,
                num_items=item_index.num_items,
                checkpoint_path=current_model_path,
                config_path=current_config_path,
            )

            tuning_results.append({"config_id": i, "val_ndcg": val_ndcg, **params})

            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                best_config = run_config
                # Remove the so-far best model checkpoint
                if best_model_path.exists():
                    best_model_path.unlink()
                    best_config_path.unlink()
                # Make the current model a new best one
                _ = current_model_path.rename(best_model_path)
                _ = current_config_path.rename(best_config_path)
                logger.info(
                    "New best model configuration found and saved",
                    val_ndcg=round(val_ndcg, 4),
                )

            # Remove the current model checkpoint (if it exists)
            current_model_path.unlink(missing_ok=True)
    finally:
        structlog.contextvars.unbind_contextvars("config_id", *param_grid.keys())

    if best_config is None:
        raise RuntimeError("No model found with NDCG > 0")

    results_path = out_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(tuning_results, f, indent=4)

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
