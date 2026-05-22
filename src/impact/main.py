import datetime
from pathlib import Path

import polars as pl
import structlog

from impact.configuration import load_config
from impact.dataset import DatasetSize
from impact.logger import configure_logging
from impact.pipeline import DataPipeline, PipelineConfig

logger = structlog.stdlib.get_logger(__name__)


def run(
    raw_data_dir: Path | None = None,
    interim_data_dir: Path | None = None,
    processed_data_dir: Path | None = None,
    base_logs_dir: Path | None = None,
    exec_time: datetime.datetime | None = None,
) -> None:
    """Run the application.

    Args:
        raw_data_dir: Root dir for raw data. If None, defaults to
            project root / 'data' / 'raw'.
        interim_data_dir: Root dir for interim data. If None, defaults to
            project root / 'data' / 'interim'.
        processed_data_dir: Root dir for interim data. If None, defaults to
            project root / 'data' / 'processed'.
        base_logs_dir: Root dir for logs. If None, defaults to project root / 'logs'.
        exec_time: Timestamp of the program execution, injected later to other
            components. Defaults to current UTC timestamp.
    """
    timestamp = exec_time or datetime.datetime.now(datetime.UTC)

    if raw_data_dir is None:
        raw_data_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    if interim_data_dir is None:
        interim_data_dir = Path(__file__).resolve().parents[2] / "data" / "interim"
    if processed_data_dir is None:
        processed_data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    if base_logs_dir is None:
        base_logs_dir = Path(__file__).resolve().parents[2] / "logs"

    logs_dir = _get_logs_dir(base_dir=base_logs_dir, timestamp=timestamp)
    log_filename = f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_impact"

    configure_logging(
        log_file=logs_dir / f"{log_filename}.log",
        json_log_file=logs_dir / f"{log_filename}.json",
        details=True,
    )

    logger.info("IMPACT application initialized", run_timestamp=timestamp.isoformat())

    _ = load_config()

    pipeline = DataPipeline(
        raw_dir=raw_data_dir,
        interim_dir=interim_data_dir,
        processed_dir=processed_data_dir,
        config=PipelineConfig(dataset_size=DatasetSize.SIZE_50M),
    )
    events_lf = pipeline.get_raw_events()
    print(f"50M events_lf: {events_lf.select(pl.len()).collect().item()}")

    sessions_lf = pipeline.get_sessions()
    print(f"50M sessions_lf: {sessions_lf.select(pl.len()).collect().item()}")

    sequences_lf = pipeline.get_sequences()
    print(f"50M sequences_lf: {sequences_lf.select(pl.len()).collect().item()}")

    train, val, test = pipeline.get_splits()
    print(f"50M train: {train.select(pl.len()).collect().item()}")
    print(f"50M val: {val.select(pl.len()).collect().item()}")
    print(f"50M test: {test.select(pl.len()).collect().item()}")

    pipeline.config = PipelineConfig(
        dataset_size=DatasetSize.SIZE_5B,
        step=20,
    )
    events_lf = pipeline.get_raw_events()
    print(f"5B events_lf: {events_lf.select(pl.len()).collect().item()}")

    sessions_lf = pipeline.get_sessions(n_partitions=32)
    print(f"5B sessions_lf: {sessions_lf.select(pl.len()).collect().item()}")

    sequences_lf = pipeline.get_sequences(n_partitions=32)
    print(f"5B sequences_lf: {sequences_lf.select(pl.len()).collect().item()}")

    train, val, test = pipeline.get_splits()
    print(f"5B train: {train.select(pl.len()).collect().item()}")
    print(f"5B val: {val.select(pl.len()).collect().item()}")
    print(f"5B test: {test.select(pl.len()).collect().item()}")

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
    run()
