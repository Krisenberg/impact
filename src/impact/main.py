import datetime
from pathlib import Path

import structlog

from impact.logger import configure_logging

logger = structlog.stdlib.get_logger(__name__)


def run(
    base_logs_dir: Path | None = None,
    exec_time: datetime.datetime | None = None,
) -> None:
    """Run the application.

    Args:
        base_logs_dir: Root dir for logs. If None, defaults to project root / 'logs'.
        exec_time: Timestamp of the program execution, injected later to other
            components. Defaults to current UTC timestamp.
    """
    timestamp = exec_time or datetime.datetime.now(datetime.UTC)

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
