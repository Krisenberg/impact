"""Module for fetching and managing Yambda datasets via HuggingFace."""

import threading
from pathlib import Path

import polars as pl
import structlog
from huggingface_hub import get_hf_file_metadata, hf_hub_url

from impact.dataset.enums import DatasetSize, DatasetType, InteractionType

logger = structlog.stdlib.get_logger(__name__)

_YAMBDA_REPO_ID: str = "yandex/yambda"


class YambdaFetcher:
    """Wrapper for downloading and loading Yambda dataset directly into Polars."""

    def __init__(
        self,
        cache_dir: Path,
        dataset_type: DatasetType = DatasetType.FLAT,
        dataset_size: DatasetSize = DatasetSize.SIZE_50M,
    ) -> None:
        """Initialize the dataset fetcher with desired dataset parameters.

        Args:
            cache_dir: Local path where downloaded Parquet files will be stored.
            dataset_type: Format of the dataset.
            dataset_size: Size of the dataset variant.
        """
        self.cache_dir: Path = cache_dir.resolve()
        self.dataset_type: DatasetType = dataset_type
        self.dataset_size: DatasetSize = dataset_size

    def get_interactions(self, event_type: InteractionType) -> pl.LazyFrame:
        """Fetch interaction data and return it as a Polars LazyFrame.

        Args:
            event_type: Type of interaction to fetch.

        Returns:
            A Polars LazyFrame containing the requested interactions, pointing to the
            cached local data.
        """
        data_dir = f"{self.dataset_type}/{self.dataset_size}"
        return self._fetch_and_cache(data_dir=data_dir, file_name=event_type)

    def _fetch_and_cache(self, data_dir: str, file_name: str) -> pl.LazyFrame:
        """Fetch the requested data from HuggingFace filesystem and cache it locally."""
        # Guard for the root-level files
        relative_path = (
            f"{data_dir}/{file_name}.parquet" if data_dir else f"{file_name}.parquet"
        )
        local_path = self.cache_dir / relative_path

        if not local_path.exists():
            total_size = self._get_hf_file_size(relative_path)
            logger.info(
                "Data not found in cache; downloading from HF and caching",
                dataset_file=relative_path,
                local_cache=local_path.as_posix(),
                total_size=_format_bytes(total_size),
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            hf_url = f"hf://datasets/{_YAMBDA_REPO_ID}/{relative_path}"

            stop_event = threading.Event()
            monitor = threading.Thread(
                target=self._log_download_progress,
                args=(local_path, total_size, stop_event),
                daemon=True,
            )
            monitor.start()
            try:
                pl.scan_parquet(hf_url).sink_parquet(local_path)
            finally:
                stop_event.set()
                monitor.join()

            logger.info("Download complete", local_cache=local_path.as_posix())
        else:
            logger.debug("Data found in local cache", local_cache=local_path.as_posix())

        return pl.scan_parquet(local_path)

    @staticmethod
    def _get_hf_file_size(relative_path: str) -> int:
        """Return the file size in bytes from HuggingFace Hub metadata."""
        url = hf_hub_url(
            repo_id=_YAMBDA_REPO_ID,
            filename=relative_path,
            repo_type="dataset",
        )
        metadata = get_hf_file_metadata(url)
        return metadata.size or 0

    @staticmethod
    def _log_download_progress(
        local_path: Path,
        total_size: int,
        stop_event: threading.Event,
        interval: int = 10,
    ) -> None:
        """Log download progress every `interval` seconds until `stop_event` is set."""
        while not stop_event.wait(timeout=interval):
            if local_path.exists() and total_size > 0:
                downloaded = local_path.stat().st_size
                pct = downloaded / total_size * 100
                logger.info(
                    "Downloading",
                    progress=f"{pct:.1f}%",
                    downloaded_size=_format_bytes(downloaded),
                    total_size=_format_bytes(total_size),
                )


def _format_bytes(num: float, suffix: str = "B", precision: int = 1) -> str:
    """Format a number of bytes into a human-readable string."""
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.{precision}f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.{precision}f}Yi{suffix}"
