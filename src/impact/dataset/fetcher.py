"""Module for fetching and managing Yambda datasets via HuggingFace."""

from pathlib import Path

import polars as pl
import structlog

from impact.dataset.enums import DatasetSize, DatasetType, InteractionType

logger = structlog.stdlib.get_logger(__name__)


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
            logger.info(
                "Data not found in cache; downloading from HF and caching",
                dataset_file=relative_path,
                local_cache=local_path.as_posix(),
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            hf_url = f"hf://datasets/yandex/yambda/{relative_path}"

            # Scan from cloud and stream directly to local disk
            pl.scan_parquet(hf_url).sink_parquet(local_path)
            logger.info("Data downloaded and saved in the local cache")
        else:
            logger.debug(
                "Data found in the local cache",
                local_cache=local_path.as_posix(),
            )

        return pl.scan_parquet(local_path)
