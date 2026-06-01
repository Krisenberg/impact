"""Module for fetching Yambda dataset files from HF Hub into Polars LazyFrames."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, final

import polars as pl
import structlog
from huggingface_hub import hf_hub_download

from impact.dataset.enums import DatasetSize, DatasetType, InteractionType

logger = structlog.stdlib.get_logger(__name__)


@final
class YambdaFetcher:
    """Fetches and caches Yambda dataset files from HuggingFace Hub.

    Downloads Parquet files on the first access and reuses the local cache on
    subsequent calls. All methods return LazyFrames pointing to cached local files.
    """

    _HF_REPO_ID: ClassVar[str] = "yandex/yambda"

    def __init__(
        self,
        cache_dir: Path,
        dataset_type: DatasetType = DatasetType.FLAT,
        dataset_size: DatasetSize = DatasetSize.SIZE_50M,
    ) -> None:
        """Initialize the dataset fetcher with desired dataset parameters.

        Args:
            cache_dir: Local directory where downloaded Parquet files will be cached.
            dataset_type: Format of the dataset variant.
            dataset_size: Size of the dataset variant to download.
        """
        self.cache_dir = cache_dir.resolve()
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size

    def get_interactions(self, event_type: InteractionType) -> pl.LazyFrame:
        """Fetch interaction data for the configured dataset variant.

        Args:
            event_type: Type of interaction to fetch (e.g. InteractionType.LISTENS).

        Returns:
            LazyFrame containing the requested interactions over the cached local
            Parquet file.
        """
        data_dir = f"{self.dataset_type}/{self.dataset_size}"
        return self._fetch_and_cache(data_dir=data_dir, file_name=event_type)

    def get_audio_embeddings(self) -> pl.LazyFrame:
        """Fetch pre-computed CNN audio embeddings mapping item IDs to embeddings."""
        return self._fetch_and_cache(data_dir="", file_name="embeddings")

    def get_album_item_mapping(self) -> pl.LazyFrame:
        """Fetch the album-to-item mapping table."""
        return self._fetch_and_cache(data_dir="", file_name="album_item_mapping")

    def get_artist_item_mapping(self) -> pl.LazyFrame:
        """Fetch the artist-to-item mapping table."""
        return self._fetch_and_cache(data_dir="", file_name="artist_item_mapping")

    def _fetch_and_cache(self, data_dir: str, file_name: str) -> pl.LazyFrame:
        """Download a Parquet file from HuggingFace, or reuse the local cache.

        Returns:
            LazyFrame pointing to the resolved local file path.
        """
        # Guard for the root-level files
        relative_path = (
            f"{data_dir}/{file_name}.parquet" if data_dir else f"{file_name}.parquet"
        )
        logger.info(
            "Downloading data from HuggingFace (if not cached)",
            repo_id=self._HF_REPO_ID,
            dataset_file=relative_path,
        )
        local_path_str = hf_hub_download(
            repo_id=self._HF_REPO_ID,
            repo_type="dataset",
            filename=relative_path,
            local_dir=self.cache_dir,
        )
        local_path = Path(local_path_str).resolve()
        logger.info("Data available in local cache", local_cache=local_path.as_posix())
        return pl.scan_parquet(local_path)
