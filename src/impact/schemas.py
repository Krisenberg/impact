from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

import polars as pl
import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.stdlib.get_logger(__name__)


@dataclass(frozen=True)
class ParquetDataset:
    """A logical dataset backed by one or more physical Parquet files.

    Provides an abstract interface for both single-file and partitioned datasets.
    All files are treated as one logical table and scanned / collected together.

    Use `from_dir()` to reconstruct from a specific directory, and `file_name()` to
    derive canonical filenames when writing, ensuring naming consistency.
    """

    _PATTERN: ClassVar[str] = "part_*.parquet"

    paths: list[Path]

    @cached_property
    def lf(self) -> pl.LazyFrame:
        """Parquet files lazily scanned into a single LazyFrame."""
        return pl.scan_parquet(self.paths)

    def collect(self) -> pl.DataFrame:
        """Eagerly materialize all Parquet files into a DataFrame.

        Warning: loads the entire dataset into RAM. Use only when the dataset is known
        to fit within available memory.
        """
        logger.warning(
            "Materializing Parquet table into RAM (might exceed available RAM)"
        )
        return self.lf.collect()

    @classmethod
    def from_dir(cls, directory: Path) -> ParquetDataset:
        """Reconstruct a ParquetDataset from a cache directory.

        Discovers all files matching the specified pattern and returns them as a single
        logical dataset (sorted for deterministic ordering).

        Raises:
            FileNotFoundError: If no matching Parquet files exist in the directory.
        """
        paths = sorted(directory.glob(cls._PATTERN))
        if not paths:
            raise FileNotFoundError(f"No parquet files found in {directory}")
        return cls(paths)

    @staticmethod
    def file_name(partition: int | None = None) -> str:
        """Return the filename for a partition.

        Args:
            partition: Partition index, or None for a single unpartitioned file.
        """
        return "part_all.parquet" if partition is None else f"part_{partition}.parquet"


@dataclass(frozen=True)
class SequenceSplit:
    """Container for temporal dataset splits and their physical representations.

    All splits are backed by cached Parquet files and share a common item-to-index
    mapping (1-based) derived from the train set. Items unseen during training map to
    index 0.
    """

    train: ParquetDataset
    val: ParquetDataset
    test: ParquetDataset
    item_to_idx: ParquetDataset  # columns: item_id (UInt32), idx (UInt32, 1-based)
