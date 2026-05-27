from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
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


@dataclass(frozen=True)
class MemmapDataset:
    """Tensorized dataset backed by memory-mapped NumPy arrays.

    Arrays are produced by `features.tensorize()` and intended for out-of-core PyTorch
    training via `np.load(..., mmap_mode="r")`.

    Use `from_dir()` to reconstruct from a directory containing the tensorized files,
    ensuring filename consistency with the constants defined on this class.
    """

    _CONTEXT_FILE: ClassVar[str] = "context.npy"
    _TARGET_FILE: ClassVar[str] = "target.npy"
    _META_FILE: ClassVar[str] = "meta.npy"

    directory: Path

    @property
    def meta_file_path(self) -> Path:
        """Path to meta.npy file."""
        return self.directory / self._META_FILE

    @property
    def context_file_path(self) -> Path:
        """Path to context.npy file."""
        return self.directory / self._CONTEXT_FILE

    @property
    def target_file_path(self) -> Path:
        """Path to target.npy file."""
        return self.directory / self._TARGET_FILE

    @cached_property
    def meta(self) -> tuple[int, int, int]:
        """Tuple of (N, L, K) from meta.npy file."""
        n, l, k = cast(tuple[int, int, int], np.load(self.meta_file_path))  # noqa: E741
        return int(n), int(l), int(k)

    @cached_property
    def context(self) -> np.ndarray:
        """Memory-mapped context array of shape [N, L, K]."""
        return cast(np.ndarray, np.load(self.context_file_path, mmap_mode="r"))

    @cached_property
    def target(self) -> np.ndarray:
        """Memory-mapped target array of shape [N, K]."""
        return cast(np.ndarray, np.load(self.target_file_path, mmap_mode="r"))

    @classmethod
    def from_dir(cls, directory: Path) -> MemmapDataset:
        """Reconstruct a MemmapDataset from a directory containing tensorized files.

        Raises:
            FileNotFoundError: If any of the expected array files are missing.
        """
        expected = {cls._CONTEXT_FILE, cls._TARGET_FILE, cls._META_FILE}
        missing = expected - {f.name for f in directory.iterdir()}
        if missing:
            raise FileNotFoundError(f"Missing files in {directory}: {missing}")
        return cls(directory)


@dataclass(frozen=True)
class MemmapSplit:
    """Container for tensorized train/val/test splits."""

    train: MemmapDataset
    val: MemmapDataset
    test: MemmapDataset
