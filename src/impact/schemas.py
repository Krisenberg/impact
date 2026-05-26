from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import polars as pl
import structlog

logger = structlog.stdlib.get_logger(__name__)


@dataclass(frozen=True)
class ParquetTable:
    """Logical table backed by one or more Parquet files.

    Supports both:
    - single-file datasets
    - partitioned datasets used for out-of-core processing

    All the files specified in `paths` are treated as one logical table.
    """

    paths: list[Path]

    @cached_property
    def lf(self) -> pl.LazyFrame:
        """Parquet files lazily scanned into a LazyFrame."""
        return pl.scan_parquet(self.paths)

    def collect(self) -> pl.DataFrame:
        """Materialize Parquet files eagerly into a DataFrame."""
        logger.warning(
            "Materializing Parquet table into RAM (might exceed available RAM)"
        )
        return self.lf.collect()


@dataclass(frozen=True)
class SequenceSplit:
    """Container for temporal dataset splits and their physical representations.

    All splits and mappings are backed by cached Parquet files.

    Train, val and test splits share a common item-to-index mapping (1-based) derived
    from the train set, available as item_to_idx. Items unseen during training map to
    index 0.
    """

    train: ParquetTable
    val: ParquetTable
    test: ParquetTable
    item_to_idx: ParquetTable  # columns: item_id (UInt32), idx (UInt32, 1-based)
