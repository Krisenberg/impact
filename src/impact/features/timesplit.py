"""Module for splitting sequences of sessions into train, validation, and test sets.

Sequences of sessions (baskets in the NBR paradigm) are split using the GTS (Global
Temporal Split) strategy.
"""

import shutil
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import cast

import polars as pl
import structlog

from impact.constants import DAY_SECONDS, MINUTE_SECONDS

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

    @property
    def df(self) -> pl.DataFrame:
        """Parquet files eagerly materialized into a DataFrame."""
        logger.warning(
            "Materializing Parquet table into RAM (might exceed available RAM)"
        )
        return self.lf.collect()


@dataclass(frozen=True)
class SequenceSplit:
    """Container for temporal dataset splits and their physical representations.

    All splits and mappings are backed by deterministic, cached Parquet files.

    Train, val and test splits share a common item-to-index mapping (1-based) derived
    from the train set, available as item_to_idx. Items unseen during training map to
    index 0.
    """

    train: ParquetTable
    val: ParquetTable
    test: ParquetTable
    item_to_idx: ParquetTable  # columns: item_id (UInt32), idx (UInt32, 1-based)


def split_sequences_train_val_test(
    sequences_lf: pl.LazyFrame,
    cache_dir: Path,
    test_days: float = 1.0,
    val_days: float = 1.0,
    gap_minutes: float = 30.0,
    n_partitions: int = 1,
) -> SequenceSplit:
    """Split sequences into train, val, and test sets using temporal boundaries (GTS).

    The split is determined strictly by the target_ts of the sequence, which represents
    the timestamp of the target session in the sequence. It follows dual-gap strategy
    from Yambda's benchmarking to mimic production deployment latency.

    A contiguous item-to-index mapping (1-based) is derived from train-set items only
    and applied uniformly to all splits, ensuring a shared evaluation ID space. Items
    appearing in val/test but absent in train are mapped to index 0.

    Val and test sets are further filtered only to users present in the train set to
    exclude cold-start users.

    Args:
        sequences_lf: LazyFrame representing user-scoped sequences of sessions, as
            produced by `sequences.extract_sequences()`.
        cache_dir: Directory used to store cached Parquet files for train, val, test
            splits and item mappings. Required because all returned datasets are
            file-backed (`ParquetTable`). When `n_partitions > 1`, it is additionally
            used for out-of-core partitioned processing.
        test_days: Number of days at the end of the timeline allocated for the test set.
        val_days: Number of days preceding test set allocated for the validation set.
        gap_minutes: Buffer time (in minutes) between dataset splits that mimics
            production system deployment latency.
        n_partitions: Number of partitions for sequential processing. Use >1 for
            large datasets that exceed available RAM (data is partitioned by user,
            i.e. uid column).

    Returns:
        SequenceSplit containing file-backed train, val and test splits together with
        the shared item-to-index mapping. Each split is represented as a `ParquetTable`
        backed by one or more cached Parquet files.
    """
    if n_partitions < 1:
        raise ValueError("Number of partitions must be greater than 0.")

    logger.info(
        "Executing Global Temporal Split (GTS) to create train, val, and test sets",
        val_days=val_days,
        test_days=test_days,
        gap_minutes=gap_minutes,
    )

    # Convert time parameters to seconds
    val_size_seconds = int(val_days * DAY_SECONDS)
    test_size_seconds = int(test_days * DAY_SECONDS)
    gap_size_seconds = int(gap_minutes * MINUTE_SECONDS)

    # Get max timestamp
    max_ts = cast(
        int | None, sequences_lf.select(pl.col("target_ts").max()).collect().item()
    )
    if max_ts is None:
        raise ValueError("Timestamp column is empty or contains only nulls.")

    # Get min timestamp
    min_ts = cast(int, sequences_lf.select(pl.col("target_ts").min()).collect().item())
    total_days = (max_ts - min_ts) / (DAY_SECONDS)  # Convert seconds to days

    # Calculate temporal boundaries for the GTS
    test_start_ts = max_ts - test_size_seconds + 1
    val_end_ts = test_start_ts - gap_size_seconds
    val_start_ts = val_end_ts - val_size_seconds
    train_end_ts = val_start_ts - gap_size_seconds

    logger.info(
        "GTS Boundaries calculated",
        total_days=total_days,
        min_ts=min_ts,
        max_ts=max_ts,
        train_end=train_end_ts,
        val_start=val_start_ts,
        val_end=val_end_ts,
        test_start=test_start_ts,
    )

    # Train split
    logger.info("Filtering sequences for the train set")
    raw_train_lf = sequences_lf.filter(pl.col("target_ts") < train_end_ts)

    logger.info("Generating item_id -> index mapping from the train set")
    item_to_idx_df: pl.DataFrame
    if n_partitions == 1:
        item_to_idx_df = _get_unique_items(raw_train_lf).sort("item_id")
    else:
        logger.info("Starting partitioned processing")
        unique_items_per_part: list[pl.DataFrame] = []

        for i in range(n_partitions):
            logger.info(f"Processing partition {i + 1}/{n_partitions}")
            part_lf = raw_train_lf.filter(pl.col("uid") % n_partitions == i)
            unique_items_per_part.append(_get_unique_items(part_lf))

        logger.info("Merging partitions and generating item_id -> index mapping")
        item_to_idx_df = pl.concat(unique_items_per_part).unique().sort("item_id")

    if item_to_idx_df.is_empty():
        raise ValueError("Train split produced no items.")

    old = item_to_idx_df.get_column("item_id").cast(pl.UInt32)
    new = item_to_idx_df.get_column("idx").cast(pl.UInt32)

    item_to_idx_cache_dir = _prepare_cache_dir(cache_dir / "item_to_idx")
    item_to_idx_df.write_parquet(item_to_idx_cache_dir / "all.parquet")
    item_to_idx_table = ParquetTable(sorted(item_to_idx_cache_dir.glob("all.parquet")))

    logger.info("Applying item_id -> index mapping to the train set")
    train_cache_dir = _prepare_cache_dir(cache_dir / "train")
    if n_partitions == 1:
        _map_item_ids(raw_train_lf, old, new).sink_parquet(
            train_cache_dir / "all.parquet"
        )
        train_table = ParquetTable(sorted(train_cache_dir.glob("all.parquet")))
    else:
        logger.info("Starting partitioned processing")
        for i in range(n_partitions):
            logger.info(f"Processing partition {i + 1}/{n_partitions}")
            (
                raw_train_lf.filter(pl.col("uid") % n_partitions == i)
                .pipe(_map_item_ids, old, new)
                .sink_parquet(train_cache_dir / f"part_{i}.parquet")
            )

        logger.info("Merging partitions (lazily!)")
        train_table = ParquetTable(sorted(train_cache_dir.glob("part_*.parquet")))

    # Identify valid users from the training set (not cold-start users)
    # Materialize them for optimized joining
    unique_train_uids = train_table.lf.select("uid").unique().collect()

    # Validation split
    logger.info("Creating validation set")
    val_cache_dir = _prepare_cache_dir(cache_dir / "val")
    (
        sequences_lf.filter(
            (pl.col("target_ts") >= val_start_ts) & (pl.col("target_ts") < val_end_ts)
        )
        # Prevent cold-start by keeping only users present in the train set
        .join(unique_train_uids.lazy(), on="uid", how="inner")
        .pipe(_map_item_ids, old, new)
        .sink_parquet(val_cache_dir / "all.parquet")
    )
    val_table = ParquetTable(sorted(val_cache_dir.glob("all.parquet")))

    # Test split
    logger.info("Creating test set")
    test_cache_dir = _prepare_cache_dir(cache_dir / "test")
    (
        sequences_lf.filter(pl.col("target_ts") >= test_start_ts)
        # Prevent cold-start by keeping only users present in the train set
        .join(unique_train_uids.lazy(), on="uid", how="inner")
        .pipe(_map_item_ids, old, new)
        .sink_parquet(test_cache_dir / "all.parquet")
    )
    test_table = ParquetTable(sorted(test_cache_dir.glob("all.parquet")))

    return SequenceSplit(train_table, val_table, test_table, item_to_idx_table)


def _prepare_cache_dir(path: Path) -> Path:
    """Recreate a cache directory.

    Existing directory contents are removed recursively before creating a new empty one.
    """
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)
    return path


def _map_item_ids(
    lf: pl.LazyFrame,
    item_ids: pl.Series,
    mapped_ids: pl.Series,
) -> pl.LazyFrame:
    """Map original item IDs to contiguous indices.

    Replaces item identifiers inside both `context` and `target` structures using the
    provided item-to-index mapping.

    Expected schema:
    - context: List(Struct({..., items: List(UInt32)}))
    - target: Struct({..., items: List(UInt32)})

    Items not present in the mapping are replaced with index 0.

    Args:
        lf: Input sequence dataset.
        item_ids: Original item IDs.
        mapped_ids: Corresponding contiguous indices.

    Returns:
        LazyFrame with remapped item identifiers.
    """
    default = pl.lit(0, dtype=pl.UInt32)
    return lf.with_columns([
        pl.col("context").list.eval(
            pl.element().struct.with_fields(
                pl.field("items").list.eval(
                    pl.element().replace_strict(item_ids, mapped_ids, default=default)
                )
            )
        ),
        pl.col("target").struct.with_fields(
            pl.field("items").list.eval(
                pl.element().replace_strict(item_ids, mapped_ids, default=default)
            )
        ),
    ])  # fmt: skip


def _get_unique_items(lf: pl.LazyFrame) -> pl.DataFrame:
    """Extract unique item IDs from nested sequence structures.

    Reads item IDs from:
    - target.items
    - context[*].items
    and returns unique values as `item_id`.

    Args:
        lf: Input sequence dataset.

    Returns:
        DataFrame containing unique item IDs in column `item_id`.
    """
    return (
        pl.concat([
            lf.select(
                pl.col("target")
                .struct.field("items")
                .explode()
                .alias("item_id")
            ),
            lf.select(
                pl.col("context")
                .explode()
                .struct.field("items")
                .explode()
                .alias("item_id")
            ),
        ])
        .drop_nulls()
        .unique()
        .collect()
    )  # fmt: skip
