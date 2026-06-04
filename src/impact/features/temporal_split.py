"""Module for splitting sequences of sessions into train, validation, and test sets.

Sequences of sessions (baskets in the NBR paradigm) are split using the GTS (Global
Temporal Split) strategy.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import polars as pl
import structlog

from impact.schemas import ItemIndex, ParquetTableSet, TemporalSplit
from impact.utils import estimate_processing_partitions

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = structlog.stdlib.get_logger(__name__)

MINUTE_SECONDS: int = 60
DAY_SECONDS: int = 24 * 60 * 60


def split(
    sequences_lf: pl.LazyFrame,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    item_to_idx_dir: Path,
    test_days: float = 1.0,
    val_days: float = 1.0,
    gap_minutes: float = 30.0,
    compute_ram_fraction: float = 0.15,
) -> tuple[TemporalSplit, ItemIndex]:
    """Compute temporal train/val/test splits and write them to Parquet files.

    Applies a Global Temporal Split (GTS) with dual gaps between splits to mimic
    production deployment latency. A 1-based item-to-index mapping is derived from
    train-set items and applied uniformly across all splits, ensuring a shared
    evaluation ID space. Items unseen in training map to index 0. Val and test sets are
    restricted to users present in the train set.

    All splits are written directly to disk partition-by-partition. This is intentional:
    the item remapping step (list.eval + replace_strict on nested structs) cannot run
    in Polars streaming mode, making sequential collect-and-write the only RAM-safe
    execution strategy. Partition counts are estimated automatically from data size
    and available RAM.

    Args:
        sequences_lf: LazyFrame of user-scoped sequences as produced by
            `sequences.extract_sequences()`.
        train_dir: Output directory for train split Parquet files.
        val_dir: Output directory for validation split Parquet files.
        test_dir: Output directory for test split Parquet files.
        item_to_idx_dir: Output directory for the item-to-index mapping Parquet files.
        test_days: Number of days at the end of the timeline allocated to the test set.
        val_days: Number of days preceding the test set allocated to validation.
        gap_minutes: Gap in minutes between splits that mimics production system
            deployment latency.
        compute_ram_fraction: Fraction of available RAM allowed per processing
            partition. It should be keept low (<=0.15) to account for memory spikes
            during struct operations.

    Returns:
        SequenceSplit pointing to the written Parquet files for all splits and mapping.

    Raises:
        ValueError: If target_ts column is empty or the train split contains no items.
    """
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

    # Ensure target directories exist
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    item_to_idx_dir.mkdir(parents=True, exist_ok=True)

    # Train split
    logger.info("Filtering sequences for the train set")
    train_lf = sequences_lf.filter(pl.col("target_ts") < train_end_ts)

    logger.info("Creating item_id -> index mapping from the train set")
    item_to_idx_df = _create_item_to_idx_mapping(
        train_lf,
        n_partitions=estimate_processing_partitions(train_lf, compute_ram_fraction),
        partition_over="uid",
    )

    if item_to_idx_df.is_empty():
        raise ValueError("Train split produced no items.")

    logger.info(f"Saving item_id -> index mapping to {item_to_idx_dir}")
    _write_partitioned(
        data=item_to_idx_df,
        out_dir=item_to_idx_dir,
        n_partitions=estimate_processing_partitions(
            item_to_idx_df, compute_ram_fraction
        ),
        partition_over="idx",
    )

    mapper = partial(_map_item_ids, item_to_idx_df=item_to_idx_df)

    logger.info(
        f"Applying item-to-index mapping to the train set and saving it to {train_dir}"
    )
    _write_partitioned(
        data=train_lf,
        out_dir=train_dir,
        n_partitions=estimate_processing_partitions(train_lf, compute_ram_fraction),
        partition_over="uid",
        transform=mapper,
    )

    # Identify valid users from the training set (not cold-start users)
    # Materialize them for optimized joining
    unique_train_uids = train_lf.select("uid").unique().collect()

    # Validation split
    logger.info("Filtering sequences for the val set")
    val_lf = (
        sequences_lf.filter(
            (pl.col("target_ts") >= val_start_ts) & (pl.col("target_ts") < val_end_ts)
        )
        # Prevent cold-start by keeping only users present in the train set
        .join(unique_train_uids.lazy(), on="uid", how="inner")
    )

    logger.info(
        f"Applying item-to-index mapping to the val set and saving it to {val_dir}"
    )
    _write_partitioned(
        data=val_lf,
        out_dir=val_dir,
        n_partitions=estimate_processing_partitions(val_lf, compute_ram_fraction),
        partition_over="uid",
        transform=mapper,
    )

    # Test split
    logger.info("Filtering sequences for the test set")
    test_lf = (
        sequences_lf.filter(pl.col("target_ts") >= test_start_ts)
        # Prevent cold-start by keeping only users present in the train set
        .join(unique_train_uids.lazy(), on="uid", how="inner")
    )

    logger.info(
        f"Applying item-to-index mapping to the test set and saving it to {test_dir}"
    )
    _write_partitioned(
        data=test_lf,
        out_dir=test_dir,
        n_partitions=estimate_processing_partitions(test_lf, compute_ram_fraction),
        partition_over="uid",
        transform=mapper,
    )

    split = TemporalSplit(
        train=ParquetTableSet.from_dir(train_dir),
        val=ParquetTableSet.from_dir(val_dir),
        test=ParquetTableSet.from_dir(test_dir),
    )
    item_index = ItemIndex(table=ParquetTableSet.from_dir(item_to_idx_dir))

    return split, item_index


def _map_item_ids(
    lf: pl.LazyFrame,
    item_to_idx_df: pl.DataFrame,
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
        item_to_idx_df: Item-to-index mapping, with item_id (UInt32) and idx (UInt32,
            1-based) columns, as created by `_create_item_to_idx_mapping()`.

    Returns:
        LazyFrame with remapped item identifiers.
    """
    item_ids = item_to_idx_df.get_column("item_id")
    mapped_ids = item_to_idx_df.get_column("idx")
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


def _create_item_to_idx_mapping(
    train_lf: pl.LazyFrame,
    n_partitions: int = 1,
    partition_over: str | None = None,
) -> pl.DataFrame:
    """Build a 1-based item_id -> idx mapping from train-set sequences.

    Collects unique item IDs from target.items and context[*].items, sorts them, and
    assigns contiguous indices starting at 1 (0 reserved for padding/cold-start).
    For large datasets, use partitioning to avoid loading all items into RAM at once.

    Args:
        train_lf: Input sequence LazyFrame (should be train set only).
        n_partitions: Number of partitions for out-of-core processing.
        partition_over: Numeric column to partition by (modulo-based). Required if
            n_partitions > 1.

    Returns:
        DataFrame with columns item_id (UInt32) and idx (UInt32, 1-based).

    Raises:
        ValueError: If n_partitions > 1 and partition_over is not provided.
        ValueError: If n_partitions > 1 and partition_over is not found in the schema.
        ValueError: If n_partitions > 1 and partition_over column is not numeric.
    """
    if n_partitions > 1:
        if partition_over is None:
            raise ValueError(
                "Cannot partition data if `partition_over` is not provided."
            )
        if partition_over not in train_lf.lazy().collect_schema():
            raise ValueError(
                f"Partitioning column {partition_over} not found in dataset schema."
            )
        dtype = cast(pl.DataType, train_lf.lazy().collect_schema().get(partition_over))
        if not dtype.is_numeric():
            raise ValueError(f"Partitioning column {partition_over} must be numeric.")

    def _get_unique_items(lf: pl.LazyFrame) -> pl.DataFrame:
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

    if n_partitions == 1:
        logger.info("Creating item_id -> index mapping without partitioning")
        item_to_idx_df = _get_unique_items(train_lf)
    else:
        logger.info("Starting partitioned processing")
        lf = train_lf.with_columns(
            (pl.col(cast(str, partition_over)) % n_partitions).alias("partition")
        )
        unique_items_per_part: list[pl.DataFrame] = []

        for i in range(n_partitions):
            logger.info(f"Processing partition {i + 1}/{n_partitions}")
            part_lf = lf.filter(pl.col("partition") == i).drop("partition")
            unique_items_per_part.append(_get_unique_items(part_lf))

        logger.info("Merging partitions and creating item_id -> index mapping")
        item_to_idx_df = pl.concat(unique_items_per_part).unique()

    return item_to_idx_df.sort("item_id").with_columns(
        pl.col("item_id").cast(pl.UInt32),
        pl.int_range(1, pl.len() + 1).alias("idx").cast(pl.UInt32),
    )


def _write_partitioned[T: pl.LazyFrame | pl.DataFrame](
    data: T,
    out_dir: Path,
    n_partitions: int = 1,
    partition_over: str | None = None,
    transform: Callable[[T], T] | None = None,
) -> None:
    """Write a dataset to partitioned Parquet files.

    Partitions are written sequentially, one at a time, so peak RAM equals one
    partition's size. For LazyFrames with non-streamable operations (e.g. list.eval),
    partitioning must happen before the `transform()` is applied. This function
    guarantees the filter-then-transform order.

    Args:
        data: Dataset to write. LazyFrames use `collect().write_parquet()` per partition
            to avoid re-executing the full query plan.
        out_dir: Output directory. Files are named via `ParquetDataset.file_name()`.
        n_partitions: Number of output files. 1 writes a single, unpartitioned file.
        partition_over: Numeric column to partition by (modulo-based). Required if
            n_partitions > 1.
        transform: Optional transformation applied per-partition after filtering,
            before writing.

    Raises:
        ValueError: If n_partitions > 1 and partition_over is not provided.
        ValueError: If n_partitions > 1 and partition_over is not found in the schema.
        ValueError: If n_partitions > 1 and partition_over column is not numeric.
    """
    if n_partitions > 1:
        if partition_over is None:
            raise ValueError(
                "Cannot partition data if `partition_over` is not provided."
            )
        if partition_over not in data.lazy().collect_schema():
            raise ValueError(
                f"Partitioning column {partition_over} not found in dataset schema."
            )
        dtype = cast(pl.DataType, data.lazy().collect_schema().get(partition_over))
        if not dtype.is_numeric():
            raise ValueError(f"Partitioning column {partition_over} must be numeric.")

    if n_partitions == 1:
        if transform is not None:
            data = transform(data)
        logger.info("Writing single file (unpartitioned)")
        if isinstance(data, pl.DataFrame):
            data.write_parquet(out_dir / ParquetTableSet.file_name(partition=None))
        else:
            data.sink_parquet(out_dir / ParquetTableSet.file_name(partition=None))
        return

    logger.info("Starting partitioned processing")
    data = cast(T, data.with_columns(
        (pl.col(cast(str, partition_over)) % n_partitions).alias("partition")
    ))  # fmt: skip
    for i in range(n_partitions):
        logger.info(f"Processing partition {i + 1}/{n_partitions}")
        part = cast(T, data.filter(pl.col("partition") == i).drop("partition"))
        if transform is not None:
            part = transform(part)

        partition_path = out_dir / ParquetTableSet.file_name(partition=i)
        if isinstance(part, pl.LazyFrame):
            part.collect().write_parquet(partition_path)
        else:
            part.write_parquet(partition_path)
