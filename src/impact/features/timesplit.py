"""Module for splitting sequences of sessions into train, validation, and test sets.

Sequences of sessions (baskets in the NBR paradigm) are split using the GTS (Global
Temporal Split) strategy.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast

import polars as pl
import structlog

from impact.constants import DAY_SECONDS, MINUTE_SECONDS
from impact.schemas import ParquetDataset, SequenceSplit
from impact.utils import estimate_processing_partitions

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = structlog.stdlib.get_logger(__name__)


def split_sequences_train_val_test(
    sequences_lf: pl.LazyFrame,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    item_to_idx_dir: Path,
    test_days: float = 1.0,
    val_days: float = 1.0,
    gap_minutes: float = 30.0,
    compute_ram_fraction: float = 0.15,  # for estimate_processing_partitions
) -> SequenceSplit:
    """Compute temporal train/validation/test splits with a global item-index mapping.

    Splits are based on timestamp thresholds with a gap between train, validation, and
    test windows. The item-to-index mapping is derived from the training set and applied
    consistently across all splits, ensuring a shared evaluation ID space. Items
    appearing in val/test but absent in train are mapped to index 0.

    Val and test sets are further filtered only to users present in the train set to
    exclude cold-start users.

    Returns:
        SequenceSplitResult that stores train, validation and test LazyFrames together
        with the item-to-index mapping derived from the training set.
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

    logger.info("Generating item_id -> index mapping from the train set")
    item_to_idx_df = _create_item_to_idx_mapping(
        train_lf,
        n_partitions=estimate_processing_partitions(train_lf, compute_ram_fraction),
        partition_over="uid",
    )

    if item_to_idx_df.is_empty():
        raise ValueError("Train split produced no items.")

    item_ids = item_to_idx_df.get_column("item_id").cast(pl.UInt32)
    mapped_ids = item_to_idx_df.get_column("idx").cast(pl.UInt32)

    logger.info(f"Save item_id -> index mapping to {item_to_idx_dir}")
    _write_partitioned(
        data=item_to_idx_df,
        out_dir=item_to_idx_dir,
        n_partitions=estimate_processing_partitions(item_to_idx_df),
        partition_over="idx",
    )

    mapper = partial(_map_item_ids, item_ids=item_ids, mapped_ids=mapped_ids)
    logger.info(
        f"Apply item_id -> index mapping to the train set and save it to {train_dir}"
    )
    _write_partitioned(
        data=train_lf,
        out_dir=train_dir,
        n_partitions=estimate_processing_partitions(train_lf),
        partition_over="uid",
        transform=mapper,
    )

    # Identify valid users from the training set (not cold-start users)
    # Materialize them for optimized joining
    unique_train_uids = train_lf.select("uid").unique().collect()

    # Validation split
    logger.info("Creating validation set")
    val_lf = (
        sequences_lf.filter(
            (pl.col("target_ts") >= val_start_ts) & (pl.col("target_ts") < val_end_ts)
        )
        # Prevent cold-start by keeping only users present in the train set
        .join(unique_train_uids.lazy(), on="uid", how="inner")
    )
    _write_partitioned(
        data=val_lf,
        out_dir=val_dir,
        n_partitions=estimate_processing_partitions(val_lf),
        partition_over="uid",
        transform=mapper,
    )

    # Test split
    logger.info("Creating test set")
    test_lf = (
        sequences_lf.filter(pl.col("target_ts") >= test_start_ts)
        # Prevent cold-start by keeping only users present in the train set
        .join(unique_train_uids.lazy(), on="uid", how="inner")
    )
    _write_partitioned(
        data=test_lf,
        out_dir=test_dir,
        n_partitions=estimate_processing_partitions(test_lf),
        partition_over="uid",
        transform=mapper,
    )

    return SequenceSplit(
        train=ParquetDataset.from_dir(train_dir),
        val=ParquetDataset.from_dir(val_dir),
        test=ParquetDataset.from_dir(test_dir),
        item_to_idx=ParquetDataset.from_dir(item_to_idx_dir),
    )

    # item_to_idx_df: pl.DataFrame
    # processing_partitions = estimate_processing_partitions(raw_train_lf)
    # logger.info(f"Estimated number of processing partitions: {processing_partitions}")

    # if processing_partitions == 1:
    #     item_to_idx_df = _get_unique_items(raw_train_lf)
    # else:
    #     logger.info("Starting partitioned processing")
    #     raw_train_lf = raw_train_lf.with_columns(
    #         (pl.col("uid") % processing_partitions).alias("processing_partition")
    #     )
    #     unique_items_per_part: list[pl.DataFrame] = []

    #     for i in range(processing_partitions):
    #         logger.info(f"Processing partition {i + 1}/{processing_partitions}")
    #         part_lf = raw_train_lf.filter(pl.col("processing_partition") == i)
    #         unique_items_per_part.append(_get_unique_items(part_lf))

    #     logger.info("Merging partitions and generating item_id -> index mapping")
    #     item_to_idx_df = pl.concat(unique_items_per_part).unique()

    # item_to_idx_df = (
    #     item_to_idx_df.sort("item_id")
    #     .with_columns(pl.int_range(1, pl.len() + 1).alias("idx"))
    # )

    # storage_partitions = estimate_storage_partitions(item_to_idx_df)
    # logger.info(f"Estimated number of storage partitions: {storage_partitions}")
    # if storage_partitions == 1:
    #     logger.info("Saving the entire dataset as a single Parquet file")
    #     item_to_idx_df.write_parquet(item_to_idx_dir / ParquetDataset.file_name())
    # else:
    #     item_to_idx_df = item_to_idx_df.with_columns(
    #         (pl.col("idx") % storage_partitions).alias("storage_partition")
    #     )
    #     for i in range(storage_partitions):
    #         logger.info(f"Saving partition {i + 1}/{storage_partitions}")
    #         item_to_idx_df.filter(pl.col("storage_partition") == i).write_parquet(
    #             item_to_idx_dir / ParquetDataset.file_name(i)
    #         )

    # logger.info(f"Applying item_id -> index mapping to the train set and saving
    # train_lf to {train_dir}")
    # processing_partitions = estimate_processing_partitions(raw_train_lf)
    # logger.info(f"Estimated number of processing partitions: {processing_partitions}")

    # if processing_partitions == 1:
    #     _map_item_ids(raw_train_lf, item_ids, mapped_ids).sink_parquet(
    #         train_dir / ParquetDataset.file_name()
    #     )
    # else:
    #     logger.info("Starting partitioned processing")
    #     part_lfs: list[pl.LazyFrame] = []
    #     for i in range(processing_partitions):
    #         logger.info(f"Processing partition {i + 1}/{processing_partitions}")
    #         part_lfs.append(
    #             raw_train_lf.filter(pl.col("uid") % processing_partitions == i).pipe(
    #                 _map_item_ids, item_ids, mapped_ids
    #             )
    #         )

    #     logger.info("Merging partitions (lazily!)")
    #     train_lf = pl.concat(part_lfs)

    # # Identify valid users from the training set (not cold-start users)
    # # Materialize them for optimized joining
    # unique_train_uids = train_lf.select("uid").unique().collect()

    # # Validation split
    # logger.info("Creating validation set")
    # val_lf = (
    #     sequences_lf.filter(
    #         (pl.col("target_ts") >= val_start_ts) & (pl.col("target_ts") < val_end_ts)
    #     )
    #     # Prevent cold-start by keeping only users present in the train set
    #     .join(unique_train_uids.lazy(), on="uid", how="inner")
    #     .pipe(_map_item_ids, item_ids, mapped_ids)
    # )

    # # Test split
    # logger.info("Creating test set")
    # test_lf = (
    #     sequences_lf.filter(pl.col("target_ts") >= test_start_ts)
    #     # Prevent cold-start by keeping only users present in the train set
    #     .join(unique_train_uids.lazy(), on="uid", how="inner")
    #     .pipe(_map_item_ids, item_ids, mapped_ids)
    # )

    # return SequenceSplitResult(train_lf, val_lf, test_lf, item_to_idx_df)


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


def _create_item_to_idx_mapping(
    lf: pl.LazyFrame,
    n_partitions: int = 1,
    partition_over: str | None = None,
) -> pl.DataFrame:
    """Extract unique item IDs from nested sequence structures.

    Reads item IDs from:
    - target.items
    - context[*].items
    and returns unique values as `item_id`.

    Returns:
        DataFrame containing unique item IDs in column `item_id`.
    """

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

    if n_partitions > 1 and partition_over is None:
        raise ValueError("Cannot partition data if `partition_over` is not provided.")

    if n_partitions == 1:
        item_to_idx_df = _get_unique_items(lf)
    else:
        logger.info("Starting partitioned processing")
        lf = lf.with_columns(
            (pl.col(cast(str, partition_over)) % n_partitions).alias("partition")
        )
        unique_items_per_part: list[pl.DataFrame] = []

        for i in range(n_partitions):
            logger.info(f"Processing partition {i + 1}/{n_partitions}")
            part_lf = lf.filter(pl.col("partition") == i).drop("partition")
            unique_items_per_part.append(_get_unique_items(part_lf))

        logger.info("Merging partitions and generating item_id -> index mapping")
        item_to_idx_df = pl.concat(unique_items_per_part).unique()

    return item_to_idx_df.sort("item_id").with_columns(
        pl.int_range(1, pl.len() + 1).alias("idx")
    )


def _write_partitioned[T: pl.LazyFrame | pl.DataFrame](
    data: T,
    out_dir: Path,
    n_partitions: int = 1,
    partition_over: str | None = None,
    transform: Callable[[T], T] | None = None,
) -> None:
    if n_partitions > 1 and partition_over is None:
        raise ValueError("Cannot partition data if `partition_over` is not provided.")

    if n_partitions == 1:
        if transform is not None:
            data = transform(data)
        if isinstance(data, pl.DataFrame):
            data.write_parquet(out_dir / ParquetDataset.file_name(partition=None))
        else:
            data.sink_parquet(out_dir / ParquetDataset.file_name(partition=None))
        return

    data = cast(
        T,
        data.with_columns(
            (pl.col(cast(str, partition_over)) % n_partitions).alias("partition")
        ),
    )
    for i in range(n_partitions):
        part = cast(T, data.filter(pl.col("partition") == i).drop("partition"))
        if transform is not None:
            part = transform(part)
        if isinstance(part, pl.LazyFrame):
            part = part.collect()
        part.write_parquet(out_dir / ParquetDataset.file_name(partition=i))
