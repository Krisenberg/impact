"""Module for splitting sequences of sessions into train, validation, and test sets.

Sequences of sessions (baskets in the NBR paradigm) are split using the GTS (Global
Temporal Split) strategy.
"""

from typing import NamedTuple, cast

import polars as pl
import structlog

from impact.constants import DAY_SECONDS, MINUTE_SECONDS

logger = structlog.stdlib.get_logger(__name__)


class SequenceSplitResult(NamedTuple):
    """Container for computed dataset splits and item mapping.

    Stores train, validation and test LazyFrames together with the item-to-index mapping
    derived from the training set.
    """

    train: pl.LazyFrame
    val: pl.LazyFrame
    test: pl.LazyFrame
    item_to_idx: pl.DataFrame


def split_sequences_train_val_test(
    sequences_lf: pl.LazyFrame,
    test_days: float = 1.0,
    val_days: float = 1.0,
    gap_minutes: float = 30.0,
    n_partitions: int = 1,
) -> SequenceSplitResult:
    """Compute temporal train/validation/test splits with a global item-index mapping.

    Splits are based on timestamp thresholds with a gap between train, validation, and
    test windows. The item-to-index mapping is derived from the training set and applied
    consistently across all splits, ensuring a shared evaluation ID space. Items
    appearing in val/test but absent in train are mapped to index 0.

    Val and test sets are further filtered only to users present in the train set to
    exclude cold-start users.

    Args:
        sequences_lf: LazyFrame representing user-scoped sequences of sessions, as
            produced by `sequences.extract_sequences()`.
        test_days: Number of days at the end of the timeline allocated for the test set.
        val_days: Number of days preceding test set allocated for the validation set.
        gap_minutes: Buffer time (in minutes) between dataset splits that mimics
            production system deployment latency.
        n_partitions: Number of partitions for sequential processing. Use >1 for
            large datasets that exceed available RAM (data is partitioned by user,
            i.e. uid column).

    Returns:
        SequenceSplitResult that stores train, validation and test LazyFrames together
        with the item-to-index mapping derived from the training set.
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

    item_ids = item_to_idx_df.get_column("item_id").cast(pl.UInt32)
    mapped_ids = item_to_idx_df.get_column("idx").cast(pl.UInt32)

    logger.info("Applying item_id -> index mapping to the train set")
    if n_partitions == 1:
        train_lf = _map_item_ids(raw_train_lf, item_ids, mapped_ids)
    else:
        logger.info("Starting partitioned processing")
        part_lfs: list[pl.LazyFrame] = []
        for i in range(n_partitions):
            logger.info(f"Processing partition {i + 1}/{n_partitions}")
            part_lfs.append(
                raw_train_lf.filter(pl.col("uid") % n_partitions == i).pipe(
                    _map_item_ids, item_ids, mapped_ids
                )
            )

        logger.info("Merging partitions (lazily!)")
        train_lf = pl.concat(part_lfs)

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
        .pipe(_map_item_ids, item_ids, mapped_ids)
    )

    # Test split
    logger.info("Creating test set")
    test_lf = (
        sequences_lf.filter(pl.col("target_ts") >= test_start_ts)
        # Prevent cold-start by keeping only users present in the train set
        .join(unique_train_uids.lazy(), on="uid", how="inner")
        .pipe(_map_item_ids, item_ids, mapped_ids)
    )

    return SequenceSplitResult(train_lf, val_lf, test_lf, item_to_idx_df)


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
