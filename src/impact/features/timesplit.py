"""Module for splitting sequences of sessions into train, validation, and test sets.

Sequences of sessions (baskets in the NBR paradigm) are split using the GTS (Global
Temporal Split) strategy.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl
import structlog

from impact.constants import DAY_SECONDS, MINUTE_SECONDS

logger = structlog.stdlib.get_logger(__name__)


@dataclass(frozen=True)
class SequenceSplit:
    """Container for train, validation, and test LazyFrames along with item mapping.

    All three splits share a common item-to-index mapping derived from the train set.
    Items unseen during training map to index 0 (padding/cold-start).
    """

    train: pl.LazyFrame
    val: pl.LazyFrame
    test: pl.LazyFrame
    item_to_idx: pl.DataFrame  # columns: item_id (UInt32), idx (UInt32, 1-based)

    @property
    def num_items(self) -> int:
        """Number of unique items in the train set (excludes padding index)."""
        return self.item_to_idx.height


def split_sequences_train_val_test(
    sequences_lf: pl.LazyFrame,
    test_days: float = 1.0,
    val_days: float = 1.0,
    gap_minutes: float = 30.0,
    n_partitions: int = 1,
    cache_dir: Path | None = None,
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
        test_days: Number of days at the end of the timeline allocated for the test set.
        val_days: Number of days preceding test set allocated for the validation set.
        gap_minutes: Buffer time (in minutes) between dataset splits that mimics
            production system deployment latency.
        n_partitions: Number of partitions for sequential processing. Use >1 for
            large datasets that exceed available RAM (data is partitioned by user,
            i.e. uid column).
        cache_dir: Directory used to store intermediate partition files during
            partitioned processing (when `n_partitions > 1`). Ignored when
            `n_partitions == 1`.

    Returns:
        SequenceSplit containing train, validation, and test LazyFrames and the
        item_to_idx DataFrame.
    """
    if n_partitions < 1:
        raise ValueError("Number of partitions must be greater than 0.")

    if n_partitions > 1 and cache_dir is None:
        raise ValueError("Cache directory must be provided for partitioned processing.")

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

    def _get_unique_items_per_part(part_lf: pl.LazyFrame) -> pl.DataFrame:
        return (
            pl.concat([
                part_lf.select(
                    pl.col("target")
                    .struct.field("items")
                    .explode()
                    .alias("item_id")
                ),
                part_lf.select(
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

    logger.info("Generating item_id -> index mapping from the train set")
    item_to_idx_df: pl.DataFrame
    if n_partitions == 1:
        item_to_idx_df = (
            _get_unique_items_per_part(raw_train_lf)
            .sort("item_id")
            .with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.UInt32).alias("idx"))
        )
    else:
        logger.info("Starting partitioned processing")
        unique_items_per_part: list[pl.DataFrame] = []

        for i in range(n_partitions):
            logger.info(f"Processing partition {i + 1}/{n_partitions}")
            part_lf = raw_train_lf.filter(pl.col("uid") % n_partitions == i)
            unique_items_per_part.append(_get_unique_items_per_part(part_lf))

        logger.info("Merging partitions and generating item_id -> index mapping")
        item_to_idx_df = (
            pl.concat(unique_items_per_part)
            .unique()
            .sort("item_id")
            .with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.UInt32).alias("idx"))
        )

    old = item_to_idx_df.get_column("item_id").cast(pl.UInt32)
    new = item_to_idx_df.get_column("idx").cast(pl.UInt32)

    def _map_item_ids(lf: pl.LazyFrame, old: pl.Series, new: pl.Series) -> pl.LazyFrame:
        default = pl.lit(0, dtype=pl.UInt32)
        return lf.with_columns([
            # context: List(Struct({..., items: List(UInt32)}))
            pl.col("context").list.eval(
                pl.element().struct.with_fields(
                    pl.field("items").list.eval(
                        pl.element().replace_strict(old, new, default=default)
                    )
                )
            ),
            # target: Struct({..., items: List(UInt32)})
            pl.col("target").struct.with_fields(
                pl.field("items").list.eval(
                    pl.element().replace_strict(old, new, default=default)
                )
            ),
        ])  # fmt: skip

    logger.info("Applying item_id -> index mapping to the train set")
    train_lf: pl.LazyFrame
    if n_partitions == 1:
        train_lf = _map_item_ids(raw_train_lf, old, new)
    else:
        logger.info("Starting partitioned processing")
        cache_dir = cast(Path, cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_partitions):
            logger.info(f"Processing partition {i + 1}/{n_partitions}")
            (
                raw_train_lf.filter(pl.col("uid") % n_partitions == i)
                .pipe(_map_item_ids, old, new)
                .sink_parquet(cache_dir / f"part_{i}.parquet")
            )

        logger.info("Merging partitions (lazily!)")
        train_lf = pl.scan_parquet(cache_dir / "part_*.parquet")

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
        .pipe(_map_item_ids, old, new)
    )

    # Test split
    logger.info("Creating test set")
    test_lf = (
        sequences_lf.filter(pl.col("target_ts") >= test_start_ts)
        # Prevent cold-start by keeping only users present in the train set
        .join(unique_train_uids.lazy(), on="uid", how="inner")
        .pipe(_map_item_ids, old, new)
    )

    return SequenceSplit(train_lf, val_lf, test_lf, item_to_idx_df)
