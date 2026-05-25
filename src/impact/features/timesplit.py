"""Module for splitting sequences of sessions into train, validation, and test sets.

Sequences of sessions (baskets in the NBR paradigm) are split using the GTS (Global
Temporal Split) strategy.
"""

from dataclasses import dataclass
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
            produced by `sequences.extract_sequences`.
        test_days: Number of days at the end of the timeline allocated for the test set.
        val_days: Number of days preceding test set allocated for the validation set.
        gap_minutes: Buffer time (in minutes) between dataset splits that mimics
            production system deployment latency.

    Returns:
        SequenceSplit containing train, validation, and test LazyFrames and the
        item_to_idx DataFrame.
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

    # Train split
    logger.info("Filtering sequences for the train set")
    raw_train_lf = sequences_lf.filter(pl.col("target_ts") < train_end_ts)

    logger.info("Generating item_id -> index mapping from the train set")
    target_items_lf = raw_train_lf.select(
        pl.col("target").struct.field("items").explode().alias("item_id")
    )
    context_items_lf = raw_train_lf.select(
        pl.col("context").explode().struct.field("items").explode().alias("item_id")
    )
    item_to_idx_df = (
        pl.concat([target_items_lf, context_items_lf])
        .drop_nulls()
        .unique()
        .sort("item_id")
        .with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.UInt32).alias("idx"))
        .collect()
    )
    old = item_to_idx_df["item_id"]
    new = item_to_idx_df["idx"]

    def _map_item_ids(lf: pl.LazyFrame, old: pl.Series, new: pl.Series) -> pl.LazyFrame:
        return lf.with_columns([
            # context: List(Struct({..., items: List(UInt32)}))
            pl.col("context").list.eval(
                pl.element().struct.with_fields(
                    pl.col("items").list.eval(
                        pl.element().replace_strict(old, new, default=0)
                    )
                )
            ),
            # target: Struct({..., items: List(UInt32)})
            pl.col("target").struct.with_fields(
                pl.col("items").list.eval(
                    pl.element().replace_strict(old, new, default=0)
                )
            ),
        ])  # fmt: skip

    logger.info("Applying item_id -> index mapping to the train set")
    train_lf = _map_item_ids(raw_train_lf, old, new)

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
