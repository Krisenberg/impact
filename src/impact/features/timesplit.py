"""Module for splitting sequences of sessions into train, validation, and test sets.

Sequences of sessions (baskets in the NBR paradigm) are split using the GTS (Global
Temporal Split) strategy.
"""

from typing import NamedTuple, cast

import polars as pl
import structlog

from impact.constants import DAY_SECONDS, MINUTE_SECONDS

logger = structlog.stdlib.get_logger(__name__)


class SequenceSplit(NamedTuple):
    """Container for the train, validation, and test LazyFrames."""

    train: pl.LazyFrame
    val: pl.LazyFrame
    test: pl.LazyFrame


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

    Args:
        sequences_lf: LazyFrame representing user-scoped sequences of sessions, as
            produced by `sequences.extract_sequences`.
        test_days: Number of days at the end of the timeline allocated for the test set.
        val_days: Number of days preceding test set allocated for the validation set.
        gap_minutes: Buffer time (in minutes) between dataset splits that mimics
            production system deployment latency.

    Returns:
        SequenceSplit containing train, validation, and test LazyFrames.
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
    train_lf = sequences_lf.filter(pl.col("target_ts") < train_end_ts)

    # Identify valid users from the training set (not cold-start users)
    # Materialize them for optimized joining
    unique_train_uids = train_lf.select("uid").unique().collect()

    # Validation split
    val_lf = (
        sequences_lf.filter(
            (pl.col("target_ts") >= val_start_ts) & (pl.col("target_ts") < val_end_ts)
        )
        # Prevent cold-start by keeping only users present in the train set
        .join(unique_train_uids.lazy(), on="uid", how="inner")
    )

    # Test split
    test_lf = (
        sequences_lf.filter(pl.col("target_ts") >= test_start_ts)
        # Prevent cold-start by keeping only users present in the train set
        .join(unique_train_uids.lazy(), on="uid", how="inner")
    )

    return SequenceSplit(train=train_lf, val=val_lf, test=test_lf)
