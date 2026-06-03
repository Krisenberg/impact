"""Module for extracting sequences of listening sessions from the sessionized data."""

from typing import Any

import polars as pl
import structlog

logger = structlog.stdlib.get_logger(__name__)


def sequence(
    sessions_lf: pl.LazyFrame,
    context_sessions: int = 20,
    step: int = 20,
    assign_index: bool = True,
) -> pl.LazyFrame:
    """Extract sequences of listening sessions using a sliding window technique.

    Each sequence contains context_sessions (L) sessions defining the listening context,
    followed by a single target session used for model evaluation.

    Args:
        sessions_lf: Sessionized LazyFrame of 'listen' events with a session_id column
            that contains assigned ID of the session that each event belongs to, as
            produced by `dataset.extract_sessions()`.
        context_sessions: Number of historical sessions used as context. Denoted as L in
            the thesis.
        step: Step size for the sliding window sequence generation.
        assign_index: Whether to assign a globally unique sequence index. Should be set
            to False when processing partitioned data, as the index must be assigned
            after merging all partitions to ensure global uniqueness.

    Returns:
        LazyFrame representing user-scoped sequences of sessions with columns:
        uid, target_ts, context, target, and optionally sequence_idx (as the first
        column) if assign_index is True.
    """
    logger.info(
        "Creating sequences of listening sessions",
        context_sessions=context_sessions,
        step=step,
    )

    session_schema = pl.Struct([
        pl.Field("session_id", pl.Int32),
        pl.Field("session_ts", pl.UInt32),
        pl.Field("items", pl.List(pl.UInt32)),
    ])  # fmt: skip

    sequence_schema = pl.List(
        pl.Struct([
            pl.Field("context", pl.List(session_schema)),
            pl.Field("target", session_schema),
            pl.Field("target_ts", pl.UInt32),
        ])
    )  # fmt: skip

    window_size = context_sessions + 1

    def _extract_sequences(user_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        windows: list[dict[str, Any]] = []
        for i in range(0, max(1, len(user_history) - window_size + 1), step):
            window = user_history[i : (i + window_size)]
            if len(window) == window_size:
                windows.append({
                    "context": window[:-1],
                    "target": window[-1],
                    "target_ts": window[-1]["session_ts"],
                })  # fmt: skip
        return windows

    # Collect listening history for each user
    user_history = (
        # Aggregate items (tracks) to sessions
        sessions_lf
        .group_by(["uid", "session_id"])
        .agg([
            pl.col("item_id").alias("items"),
            pl.col("timestamp").min().alias("session_ts"),
        ])
        .sort(["uid", "session_ts"])

        # Aggregate sessions to user history
        .group_by("uid", maintain_order=True)
        .agg([pl.struct(["session_id", "session_ts", "items"]).alias("all_sessions")])
    )  # fmt: skip

    # Form sequences of sessions
    sequences = (
        user_history.with_columns(
            pl.col("all_sessions")
            .map_elements(_extract_sequences, return_dtype=sequence_schema)
            .alias("seq_data")
        )

        # Explode seq_data to long format
        .explode("seq_data")

        # Remove null sequences, i.e. users with less than L+1 sessions
        .drop_nulls("seq_data")

        # Promote each field of the seq_data struct to its own column
        .unnest("seq_data")

        # Remove aggregated sessions column which is no longer needed
        .drop("all_sessions")

        # Reorder columns
        .select(["uid", "target_ts", "context", "target"])
    )  # fmt: skip

    if assign_index:
        # Assign sequence ID (globally unique)
        sequences = sequences.with_row_index("sequence_idx", offset=1)

    logger.info("Sequence creation complete")
    return sequences
