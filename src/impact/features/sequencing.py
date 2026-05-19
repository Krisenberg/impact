"""Module for transforming sessionized data into sliding window sequences."""

from typing import Any

import polars as pl
import structlog

logger = structlog.stdlib.get_logger(__name__)


def create_sequences(
    sessions_lf: pl.LazyFrame,
    context_sessions: int = 20,  # L
    step: int = 5,
) -> pl.LazyFrame:
    """Create sequences of context_sessions + 1 (L+1) sessions for each user.

    A sliding window technique is used for creating the sequences. Each sequence
    contains context_sessions (L) context sessions and 1 target session.

    Args:
        sessions_lf: Sessionized LazyFrame containing 'listen' events and the following
            columns: uid, session_id, item_id, timestamp.
        context_sessions: Number of context sessions per sequence. Denoted as L in the
            thesis.
        step: Sliding window step size.

    Returns:
        LazyFrame representing user-scoped sequences of sessions with columns:
        uid, sequence_idx, target_ts, context, target.
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
    ])

    sequence_schema = pl.List(
        pl.Struct([
            pl.Field("context", pl.List(session_schema)),
            pl.Field("target", session_schema),
            pl.Field("target_ts", pl.UInt32),
        ])
    )

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
                })
        return windows

    # Collect listening history for each user
    user_history = (
        # Aggregate items (tracks) to sessions
        sessions_lf
        .group_by(["uid", "session_id"])
        .agg([
            pl.col("item_id").alias("items"),
            pl.col("timestamp").max().alias("session_ts"),
        ])
        .sort(["uid", "session_ts"])

        # Aggregate sessions to user history
        .group_by("uid", maintain_order=True)
        .agg([pl.struct(["session_id", "session_ts", "items"]).alias("all_sessions")])
    )

    # Form sequences of sessions
    sequences = (
        user_history
        .with_columns(
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

        # Assign sequence ID (globally unique)
        .with_row_index("sequence_idx", offset=1)
    )

    logger.info("Sequence creation complete")
    return sequences
