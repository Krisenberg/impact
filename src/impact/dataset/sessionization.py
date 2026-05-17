"""Module for preprocessing raw events from the dataset into sessions."""

import polars as pl
import structlog

logger = structlog.stdlib.get_logger(__name__)


def preprocess_sessions(
    events_lf: pl.LazyFrame,
    inactivity_gap_seconds: int = 1200,
    tracks_per_session: int = 10,  # K
    sessions_per_user: int = 50,   # L
    min_listen_seconds: int = 30,  # Treshold defining a valid listening event
) -> pl.LazyFrame:
    """Preprocess raw data containing all event types into user-scoped sessions.

    Args:
        events_lf: A Polars LazyFrame containing all event types.
        inactivity_gap_seconds: Number of seconds of inactivity before a session is
            considered finished. Defaults to 1200 (20 minutes).
        tracks_per_session: Number of tracks that consitute a session. Denoted as K in
            the paper. Sessions with less than K tracks are discarded. Sessions with
            more than K tracks are truncated. Defaults to 10.
        sessions_per_user: Number of sessions per user. Denoted as L in the paper.
            Users with less than L sessions are discarded. Users with more than L
            sessions are trimmed to their L most recent sessions. Defaults to 50.
        min_listen_seconds: Minimum number of seconds a track must have been played
            to be considered as a valid listening event. Defaults to 30 (industry
            standard used by authors of other publications).
    """
    logger.info(
        "Preprocessing raw events into sessions",
        inactivity_gap_seconds=inactivity_gap_seconds,
        k_tracks=tracks_per_session,
        l_sessions=sessions_per_user,
    )

    # Convert seconds to 5-second units
    inactivity_gap_units = inactivity_gap_seconds // 5

    sessions_lf = (
        events_lf
        # Filter out non-listen events
        .filter(pl.col("event_type") == "listen")

        # Calculate actual played time in seconds
        .with_columns(
            (pl.col("track_length_seconds") * pl.col("played_ratio_pct") / 100)
            .alias("actual_played_seconds")
        )

        # Filter valid listening events
        .filter(pl.col("actual_played_seconds") >= min_listen_seconds)

        # Sort temporally
        .sort(["uid", "timestamp"])

        # Detect session boundaries
        .with_columns(pl.col("timestamp").diff().over("uid").alias("time_diff"))
        .with_columns(
            pl.when(
                (pl.col("time_diff").is_null())
                .or_(pl.col("time_diff") > inactivity_gap_units)
            )
            .then(1)
            .otherwise(0)
            .alias("is_new_session")
        )

        # Create unique session_id per user
        .with_columns(
            pl.col("is_new_session").cum_sum().over("uid").alias("session_id")
        )

        # Truncate sessions to top K tracks
        .filter(
            pl.int_range(1, pl.len() + 1)
            .over(["uid", "session_id"])
            .le(tracks_per_session)
        )

        # Discard sessions with fewer than K tracks
        .with_columns(pl.len().over(["uid", "session_id"]).alias("session_track_count"))
        .filter(pl.col("session_track_count") == tracks_per_session)

        # Filter out users with insufficient sessions
        .with_columns(
            pl.col("session_id").n_unique().over("uid").alias("total_user_sessions")
        )
        .filter(pl.col("total_user_sessions") >= sessions_per_user)

        # Trim to L most recent sessions per user
        .with_columns(
            pl.col("session_id")
            .rank(method="dense", descending=True)
            .over("uid")
            .alias("session_recency_rank")
        )
        .filter(pl.col("session_recency_rank") <= sessions_per_user)

        # Clean up temporary processing columns
        .drop([
            "time_diff",
            "is_new_session",
            "session_track_count",
            "total_user_sessions",
            "session_recency_rank",
        ])
    )

    logger.info("Raw events data successfully preprocessed into sessions (lazily!)")
    return sessions_lf
