"""Module for processing raw events from the dataset and extracting sessions."""

import polars as pl
import structlog

logger = structlog.stdlib.get_logger(__name__)


def extract_sessions(
    events_lf: pl.LazyFrame,
    inactivity_gap_seconds: int = 1200,
    tracks_per_session: int = 10,  # K
    min_listen_seconds: int = 30,  # Threshold defining a valid listening event
) -> pl.LazyFrame:
    """Process raw data containing all event types and extract user-scoped sessions.

    Args:
        events_lf: A Polars LazyFrame containing all event types.
        inactivity_gap_seconds: Number of seconds of inactivity before a session is
            considered finished. Defaults to 1200 (20 minutes).
        tracks_per_session: Number of tracks that consitute a session. Denoted as K in
            the thesis. Sessions with less than K tracks are discarded. Sessions with
            more than K tracks are truncated. Defaults to 10.
        min_listen_seconds: Minimum number of seconds a track must have been played
            to be considered as a valid listening event. Defaults to 30 (industry
            standard used by authors of other publications).

    Returns:
        A Polars LazyFrame extending the events_lf by the session_id column containing
        assigned session IDs, unique per user (not globally unique).
    """
    logger.info(
        "Preprocessing raw events into sessions",
        inactivity_gap_seconds=inactivity_gap_seconds,
        k_tracks=tracks_per_session,
    )

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
                | (pl.col("time_diff") > inactivity_gap_seconds)
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

        # Clean up temporary processing columns
        .drop([
            "actual_played_seconds",
            "time_diff",
            "is_new_session",
            "session_track_count",
        ])
    )

    logger.info("Raw events data successfully preprocessed into sessions (lazily!)")
    return sessions_lf
