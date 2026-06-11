import polars as pl
import structlog

logger = structlog.stdlib.get_logger(__name__)


def sessionize(
    listens_lf: pl.LazyFrame,
    inactivity_gap_seconds: int = 1200,
    tracks_per_session: int = 10,  # K
) -> pl.LazyFrame:
    """Extract user-scoped sessions from pre-filtered listen events.

    Args:
        listens_lf: LazyFrame of valid listen events (event_type == "listen",
            actual_played_seconds already computed and filtered).
        inactivity_gap_seconds: Number of seconds of inactivity before a session is
            considered finished. Defaults to 1200 (20 minutes).
        tracks_per_session: Number of tracks that constitute a session. Denoted as K in
            the thesis. Sessions with less than K tracks are discarded. Sessions with
            more than K tracks are truncated. Defaults to 10.

    Returns:
        A Polars LazyFrame extending the listens_lf by the session_id column containing
        assigned session IDs, unique per user (not globally unique).
    """
    logger.info(
        "Extracting user-scoped sessions from listen events",
        inactivity_gap_seconds=inactivity_gap_seconds,
        tracks_per_session=tracks_per_session,
    )

    sessions_lf = (
        listens_lf

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
    )  # fmt: skip

    logger.info("Raw events data successfully preprocessed into sessions (lazily!)")
    return sessions_lf
