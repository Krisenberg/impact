"""Raw event preprocessing applied before any feature-engineering step."""

import polars as pl
import structlog

logger = structlog.stdlib.get_logger(__name__)

MIN_LISTEN_SECONDS: int = 30
"""Minimum playback duration (in seconds) to consider an event as a valid listen.
Below this threshold events are treated as noise (accidental plays, skips).
Follows the industry standard established in the literature.
"""


def filter_listen_events(
    events_lf: pl.LazyFrame,
    min_listen_seconds: int = MIN_LISTEN_SECONDS,
) -> pl.LazyFrame:
    """Filter raw events of all types down to valid listening events only.

    This is the only place that decides what constitutes a valid listen, so the
    definition stays consistent throughout the entire codebase.

    Args:
        events_lf: Raw LazyFrame containing all event types.
        min_listen_seconds: Minimum playback duration (in seconds) to consider an event
            as a valid listen. Below this threshold events are treated as noise
            (accidental plays, skips).

    Returns:
        A LazyFrame containing only `event_type == "listen"` rows where the user
        actually played enough of the track, with an `actual_played_seconds` column
        added.
    """
    logger.debug(
        "Filtering raw events to valid listen events",
        min_listen_seconds=min_listen_seconds,
    )
    return (
        events_lf

        # Filter out non-listen events
        .filter(pl.col("event_type") == "listen")

        # Calculate actual played time in seconds
        .with_columns(
            (pl.col("track_length_seconds") * pl.col("played_ratio_pct") / 100).alias(
                "actual_played_seconds"
            )
        )

        # Filter valid listening events
        .filter(pl.col("actual_played_seconds") >= min_listen_seconds)
    )  # fmt: skip
