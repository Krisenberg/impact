from dataclasses import dataclass

from impact.dataset import DatasetSize, DatasetType, InteractionType


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration parameters for the IMPACT data processing pipeline.

    This class acts as the single source of truth for the entire pipeline configuration.
    All parameters are immutable to ensure reproducibility and deterministic data
    caching strategy across pipeline runs.
    """

    # === Dataset ===
    dataset_size: DatasetSize
    """Yambda dataset variant to process (50M, 500M, or 5B)."""

    dataset_type: str = DatasetType.FLAT
    """Dataset format type. Currently only FLAT is supported."""

    interaction_type: InteractionType = InteractionType.MULTI_EVENT
    """Interaction file to load.

    MULTI_EVENT includes all event types (listen, like, etc.); other variants contain
    only a subset.
    """

    # === Sessions ===
    inactivity_gap_seconds: int = 1200
    """Number of seconds of inactivity before a session is considered finished."""

    tracks_per_session: int = 10
    """Number of tracks that consitute a session. Denoted as K in the thesis.

    Sessions with less than K tracks are discarded. Sessions with more than K tracks are
    truncated.
    """

    min_listen_seconds: int = 30
    """Minimum playback duration (in seconds) to consider an event as a valid listen.

    Defaults to 30 (industry standard used by authors of other publications).
    """

    # === Sequences ===
    context_sessions: int = 20
    """Number of historical sessions used as context. Denoted as L in the thesis."""

    step: int = 5
    """Step size for the sliding window sequence generation."""

    # === Timesplit (GTS) ===
    test_days: float = 1.0
    """Number of days at the end of the timeline allocated for the test set."""

    val_days: float = 1.0
    """Number of days preceding test set allocated for the validation set."""

    gap_minutes: float = 30.0
    """Buffer time (in minutes) between dataset splits.

    The purpose of this parameter is to mimic production system deployment latency.
    """
