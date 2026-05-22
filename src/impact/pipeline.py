"""Orchestrator for the IMPACT data processing pipeline."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import structlog

from impact.dataset import DatasetSize, DatasetType, InteractionType

logger = structlog.stdlib.get_logger(__name__)


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
    dataset_type: DatasetType = DatasetType.FLAT
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

    def __post_init__(self) -> None:
        """Verify that the selected dataset type is supported."""
        if self.dataset_type != DatasetType.FLAT:
            raise ValueError(f"Dataset type {self.dataset_type} not supported.")

    @property
    def dataset_id(self) -> str:
        """Cache key for the raw data fetching stage."""
        return "_".join([
            "dataset",
            self.dataset_size,
            self.dataset_type,
            self.interaction_type,
        ])

    @property
    def sessionization_id(self) -> str:
        """Cache key for the session extraction stage (Algorithm 1 in the thesis)."""
        return "_".join([
            "sessions",
            f"gap{self.inactivity_gap_seconds}",
            f"k{self.tracks_per_session}",
            f"min{self.min_listen_seconds}",
        ])

    @property
    def sequencing_id(self) -> str:
        """Cache key for the sequence extraction stage (Algorithm 2 in the thesis)."""
        return "_".join([
            "sequences",
            f"l{self.context_sessions}",
            f"step{self.step}",
        ])

    @property
    def timesplit_id(self) -> str:
        """Cache key for the temporal split (GTS) stage (Algorithm 3 in the thesis)."""
        return "_".join([
            "timesplit",
            f"val{self.val_days}d",
            f"test{self.test_days}d",
            f"gap{self.gap_minutes}m",
        ])

    @property
    def run_id(self) -> str:
        """Deterministic key representing the full pipeline configuration."""
        return "__".join([
            self.dataset_id,
            self.sessionization_id,
            self.sequencing_id,
            self.timesplit_id,
        ])


class DataPipeline:
    """Orchestrates IMPACT data processing with a cache-aware API.

    This pipeline manages the dataset acquisition and transformation of raw events into
    ML-ready temporal splits. It ensures reproducibility and avoids redundant
    computations by caching intermediate results in nested directories based on the
    pipeline configuration.
    """

    def __init__(
        self,
        raw_dir: Path,
        interim_dir: Path,
        processed_dir: Path,
        config: PipelineConfig,
    ) -> None:
        """Initialize the data pipeline and prepare cache directories.

        Args:
            raw_dir: Path to the directory for raw dataset files.
            interim_dir: Path to the directory for intermediate processing steps
                (e.g., after extracting sessions or sequences).
            processed_dir: Path to the directory for final, ML-ready datasets
                (e.g., train/val/test splits).
            config: The configuration object defining parameters for all pipeline steps.
        """
        self.raw_dir: Path = raw_dir.resolve()
        self.interim_dir: Path = interim_dir.resolve()
        self.processed_dir: Path = processed_dir.resolve()
        self.config: PipelineConfig = config

    def get_raw_events(self) -> pl.LazyFrame:
        """Fetch raw interactions (Step 1)."""
        from impact.dataset import YambdaFetcher

        fetcher = YambdaFetcher(
            cache_dir=self.raw_dir,
            dataset_type=self.config.dataset_type,
            dataset_size=self.config.dataset_size,
        )
        return fetcher.get_interactions(event_type=self.config.interaction_type)

    def get_sessions(self) -> pl.LazyFrame:
        """Get sessionized data, computing it if not cached (Step 2)."""
        from impact.dataset import extract_sessions

        sessions_dir = self.interim_dir / self.config.sessionization_id
        success_marker = sessions_dir / "_SUCCESS"

        partitions_by_dataset_size: dict[DatasetSize, int] = {
            DatasetSize.SIZE_50M: 1,
            DatasetSize.SIZE_500M: 4,
            DatasetSize.SIZE_5B: 32,
        }
        n_partitions = partitions_by_dataset_size[self.config.dataset_size]

        # Atomicity check
        if not success_marker.exists():
            logger.info("No complete sessionized data found in cache, computing...")

            # Clean up potentially corrupted partial runs
            if sessions_dir.exists():
                shutil.rmtree(sessions_dir)
            sessions_dir.mkdir(parents=True)

            events_lf = self.get_raw_events()

            for i in range(n_partitions):
                logger.info("Computing partition", partition=i + 1, total=n_partitions)
                extract_sessions(
                    events_lf=events_lf.filter(pl.col("uid") % n_partitions == i),
                    inactivity_gap_seconds=self.config.inactivity_gap_seconds,
                    tracks_per_session=self.config.tracks_per_session,
                    min_listen_seconds=self.config.min_listen_seconds,
                ).sink_parquet(sessions_dir / f"sessions_part_{i}.parquet")

            # Mark as completed
            success_marker.touch()

        return pl.scan_parquet(sessions_dir / "sessions_part_*.parquet")
