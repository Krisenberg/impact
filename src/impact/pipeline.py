"""Orchestrator for the IMPACT data processing pipeline."""

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from impact.dataset import DatasetSize, DatasetType, InteractionType
from impact.schemas import ParquetDataset, SequenceSplit

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
    """Type of interactions to load from the Yambda dataset.

    MULTI_EVENT includes all event types (listen, like, etc.); other variants contain
    only a specific subset.
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

    @staticmethod
    def _format_float_value(val: float) -> str:
        """Format float values to be filesystem-friendly (no dots)."""
        return f"{int(val)}" if val.is_integer() else str(val).replace(".", "-")

    @property
    def dataset_id(self) -> str:
        """Cache key for the raw data fetching stage."""
        return "_".join([
            "dataset",
            self.dataset_size,
            self.dataset_type,
            self.interaction_type,
        ])  # fmt: skip

    @property
    def sessionization_id(self) -> str:
        """Cache key for the session extraction stage (Algorithm 1 in the thesis)."""
        return "_".join([
            "sessions",
            f"gap{self.inactivity_gap_seconds}",
            f"k{self.tracks_per_session}",
            f"min{self.min_listen_seconds}",
        ])  # fmt: skip

    @property
    def sequencing_id(self) -> str:
        """Cache key for the sequence extraction stage (Algorithm 2 in the thesis)."""
        return "_".join([
            "sequences",
            f"l{self.context_sessions}",
            f"step{self.step}",
        ])  # fmt: skip

    @property
    def timesplit_id(self) -> str:
        """Cache key for the temporal split (GTS) stage (Algorithm 3 in the thesis)."""
        val_str = self._format_float_value(self.val_days)
        test_str = self._format_float_value(self.test_days)
        gap_str = self._format_float_value(self.gap_minutes)
        return "_".join([
            "split",
            f"val{val_str}d",
            f"test{test_str}d",
            f"gap{gap_str}m",
        ])  # fmt: skip

    @property
    def run_id(self) -> str:
        """Deterministic key representing the full pipeline configuration."""
        return "__".join([
            self.dataset_id,
            self.sessionization_id,
            self.sequencing_id,
            self.timesplit_id,
        ])  # fmt: skip

    @property
    def config_hash(self) -> str:
        """SHA-256 hash of the full run_id."""
        return hashlib.sha256(self.run_id.encode()).hexdigest()

    @property
    def display_id(self) -> str:
        """Short human-readable ID, e.g. '5B-10K-20L-20S-a1b2c3d'.

        Human-readable ID includes the following components:
        - Dataset size, uppercased (50M, 500M, 5B),
        - Tracks per session (K),
        - Context sessions (L),
        - Step size (S).
        - Last 7 characters of the SHA-256 hash of the run_id (e.g. 'a1b2c3d').
        """
        return "-".join([
            self.dataset_size.value.upper(),
            f"{self.tracks_per_session}K",
            f"{self.context_sessions}L",
            f"{self.step}S",
            self.config_hash[:7],
        ])  # fmt: skip

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Serialize the flat configuration into a structured, nested dictionary."""
        return {
            "dataset": {
                "dataset_size": self.dataset_size.value,
                "dataset_type": self.dataset_type.value,
                "interaction_type": self.interaction_type.value,
            },
            "sessions": {
                "inactivity_gap_seconds": self.inactivity_gap_seconds,
                "tracks_per_session": self.tracks_per_session,
                "min_listen_seconds": self.min_listen_seconds,
            },
            "sequences": {
                "context_sessions": self.context_sessions,
                "step": self.step,
            },
            "timesplit": {
                "test_days": self.test_days,
                "val_days": self.val_days,
                "gap_minutes": self.gap_minutes,
            },
        }

    def write_yaml(self, output_path: Path) -> None:
        """Write the configuration to a YAML file.

        Args:
            output_path: Target path where the YAML file with the configuration snapshot
                should be saved.
        """
        import yaml

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                data=self.to_dict(),
                stream=f,
                default_flow_style=False,
                sort_keys=False,  # Prevents from sorting keys alphabetically
                allow_unicode=True,
            )


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
        """Fetch raw interactions."""
        from impact.dataset import YambdaFetcher

        fetcher = YambdaFetcher(
            cache_dir=self.raw_dir,
            dataset_type=self.config.dataset_type,
            dataset_size=self.config.dataset_size,
        )
        return fetcher.get_interactions(event_type=self.config.interaction_type)

    def get_sessions(self, n_partitions: int = 1) -> pl.LazyFrame:
        """Get sessionized data, computing it if not cached.

        Args:
            n_partitions: Number of partitions for sequential processing. Use >1 for
                large datasets that exceed available RAM (data is partitioned by user,
                i.e. uid column).
        """
        if n_partitions < 1:
            raise ValueError("Number of partitions must be greater than 0.")

        from impact.dataset import extract_sessions

        sessions_dir = (
            self.interim_dir / self.config.dataset_id / self.config.sessionization_id
        )
        success_marker = sessions_dir / "_SUCCESS"

        # Atomicity check
        if not success_marker.exists():
            logger.info(
                "No complete sessionized data found in cache, computing...",
                partitions=n_partitions,
            )

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

    def get_sequences(self, n_partitions: int = 1) -> pl.LazyFrame:
        """Get extracted sequences of sessions, computing them if not cached.

        Args:
            n_partitions: Number of partitions for sequential processing. Use >1 for
                large datasets that exceed available RAM (data is partitioned by user,
                i.e. uid column).
        """
        if n_partitions < 1:
            raise ValueError("Number of partitions must be greater than 0.")

        from impact.features import extract_sequences

        sequences_dir = (
            self.interim_dir
            / self.config.dataset_id
            / self.config.sessionization_id
            / self.config.sequencing_id
        )
        success_marker = sequences_dir / "_SUCCESS"

        # Atomicity check
        if not success_marker.exists():
            logger.info(
                "No complete sequenced data found in cache, computing...",
                partitions=n_partitions,
            )

            # Clean up potentially corrupted partial runs
            if sequences_dir.exists():
                shutil.rmtree(sequences_dir)
            sequences_dir.mkdir(parents=True)

            sessions_lf = self.get_sessions(n_partitions=n_partitions)

            for i in range(n_partitions):
                logger.info("Computing partition", partition=i + 1, total=n_partitions)

                extract_sequences(
                    sessions_lf=sessions_lf.filter(pl.col("uid") % n_partitions == i),
                    context_sessions=self.config.context_sessions,
                    step=self.config.step,
                    assign_index=False,  # Assign global index after merging
                ).sink_parquet(sequences_dir / f"sequences_part_{i}.parquet")

            # Mark as completed
            success_marker.touch()

        # Assign global index
        return pl.scan_parquet(
            sequences_dir / "sequences_part_*.parquet"
        ).with_row_index("sequence_idx", offset=1)

    def get_splits(
        self,
        n_partitions: int = 1,
        compute_ram_fraction: float = 0.15,
    ) -> SequenceSplit:
        """Get final train/val/test splits, computing them if not cached.

        Args:
            n_partitions: Number of partitions for sequential processing. Use >1 for
                large datasets that exceed available RAM (data is partitioned by user,
                i.e. uid column).
            compute_ram_fraction: Fraction of available RAM allowed per processing
                partition. It should be keept low (<=0.15) to account for memory spikes
                during joins, explodes, and struct (non-streamable) Polars operations.
        """
        if n_partitions < 1:
            raise ValueError("Number of partitions must be greater than 0.")

        from impact.features import split_sequences_train_val_test

        splits_dir = (
            self.processed_dir
            / self.config.dataset_id
            / self.config.sessionization_id
            / self.config.sequencing_id
            / self.config.timesplit_id
        )
        item_to_idx_cache_dir = splits_dir / "item_to_idx"
        train_cache_dir = splits_dir / "train"
        val_cache_dir = splits_dir / "val"
        test_cache_dir = splits_dir / "test"
        success_marker = splits_dir / "_SUCCESS"

        # Atomicity check
        if not success_marker.exists():
            logger.info("No complete temporal splits found in cache, computing...")

            # Clean up potentially corrupted partial runs
            if splits_dir.exists():
                shutil.rmtree(splits_dir)

            item_to_idx_cache_dir.mkdir(parents=True)
            train_cache_dir.mkdir(parents=True)
            val_cache_dir.mkdir(parents=True)
            test_cache_dir.mkdir(parents=True)

            sequences_lf = self.get_sequences(n_partitions=n_partitions)

            # Discard the result as it has to be recreated at the end either way
            # (because of caching)
            _ = split_sequences_train_val_test(
                sequences_lf=sequences_lf,
                train_dir=train_cache_dir,
                val_dir=val_cache_dir,
                test_dir=test_cache_dir,
                item_to_idx_dir=item_to_idx_cache_dir,
                test_days=self.config.test_days,
                val_days=self.config.val_days,
                gap_minutes=self.config.gap_minutes,
                compute_ram_fraction=compute_ram_fraction,
            )

            # Mark as completed
            success_marker.touch()
            logger.info("Temporal splits computed successfully")

        return SequenceSplit(
            train=ParquetDataset.from_dir(train_cache_dir),
            val=ParquetDataset.from_dir(val_cache_dir),
            test=ParquetDataset.from_dir(test_cache_dir),
            item_to_idx=ParquetDataset.from_dir(item_to_idx_cache_dir),
        )

    def _save_to_cache(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        cache_dir: Path,
        storage_partitions: int = 1,
    ) -> None:
        """Write dataset to Parquet cache using predefined partitioning strategy.

        The function persists the input dataset to disk either as a single Parquet file
        or as multiple partitioned files, depending on the provided `storage_partitions`
        parameter.

        If multiple partitions are requested and the dataset contains a `uid` column,
        data is split deterministically using a modulo-based partitioning strategy.
        Otherwise, the dataset is written as a single Parquet file with a warning.

        Args:
            data: Dataset to be cached as Parquet files.
            cache_dir: Directory where Parquet files will be saved.
            storage_partitions: Number of storage partitions (number of Parquet files).
        """
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Saving dataset to cache",
            cache_dir=cache_dir.as_posix(),
            storage_partitions=storage_partitions,
        )

        if storage_partitions == 1:
            path = cache_dir / "part_all.parquet"

            logger.info("Writing to a single Parquet file...")
            if isinstance(data, pl.LazyFrame):
                data.sink_parquet(path)
            else:
                data.write_parquet(path)

            return

        lf = data.lazy() if isinstance(data, pl.DataFrame) else data
        schema = lf.collect_schema()

        if "uid" not in schema.names():
            logger.warning(
                "Partitioning by uid not available, trying to write to a single file..."
            )
            lf.sink_parquet(cache_dir / "part_all.parquet")
            return

        lf = lf.with_columns((pl.col("uid") % storage_partitions).alias("partition"))
        for i in range(storage_partitions):
            logger.info(f"Saving partition {i + 1}/{storage_partitions}...")
            lf.filter(pl.col("partition") == i).collect().write_parquet(
                cache_dir / f"part_{i}.parquet"
            )
