"""Orchestrator for the IMPACT data processing pipeline."""

import shutil
from pathlib import Path

import polars as pl
import structlog

from impact.config import ImpactConfig
from impact.schemas import (
    ItemIndex,
    ParquetTableSet,
    TemporalSplit,
    TensorizedDataset,
    TensorizedSplit,
)

logger = structlog.stdlib.get_logger(__name__)


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
        config: ImpactConfig,
    ) -> None:
        """Initialize the data pipeline and prepare cache directories.

        Args:
            raw_dir: Path to the directory for raw dataset files.
            interim_dir: Path to the directory for intermediate processing steps
                (e.g., after extracting sessions or sequences).
            processed_dir: Path to the directory for final, ML-ready datasets
                (e.g., train/val/test splits).
            config: Complete IMPACT Framework configuration.
        """
        self.raw_dir: Path = raw_dir.resolve()
        self.interim_dir: Path = interim_dir.resolve()
        self.processed_dir: Path = processed_dir.resolve()
        self.config: ImpactConfig = config

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

        from impact.features import sessionize

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

            events_lf = self.get_raw_events()

            sessions_dir.mkdir(parents=True)

            for i in range(n_partitions):
                logger.info("Computing partition", partition=i + 1, total=n_partitions)
                sessionize(
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

        from impact.features import sequence

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

            sessions_lf = self.get_sessions(n_partitions=n_partitions)

            sequences_dir.mkdir(parents=True)

            for i in range(n_partitions):
                logger.info("Computing partition", partition=i + 1, total=n_partitions)

                sequence(
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
    ) -> tuple[TemporalSplit, ItemIndex]:
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

        from impact.features import split

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

            sequences_lf = self.get_sequences(n_partitions=n_partitions)

            item_to_idx_cache_dir.mkdir(parents=True)
            train_cache_dir.mkdir(parents=True)
            val_cache_dir.mkdir(parents=True)
            test_cache_dir.mkdir(parents=True)

            # Discard the result as it has to be recreated at the end either way
            # (because of caching)
            _ = split(
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

        split = TemporalSplit(
            train=ParquetTableSet.from_dir(train_cache_dir),
            val=ParquetTableSet.from_dir(val_cache_dir),
            test=ParquetTableSet.from_dir(test_cache_dir),
        )
        item_index = ItemIndex(table=ParquetTableSet.from_dir(item_to_idx_cache_dir))

        return split, item_index

    def tensorize(self, split: TemporalSplit) -> TensorizedSplit:
        """Tensorize a given TemporalSplit into memory-mapped NumPy arrays.

        Converts each Parquet-backed split into dense [N, L, K] arrays suitable for
        out-of-core PyTorch training. Results are cached under the pipeline's
        `processed_dir`; subsequent calls with a complete cache skip recomputation.

        Args:
            split: Temporal dataset splits, as returned by `get_splits()`.

        Returns:
            MemmapSplit with memory-mapped train, val, and test arrays.
        """
        from impact.features import tensorize

        tensors_dir = (
            self.processed_dir
            / self.config.dataset_id
            / self.config.sessionization_id
            / self.config.sequencing_id
            / self.config.timesplit_id
            / "tensors"
        )
        success_marker = tensors_dir / "_SUCCESS"

        # Atomicity check
        if not success_marker.exists():
            logger.info("No complete tensorized splits found in cache, computing...")

            # Clean up potentially corrupted partial runs
            if tensors_dir.exists():
                shutil.rmtree(tensors_dir)

            dataset_by_name = {
                "train": split.train,
                "val": split.val,
                "test": split.test,
            }
            for name, dataset in dataset_by_name.items():
                out_dir = tensors_dir / name
                out_dir.mkdir(parents=True)

                # Discard the result as it has to be recreated at the end either way
                # (because of caching)
                _ = tensorize(
                    dataset=dataset,
                    out_dir=tensors_dir / name,
                    tracks_per_session=self.config.tracks_per_session,
                    context_sessions=self.config.context_sessions,
                )

            # Mark as completed
            success_marker.touch()
            logger.info("Tensorized splits computed successfully")

        return TensorizedSplit(
            train=TensorizedDataset.from_dir(tensors_dir / "train"),
            val=TensorizedDataset.from_dir(tensors_dir / "val"),
            test=TensorizedDataset.from_dir(tensors_dir / "test"),
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
