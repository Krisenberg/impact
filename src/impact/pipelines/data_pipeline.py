import shutil
from pathlib import Path

import polars as pl
import structlog
from tqdm import tqdm

from impact.config import ImpactConfig
from impact.dataset import YambdaFetcher
from impact.features import (
    extract_sequences,
    filter_listen_events,
    sessionize,
    split,
    tensorize,
)
from impact.schemas import (
    ItemIndex,
    ParquetTableSet,
    TemporalSplit,
    TensorizedDataset,
    TensorizedSplit,
)
from impact.utils import estimate_processing_partitions

logger = structlog.stdlib.get_logger(__name__)


class DataPipeline:
    """Orchestrates IMPACT data processing with a cache-aware API.

    Manages dataset acquisition and transformation of raw events into ML-ready tensors.
    Intermediate results are cached in nested directories keyed by IMPACT Framework
    configuration, ensuring reproducibility and avoiding redundant computation across
    runs.

    Processing stages in order:
    1. `get_raw_events()`    - fetch and cache raw Yambda events (multi-event type).
    2. `get_listen_events()` - filter to valid listen events only.
    3. `get_sessions()`      - extract user-scoped listening sessions.
    4. `get_sequences()`     - build context/target sequence pairs.
    5. `get_splits()`        - apply temporal train/val/test split.
    6. `tensorize()`         - convert to memory-mapped NumPy arrays.
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
        """Get raw interaction events, downloading and caching if not already cached.

        Caching is handled by the Hugging Face Hub client, which persists the dataset
        under `<raw_dir>` and skips downloading on subsequent calls.

        Returns:
            A LazyFrame of raw interaction events of all types, scanned lazily from the
            cached arquet files.
        """
        fetcher = YambdaFetcher(
            cache_dir=self.raw_dir,
            dataset_type=self.config.dataset.format,
            dataset_size=self.config.dataset.size,
        )
        return fetcher.get_interactions(event_type=self.config.dataset.interaction_type)

    def get_listen_events(self) -> pl.LazyFrame:
        """Get valid listen events, computing and caching to disk if not already cached.

        Filters raw events down to the listening events that meet the minimum playback
        threshold defined in config. Results are persisted under
        `<interim_dir>/<dataset_id>/listen_events/` and reused on subsequent calls.

        Returns:
            A LazyFrame of valid listen events with an `actual_played_seconds` column
            added. Scanned lazily from the cached parquet file.
        """
        cache_dir = self.interim_dir / self.config.dataset_id / "listen_events"
        success_marker = cache_dir / "_SUCCESS"

        # Atomicity check
        if not success_marker.exists():
            # Clean up potentially corrupted partial runs
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)

            logger.info("No complete cached listen events found, computing...")

            with tqdm(
                desc="Computing listen events",
                unit="",
                bar_format="{desc}: {elapsed} {postfix}",
            ) as bar:
                filter_listen_events(events_lf=self.get_raw_events()).sink_parquet(
                    path=cache_dir / "listen_events.parquet",
                )
                bar.set_postfix_str("done")

            success_marker.touch()
            logger.info("Listen events computed and cached.")

        return pl.scan_parquet(cache_dir / "listen_events.parquet")

    def get_sessions(self) -> pl.LazyFrame:
        """Get user-scoped sessions, computing and caching if not already cached.

        Partitions the listen events by user (`uid % n_partitions`) and processes each
        partition sequentially to fit within available RAM. The number of partitions
        is estimated automatically based on dataset size.

        Results are persisted under `<interim_dir>/<dataset_id>/<sessionization_id>/`
        and reused on subsequent calls.

        Returns:
            A LazyFrame of extracted user-scoped listening sessions, scanned lazily from
            the cached parquet files. Each row is a listen event annotated with its
            `session_id` (it's unique per user, not globally).
        """
        cache_dir = (
            self.interim_dir / self.config.dataset_id / self.config.sessionization_id
        )
        success_marker = cache_dir / "_SUCCESS"

        # Atomicity check
        if not success_marker.exists():
            # Clean up potentially corrupted partial runs
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)

            listens_lf = self.get_listen_events()
            n_partitions = estimate_processing_partitions(data=listens_lf)
            logger.info(
                "No complete sessionized data found in cache, computing...",
                partitions=n_partitions,
            )

            for i in tqdm(range(n_partitions), desc="Sessionizing", unit="partition"):
                logger.info("Computing partition", partition=i + 1, total=n_partitions)
                sessionize(
                    listens_lf=listens_lf.filter(pl.col("uid") % n_partitions == i),
                    tracks_per_session=self.config.session.tracks_per_session,
                ).sink_parquet(cache_dir / f"sessions_part_{i}.parquet")

            # Mark as completed
            success_marker.touch()

        return pl.scan_parquet(cache_dir / "sessions_part_*.parquet")

    def get_sequences(self) -> pl.LazyFrame:
        """Get extracted sequences of sessions, computing if not already cached.

        Partitions the sessions by user (`uid % n_partitions`) and processes each
        partition sequentially to fit within available RAM. The number of partitions
        is estimated automatically based on dataset size.

        Results are persisted under
        `<interim_dir>/<dataset_id>/<sessionization_id>/<sequence_extraction_id>/`
        and reused on subsequent calls.

        Returns:
            A LazyFrame of sequences scanned lazily from the cached parquet files,
            with a globally unique `sequence_idx` column assigned across all partitions
            (1-based).
        """
        cache_dir = (
            self.interim_dir
            / self.config.dataset_id
            / self.config.sessionization_id
            / self.config.sequence_extraction_id
        )
        success_marker = cache_dir / "_SUCCESS"

        # Atomicity check
        if not success_marker.exists():
            # Clean up potentially corrupted partial runs
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)

            sessions_lf = self.get_sessions()
            n_partitions = estimate_processing_partitions(data=sessions_lf)
            logger.info(
                "No complete sequenced data found in cache, computing...",
                partitions=n_partitions,
            )

            for i in tqdm(range(n_partitions), desc="Sequencing", unit="partition"):
                logger.info("Computing partition", partition=i + 1, total=n_partitions)
                extract_sequences(
                    sessions_lf=sessions_lf.filter(pl.col("uid") % n_partitions == i),
                    context_sessions=self.config.sequence.context_sessions,
                    step=self.config.sequence.step,
                    assign_index=False,  # Assign global index after merging
                ).sink_parquet(cache_dir / f"sequences_part_{i}.parquet")

            # Mark as completed
            success_marker.touch()

        # Assign global index
        return pl.scan_parquet(cache_dir / "sequences_part_*.parquet").with_row_index(
            "sequence_idx", offset=1
        )

    def get_splits(self) -> tuple[TemporalSplit, ItemIndex]:
        """Get temporal train/val/test splits and item index, computing if not cached.

        Unlike other pipeline stages, partitioning and disk writes are handled
        internally by `impact.features.split()`. This is intentional because the
        item-to-index mapping is derived from the train set and must be known before
        val/test sets can be written, making the entire operation atomic. Partition
        count and RAM management are estimated automatically inside `split()`.

        Results are persisted under `<processed_dir>/<dataset_id>/<sessionization_id>/
        <sequence_extraction_id>/<temporal_split_id>/` and reused on subsequent calls.

        Returns:
            A tuple of (TemporalSplit, ItemIndex) where splits are scanned lazily
            from cached parquet files and the item index maps train-set item IDs to
            contiguous 1-based indices (unseen items map to 0 which serves as padding).
        """
        cache_dir = (
            self.processed_dir
            / self.config.dataset_id
            / self.config.sessionization_id
            / self.config.sequence_extraction_id
            / self.config.temporal_split_id
        )
        item_to_idx_cache_dir = cache_dir / "item_to_idx"
        train_cache_dir = cache_dir / "train"
        val_cache_dir = cache_dir / "val"
        test_cache_dir = cache_dir / "test"
        success_marker = cache_dir / "_SUCCESS"

        # Atomicity check
        if not success_marker.exists():
            # Clean up potentially corrupted partial runs
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)
            item_to_idx_cache_dir.mkdir(parents=True)
            train_cache_dir.mkdir(parents=True)
            val_cache_dir.mkdir(parents=True)
            test_cache_dir.mkdir(parents=True)

            sequences_lf = self.get_sequences()
            logger.info("No complete temporal splits found in cache, computing...")

            # Discard the result as it has to be recreated at the end either way
            # (because of caching)
            _ = split(
                sequences_lf=sequences_lf,
                train_dir=train_cache_dir,
                val_dir=val_cache_dir,
                test_dir=test_cache_dir,
                item_to_idx_dir=item_to_idx_cache_dir,
                test_days=self.config.gts.test_days,
                val_days=self.config.gts.val_days,
                gap_minutes=self.config.gts.gap_minutes,
            )

            # Mark as completed
            success_marker.touch()
            logger.info("Temporal splits computed successfully")

        temporal_split = TemporalSplit(
            train=ParquetTableSet.from_dir(train_cache_dir),
            val=ParquetTableSet.from_dir(val_cache_dir),
            test=ParquetTableSet.from_dir(test_cache_dir),
        )
        item_index = ItemIndex(table=ParquetTableSet.from_dir(item_to_idx_cache_dir))

        return temporal_split, item_index

    def tensorize(self, split: TemporalSplit) -> TensorizedSplit:
        """Get tensorized splits, computing and caching to disk if not already cached.

        Converts each parquet-backed split into dense `[N, L, K]` memory-mapped NumPy
        arrays suitable for out-of-core PyTorch training, where N is the number of
        sequences (samples), L is the number of context sessions, and K is the number
        of tracks per session.

        Unlike earlier pipeline stages, disk writes are handled internally by
        `impact.features.tensorize()`. As memory-mapped arrays are files by definition,
        writing to disk and creating the tensor are the same operation.

        Results are persisted under
        `<processed_dir>/<dataset_id>/<sessionization_id>/<sequence_extraction_id>/
        <temporal_split_id>/tensors/` and reused on subsequent calls.

        Args:
            split: Temporal dataset splits, as returned by `get_splits()`.

        Returns:
            A TensorizedSplit with memory-mapped train, val, and test arrays loaded
            from the cached directory.
        """
        cache_dir = (
            self.processed_dir
            / self.config.dataset_id
            / self.config.sessionization_id
            / self.config.sequence_extraction_id
            / self.config.temporal_split_id
            / "tensors"
        )
        success_marker = cache_dir / "_SUCCESS"

        # Atomicity check
        if not success_marker.exists():
            # Clean up potentially corrupted partial runs
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True)

            logger.info("No complete tensorized splits found in cache, computing...")

            dataset_by_name = {
                "train": split.train,
                "val": split.val,
                "test": split.test,
            }
            for name, dataset in tqdm(
                dataset_by_name.items(),
                desc="Tensorizing",
                unit="split",
            ):
                dataset_cache_dir = cache_dir / name
                dataset_cache_dir.mkdir(parents=True)

                # Discard the result as it has to be recreated at the end either way
                # (because of caching)
                _ = tensorize(
                    dataset=dataset,
                    out_dir=dataset_cache_dir,
                    tracks_per_session=self.config.session.tracks_per_session,
                    context_sessions=self.config.sequence.context_sessions,
                )

            # Mark as completed
            success_marker.touch()
            logger.info("Tensorized splits computed successfully")

        return TensorizedSplit(
            train=TensorizedDataset.from_dir(cache_dir / "train"),
            val=TensorizedDataset.from_dir(cache_dir / "val"),
            test=TensorizedDataset.from_dir(cache_dir / "test"),
        )

    def _save_to_cache(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        cache_dir: Path,
        storage_partitions: int = 1,
    ) -> None:
        """Write dataset to parquet cache using predefined partitioning strategy.

        The function persists the input dataset to disk either as a single parquet file
        or as multiple partitioned files, depending on the provided `storage_partitions`
        parameter.

        If multiple partitions are requested and the dataset contains a `uid` column,
        data is split deterministically using a modulo-based partitioning strategy.
        Otherwise, the dataset is written as a single parquet file with a warning.

        Args:
            data: Dataset to be cached as parquet files.
            cache_dir: Directory where parquet files will be saved.
            storage_partitions: Number of storage partitions (number of parquet files).
        """
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Saving dataset to cache",
            cache_dir=cache_dir.as_posix(),
            storage_partitions=storage_partitions,
        )

        if storage_partitions == 1:
            path = cache_dir / "part_all.parquet"

            logger.info("Writing to a single parquet file...")
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
