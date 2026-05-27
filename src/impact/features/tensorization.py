from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import structlog

logger = structlog.stdlib.get_logger(__name__)


def tensorize(
    out_dir: Path,
    tracks_per_session: int,  # K
    context_sessions: int,  # L
    dataset_lf: pl.LazyFrame | None = None,
    dataset_parquet_parts: list[Path] | None = None,
    chunk_size: int = 100_000,
) -> None:
    """Convert fixed-size sequences of listening session into mem-mapped NumPy tensors.

    This function performs an offline tensorization step that transforms sequence
    datasets stored as Polars LazyFrame or partitioned Parquet files into dense NumPy
    arrays suitable for efficient PyTorch training.

    The output representation is:
        `context.npy`: uint32 tensor of shape [N, L, K]
        `target.npy` : uint32 tensor of shape [N, K]
        `meta.npy`   : int64 array [N, L, K]

    where:
        N : number of sequence samples
        L : number of context sessions
        K : number of tracks per session

    Generated arrays are stored using NumPy memory-mapped format and are
    intended for loading via:

        np.load(..., mmap_mode="r")

    which enables out-of-core training without materializing the entire
    dataset in RAM.

    Input datasets are assumed to contain fixed-size sequences produced
    during preprocessing:

        - every context contains exactly L sessions
        - every session contains exactly K tracks
        - shorter sessions were removed
        - longer sessions were truncated

    Args:
        out_dir:
            Directory where tensorized arrays will be written.
        tracks_per_session: Number of tracks that consitute a session. Denoted as K in
            the thesis.
        context_sessions: Number of historical sessions used as context. Denoted as L in
            the thesis.
        dataset_lf:
            LazyFrame containing sequence samples. Intended for smaller datasets.
            Tensorization is performed in chunks using repeated slicing.
        dataset_parquet_parts:
            Partitioned Parquet files used for large-scale processing. Recommended for
            datasets exceeding RAM capacity.
        chunk_size:
            Number of rows processed simultaneously during tensorization.

    Raises:
        ValueError: If neither or both input sources are provided.

    Notes:
        Partitioned Parquet processing is preferred for large datasets because repeated
        LazyFrame slicing may trigger multiple scans and increase execution time.
    """
    if (dataset_lf is None) == (dataset_parquet_parts is None):
        raise ValueError(
            "Either dataset_lf or dataset_parquet_parts must be provided, but not both"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    K, L = tracks_per_session, context_sessions  # noqa: N806

    if dataset_lf is not None:
        logger.info("Inferring number of samples from the dataset_lf (LazyFrame)")
        n_samples = cast(int, dataset_lf.select(pl.len()).collect().item())
        logger.info(f"Number of samples: {n_samples}")
    else:
        logger.info("Inferring number of samples from the dataset_parquet_parts")
        n_samples = sum(
            pl.scan_parquet(f).select(pl.len()).collect().item()
            for f in cast(list[Path], dataset_parquet_parts)
        )
        logger.info(f"Number of samples: {n_samples}")

    context_mm = np.lib.format.open_memmap(
        filename=out_dir / "context.npy",
        mode="w+",
        dtype=np.uint32,
        shape=(n_samples, L, K),
    )

    target_mm = np.lib.format.open_memmap(
        out_dir / "target.npy",
        mode="w+",
        dtype=np.uint32,
        shape=(n_samples, K),
    )

    def _tensorize_slice(slice_df: pl.DataFrame, offset: int) -> int:
        contexts = slice_df["context"].to_list()
        targets = slice_df["target"].to_list()
        n = slice_df.height

        context_mm[offset : (offset + n)] = [
            [s["items"] for s in ctx] for ctx in contexts
        ]
        target_mm[offset : (offset + n)] = [tgt["items"] for tgt in targets]

        return n

    if dataset_lf is not None:
        logger.info("Tensorizing dataset using the given LazyFrame")
        offset = 0
        while offset < n_samples:
            slice_df = dataset_lf.slice(offset, length=chunk_size).collect()
            n_tensorized = _tensorize_slice(slice_df, offset)
            offset += n_tensorized
    else:
        logger.info("Tensorizing dataset using the given Parquet files")
        offset = 0
        for f in cast(list[Path], dataset_parquet_parts):
            for slice_df in pl.read_parquet(f).iter_slices(chunk_size):
                n_tensorized = _tensorize_slice(slice_df, offset)
                offset += n_tensorized

    logger.info(
        "Tensorization complete, flushing memory-mapped arrays and saving metadata"
    )
    context_mm.flush()
    target_mm.flush()
    np.save(out_dir / "meta.npy", np.array([n_samples, L, K], dtype=np.int64))
