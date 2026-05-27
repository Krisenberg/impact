from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import structlog

from impact.schemas import MemmapDataset, ParquetDataset

if TYPE_CHECKING:
    from pathlib import Path


logger = structlog.stdlib.get_logger(__name__)


def tensorize(
    dataset: ParquetDataset,
    out_dir: Path,
    tracks_per_session: int,  # K
    context_sessions: int,  # L
    batch_size: int = 100_000,
) -> MemmapDataset:
    """Convert fixed-size sequences of listening session into mem-mapped NumPy tensors.

    This function performs an offline tensorization step that transforms sequence
    datasets stored as partitioned Parquet files into dense NumPy arrays suitable for
    efficient PyTorch training.

    The output representation is:
        `context.npy`: uint32 tensor of shape [N, L, K]
        `target.npy` : uint32 tensor of shape [N, K]
        `meta.npy`   : int64 array of shape [3] storing [N, L, K]

    where:
        N : number of sequence samples
        L : number of context sessions
        K : number of tracks per session

    Generated arrays are stored using NumPy memory-mapped format and are
    intended for loading via: `np.load(..., mmap_mode="r")`, which enables out-of-core
    training without materializing the entire dataset in RAM.

    Input datasets are assumed to contain fixed-size sequences produced during
    preprocessing:
        - every context sequence contains exactly L sessions
        - every session contains exactly K tracks

    Parquet files are read in row batches rather than whole-file to bound peak RAM
    consumption. Each batch is unpacked from Polars structs into Python lists, which
    expands memory significantly relative to the on-disk size.

    Args:
        dataset: Partitioned Parquet dataset containing fixed-size listening sequences,
            as produced by the preprocessing pipeline.
        out_dir: Directory where the output `.npy` files will be written.
        tracks_per_session: Number of tracks that consitute a session. Denoted as K in
            the thesis.
        context_sessions: Number of historical sessions used as context. Denoted as L in
            the thesis.
        batch_size: Number of rows processed per iteration when reading Parquet
            partitions. Reducing this lowers peak RAM usage at the cost of more Python
            loop iterations.

    Returns:
        MemmapDataset object representing tensorized dataset backed by memory-mapped
        NumPy arrays (`context.npy`, `target.npy`, `meta.npy`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    K, L = tracks_per_session, context_sessions  # noqa: N806

    # Validate dataset shape (including the nested structures)
    first_df = pl.read_parquet(dataset.paths[0])
    first_context: list[dict[str, Any]] = first_df["context"][0]
    first_target: dict[str, Any] = first_df["target"][0]
    if len(first_context) != L or len(first_context[0]["items"]) != K:
        raise ValueError(f"Context shape mismatch: expected ({L}, {K})")
    if len(first_target["items"]) != K:
        raise ValueError(f"Target shape mismatch: expected K={K}")

    logger.info("Inferring total number of samples in dataset...")
    n_samples = sum(
        pl.scan_parquet(f).select(pl.len()).collect().item() for f in dataset.paths
    )
    logger.info(f"Inferred number of samples: {n_samples}")

    mm_dataset = MemmapDataset(directory=out_dir)

    context_mm = np.lib.format.open_memmap(
        filename=mm_dataset.context_file_path,
        mode="w+",
        dtype=np.uint32,
        shape=(n_samples, L, K),
    )

    target_mm = np.lib.format.open_memmap(
        filename=mm_dataset.target_file_path,
        mode="w+",
        dtype=np.uint32,
        shape=(n_samples, K),
    )

    logger.info("Tensorizing dataset using the underlying partitioned Parquet files...")
    offset = 0
    for i, path in enumerate(dataset.paths, start=1):
        logger.info(f"Processing partition {i}/{len(dataset.paths)}", path=path)
        partition_size = pl.scan_parquet(path).select(pl.len()).collect().item()

        for j, batch in enumerate(pl.read_parquet(path).iter_slices(n_rows=batch_size)):
            n = batch.height
            batch_range = f"{j * batch_size} - {(j * batch_size) + n}"
            logger.info(f"Tensorizing samples {batch_range} (out of {partition_size})")

            contexts = batch["context"].to_list()
            targets = batch["target"].to_list()
            context_mm[offset : (offset + n)] = [
                [session["items"] for session in context] for context in contexts
            ]
            target_mm[offset : (offset + n)] = [target["items"] for target in targets]
            offset += n

    logger.info(
        "Tensorization complete, flushing memory-mapped arrays and saving metadata"
    )
    context_mm.flush()
    target_mm.flush()
    np.save(mm_dataset.meta_file_path, np.array([n_samples, L, K], dtype=np.int64))

    return mm_dataset
