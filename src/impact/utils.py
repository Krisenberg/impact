from __future__ import annotations

import math
import os
import random
from typing import cast

import numpy as np
import polars as pl
import psutil
import structlog
import torch

logger = structlog.stdlib.get_logger(__name__)

WORKER_SEED_MODULUS: int = 2**32


def seed_everything(seed: int) -> None:
    """Seed all RNG sources in the main process for reproducibility.

    Sets seeds for Python's random, NumPy, and PyTorch (CPU, CUDA, MPS) modules.
    Also disables cuDNN benchmark mode and enables deterministic operations.
    Should be called once at the top-level entry point before any data loading or model
    initialization.

    Inspired by: https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964

    Args:
        seed: Global random seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    torch.mps.manual_seed(seed)  # For Apple Chip
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_: int) -> None:  # Ignore the argument (worker_id)
    """Seed RNG sources inside a PyTorch's DataLoader worker process.

    Each worker is forked from the main process and requires independent re-seeding to
    avoid all workers producing identical random sequences (e.g. for negative sampling).
    The per-worker seed is derived from PyTorch's initial worker seed, ensuring workers
    are both independent and deterministic given the same global seed.

    Pass this function as `worker_init_fn` to PyTorch's DataLoader.

    Args:
        worker_id: Worker index provided automatically by DataLoader.
    """
    worker_seed = torch.initial_seed() % WORKER_SEED_MODULUS
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def estimate_processing_partitions(
    data: pl.LazyFrame | pl.DataFrame,
    target_ram_fraction: float = 0.15,
) -> int:
    """Estimate the number of partitions needed for RAM-safe processing.

    Computes how many dataset partitions are required so that each partition fits
    within the available RAM. Intended for operations that cannot run in Polars
    streaming mode (e.g. list.eval or replace_strict on nested structs).

    Args:
        data: Dataset to be processed. For LazyFrames, size is estimated by sampling
            the first 1000 rows.
        target_ram_fraction: Fraction of currently available RAM to target per
            partition. It should be kept low (<=0.2) to account for memory spikes
            during joins, explodes, and struct operations.

    Returns:
        Number of partitions in range [1, 64].
    """
    estimated_size = _estimate_size_bytes(data)
    available_ram = cast(int, psutil.virtual_memory().available)  # type: ignore[redundant-cast]
    usable_ram = available_ram * target_ram_fraction
    partitions = max(1, min(64, math.ceil(estimated_size / usable_ram)))

    logger.debug(
        f"Estimated {partitions} processing partitions",
        estimated_data_size=estimated_size,
        target_ram_fraction=target_ram_fraction,
        available_ram=available_ram,
        usable_ram=usable_ram,
    )

    return partitions


def estimate_storage_partitions(
    data: pl.LazyFrame | pl.DataFrame,
    target_file_size_mb: int = 256,
    max_partitions: int = 64,
) -> int:
    """Estimate the number of Parquet files for partitioned storage.

    Determines how many Parquet files to produce so that each file approximates
    `target_file_size_mb` in uncompressed size. Actual files will be smaller due to
    Parquet compression.

    Args:
        data: Dataset to be written as partitioned Parquet files. For LazyFrames, size
            is estimated by sampling the first 1000 rows.
        target_file_size_mb: Target uncompressed size per output file in MB.
        max_partitions: Upper bound on the number of output files.

    Returns:
        Number of partitions in range [1, max_partitions].
    """
    estimated_size = _estimate_size_bytes(data)
    target_bytes = target_file_size_mb * 1024**2
    partitions = max(1, min(max_partitions, math.ceil(estimated_size / target_bytes)))

    logger.debug(
        f"Estimated {partitions} storage partitions",
        estimated_data_size=estimated_size,
        target_file_size_mb=target_file_size_mb,
    )

    return partitions


def _estimate_size_bytes(data: pl.LazyFrame | pl.DataFrame) -> float:
    """Estimate size of the given dataset in bytes."""
    if isinstance(data, pl.DataFrame):
        return data.estimated_size()

    n_rows = cast(int, data.select(pl.len()).collect().item())
    sample = data.limit(1000).collect()
    if sample.height == 0:
        return 0.0
    return n_rows * (sample.estimated_size() / sample.height)
