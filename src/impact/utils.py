from __future__ import annotations

from typing import cast

import polars as pl
import structlog

logger = structlog.stdlib.get_logger(__name__)


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
            partition. It should be keept low (<=0.2) to account for memory spikes
            during joins, explodes, and struct operations.

    Returns:
        Number of partitions in range [1, 64].
    """
    import math
    from typing import cast

    import psutil

    estimated_size = _estimate_size_bytes(data)
    available_ram = cast(int, psutil.virtual_memory().available)
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

    Dataset partitioning can be especially useful for torch dataloaders or numpy memmap.

    Args:
        data: Dataset to be written as partitioned Parquet files. For LazyFrames, size
            is estimated by sampling the first 1000 rows.
        target_file_size_mb: Target uncompressed size per output file in MB.
        max_partitions: Upper bound on the number of output files.

    Returns:
        Number of partitions in range [1, max_partitions].
    """
    import math

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
