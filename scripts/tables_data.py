from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pandera.polars as pa
import polars as pl
import structlog
from pandera.typing.polars import Series  # noqa: TC002

from impact.dataset import DatasetSize
from impact.pipelines import DataPipeline

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pandera.typing.polars import DataFrame

    from impact.config import ImpactConfig

logger = structlog.stdlib.get_logger(__name__)


class DataFunnelStatsSchema(pa.DataFrameModel):
    """Schema for the data funnel stats table."""

    dataset_variant: Series[pl.String] = pa.Field()
    config_display_id: Series[pl.String] = pa.Field()
    all_events: Series[pl.Int64] = pa.Field()
    valid_listen_events: Series[pl.Int64] = pa.Field()
    listening_sessions: Series[pl.Int64] = pa.Field()
    sequences_of_sessions: Series[pl.Int64] = pa.Field()
    train_split: Series[pl.Int64] = pa.Field()
    val_split: Series[pl.Int64] = pa.Field()
    test_split: Series[pl.Int64] = pa.Field()


class DataFunnelVariant(NamedTuple):
    """Dataset variant to be included in data funnel stats table, e.g. Yambda-50M."""

    dataset_size: DatasetSize
    sequence_step: int
    display_name: str  # e.g. "Yambda-50M"


DEFAULT_VARIANTS: Final[list[DataFunnelVariant]] = [
    DataFunnelVariant(DatasetSize.SIZE_50M, sequence_step=5, display_name="Yambda-50M"),
    DataFunnelVariant(DatasetSize.SIZE_5B, sequence_step=20, display_name="Yambda-5B"),
]


def compute_data_funnel_stats_for_variants(
    base_impact_config: ImpactConfig,
    raw_data_dir: Path,
    interim_data_dir: Path,
    processed_data_dir: Path,
    variants: Sequence[DataFunnelVariant] = DEFAULT_VARIANTS,
) -> tuple[DataFrame[DataFunnelStatsSchema], dict[str, ImpactConfig]]:
    """Compute data funnel stats for the given variants (one row for each variant)."""
    rows: list[pl.DataFrame] = []
    resolved_configs: dict[str, ImpactConfig] = {}

    for variant in variants:
        variant_config = dataclasses.replace(
            base_impact_config,
            dataset=dataclasses.replace(
                base_impact_config.dataset,
                size=variant.dataset_size,
            ),
            sequence=dataclasses.replace(
                base_impact_config.sequence,
                step=variant.sequence_step,
            ),
        )
        logger.info("Computing data funnel stats", variant=variant.display_name)
        stats_df = compute_data_funnel_stats(
            variant_config, raw_data_dir, interim_data_dir, processed_data_dir
        )
        rows.append(
            stats_df.with_columns(
                pl.lit(variant.display_name).alias("dataset_variant"),
                pl.lit(variant_config.display_id).alias("config_display_id"),
            )
        )
        resolved_configs[variant_config.display_id] = variant_config

    return DataFunnelStatsSchema.validate(pl.concat(rows)), resolved_configs


def compute_data_funnel_stats(
    impact_config: ImpactConfig,
    raw_data_dir: Path,
    interim_data_dir: Path,
    processed_data_dir: Path,
) -> pl.DataFrame:
    """Compute data funnel stats."""
    pipeline = DataPipeline(
        raw_dir=raw_data_dir,
        interim_dir=interim_data_dir,
        processed_dir=processed_data_dir,
        config=impact_config,
    )

    events_lf = pipeline.get_raw_events()
    listen_events_lf = pipeline.get_listen_events()
    sessions_lf = pipeline.get_sessions()
    sequences_lf = pipeline.get_sequences()
    splits, _ = pipeline.get_splits()

    return pl.DataFrame(
        {
            "all_events": _lf_len(events_lf),
            "valid_listen_events": _lf_len(listen_events_lf),
            "listening_sessions": _lf_len(sessions_lf),
            "sequences_of_sessions": _lf_len(sequences_lf),
            "train_split": _lf_len(splits.train.lf),
            "val_split": _lf_len(splits.val.lf),
            "test_split": _lf_len(splits.test.lf),
        }
    )


def _lf_len(lf: pl.LazyFrame) -> int:
    """Return the number of rows in the given LazyFrame."""
    return cast(int, lf.select(pl.len()).collect().item())
