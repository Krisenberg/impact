from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandera.polars as pa
import polars as pl
import structlog
from pandera.typing.polars import Series  # noqa: TC002

from impact.pipelines import DataPipeline, ImpactEvaluationPipeline
from impact.utils import seed_everything

if TYPE_CHECKING:
    from pathlib import Path

    from pandera.typing.polars import DataFrame

    from impact.config import ImpactConfig

logger = structlog.stdlib.get_logger(__name__)


class AlphaCurvePlotSchema(pa.DataFrameModel):
    """Schema for the alpha curve plot data."""

    alpha: Series[pl.Float64] = pa.Field()
    metric: Series[pl.String] = pa.Field()
    score: Series[pl.Float64] = pa.Field()


def compute_alpha_curve_plot_data(
    impact_config: ImpactConfig,
    sasrec_checkpoint_path: Path,
    raw_data_dir: Path,
    interim_data_dir: Path,
    processed_data_dir: Path,
    alpha_start: float,
    alpha_stop: float,
    alpha_step: float,
    metrics_at_k: int,
) -> DataFrame[AlphaCurvePlotSchema]:
    """Compute data for the alpha curve plot."""
    seed_everything(impact_config.system.seed)

    data_pipeline = DataPipeline(
        raw_dir=raw_data_dir,
        interim_dir=interim_data_dir,
        processed_dir=processed_data_dir,
        config=impact_config,
    )

    logger.info("Fetching listening events data...")
    listens_lf = data_pipeline.get_listen_events()

    logger.info("Fetching train/val/test splits...")
    splits, item_index = data_pipeline.get_splits()

    logger.info("Tensorizing splits...")
    tensorized_splits = data_pipeline.tensorize(splits)

    alphas = np.round(
        a=np.arange(alpha_start, alpha_stop + alpha_step / 2, alpha_step),
        decimals=2,
    )
    results: list[dict[str, float]] = []

    for alpha in alphas:
        eval_config = dataclasses.replace(
            impact_config,
            eval=dataclasses.replace(
                impact_config.eval,
                alpha=float(alpha),
                k_cutoffs=(metrics_at_k,),
            ),
        )

        logger.info(f"Running IMPACT evaluation for {alpha=}...")
        eval_pipeline = ImpactEvaluationPipeline(eval_config, item_index, listens_lf)
        eval_result = eval_pipeline.evaluate(
            split_partition=tensorized_splits.val,
            sasrec_state_dict_path=sasrec_checkpoint_path,
        )

        row = {"alpha": float(alpha), **eval_result.metrics}
        results.append(row)

    df = (
        pl.DataFrame(results)
        .unpivot(
            on=[r for r in results[0] if r != "alpha"],
            index="alpha",
            variable_name="metric",
            value_name="score",
        )
        .filter(pl.col("metric").str.contains("(?i)ndcg"))  # only NDCG
    )
    return AlphaCurvePlotSchema.validate(df)
