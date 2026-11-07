"""Statistical significance testing for IMPACT evaluation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import structlog
from scipy import stats

if TYPE_CHECKING:
    import polars as pl
    from numpy.typing import NDArray

logger = structlog.stdlib.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WilcoxonResult:
    """Result of a Wilcoxon signed-rank test comparing two model runs (paired).

    Attributes:
        metric: Name of the compared metric (e.g., ndcg_at_10).
        statistic: Wilcoxon signed-rank statistic.
        p_value: One-sided p-value.
        effect_size_r: Standardized effect size r = |Z| / sqrt(n).
        n_total: Number of all valid pairs, including zero differences.
        n_nonzero: Number or pairs with d != 0.
        n_improved: Number of pairs with d > 0.
        n_worsened: Number of pairs with d < 0.
    """

    metric: str
    statistic: float
    p_value: float
    effect_size_r: float
    n_total: int  # all valid pairs (including zero differences)
    n_nonzero: int  # pairs with d != 0, used for the test
    n_improved: int  # pairs with d > 0
    n_worsened: int  # pairs with d < 0


def compare_models(
    raw_f_df: pl.DataFrame,
    raw_fplus_df: pl.DataFrame,
    metric: str = "ndcg_at_10",
) -> WilcoxonResult:
    """Compare two paired evaluation runs using the Wilcoxon signed-rank test.

    Both dataframes must contain raw per-sample metric values, one row per (uid,
    target_ts) test sample, as produced by
    `ImpactEvaluationPipeline.evaluate(collect_raw=True)`.

    Args:
        raw_f_df: Raw per-sample metrics for the baseline model f (pure SASRec).
        raw_fplus_df: Raw per-sample metrics for the extended model f+ (IMPACT, with
            0.0 < alpha < 1.0).
        metric: Name of the per-sample metric column to compare.

    Returns:
        WilcoxonResult with the test statistic, p-value, effect size, and the number
        of paired samples used.

    Raises:
        ValueError: If no valid paired samples remain after joining and dropping NaN
            values (e.g. due to mismatched (uid, target_ts) keys or the metric being
            undefined for all samples in a given scope).
    """
    merged_df = raw_fplus_df.join(
        raw_f_df, on=["uid", "target_ts"], suffix="_f", how="inner"
    )

    a = merged_df[metric].to_numpy()
    b = merged_df[f"{metric}_f"].to_numpy()

    # Drop pairs where the metric is undefined for either run (e.g. scoped metrics like
    # "ndcg_local_exp_at_10" where the sample has no items in that scope)
    valid_mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid_mask], b[valid_mask]

    if a.size == 0:
        raise ValueError(
            f"No valid paired samples for metric '{metric}' after dropping NaNs"
        )

    d: NDArray[np.floating] = a - b

    # Number of non-zero paired differences used by the Wilcoxon test
    n = int(np.count_nonzero(d))
    if n == 0:
        raise ValueError(
            f"All paired differences for metric '{metric}' are zero (test is undefined)"
        )

    logger.info(
        "Running Wilcoxon signed-rank test",
        metric=metric,
        raw_f_df_shape=raw_f_df.shape,
        raw_fplus_df_shape=raw_fplus_df.shape,
        n_pairs=merged_df.height,
        n_valid=a.size,
        n_nonzero=n,
    )

    # Alternative refers to the direction of the alternative hypothesis. "greater" tests
    # whether f+ outperforms f (median of f+ - f > 0).
    res = stats.wilcoxon(
        d,
        alternative="greater",
        zero_method="wilcox",
        method="approx",
    )
    z_stat = cast(float, res.zstatistic)
    effect_size_r = float(abs(z_stat) / np.sqrt(n))

    return WilcoxonResult(
        metric=metric,
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        effect_size_r=effect_size_r,
        n_total=a.size,
        n_nonzero=n,
        n_improved=int(np.count_nonzero(d > 0)),
        n_worsened=int(np.count_nonzero(d < 0)),
    )
