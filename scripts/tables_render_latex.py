from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import structlog

from .tables_data import DataFunnelStatsSchema

if TYPE_CHECKING:
    from .types import Language

logger = structlog.stdlib.get_logger(__name__)


def render_data_funnel_stats(
    data_funnel_stats_df: pl.DataFrame,
    lang: Language = "en",
) -> str:
    """Generate a LaTeX table of record counts across data preparation stages."""
    translations: dict[Language, dict[str, str]] = {
        "pl": {
            "stage": "Etap przygotowania danych",
            "all_events": "Wszystkie interakcje $E$",
            "valid_listen_events": "Odsłuchy $E_{\\text{listen}}$",
            "listening_sessions": "Sesje odsłuchowe $S_{\\text{ses}}$",
            "sequences_of_sessions": "Sekwencje $S_{\\text{sek}}$",
            "train_split": "Zbiór treningowy $S_{\\text{sek | train}}$",
            "val_split": "Zbiór walidacyjny $S_{\\text{sek | val}}$",
            "test_split": "Zbiór testowy $S_{\\text{sek | test}}$",
            "caption": "Liczba rekordów na poszczególnych etapach przygotowania danych dla obu wariantów zbioru Yambda.",  # noqa: E501
        },
        "en": {
            "stage": "Data preparation stage",
            "all_events": "All events $E$",
            "valid_listen_events": "Valid listen events $E_{\\text{listen}}$",
            "listening_sessions": "Listening sessions $S_{\\text{ses}}$",
            "sequences_of_sessions": "Sequences $S_{\\text{sek}}$",
            "train_split": "Train split $S_{\\text{sek | train}}$",
            "val_split": "Validation split $S_{\\text{sek | val}}$",
            "test_split": "Test split $S_{\\text{sek | test}}$",
            "caption": "The number of records at each stage of data preparation for both variants of the Yambda dataset.",  # noqa: E501
        },
    }
    t = translations[lang]

    table_rows: list[str] = [
        "all_events",
        "valid_listen_events",
        "listening_sessions",
        "sequences_of_sessions",
        "train_split",
        "val_split",
        "test_split",
    ]

    rows: list[dict[str, str | int]] = [
        {
            "dataset": row[DataFunnelStatsSchema.dataset_variant],
            **{header: _format_num(row[header]) for header in table_rows},
        }
        for row in data_funnel_stats_df.iter_rows(named=True)
    ]
    df = pl.DataFrame(
        {
            t["stage"]: [*[t[header] for header in table_rows]],
            **{r["dataset"]: [*[r[header] for header in table_rows]] for r in rows},
        }
    )

    return (
        df.to_pandas()
        .style.hide(axis="index")
        # Generate LaTeX table using Pandas Styler
        .to_latex(
            hrules=True,
            environment="table",
            position="!htb",
            position_float="centering",
            label="tab:data_funnel_stats",
            caption=" ".join(t["caption"].split()),
            column_format="lrr",  # 'l' for the text, 'r' for the numbers
        )
    )


def _format_num(n: int) -> str:
    """Format integer with space as thousands separator."""
    return f"{n:,}".replace(",", "\\,")
