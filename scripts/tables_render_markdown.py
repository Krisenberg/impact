from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import structlog

from .tables_data import DataFunnelStatsSchema

if TYPE_CHECKING:
    from pandera.typing.polars import DataFrame

    from .types import Language

logger = structlog.stdlib.get_logger(__name__)


def render_data_funnel_stats(
    data_funnel_stats_df: DataFrame[DataFunnelStatsSchema],
    lang: Language = "en",
) -> str:
    """Generate a Markdown table of record counts across data preparation stages."""
    translations: dict[Language, dict[str, str]] = {
        "pl": {
            "stage": "Etap przygotowania danych",
            "all_events": "Wszystkie interakcje **E**",
            "valid_listen_events": "Odsłuchy **E_listen**",
            "listening_sessions": "Sesje odsłuchowe **S_ses**",
            "sequences_of_sessions": "Sekwencje **S_sek**",
            "train_split": "Zbiór treningowy **S_sek_train**",
            "val_split": "Zbiór walidacyjny **S_sek_val**",
            "test_split": "Zbiór testowy **S_sek_test**",
        },
        "en": {
            "stage": "Data preparation stage",
            "all_events": "All events **E**",
            "valid_listen_events": "Valid listen events **E_listen**",
            "listening_sessions": "Listening sessions **S_ses**",
            "sequences_of_sessions": "Sequences **S_sek**",
            "train_split": "Train split **S_sek_train**",
            "val_split": "Validation split **S_sek_val**",
            "test_split": "Test split **S_sek_test**",
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
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        fmt_str_lengths=99,
    ):
        return str(df)


def _format_num(n: int) -> str:
    return f"{n:,}".replace(",", " ")
