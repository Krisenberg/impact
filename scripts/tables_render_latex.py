"""Module for creating tables in LaTeX format that are presented in the thesis."""

from __future__ import annotations

import string
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
            "reproducibility_note": (
                "Konfiguracja IMPACT Framework: \\texttt{$display_id}~"
                "(\\cref{$appendix_label})."
            ),
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
            "reproducibility_note": (
                "The IMPACT Framework configuration: \\texttt{$display_id}~"
                "(\\cref{$appendix_label})."
            ),
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

    rows: list[dict[str, str | int]] = []
    footnotes: list[str] = []
    markers: str = string.ascii_lowercase

    for i, row in enumerate(data_funnel_stats_df.iter_rows(named=True)):
        marker = markers[i]
        tnote = f"\\tnote{{\\hyperlink{{note:{marker}}}{{{marker}}}}}"

        config_display_id = row[DataFunnelStatsSchema.config_display_id]
        rows.append(
            {
                "dataset": f"{row[DataFunnelStatsSchema.dataset_variant]}{tnote}",
                **{header: _format_num(row[header]) for header in table_rows},
            }
        )

        hypertarget = f"\\hypertarget{{note:{marker}}}{{{marker}}}"
        note_text = string.Template(t["reproducibility_note"]).substitute(
            display_id=config_display_id,
            appendix_label=f"appendix:impactconfig:{config_display_id}",
        )
        footnotes.append(f"\\item[{hypertarget}]: {note_text}")

    df = pl.DataFrame(
        {
            t["stage"]: [*[t[header] for header in table_rows]],
            **{r["dataset"]: [*[r[header] for header in table_rows]] for r in rows},
        }
    )

    tablenotes_latex = (
        "\\begin{tablenotes}[flushleft]\n"
        "\\footnotesize\n"
        f"{'\n'.join(footnotes)}\n"
        "\\end{tablenotes}\n"
    )

    return (
        df.to_pandas()
        .style.hide(axis="index")
        # Generate LaTeX table using Pandas Styler
        .to_latex(
            hrules=True,
            environment="threeparttable",  # To allow table footnotes
            position="H",
            position_float="centering",
            label="tab:data_funnel_stats",
            caption=" ".join(t["caption"].split()),
            column_format="lrr",  # 'l' for the text, 'r' for the numbers
        )
        .replace("\\end{tabular}\n", f"\\end{{tabular}}\n{tablenotes_latex}")
    )


def _format_num(n: int) -> str:
    """Format integer with space as thousands separator."""
    return f"{n:,}".replace(",", " ")
