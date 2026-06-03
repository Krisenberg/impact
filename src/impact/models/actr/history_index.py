from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog
import torch

if TYPE_CHECKING:
    from impact.schemas import ItemIndex

logger = structlog.stdlib.get_logger(__name__)


class ACTRHistoryIndex:
    """In-memory index of user event histories for ACT-R scoring.

    Indexed once at construction from the raw event log; individual user histories are
    retrieved via `query(uid, t_cutoff)`. Only eval-split users are indexed to minimize
    RAM usage.
    """

    def __init__(
        self,
        events_lf: pl.LazyFrame,
        eval_uids: list[int] | pl.Series,
        item_index: ItemIndex,
    ) -> None:
        """Index event histories for the given evaluation users.

        Filters down to listen events, maps item IDs to the contiguous training items
        index space (dropping cold-start items), and stores per-user arrays sorted by
        timestamp for efficient cutoff slicing.

        Args:
            events_lf: LazyFrame of all interaction events.
            eval_uids: User IDs in the evaluation split; only these are indexed.
            item_index: Item-to-index mapping for ID translation derived from the
                training split.
        """
        logger.info(
            "Filtering, mapping, and indexing raw event history for eval users...",
            num_eval_users=len(eval_uids),
        )

        # Filter and map item_id -> idx
        eval_history_df = (
            events_lf.filter(
                pl.col("event_type") == "listen",
                pl.col("uid").is_in(eval_uids),
            )
            .join(item_index.table.lf, on="item_id", how="inner")
            .select(["uid", "idx", "timestamp", "is_organic"])
            .collect()
        )
        # TODO: remove / add logging
        logger.warning(
            f"eval_history_df size: {eval_history_df.estimated_size() / 1024**2:.2f} MB"
        )
        # if len(eval_uids) > 50_000:
        #     msg = (
        #         "Large eval_uids set passed to ACTRHistoryIndex, "
        #         "ensure this is intentional to avoid RAM exhaustion."
        #     )
        #     logger.warning(msg, num_eval_users=len(eval_uids))

        # Build a dictionary of NumPy arrays for O(1) lookups
        self._history: dict[int, dict[str, np.ndarray]] = {}
        for uid, df in eval_history_df.partition_by("uid", as_dict=True).items():
            df = df.sort("timestamp")
            self._history[uid[0]] = {
                "item_ids": df["idx"].to_numpy(),
                "timestamps": df["timestamp"].to_numpy(),
                "is_organic": df["is_organic"].to_numpy(),
            }

        logger.info("ACT-R History indexing complete", indexed_users=len(self._history))

    def query(
        self,
        uid: int,
        t_cutoff: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ACT-R input tensors for user events strictly before t_cutoff.

        Args:
            uid: User to query.
            t_cutoff: Exclusive timestamp cutoff, typically a timestamp of the first
                event of the target session.

        Returns:
            (item_ids, time_deltas, is_organic) as 1D tensors. All three are empty if
            the user has no history before the cutoff.
        """
        empty_history = (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.bool),
        )

        user_data = self._history.get(uid)
        if user_data is None:
            return empty_history

        # Get events strictly before the cutoff (target session start)
        timestamps = user_data["timestamps"]
        cutoff_idx = np.searchsorted(timestamps, t_cutoff, side="left")
        if cutoff_idx == 0:
            return empty_history

        item_ids_raw = user_data["item_ids"][:cutoff_idx]
        timestamps_raw = timestamps[:cutoff_idx]
        is_organic_raw = user_data["is_organic"][:cutoff_idx]

        # Delta_t = t_ref - t_k
        time_deltas_raw = (t_cutoff - timestamps_raw).astype(np.float32)

        # Avoid time_delta = 0 due to 5-second quantization
        time_deltas_raw = np.clip(time_deltas_raw, a_min=1.0, a_max=None)

        # Convert directly to PyTorch tensors
        item_ids = torch.from_numpy(item_ids_raw).long()
        time_deltas = torch.from_numpy(time_deltas_raw).float()
        is_organic = torch.from_numpy(is_organic_raw).bool()

        return item_ids, time_deltas, is_organic
