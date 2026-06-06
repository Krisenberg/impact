from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast, final

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from impact.evaluation.metrics import compute_ndcg_at_k, compute_recall_at_k
from impact.models.actr import ACTR, ACTRHistoryIndex
from impact.models.sasrec import (
    EvalSequenceBatch,
    EvalSequenceDataset,
    EvalSequenceSample,
    SASRecNBR,
    get_eval_collate_fn,
)
from impact.utils import seed_worker

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

    from impact.config import ImpactConfig
    from impact.schemas import ItemIndex, TensorizedSplit

logger = structlog.stdlib.get_logger(__name__)


@final
class IMPACTEvaluationPipeline:
    """IMPACT core: late-fusion evaluation of SASRecNBR and ACT-R on held-out test data.

    Loads a trained SASRecNBR checkpoint, builds an ACT-R history index for test users,
    and scores both models against the full item catalog. Final scores are a weighted
    combination controlled by `ImpactConfig.eval.alpha` (1 = SASRec only, 0 = ACT-R
    only). Global (vanilla), repetition-focused, and exploration-focused NDCG@K and
    Recall@K are reported for each configured k cutoff.
    """

    def __init__(
        self,
        config: ImpactConfig,
        item_index: ItemIndex,
        events_lf: pl.LazyFrame,
    ) -> None:
        """Initialize the evaluation pipeline.

        Args:
            config: Complete IMPACT Framework configuration.
            item_index: Train-derived index mapping raw item (track) IDs to contiguous
                1-based indices.
            events_lf: LazyFrame of all interaction events; used to build the ACT-R
                history index.
        """
        self._config = config
        self._item_index = item_index
        self._events_lf = events_lf

    @torch.no_grad()
    def evaluate(
        self,
        split: TensorizedSplit,
        sasrec_state_dict_path: Path,
    ) -> dict[str, float]:
        """Evaluate the IMPACT model on the test split at multiple cutoff levels.

        Builds the ACT-R history index for test users, runs batched inference,
        and computes sample-weighted metrics at all configured cutoff levels.

        Args:
            split: Tensorized train/val/test temporal splits; only the test partition
                is used here.
            sasrec_state_dict_path: Path to the saved SASRecNBR checkpoint.

        Returns:
            Dictionary of ndcg_at_{k}, recall_at_{k}, and all local/global
            repetition- and exploration-focused variants for each configured k cutoff.
        """
        logger.info("Starting evaluation pipeline")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha = self._config.eval.alpha

        sasrec = self._load_sasrec(sasrec_state_dict_path, device)

        actr_history_index = ACTRHistoryIndex(
            events_lf=self._events_lf,
            eval_uids=np.unique(split.test.uid).tolist(),
            item_index=self._item_index,
        )
        actr = ACTR(
            num_items=self._item_index.num_items,
            decay=self._config.actr.decay,
            organic_weight=self._config.actr.organic_weight,
            autoplay_weight=self._config.actr.autoplay_weight,
            eps=self._config.actr.eps,
        )

        metric_prefixes = (
            "ndcg",
            "recall",
            "ndcg_loc_rep",
            "recall_loc_rep",
            "ndcg_loc_exp",
            "recall_loc_exp",
            "ndcg_glb_rep",
            "recall_glb_rep",
            "ndcg_glb_exp",
            "recall_glb_exp",
        )

        metric_names: list[str] = [
            f"{prefix}_at_{k}"
            for k in self._config.eval.k_cutoffs
            for prefix in metric_prefixes
        ]
        batch_metric_sums: dict[str, float] = dict.fromkeys(metric_names, 0.0)
        total_counts: dict[str, int] = dict.fromkeys(
            ("global", "loc_rep", "loc_exp", "glb_rep", "glb_exp"), 0
        )

        test_loader = DataLoader[EvalSequenceSample](
            dataset=EvalSequenceDataset(split.test),
            batch_size=self._config.eval.batch_size,
            shuffle=False,
            collate_fn=get_eval_collate_fn(),  # No negative sampling
            num_workers=self._config.system.num_workers,
            persistent_workers=self._config.system.num_workers > 0,
            worker_init_fn=seed_worker,
            pin_memory=torch.cuda.is_available(),
        )

        iterator = cast(Iterator[EvalSequenceBatch], iter(test_loader))
        for batch in tqdm(iterator, desc="Evaluation", leave=False):
            batch = batch.to(device)

            sasrec_logits = sasrec.score_all_items(sasrec.encode(context=batch.context))

            # Single pass: compute global-repetition mask and optionally BLA scores
            is_global_repeated, actr_scores = self._score_actr_batch(
                batch=batch,
                actr=actr,
                actr_history_index=actr_history_index,
                device=device,
                compute_bla=alpha != 1.0,
            )

            # Late fusion
            if alpha == 1.0:
                final_scores = sasrec_logits
            else:
                if actr_scores is None:
                    raise ValueError("ACT-R scores must be computed when alpha != 1.0")

                # Normalize both logits to the same scale and then combine (late fusion)
                sasrec_probs = torch.softmax(sasrec_logits, dim=-1)
                del sasrec_logits  # Free up memory

                actr_probs = torch.softmax(actr_scores, dim=-1)
                del actr_scores  # Free up memory

                # final_scores = alpha * sasrec_probs + (1 - alpha) * actr_probs
                final_scores = sasrec_probs.mul_(alpha).add_(
                    actr_probs, alpha=1.0 - alpha
                )
                del actr_probs

            batch_metrics, counts = self._compute_batch_metrics(
                final_scores=final_scores,
                targets=batch.positive,
                context=batch.context,
                is_global_repeated=is_global_repeated,
            )

            for key, value in batch_metrics.items():
                batch_metric_sums[key] += value
            for key, value in counts.items():
                total_counts[key] += value

        logger.info("Aggregating metrics across batches")

        count_key_by_prefix: dict[str, str] = {
            "ndcg": "global",
            "recall": "global",
            "ndcg_loc_rep": "loc_rep",
            "recall_loc_rep": "loc_rep",
            "ndcg_loc_exp": "loc_exp",
            "recall_loc_exp": "loc_exp",
            "ndcg_glb_rep": "glb_rep",
            "recall_glb_rep": "glb_rep",
            "ndcg_glb_exp": "glb_exp",
            "recall_glb_exp": "glb_exp",
        }

        metrics: dict[str, float] = {
            f"{prefix}_at_{k}": (
                batch_metric_sums[f"{prefix}_at_{k}"]
                / max(total_counts[count_key], 1)  # guard against empty split
            )
            for k in self._config.eval.k_cutoffs
            for prefix, count_key in count_key_by_prefix.items()
        }

        logger.info("Evaluation complete")
        return metrics

    def _load_sasrec(self, state_dict_path: Path, device: torch.device) -> SASRecNBR:
        """Load SASRecNBR from a checkpoint and set the model to eval mode."""
        sasrec = SASRecNBR(
            num_items=self._item_index.num_items,
            context_sessions=self._config.sequence.context_sessions,
            embedding_dim=self._config.sasrec.embedding_dim,
            num_heads=self._config.sasrec.num_heads,
            num_layers=self._config.sasrec.num_layers,
            dropout=self._config.sasrec.dropout_rate,
        )

        logger.info("Loading best SASRecNBR weights", path=state_dict_path.as_posix())
        sasrec.load_state_dict(torch.load(state_dict_path, map_location=device))
        sasrec.to(device)
        sasrec.eval()

        return sasrec

    def _score_actr_batch(
        self,
        batch: EvalSequenceBatch,
        actr: ACTR,
        actr_history_index: ACTRHistoryIndex,
        device: torch.device,
        *,
        compute_bla: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Resolve ACT-R history for each user and build per-batch scoring artifacts.

        The bottleneck of the ACT-R scoring is querying the ACT-R history index.
        A single pass per batch resolves both the global-repetition mask (which is
        always required regardless of alpha) and the BLA score matrix (required only
        when alpha < 1.0), avoiding redundant traversals.

        Args:
            batch: Current evaluation batch.
            actr: Instantiated ACTR model.
            actr_history_index: Pre-indexed event histories of the evaluation users.
            device: Target compute device.
            compute_bla: When False (i.e. pure SASRec, alpha == 1.0), skip BLA
                computation, actr_scores in the return tuple will be None then.

        Returns:
            is_global_repeated: Bool tensor of shape [B, K]. True where a target item
                appears anywhere in the user's global history before the eval cutoff.
            actr_scores: Float32 tensor of shape [B, num_items + 1] containing BLA
                scores, or None when compute_bla is False.
        """
        batch_size = batch.positive.size(0)
        batch_uids = cast(list[int], batch.uid.tolist())
        batch_timestamps = cast(list[int], batch.target_ts.tolist())

        is_global_repeated = torch.zeros_like(batch.positive, dtype=torch.bool)

        # Create empty matrix of shape [B, num_items + 1] for ACT-R scores (if required)
        actr_scores: torch.Tensor | None = (
            torch.empty(
                batch_size,
                self._item_index.num_items + 1,
                dtype=torch.float32,
                device=device,
            )
            if compute_bla
            else None
        )

        # Calculate the BLA for each row in the batch
        for b_idx in range(batch_size):
            item_ids, time_deltas, is_organic = actr_history_index.query(
                uid=batch_uids[b_idx],
                t_cutoff=batch_timestamps[b_idx],
            )
            item_ids = item_ids.to(device)
            is_global_repeated[b_idx] = torch.isin(batch.positive[b_idx], item_ids)

            # Update ACT-R scores (if required)
            if actr_scores is not None:
                actr_scores[b_idx] = actr.score_all_items(
                    item_ids=item_ids,
                    time_deltas=time_deltas.to(device),
                    is_organic=is_organic.to(device),
                )

        return is_global_repeated, actr_scores

    def _compute_batch_metrics(
        self,
        final_scores: torch.Tensor,
        targets: torch.Tensor,
        context: torch.Tensor,
        is_global_repeated: torch.Tensor,
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Compute weighted metric sums for one batch across all k cutoffs.

        Derives repetition and exploration target subsets for both local (relative
        to the L context sessions) and global (relative to the full user history)
        scopes. Returns unnormalized sums (value * count); division by total counts
        should happen in evaluate().

        Args:
            final_scores: [B, num_items + 1] Logit scores for the entire item catalog.
            targets: [B, K] Ground-truth target sessions.
            context: [B, L, K] The L context sessions for local scope evaluation.
            is_global_repeated: [B, K] Boolean mask indicating if target items exist in
                the user's global history.

        Returns:
            Tuple of (metric_sums, counts) where metric_sums contains weighted sums
            keyed by metric name, and counts contains the number of valid samples
            for global, loc_rep, loc_exp, glb_rep, and glb_exp splits.
        """
        # Local scope
        context_flat = context.flatten(start_dim=1)  # [B, L * K]

        # Mask padding (unknown items)
        context_valid = context_flat.masked_fill(context_flat == 0, -1)

        # [B, K, 1] == [B, 1, L*K] --broadcast--> [B, K, L*K] -> [B, K]
        _is_loc_rep = (targets.unsqueeze(2) == context_valid.unsqueeze(1)).any(dim=-1)
        is_loc_rep = _is_loc_rep & (targets != 0)  # [B, K]
        is_loc_exp = (~_is_loc_rep) & (targets != 0)  # [B, K]

        # Global scope
        is_glb_rep = is_global_repeated & (targets != 0)
        is_glb_exp = (~is_global_repeated) & (targets != 0)

        # Target subsets
        targets_loc_rep = targets.masked_fill(~is_loc_rep, 0)
        targets_loc_exp = targets.masked_fill(~is_loc_exp, 0)
        targets_glb_rep = targets.masked_fill(~is_glb_rep, 0)
        targets_glb_exp = targets.masked_fill(~is_glb_exp, 0)

        counts = {
            "global": targets.size(0),
            "loc_rep": int((targets_loc_rep != 0).any(dim=-1).sum().item()),
            "loc_exp": int((targets_loc_exp != 0).any(dim=-1).sum().item()),
            "glb_rep": int((targets_glb_rep != 0).any(dim=-1).sum().item()),
            "glb_exp": int((targets_glb_exp != 0).any(dim=-1).sum().item()),
        }

        batch_metrics: dict[str, float] = {}

        for k in self._config.eval.k_cutoffs:
            # General (global)
            batch_metrics[f"ndcg_at_{k}"] = (
                compute_ndcg_at_k(final_scores, targets, k) * counts["global"]
            )
            batch_metrics[f"recall_at_{k}"] = (
                compute_recall_at_k(final_scores, targets, k) * counts["global"]
            )

            # Local
            batch_metrics[f"ndcg_loc_rep_at_{k}"] = (
                compute_ndcg_at_k(final_scores, targets_loc_rep, k) * counts["loc_rep"]
            )
            batch_metrics[f"recall_loc_rep_at_{k}"] = (
                compute_recall_at_k(final_scores, targets_loc_rep, k)
                * counts["loc_rep"]
            )
            batch_metrics[f"ndcg_loc_exp_at_{k}"] = (
                compute_ndcg_at_k(final_scores, targets_loc_exp, k) * counts["loc_exp"]
            )
            batch_metrics[f"recall_loc_exp_at_{k}"] = (
                compute_recall_at_k(final_scores, targets_loc_exp, k)
                * counts["loc_exp"]
            )

            # Global
            batch_metrics[f"ndcg_glb_rep_at_{k}"] = (
                compute_ndcg_at_k(final_scores, targets_glb_rep, k) * counts["glb_rep"]
            )
            batch_metrics[f"recall_glb_rep_at_{k}"] = (
                compute_recall_at_k(final_scores, targets_glb_rep, k)
                * counts["glb_rep"]
            )
            batch_metrics[f"ndcg_glb_exp_at_{k}"] = (
                compute_ndcg_at_k(final_scores, targets_glb_exp, k) * counts["glb_exp"]
            )
            batch_metrics[f"recall_glb_exp_at_{k}"] = (
                compute_recall_at_k(final_scores, targets_glb_exp, k)
                * counts["glb_exp"]
            )

        return batch_metrics, counts
