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
            Dictionary of ndcg_at_{k}, recall_at_{k}, ndcg_rep_at_{k},
            recall_rep_at_{k}, ndcg_exp_at_{k}, recall_exp_at_{k} for each k cutoff.
        """
        logger.info("Starting evaluation pipeline")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            "ndcg_rep",
            "recall_rep",
            "ndcg_exp",
            "recall_exp",
        )
        metric_names: list[str] = [
            f"{prefix}_at_{k}"
            for k in self._config.eval.k_cutoffs
            for prefix in metric_prefixes
        ]

        total_samples = 0
        total_rep_samples = 0
        total_exp_samples = 0
        batch_metric_sums: dict[str, float] = dict.fromkeys(metric_names, 0.0)

        test_loader = DataLoader[EvalSequenceSample](
            dataset=EvalSequenceDataset(split.test),
            batch_size=self._config.eval.batch_size,
            shuffle=False,
            collate_fn=get_eval_collate_fn(),  # No negative sampling
            num_workers=self._config.system.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        iterator = cast(Iterator[EvalSequenceBatch], iter(test_loader))
        for batch in tqdm(iterator, desc="Evaluation", leave=False):
            batch = batch.to(device)

            # Get SASRecNBR logits
            logger.info("SASRecNBR scoring...")
            sasrec_logits = sasrec.score_all_items(sasrec.encode(context=batch.context))

            if self._config.eval.alpha == 1.0:
                final_scores = sasrec_logits
            else:
                # Get ACT-R logits
                logger.info("ACT-R scoring...")
                actr_scores = self._get_actr_scores(
                    batch, actr, actr_history_index, device
                )

                # Normalize both logits to the same scale and then combine (late fusion)
                sasrec_probs = torch.softmax(sasrec_logits, dim=-1)
                del sasrec_logits  # Free up memory

                actr_probs = torch.softmax(actr_scores, dim=-1)
                del actr_scores  # Free up memory

                sasrec_probs.mul_(self._config.eval.alpha)
                sasrec_probs.add_(actr_probs, alpha=1.0 - self._config.eval.alpha)
                del actr_probs

                final_scores = sasrec_probs

            logger.info("Computing metrics (batch level)...")
            batch_metrics, num_rep, num_exp = self._compute_batch_metrics(
                final_scores=final_scores,
                targets=batch.positive,
                context=batch.context,
            )
            for key, value in batch_metrics.items():
                batch_metric_sums[key] += value

            total_samples += batch.positive.size(0)
            total_rep_samples += num_rep
            total_exp_samples += num_exp

        logger.info("Computing final (aggregated) metrics...")
        metric_prefixes = ("ndcg", "recall")
        splits = {
            "": max(total_samples, 1),  # max(..., 1) to avoid division by zero
            "_rep": max(total_rep_samples, 1),
            "_exp": max(total_exp_samples, 1),
        }

        metrics: dict[str, float] = {}
        for k in self._config.eval.k_cutoffs:
            for prefix in metric_prefixes:
                for suffix, denom in splits.items():
                    key = f"{prefix}{suffix}_at_{k}"
                    metrics[key] = batch_metric_sums[key] / denom

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

    def _get_actr_scores(
        self,
        batch: EvalSequenceBatch,
        actr: ACTR,
        actr_history_index: ACTRHistoryIndex,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute per-user ACT-R scores and return the [B, num_items + 1] matrix."""
        batch_size = batch.positive.size(0)

        # Create empty matrix of shape [B, num_items + 1] for ACT-R scores
        actr_scores = torch.empty(
            batch_size,
            self._item_index.num_items + 1,
            dtype=torch.float32,
            device=device,
        )

        # Get batch samples metadata
        batch_uids = cast(list[int], batch.uid.tolist())
        batch_timestamps = cast(list[int], batch.target_ts.tolist())

        # Calculate the BLA for each row in the batch
        for b_idx in range(batch_size):
            uid = batch_uids[b_idx]
            t_cutoff = batch_timestamps[b_idx]
            item_ids, time_deltas, is_organic = actr_history_index.query(uid, t_cutoff)

            # Calculate BLA for the user (on the correct device)
            user_bla = actr.score_all_items(
                item_ids=item_ids.to(device),
                time_deltas=time_deltas.to(device),
                is_organic=is_organic.to(device),
            )

            # Update ACT-R scores
            actr_scores[b_idx] = user_bla

        return actr_scores

    def _compute_batch_metrics(
        self,
        final_scores: torch.Tensor,
        targets: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[dict[str, float], int, int]:
        """Compute weighted metric sums for one batch across all k cutoffs.

        Derives repetition and exploration target subsets internally from targets
        and context. Returns unnormalized sums (value * count); division by total
        counts happens in evaluate().

        Returns:
            Tuple of (metric_sums, num_rep, num_exp) where metric_sums contains
            weighted sums keyed by metric name, num_rep and num_exp are the numbers of
            samples with at least one repetition/exploration target in the batch.
        """
        context_flat = context.flatten(start_dim=1)  # [B, L*K]

        # Mask padding (unknown items)
        context_valid = context_flat.masked_fill(context_flat == 0, -1)

        # [B, K, 1] == [B, 1, L*K] --broadcast--> [B, K, L*K] -> [B, K]
        is_repeated = (targets.unsqueeze(2) == context_valid.unsqueeze(1)).any(
            dim=-1
        ) & (targets != 0)
        is_explored = (~is_repeated) & (targets != 0)  # [B, K]

        targets_rep = targets.masked_fill(~is_repeated, 0)
        targets_exp = targets.masked_fill(~is_explored, 0)

        num_rep = int((targets_rep != 0).any(dim=-1).sum().item())
        num_exp = int((targets_exp != 0).any(dim=-1).sum().item())

        batch_len = targets.size(0)
        batch_metrics: dict[str, float] = {}

        for k in self._config.eval.k_cutoffs:
            # Global
            batch_metrics[f"ndcg_at_{k}"] = (
                compute_ndcg_at_k(final_scores, targets, k=k) * batch_len
            )
            batch_metrics[f"recall_at_{k}"] = (
                compute_recall_at_k(final_scores, targets, k=k) * batch_len
            )

            # Repetition (P-Pop)
            batch_metrics[f"ndcg_rep_at_{k}"] = (
                compute_ndcg_at_k(final_scores, targets_rep, k=k) * num_rep
            )
            batch_metrics[f"recall_rep_at_{k}"] = (
                compute_recall_at_k(final_scores, targets_rep, k=k) * num_rep
            )

            # Exploration (Discovery)
            batch_metrics[f"ndcg_exp_at_{k}"] = (
                compute_ndcg_at_k(final_scores, targets_exp, k=k) * num_exp
            )
            batch_metrics[f"recall_exp_at_{k}"] = (
                compute_recall_at_k(final_scores, targets_exp, k=k) * num_exp
            )

        return batch_metrics, num_rep, num_exp
