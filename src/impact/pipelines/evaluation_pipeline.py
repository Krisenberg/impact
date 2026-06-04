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
class EvaluationPipeline:
    """Late-fusion evaluation of SASRecNBR and ACT-R on held-out test data.

    Loads a trained SASRecNBR checkpoint, builds an ACT-R history index for test users,
    and scores both models against the full item catalog. Final scores are a weighted
    combination controlled by `ImpactConfig.alpha` (1 = SASRec only, 0 = ACT-R only).
    Global (vanilla), repetition-focused, and exploration-focused NDCG@K and Recall@K
    are reported for each configured k cutoff.
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
        and computes sample-weighted metrics across all configured cutoff levels.

        Args:
            split: Tensorized train/val/test temporal splits; only the test partition
                is used here.
            sasrec_state_dict_path: Path to the saved SASRecNBR checkpoint.

        Returns:
            Dictionary of ndcg_at_{k}, recall_at_{k}, ndcg_rep_at_{k},
            recall_rep_at_{k}, ndcg_exp_at_{k}, recall_exp_at_{k} for each k cutoff.
        """
        logger.info("Starting evaluation pipeline")

        test_loader = DataLoader[EvalSequenceSample](
            dataset=EvalSequenceDataset(split.test),
            batch_size=self._config.sasrec.batch_size,
            shuffle=False,
            collate_fn=get_eval_collate_fn(),  # No negative sampling
            num_workers=self._config.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sasrec = SASRecNBR(
            num_items=self._item_index.num_items,
            context_sessions=self._config.context_sessions,
            embedding_dim=self._config.sasrec.embedding_dim,
            num_heads=self._config.sasrec.num_heads,
            num_layers=self._config.sasrec.num_layers,
            dropout=self._config.sasrec.dropout_rate,
        )

        logger.info(
            "Loading best SASRecNBR weights",
            path=sasrec_state_dict_path.as_posix(),
        )
        sasrec.load_state_dict(torch.load(sasrec_state_dict_path, map_location=device))
        sasrec.to(device)
        sasrec.eval()

        actr_history_index = ACTRHistoryIndex(
            events_lf=self._events_lf,
            eval_uids=np.unique(split.test.uid).tolist(),
            item_index=self._item_index,
        )
        actr = ACTR(
            num_items=self._item_index.num_items,
            # decay=self._config.actr.decay,
            # organic_weight=self._config.actr.organic_weight,
            # autoplay_weight=self._config.actr.autoplay_weight,
            # eps=self._config.actr.eps,
        )

        ndcg_sums = dict.fromkeys(self._config.eval_ks, 0.0)
        recall_sums = dict.fromkeys(self._config.eval_ks, 0.0)
        ndcg_rep_sums = dict.fromkeys(self._config.eval_ks, 0.0)
        recall_rep_sums = dict.fromkeys(self._config.eval_ks, 0.0)
        ndcg_exp_sums = dict.fromkeys(self._config.eval_ks, 0.0)
        recall_exp_sums = dict.fromkeys(self._config.eval_ks, 0.0)
        total_samples = 0

        pbar = tqdm(
            iterable=cast(Iterator[EvalSequenceBatch], iter(test_loader)),
            desc="Evaluation",
            leave=False,
        )
        for batch in pbar:
            batch = batch.to(device)
            context = batch.context
            target = batch.positive
            batch_len = target.size(0)

            context_flat = context.flatten(start_dim=1)
            is_repeated = (target.unsqueeze(2) == context_flat.unsqueeze(1)).any(dim=-1)
            is_explored = (~is_repeated) & (target != 0)

            targets_rep = torch.where(is_repeated, target, 0)
            targets_exp = torch.where(is_explored, target, 0)

            # Get SASRecNBR logits
            sasrec_logits = sasrec.score_all_items(sasrec.encode(context))

            if self._config.alpha == 1.0:
                final_scores = sasrec_logits
                # del sasrec_logits
            else:
                # Create empty matrix of size [B, num_items + 1] for ACT-R scores
                actr_scores = torch.zeros_like(sasrec_logits)

                # Get batch samples metadata
                batch_uids = cast(list[int], batch.uid.tolist())
                batch_timestamps = cast(list[int], batch.target_ts.tolist())

                # Calculate the BLA for each row in the batch
                for b_idx in range(batch_len):
                    uid = batch_uids[b_idx]
                    t_cutoff = batch_timestamps[b_idx]
                    item_ids, time_deltas, is_organic = actr_history_index.query(
                        uid, t_cutoff
                    )

                    # Calculate BLA for the user (on the correct device)
                    user_bla = actr.score_all_items(
                        item_ids=item_ids.to(device),
                        time_deltas=time_deltas.to(device),
                        is_organic=is_organic.to(device),
                    )

                    # Update ACT-R scores
                    actr_scores[b_idx] = user_bla

                # Normalize both logits to the same scale and then combine (Late Fusion)
                sasrec_probs = torch.softmax(sasrec_logits, dim=-1)
                del sasrec_logits  # Free up memory

                actr_probs = torch.softmax(actr_scores, dim=-1)
                del actr_scores  # Free up memory

                sasrec_probs.mul_(self._config.alpha)
                sasrec_probs.add_(actr_probs, alpha=1.0 - self._config.alpha)
                del actr_probs

                final_scores = sasrec_probs

            # Calculate final metrics
            for k in self._config.eval_ks:
                # Global
                ndcg_sums[k] += compute_ndcg_at_k(final_scores, target, k=k) * batch_len
                recall_sums[k] += (
                    compute_recall_at_k(final_scores, target, k=k) * batch_len
                )

                # Repetition (P-Pop)
                ndcg_rep_sums[k] += (
                    compute_ndcg_at_k(final_scores, targets_rep, k=k) * batch_len
                )
                recall_rep_sums[k] += (
                    compute_recall_at_k(final_scores, targets_rep, k=k) * batch_len
                )

                # Exploration (Discovery)
                ndcg_exp_sums[k] += (
                    compute_ndcg_at_k(final_scores, targets_exp, k=k) * batch_len
                )
                recall_exp_sums[k] += (
                    compute_recall_at_k(final_scores, targets_exp, k=k) * batch_len
                )

            total_samples += batch_len

        metrics: dict[str, float] = {}
        for k in self._config.eval_ks:
            metrics[f"ndcg_at_{k}"] = ndcg_sums[k] / total_samples
            metrics[f"recall_at_{k}"] = recall_sums[k] / total_samples

            metrics[f"ndcg_rep_at_{k}"] = ndcg_rep_sums[k] / total_samples
            metrics[f"recall_rep_at_{k}"] = recall_rep_sums[k] / total_samples

            metrics[f"ndcg_exp_at_{k}"] = ndcg_exp_sums[k] / total_samples
            metrics[f"recall_exp_at_{k}"] = recall_exp_sums[k] / total_samples

        logger.info("Evaluation complete", metrics=metrics)
        return metrics
