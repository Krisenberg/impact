"""Orchestrator for the final model evaluation."""

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
    """Orchestrates final model evaluation on held-out test data.

    Loads a trained SASRecNBR checkpoint, runs inference over the test split,
    and returns sample-weighted NDCG@K and Recall@K. Structured for extension with late
    fusion scoring when additional models are introduced.
    """

    def __init__(
        self,
        config: ImpactConfig,
        item_index: ItemIndex,
        events_lf: pl.LazyFrame,
    ) -> None:
        """Initialize the evaluation pipeline environment.

        Args:
            config: Complete IMPACT Framework configuration.
            item_index: ...
            events_lf: ...
        """
        self.config = config
        self.item_index = item_index
        self.events_lf = events_lf

    @torch.no_grad()
    def evaluate(
        self,
        split: TensorizedSplit,
        sasrec_state_dict_path: Path,
    ) -> dict[str, float]:
        """Run evaluation on the test split for multiple K cut-off levels.

        Scores are computed against the full item catalog. Metrics are sample-weighted
        averages across all batches.

        Args:
            split: The memory-mapped, tensorized train/val/test temporal splits; test
                split is used for evaluation, num_items for model and dataset
                initialisation.
            sasrec_state_dict_path: Path to the saved SASRecNBR model state dict.

        Returns:
            A dictionary of NDCG@K and Recall@K metrics.
        """
        logger.info("Starting evaluation pipeline")

        test_loader = DataLoader[EvalSequenceSample](
            dataset=EvalSequenceDataset(split.test),
            batch_size=self.config.sasrec.batch_size,
            shuffle=False,
            collate_fn=get_eval_collate_fn(),  # No negative sampling
            num_workers=self.config.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sasrec = SASRecNBR(
            num_items=self.item_index.num_items,
            context_sessions=self.config.context_sessions,
            embedding_dim=self.config.sasrec.embedding_dim,
            num_heads=self.config.sasrec.num_heads,
            num_layers=self.config.sasrec.num_layers,
            dropout=self.config.sasrec.dropout_rate,
        )

        logger.info(
            "Loading best SASRecNBR weights",
            path=sasrec_state_dict_path.as_posix(),
        )
        sasrec.load_state_dict(torch.load(sasrec_state_dict_path, map_location=device))
        sasrec.to(device)
        sasrec.eval()

        actr_history_index = ACTRHistoryIndex(
            events_lf=self.events_lf,
            eval_uids=np.unique(split.test.uid).tolist(),
            item_index=self.item_index,
        )
        actr = ACTR(
            num_items=self.item_index.num_items,
            # decay=self.config.actr.decay,
            # organic_weight=self.config.actr.organic_weight,
            # autoplay_weight=self.config.actr.autoplay_weight,
            # eps=self.config.actr.eps,
        )

        ndcg_sums = dict.fromkeys(self.config.eval_ks, 0.0)
        recall_sums = dict.fromkeys(self.config.eval_ks, 0.0)
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

            # Get SASRecNBR logits
            sasrec_logits = sasrec.score_all_items(sasrec.encode(context))

            # TODO: Late Fusion
            if self.config.alpha == 1.0:
                final_scores = sasrec_logits
            else:
                # Create empty matrix of size [B, num_items + 1] for ACT-R scores
                actr_scores = torch.zeros_like(sasrec_logits, device=device)

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
                actr_probs = torch.softmax(actr_scores, dim=-1)

                final_scores = (
                    self.config.alpha * sasrec_probs
                    + (1.0 - self.config.alpha) * actr_probs
                )

            # Calculate final metrics
            for k in self.config.eval_ks:
                batch_ndcg = compute_ndcg_at_k(final_scores, target, k=k)
                batch_recall = compute_recall_at_k(final_scores, target, k=k)

                ndcg_sums[k] += batch_ndcg * batch_len
                recall_sums[k] += batch_recall * batch_len

            total_samples += batch_len

            # final_scores = sasrec_logits

            # # Calculate final metrics
            # for k in self.config.eval_ks:
            #     batch_ndcg = compute_ndcg_at_k(final_scores, target, k=k)
            #     batch_recall = compute_recall_at_k(final_scores, target, k=k)

            #     ndcg_sums[k] += batch_ndcg * batch_len
            #     recall_sums[k] += batch_recall * batch_len

            # total_samples += batch_len

        metrics: dict[str, float] = {}
        for k in self.config.eval_ks:
            metrics[f"ndcg_at_{k}"] = ndcg_sums[k] / total_samples
            metrics[f"recall_at_{k}"] = recall_sums[k] / total_samples

        logger.info("Evaluation complete", metrics=metrics)
        return metrics
