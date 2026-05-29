"""Orchestrator for the final model evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast, final

import structlog
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from impact.evaluation.metrics import compute_ndcg_at_k, compute_recall_at_k
from impact.models.sasrec import (
    SASRecNBR,
    SequenceBatch,
    SequenceDataset,
    get_eval_collate_fn,
)
from impact.models.sasrec.dataset import SequenceSample

if TYPE_CHECKING:
    from pathlib import Path

    from impact.config import ImpactConfig
    from impact.schemas import MemmapSplit

logger = structlog.stdlib.get_logger(__name__)


@final
class EvaluationPipeline:
    """Orchestrates final model evaluation on held-out test data.

    Loads a trained SASRecNBR checkpoint, runs inference over the test split,
    and returns sample-weighted NDCG@K and Recall@K. Structured for extension with late
    fusion scoring when additional models are introduced.
    """

    def __init__(self, config: ImpactConfig) -> None:
        """Initialize the evaluation pipeline.

        Args:
            config: Complete IMPACT Framework configuration.
        """
        self._config = config

    @torch.no_grad()
    def evaluate(
        self,
        split: MemmapSplit,
        sasrec_state_dict_path: Path,
        ks: list[int] | None = None,
    ) -> dict[str, float]:
        """Run evaluation on the test split for multiple K cut-off levels.

        Scores are computed against the full item catalog. Metrics are sample-weighted
        averages across all batches.

        Args:
            split: The memory-mapped, tensorized train/val/test temporal splits; test
                split is used for evaluation, num_items for model and dataset
                initialisation.
            sasrec_state_dict_path: Path to the saved SASRecNBR model state dict.
            ks: List of K values (cutoffs) for evaluation. Defaults to [5, 10, 40, 100].

        Returns:
            A dictionary of NDCG@K and Recall@K metrics.
        """
        ks = ks or [5, 10, 40, 100]
        logger.info(
            "Starting evaluation pipeline",
            # alpha=self.config.alpha,
        )

        test_ds = SequenceDataset(split.test, split.num_items)

        test_loader = DataLoader[SequenceSample](
            test_ds,
            batch_size=self._config.sasrec.batch_size,
            shuffle=False,
            collate_fn=get_eval_collate_fn(),  # No negative sampling
            num_workers=4,
            pin_memory=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SASRecNBR(
            num_items=split.num_items,
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
        model.load_state_dict(torch.load(sasrec_state_dict_path, map_location=device))
        model.to(device)
        model.eval()

        ndcg_sums = dict.fromkeys(ks, 0.0)
        recall_sums = dict.fromkeys(ks, 0.0)
        total_samples = 0

        pbar = tqdm(
            iterable=cast(Iterator[SequenceBatch], iter(test_loader)),
            desc="Evaluation",
            leave=False,
        )
        for batch in pbar:
            batch = batch.to(device)
            context = batch.context
            target = batch.positive
            batch_len = target.size(0)

            # Get SASRecNBR logits
            hidden_state = model.encode(context)
            sasrec_logits = model.score_all_items(hidden_state)

            # TODO: Late Fusion
            # if self._config.alpha == 1.0:
            #     final_scores = sasrec_logits
            # else:
            #     final_scores = (
            #         (self.config.alpha * sasrec_logits)
            #         + (1.0 - self.config.alpha) * actr_scores
            #     )
            #     final_scores = sasrec_logits

            final_scores = sasrec_logits

            # Calculate final metrics
            for k in ks:
                batch_ndcg = compute_ndcg_at_k(final_scores, target, k=k)
                batch_recall = compute_recall_at_k(final_scores, target, k=k)

                ndcg_sums[k] += batch_ndcg * batch_len
                recall_sums[k] += batch_recall * batch_len

            total_samples += batch_len

        metrics: dict[str, float] = {}
        for k in ks:
            metrics[f"ndcg_at_{k}"] = ndcg_sums[k] / total_samples
            metrics[f"recall_at_{k}"] = recall_sums[k] / total_samples

        logger.info("Evaluation complete", metrics=metrics)
        return metrics
