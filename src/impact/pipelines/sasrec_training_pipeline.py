from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, cast, final

import structlog
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from impact.evaluation.metrics import compute_ndcg_at_k, compute_recall_at_k
from impact.models.sasrec import (
    EvalSequenceBatch,
    EvalSequenceDataset,
    EvalSequenceSample,
    SASRecNBR,
    TrainSequenceBatch,
    TrainSequenceDataset,
    TrainSequenceSample,
    bce_loss,
    get_eval_collate_fn,
    get_train_collate_fn,
)
from impact.utils import seed_worker

if TYPE_CHECKING:
    from pathlib import Path

    from impact.config import ImpactConfig
    from impact.schemas import TensorizedSplit

logger = structlog.stdlib.get_logger(__name__)


@final
class SASRecTrainingPipeline:
    """Orchestrates SASRecNBR initialization, data loading, and training."""

    def __init__(self, config: ImpactConfig) -> None:
        """Initialize the SASRecNBR training pipeline.

        Args:
            config: Complete IMPACT Framework configuration.
        """
        self._config = config

    def train(
        self,
        split: TensorizedSplit,
        num_items: int,
        save_path: Path | None = None,
    ) -> float:
        """Initialise and train SASRecNBR with early stopping monitored via NDCG@K.

        Training stops either when `ImpactConfig.sasrec.max_epochs` is reached or after
        `ImpactConfig.sasrec.patience` consecutive epochs without improvement.

        Args:
            split: Tensorized train/val/test temporal splits; train is used for
                training, val for early stopping.
            num_items: Catalog size, passed to the model and negative sampler.
            save_path: Where to write the best-performing model checkpoint. If None, no
                checkpoint is saved.

        Returns:
            Best NDCG@K (calculated on the validation set) reached before training
            terminated.
        """
        if save_path is not None:
            if save_path.exists():
                raise ValueError(f"Checkpoint already exists: {save_path}")
            save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing SASRecNBR training run",
            batch_size=self._config.sasrec.batch_size,
            max_epochs=self._config.sasrec.max_epochs,
            learning_rate=self._config.sasrec.learning_rate,
            patience=self._config.sasrec.patience,
        )

        logger.info(
            "Dataset statistics",
            num_items=num_items,
            train_sequences=split.train.meta[0],
            val_sequences=split.val.meta[0],
        )

        # Setup deterministic random state
        g = torch.Generator()
        _ = g.manual_seed(self._config.system.seed)

        # Configure DataLoaders with multi-processing seed protection
        train_loader = DataLoader[TrainSequenceSample](
            dataset=TrainSequenceDataset(split.train),
            batch_size=self._config.sasrec.batch_size,
            shuffle=True,
            collate_fn=get_train_collate_fn(num_items),
            num_workers=self._config.system.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_loader = DataLoader[EvalSequenceSample](
            dataset=EvalSequenceDataset(split.val),
            batch_size=self._config.sasrec.batch_size,
            shuffle=False,
            collate_fn=get_eval_collate_fn(),
            num_workers=self._config.system.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        # Initialize model and optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SASRecNBR(
            num_items=num_items,
            context_sessions=self._config.sequence.context_sessions,
            embedding_dim=self._config.sasrec.embedding_dim,
            num_heads=self._config.sasrec.num_heads,
            num_layers=self._config.sasrec.num_layers,
            dropout=self._config.sasrec.dropout_rate,
        )
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=self._config.sasrec.learning_rate,
            weight_decay=1e-4,
        )

        logger.info("Starting training loop", device=str(device))
        best_ndcg = -1.0
        epochs_without_improvement = 0

        for epoch in range(1, self._config.sasrec.max_epochs + 1):
            train_loss = self._train_epoch(
                model=model,
                loader=cast(DataLoader[TrainSequenceBatch], train_loader),
                optimizer=optimizer,
                device=device,
                epoch=epoch,
            )
            val_ndcg, val_recall = self._evaluate(
                model=model,
                loader=cast(DataLoader[EvalSequenceBatch], val_loader),
                device=device,
            )

            logger.info(
                "Epoch completed",
                epoch=epoch,
                train_loss=round(train_loss, 4),
                val_ndcg=round(val_ndcg, 4),
                val_recall=round(val_recall, 4),
            )

            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg
                epochs_without_improvement = 0
                logger.info("New best model found", best_ndcg=round(best_ndcg, 4))
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self._config.sasrec.patience:
                logger.info(
                    "Early stopping triggered", epoch=epoch, best_ndcg=best_ndcg
                )
                break

        if save_path is not None:
            logger.info(
                "SASRecNBR training complete, best-performing model checkpoint saved",
                best_model_path=save_path,
            )
        else:
            logger.info("SASRecNBR training complete")

        return best_ndcg

    def _train_epoch(
        self,
        model: SASRecNBR,
        loader: DataLoader[TrainSequenceBatch],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
    ) -> float:
        """Run one training epoch, returning mean loss over all batches."""
        _ = model.train()
        total_loss = 0.0

        iterator = cast(Iterator[TrainSequenceBatch], iter(loader))
        for batch in tqdm(iterator, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass (get intent vectors)
            hidden_state = cast(torch.Tensor, model(batch.context))

            # Score positive and negative items
            pos_scores = model.score_items(hidden_state, batch.positive)
            neg_scores = model.score_items(hidden_state, batch.negative)

            # Compute BCE Loss
            loss = bce_loss(pos_scores, neg_scores)

            # Backward pass
            _ = loss.backward()
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def _evaluate(
        self,
        model: SASRecNBR,
        loader: DataLoader[EvalSequenceBatch],
        device: torch.device,
    ) -> tuple[float, float]:
        """Score the model over given loader against the full catalog.

        Returns:
            (NDCG@K, Recall@K) as sample-weighted averages over all batches.
        """
        _ = model.eval()
        ndcg_sum, recall_sum = 0.0, 0.0
        total_samples = 0

        k_cutoff = self._config.sasrec.val_k_cutoff

        for batch in cast(Iterator[EvalSequenceBatch], iter(loader)):
            batch = batch.to(device)
            target = batch.positive
            batch_len = target.size(0)

            # Forward pass (get intent vectors)
            hidden_state = cast(torch.Tensor, model(batch.context))
            logits = model.score_all_items(hidden_state)

            # Compute NDCG@K and Recall@K
            batch_ndcg = compute_ndcg_at_k(logits, target, k=k_cutoff)
            batch_recall = compute_recall_at_k(logits, target, k=k_cutoff)

            # Weight by batch size
            ndcg_sum += batch_ndcg * batch_len
            recall_sum += batch_recall * batch_len
            total_samples += batch_len

        # Return average NDCG@K and Recall@K
        return ndcg_sum / total_samples, recall_sum / total_samples
