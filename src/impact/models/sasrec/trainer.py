"""Module defining the training loop for the SASRecNBR model."""

from collections.abc import Iterator
from pathlib import Path
from typing import cast, final

import structlog
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from impact.evaluation.metrics import compute_ndcg_at_k, compute_recall_at_k
from impact.models.sasrec.collate import EvalSequenceBatch, TrainSequenceBatch
from impact.models.sasrec.loss import bce_loss
from impact.models.sasrec.model import SASRecNBR

logger = structlog.stdlib.get_logger(__name__)


@final
class SASRecTrainer:
    """Training and evaluation loop for SASRecNBR."""

    def __init__(
        self,
        model: SASRecNBR,
        train_loader: DataLoader[TrainSequenceBatch],
        val_loader: DataLoader[EvalSequenceBatch],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        k: int = 10,
        patience: int = 10,
    ) -> None:
        """Initialize the SASRecTrainer.

        Args:
            model: SASRecNBR instance to train.
            train_loader: DataLoader yielding SequenceBatch with context, positive
                (target) and negative samples.
            val_loader: DataLoader used for early stopping evaluation.
            optimizer: Configured optimizer (e.g. AdamW).
            device: Device to move the model and batches to.
            k: Cutoff for NDCG@K and Recall@K.
            patience: Number of epochs without NDCG improvement before stopping.
        """
        self._model = model.to(device)
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._optimizer = optimizer
        self._device = device
        self._k = k
        self._patience = patience

    def fit(self, max_epochs: int, save_path: Path | None = None) -> float:
        """Train the model with early stopping on validation NDCG@k.

        Training stops either when `max_epochs` is reached or after `patience`
        consecutive epochs without improvement. If `save_path` is provided, the
        best model state dict (checkpoint) is saved whenever validation NDCG@k improves;
        if None, no checkpoint is written.

        Args:
            max_epochs: Maximum number of training epochs.
            save_path: Path where the best model checkpoint will be written, or None to
                skip saving.

        Returns:
            Best validation NDCG@k achieved before training terminated.
        """
        if save_path is not None:
            if save_path.exists():
                raise ValueError(f"Checkpoint already exists: {save_path}")
            save_path.parent.mkdir(parents=True, exist_ok=True)

        best_ndcg = -1.0
        epochs_without_improvement = 0

        logger.info(
            "Starting training loop",
            max_epochs=max_epochs,
            device=str(self._device),
        )

        for epoch in range(1, max_epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_ndcg, val_recall = self._evaluate(self._val_loader)

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
                if save_path is not None:
                    torch.save(self._model.state_dict(), save_path)
                logger.info("New best model found", best_ndcg=round(best_ndcg, 4))
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self._patience:
                logger.info(
                    "Early stopping triggered",
                    epoch=epoch,
                    best_ndcg=best_ndcg,
                )
                break

        return best_ndcg

    def _train_epoch(self, epoch: int) -> float:
        """Run a single training epoch and return mean loss over all batches."""
        _ = self._model.train()
        total_loss = 0.0

        pbar = tqdm(
            iterable=cast(Iterator[TrainSequenceBatch], iter(self._train_loader)),
            desc=f"Epoch {epoch}",
            leave=False,
        )
        for batch in pbar:
            batch = batch.to(self._device)

            self._optimizer.zero_grad()

            # Forward pass (get intent vectors)
            hidden_state = cast(torch.Tensor, self._model(batch.context))

            # Score positive and negative items
            pos_scores = self._model.score_items(hidden_state, batch.positive)
            neg_scores = self._model.score_items(hidden_state, batch.negative)

            # Compute BCE Loss
            loss = bce_loss(pos_scores, neg_scores)

            # Backward pass
            _ = loss.backward()  # type: ignore[no-untyped-call]
            _ = torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5.0)
            self._optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self._train_loader)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader[EvalSequenceBatch]) -> tuple[float, float]:
        """Evaluate the model over a given dataloader.

        Negatives are not used during evaluation - scoring is against the full item
        catalog via `score_all_items()`.

        Args:
            loader: DataLoader to evaluate on.

        Returns:
            Tuple of (NDCG@K, Recall@K) as sample-weighted averages over all batches.
        """
        _ = self._model.eval()
        ndcg_sum, recall_sum = 0.0, 0.0
        total_samples = 0

        for batch in cast(Iterator[EvalSequenceBatch], iter(loader)):
            batch = batch.to(self._device)
            context = batch.context
            target = batch.positive
            batch_len = target.size(0)

            # Forward pass (get intent vectors)
            hidden_state = cast(torch.Tensor, self._model(context))
            logits = self._model.score_all_items(hidden_state)

            batch_ndcg = compute_ndcg_at_k(logits, target, k=self._k)
            batch_recall = compute_recall_at_k(logits, target, k=self._k)

            ndcg_sum += batch_ndcg * batch_len
            recall_sum += batch_recall * batch_len

            total_samples += batch_len

        return ndcg_sum / total_samples, recall_sum / total_samples
