"""Orchestrator for the SASRecNBR model training pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, final

import structlog
import torch
from torch.utils.data import DataLoader

from impact.models.sasrec import (
    EvalSequenceBatch,
    EvalSequenceDataset,
    EvalSequenceSample,
    SASRecNBR,
    SASRecTrainer,
    TrainSequenceBatch,
    TrainSequenceDataset,
    TrainSequenceSample,
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
    """Orchestrates SASRecNBR model initialization, data loading, and training."""

    def __init__(self, config: ImpactConfig) -> None:
        """Initialize the SASRecNBR training pipeline.

        Args:
            config: Complete IMPACT Framework configuration.
        """
        self._config = config

    def train(
        self, split: TensorizedSplit, num_items: int, save_path: Path | None = None
    ) -> float:
        """Execute the training and validation loop.

        Args:
            split: The memory-mapped, tensorized train/val/test temporal splits; train
                split is used for model training, val split for validation and early
                stopping, num_items for model and dataset initialisation.
            num_items: ...
            save_path: If provided, the best-performing model checkpoint is written
                there. If None, no checkpoint is saved.

        Returns:
            Best validation NDCG@k achieved before training terminated.
        """
        if save_path is not None:
            if save_path.exists():
                raise ValueError(f"Checkpoint already exists: {save_path}")
            save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing SASRecNBR training run",
            batch_size=self._config.sasrec.batch_size,
            learning_rate=self._config.sasrec.learning_rate,
            patience=self._config.sasrec.patience,
            max_epochs=self._config.sasrec.max_epochs,
        )

        # num_items = split.num_items
        logger.info(
            "Dataset statistics loaded",
            num_items=num_items,
            train_sequences=split.train.meta[0],
        )

        # Setup deterministic random state
        g = torch.Generator()
        _ = g.manual_seed(self._config.seed)

        # Configure DataLoaders with multi-processing seed protection
        train_loader = DataLoader[TrainSequenceSample](
            dataset=TrainSequenceDataset(split.train),
            batch_size=self._config.sasrec.batch_size,
            shuffle=True,
            collate_fn=get_train_collate_fn(num_items),
            num_workers=self._config.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        val_loader = DataLoader[EvalSequenceSample](
            dataset=EvalSequenceDataset(split.val),
            batch_size=self._config.sasrec.batch_size,
            shuffle=False,
            collate_fn=get_eval_collate_fn(),
            num_workers=self._config.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

        # Device and Model allocation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = SASRecNBR(
            num_items=num_items,
            context_sessions=self._config.context_sessions,
            embedding_dim=self._config.sasrec.embedding_dim,
            num_heads=self._config.sasrec.num_heads,
            num_layers=self._config.sasrec.num_layers,
            dropout=self._config.sasrec.dropout_rate,
        )

        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=self._config.sasrec.learning_rate,
            weight_decay=1e-4,
        )

        # Trainer initialization and execution
        trainer = SASRecTrainer(
            model=model,
            train_loader=cast(DataLoader[TrainSequenceBatch], train_loader),
            val_loader=cast(DataLoader[EvalSequenceBatch], val_loader),
            optimizer=optimizer,
            device=device,
            k=self._config.sasrec.k,
            patience=self._config.sasrec.patience,
        )

        best_val_ndcg = trainer.fit(
            max_epochs=self._config.sasrec.max_epochs,
            save_path=save_path,
        )

        if save_path is not None:
            logger.info(
                "SASRecNBR training complete, best-performing model checkpoint saved",
                best_model_path=save_path,
            )
        else:
            logger.info("SASRecNBR training complete")

        return best_val_ndcg
