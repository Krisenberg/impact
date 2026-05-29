"""Orchestrator for the SASRecNBR model training pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, final

import structlog
import torch
from torch.utils.data import DataLoader

from impact.models.sasrec import (
    SASRecNBR,
    SASRecTrainer,
    SequenceBatch,
    SequenceDataset,
    SequenceSample,
    get_collate_fn,
    get_eval_collate_fn,
)
from impact.utils import seed_worker

if TYPE_CHECKING:
    from pathlib import Path

    from impact.config import ImpactConfig
    from impact.schemas import MemmapSplit

logger = structlog.stdlib.get_logger(__name__)


@final
class SASRecTrainingPipeline:
    """Orchestrates SASRecNBR model initialization, data loading, and training."""

    def __init__(self, models_dir: Path, config: ImpactConfig) -> None:
        """Initialize the SASRecNBR training pipeline.

        Args:
            models_dir: Base directory where model checkpoints are saved.
            config: Complete IMPACT Framework configuration.
        """
        self._config = config
        self._run_models_dir = (
            models_dir
            / config.dataset_id
            / config.sessionization_id
            / config.sequencing_id
            / config.timesplit_id
        )
        self._run_models_dir.mkdir(parents=True, exist_ok=True)
        self._model_save_path = self._run_models_dir / "sasrec_best.pt"

    def train(self, split: MemmapSplit) -> Path:
        """Execute the training and validation loop.

        Args:
            split: The memory-mapped, tensorized train/val/test temporal splits; train
                split is used for model training, val split is used for validation and
                early stopping, num_items for model and dataset initialisation.

        Returns:
            Path to the best-performing model checkpoint (state dict).
        """
        # return self._model_save_path

        logger.info(
            "Initializing SASRecNBR training run",
            batch_size=self._config.sasrec.batch_size,
            learning_rate=self._config.sasrec.learning_rate,
            patience=self._config.sasrec.patience,
            max_epochs=self._config.sasrec.max_epochs,
        )

        num_items = split.num_items
        logger.info(
            "Dataset statistics loaded",
            num_items=num_items,
            train_sequences=split.train.meta[0],
        )

        # Instantiate PyTorch Datasets
        train_ds = SequenceDataset(split.train, num_items)
        val_ds = SequenceDataset(split.val, num_items)

        # Setup deterministic random state
        g = torch.Generator()
        _ = g.manual_seed(self._config.seed)

        # Configure DataLoaders with multi-processing seed protection
        train_loader = DataLoader[SequenceSample](
            train_ds,
            batch_size=self._config.sasrec.batch_size,
            shuffle=True,
            collate_fn=get_collate_fn(num_items),
            num_workers=self._config.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        val_loader = DataLoader[SequenceSample](
            val_ds,
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
            train_loader=cast(DataLoader[SequenceBatch], train_loader),
            val_loader=cast(DataLoader[SequenceBatch], val_loader),
            optimizer=optimizer,
            device=device,
            k=self._config.sasrec.k,
            patience=self._config.sasrec.patience,
        )

        trainer.fit(
            max_epochs=self._config.sasrec.max_epochs,
            save_path=self._model_save_path,
        )

        logger.info(
            "SASRecNBR training complete, best model saved",
            best_model_path=self._model_save_path,
        )

        return self._model_save_path
