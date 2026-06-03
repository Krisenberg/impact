"""Module for collating sequence samples into batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from impact.models.sasrec.dataset import EvalSequenceSample, TrainSequenceSample


@dataclass(frozen=True)
class TrainSequenceBatch:
    """One training batch with negatives sampled in-batch.

    Attributes:
        context: [B, L, K] context session item indices.
        positive: [B, K] target session item indices.
        negative: [B, K] uniformly sampled negative item indices.
    """

    context: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor

    def to(
        self,
        device: torch.device,
        non_blocking: bool = True,
    ) -> TrainSequenceBatch:
        """Move all tensors to the given device, returning a new TrainSequenceBatch."""
        return TrainSequenceBatch(
            context=self.context.to(device, non_blocking=non_blocking),
            positive=self.positive.to(device, non_blocking=non_blocking),
            negative=self.negative.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> TrainSequenceBatch:
        """Pin memory of all underlying tensors, returning a new TrainSequenceBatch.

        This method should be automatically detected and executed by PyTorch's
        DataLoader when pin_memory=True is configured.
        """
        return TrainSequenceBatch(
            context=self.context.pin_memory(),
            positive=self.positive.pin_memory(),
            negative=self.negative.pin_memory(),
        )


@dataclass(frozen=True)
class EvalSequenceBatch:
    """One evaluation batch, carrying per-sample metadata needed for ACT-R scoring.

    No negative samples here as evaluation ranks against the full catalog.

    Attributes:
        uid: [B] user IDs, used to look up ACT-R history.
        target_ts: [B] timestamps of the target session (temporal cutoff for ACT-R).
        context: [B, L, K] context session item indices.
        positive: [B, K] target session item indices.
    """

    uid: torch.Tensor
    target_ts: torch.Tensor
    context: torch.Tensor
    positive: torch.Tensor

    def to(
        self,
        device: torch.device,
        non_blocking: bool = True,
    ) -> EvalSequenceBatch:
        """Move all tensors to the given device, returning a new EvalSequenceBatch."""
        return EvalSequenceBatch(
            uid=self.uid.to(device, non_blocking=non_blocking),
            target_ts=self.target_ts.to(device, non_blocking=non_blocking),
            context=self.context.to(device, non_blocking=non_blocking),
            positive=self.positive.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> EvalSequenceBatch:
        """Pin memory of all underlying tensors, returning a new EvalSequenceBatch.

        This method should be automatically detected and executed by PyTorch's
        DataLoader when pin_memory=True is configured.
        """
        return EvalSequenceBatch(
            uid=self.uid.pin_memory(),
            target_ts=self.target_ts.pin_memory(),
            context=self.context.pin_memory(),
            positive=self.positive.pin_memory(),
        )


def get_train_collate_fn(
    num_items: int,
) -> Callable[[list[TrainSequenceSample]], TrainSequenceBatch]:
    """Return a collate function that assembles training batches with sampled negatives.

    Uniform sampling from [1, num_items] is valid because item IDs have been mapped
    to a contiguous index space during preprocessing, so every integer in that range
    corresponds to a real item. Strict positive exclusion is omitted; given the catalog
    size (number of tracks) the collision rate is negligible.

    Args:
        num_items: Catalog size; defines the upper bound for negative sampling.
    """

    def _collate(batch: list[TrainSequenceSample]) -> TrainSequenceBatch:
        contexts = torch.stack([s.context for s in batch])
        positives = torch.stack([s.target for s in batch])
        negatives = torch.randint(1, num_items + 1, positives.shape, dtype=torch.long)

        # Possible way to enforce strict negative sampling
        # mask = torch.isin(negatives, positives)
        # while mask.any():
        #     negatives[mask] = torch.randint(1, num_items + 1, negatives[mask].shape)
        #     mask = torch.isin(negatives, positives)

        return TrainSequenceBatch(contexts, positives, negatives)

    return _collate


def get_eval_collate_fn() -> Callable[[list[EvalSequenceSample]], EvalSequenceBatch]:
    """Return a collate function that assembles evaluation batches.

    Includes uid and target_ts metadata required for ACT-R scoring. Skips negative
    sampling as evaluation scores against the full catalog directly.
    """

    def _collate(batch: list[EvalSequenceSample]) -> EvalSequenceBatch:
        return EvalSequenceBatch(
            uid=torch.tensor([s.uid for s in batch], dtype=torch.long),
            target_ts=torch.tensor([s.target_ts for s in batch], dtype=torch.long),
            context=torch.stack([s.context for s in batch]),
            positive=torch.stack([s.target for s in batch]),
        )

    return _collate
