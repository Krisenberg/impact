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
    """Batch of sequence samples with in-batch negative sampling.

    Attributes:
        context:  Item indices for L context sessions of shape [B, L, K].
        positive: Item indices for the target session of shape [B, K].
        negative: Sampled negative item indices of shape [B, K].
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
    """..."""

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
    """Return a collate function that assembles a SequenceBatch with sampled negatives.

    In the current implementation, negatives are sampled uniformly from [1, num_items]
    without strict exclusion of positives. Given the catalog size (number of tracks),
    collisions are rare enough to be acceptable.

    Args:
        num_items: Total number of unique items (tracks); defines the sampling range.
    """

    def _collate(batch: list[TrainSequenceSample]) -> TrainSequenceBatch:
        contexts = torch.stack([s.context for s in batch])
        positives = torch.stack([s.target for s in batch])

        # Currently not a 'strict negative sampling' but taking into account the catalog
        # size, it might be enough
        negatives = torch.randint(1, num_items + 1, positives.shape, dtype=torch.long)

        # Possible way to enforce strict negative sampling
        # mask = torch.isin(negatives, positives)
        # while mask.any():
        #     negatives[mask] = torch.randint(1, num_items + 1, negatives[mask].shape)
        #     mask = torch.isin(negatives, positives)

        return TrainSequenceBatch(contexts, positives, negatives)

    return _collate


def get_eval_collate_fn() -> Callable[[list[EvalSequenceSample]], EvalSequenceBatch]:
    """Return a collate function for evaluation, omitting negative sampling.

    The returned batch has an empty `negative` tensor, as evaluation scores against
    the full item catalog either way.
    """

    def _collate(batch: list[EvalSequenceSample]) -> EvalSequenceBatch:
        uids = torch.tensor([s.uid for s in batch], dtype=torch.long)
        target_tss = torch.tensor([s.target_ts for s in batch], dtype=torch.long)

        contexts = torch.stack([s.context for s in batch])
        positives = torch.stack([s.target for s in batch])

        return EvalSequenceBatch(
            uid=uids,
            target_ts=target_tss,
            context=contexts,
            positive=positives,
        )

    return _collate
