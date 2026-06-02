"""Module defining PyTorch Datasets for sequence-based (NBR) music recommendation."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, override

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import numpy as np

    from impact.schemas import TensorizedDataset


class TrainSequenceSample(NamedTuple):
    """Sample containing only tensors required for SASRecNBR training.

    Attributes:
        context: Tensor of shape [L, K] with track indices from L context sessions.
        target: Tensor of shape [K] with track indices from the target session (L+1).
    """

    context: torch.Tensor
    target: torch.Tensor


class EvalSequenceSample(NamedTuple):
    """Sample containing tensors and sequence metadata required for evaluation.

    Attributes:
        uid: User ID.
        target_ts: Timestamp of the target session (first listen event).
        context: Tensor of shape [L, K] with track indices from L context sessions.
        target: Tensor of shape [K] with track indices from the target session (L+1).
    """

    uid: int
    target_ts: int
    context: torch.Tensor
    target: torch.Tensor


class TrainSequenceDataset(Dataset[TrainSequenceSample]):
    """PyTorch Dataset over memory-mapped (context, target) sequence arrays.

    Samples are loaded lazily via memory mapping. Each tensor is copied on access for
    safe use with multi-process DataLoader workers.
    """

    def __init__(self, dataset: TensorizedDataset) -> None:
        """Initialize the training dataset.

        Args:
            dataset: Tensorized split with memory-mapped context and target arrays.
        """
        super().__init__()
        self.context: np.ndarray = dataset.context  # [N, L, K]
        self.target: np.ndarray = dataset.target  # [N, K]

    def __len__(self) -> int:
        return len(self.target)

    @override
    def __getitem__(self, idx: int) -> TrainSequenceSample:
        return TrainSequenceSample(
            torch.from_numpy(self.context[idx].copy()).long(),
            torch.from_numpy(self.target[idx].copy()).long(),
        )


class EvalSequenceDataset(Dataset[EvalSequenceSample]):
    """PyTorch Dataset over memory-mapped sequence arrays for evaluation.

    Extends the training view with per-sample metadata (uid, target_ts) required for
    temporal cutoff filtering in the ACT-R scorer and for aggregating per-user metrics
    during evaluation.

    Samples are loaded lazily via memory mapping. Each tensor is copied on access for
    safe use with multi-process DataLoader workers.
    """

    def __init__(self, dataset: TensorizedDataset) -> None:
        """Initialize the evaluation dataset.

        Args:
            dataset: Tensorized split with memory-mapped context and target arrays.
        """
        super().__init__()
        self.uids: np.ndarray = dataset.uid  # [N]
        self.target_tss: np.ndarray = dataset.target_ts  # [N]
        self.context: np.ndarray = dataset.context  # [N, L, K]
        self.target: np.ndarray = dataset.target  # [N, K]

    def __len__(self) -> int:
        return len(self.target)

    @override
    def __getitem__(self, idx: int) -> EvalSequenceSample:
        return EvalSequenceSample(
            uid=int(self.uids[idx]),
            target_ts=int(self.target_tss[idx]),
            context=torch.from_numpy(self.context[idx].copy()).long(),
            target=torch.from_numpy(self.target[idx].copy()).long(),
        )
