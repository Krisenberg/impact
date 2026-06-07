from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, override

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import numpy as np

    from impact.schemas import TensorizedDataset


class TrainSequenceSample(NamedTuple):
    """One training example: a context sequence and its target session.

    Attributes:
        context: [L, K] context session item indices.
        target: [K] target session item indices (session L+1).
    """

    context: torch.Tensor
    target: torch.Tensor


class EvalSequenceSample(NamedTuple):
    """One evaluation example (context+target), with metadata needed for ACT-R scoring.

    Attributes:
        uid: User ID, used to query ACT-R history.
        target_ts: Timestamp denoting the start of target session (cutoff for ACT-R).
        context: [L, K] context session item indices.
        target: [K] target session item indices (session L+1).
    """

    uid: int
    target_ts: int
    context: torch.Tensor
    target: torch.Tensor


class TrainSequenceDataset(Dataset[TrainSequenceSample]):
    """Memory-mapped dataset for training.

    Samples are loaded lazily via  memory mapping and copied on access for safe use
    with multi-process DataLoader workers.
    """

    def __init__(self, dataset: TensorizedDataset) -> None:
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
    """Memory-mapped dataset for evaluation.

    Extends the training view with uid and target_ts metadata required for ACT-R scoring
    and per-user metric aggregation. Samples are loaded lazily via  memory mapping and
    copied on access for safe use with multi-process DataLoader workers.
    """

    def __init__(self, dataset: TensorizedDataset) -> None:
        super().__init__()
        self.uid: np.ndarray = dataset.uid  # [N]
        self.target_ts: np.ndarray = dataset.target_ts  # [N]
        self.context: np.ndarray = dataset.context  # [N, L, K]
        self.target: np.ndarray = dataset.target  # [N, K]

    def __len__(self) -> int:
        return len(self.target)

    @override
    def __getitem__(self, idx: int) -> EvalSequenceSample:
        return EvalSequenceSample(
            uid=int(self.uid[idx]),
            target_ts=int(self.target_ts[idx]),
            context=torch.from_numpy(self.context[idx].copy()).long(),
            target=torch.from_numpy(self.target[idx].copy()).long(),
        )
