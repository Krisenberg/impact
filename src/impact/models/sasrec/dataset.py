"""Module defining PyTorch Datasets for sequence-based (NBR) music recommendation."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, override

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import numpy as np

    from impact.schemas import TensorizedDataset


class SequenceSample(NamedTuple):
    """Immutable container representing a single sequence sample.

    Attributes:
        context: A 2D tensor of shape [L, K] holding item IDs from L context sessions.
        target: A 1D tensor of shape [K] holding item IDs from the target session (L+1).
    """

    context: torch.Tensor
    target: torch.Tensor


class SequenceDataset(Dataset[SequenceSample]):
    """PyTorch dataset over memory-mapped (context, target) sequence arrays.

    Wraps a MemmapDataset produced by `features.tensorize()` for use with DataLoader.
    Samples are loaded lazily via mmap, so no data is held in RAM.
    """

    def __init__(self, dataset: TensorizedDataset, num_items: int) -> None:
        """Initialize the dataset.

        Args:
            dataset: Tensorized split with memory-mapped context and target arrays.
            num_items: Total number of unique items (tracks) in the catalog. Used
                downstream for negative sampling and embedding table sizing.
        """
        self.context: np.ndarray = dataset.context  # [N, L, K]
        self.target: np.ndarray = dataset.target  # [N, K]
        self.num_items: int = num_items

    def __len__(self) -> int:
        return len(self.target)

    @override
    def __getitem__(self, idx: int) -> SequenceSample:
        return SequenceSample(
            torch.from_numpy(self.context[idx].copy()).long(),
            torch.from_numpy(self.target[idx].copy()).long(),
        )
