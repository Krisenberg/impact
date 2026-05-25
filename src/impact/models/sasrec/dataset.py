"""Module defining PyTorch Datasets for sequence-based (NBR) music recommendation."""

from dataclasses import dataclass
from typing import TypeVar, override

import polars as pl
import torch
from torch.utils.data import Dataset

# Type variable representing the specific sequence sample type
T = TypeVar("T")


@dataclass(frozen=True)
class TrainSequenceSample:
    """Immutable container representing a single training sequence sample.

    Attributes:
        context: A 2D tensor of shape [L, K] holding item IDs from L context sessions.
        target: A 1D tensor of shape [K] holding item IDs from the target session (L+1).
        negatives: A 1D tensor of shape [K] holding randomly sampled negative item IDs.
    """

    context: torch.Tensor
    target: torch.Tensor
    negatives: torch.Tensor


@dataclass(frozen=True)
class EvalSequenceSample:
    """Immutable container representing a single evaluation sequence sample.

    Attributes:
        context: A 2D tensor of shape [L, K] holding item IDs from L context sessions.
        target: A 1D tensor of shape [K] holding item IDs from the target session (L+1).
    """

    context: torch.Tensor
    target: torch.Tensor


class _BaseDataset[T](Dataset[T]):
    """Abstract base dataset providing common initialization and methods."""

    def __init__(self, lf: pl.LazyFrame) -> None:
        """Initialize the base dataset.

        Args:
            lf: The Polars LazyFrame containing user-scoped sequences of sessions.
        """
        super().__init__()
        self._dataset: pl.DataFrame = lf.collect()  # For fast O(1) row lookups

    @property
    def dataset(self) -> pl.DataFrame:
        """Polars DataFrame containing underlying dataset (materialized)."""
        return self._dataset

    def __len__(self) -> int:
        """Return the total number of sequences."""
        return self._dataset.height


class TrainDataset(_BaseDataset[TrainSequenceSample]):
    """Dataset adjusted for model training (includes negative sampling)."""

    def __init__(self, dataset_lf: pl.LazyFrame, num_items: int) -> None:
        super().__init__(dataset_lf)
        self.num_items: int = num_items

    @override
    def __getitem__(self, idx: int) -> TrainSequenceSample:
        row = self._dataset.row(idx, named=True)

        context_tensor = torch.tensor(
            data=[session["items"] for session in row["context"]],
            dtype=torch.long,
        )

        target_raw = row["target"]["items"]
        target_tensor = torch.tensor(
            data=row["target"]["items"],
            dtype=torch.long,
        )

        negatives_tensor = torch.randint(
            low=1,
            high=self.num_items + 1,
            size=(len(target_raw),),
            dtype=torch.long,
        )

        return TrainSequenceSample(
            context=context_tensor,
            target=target_tensor,
            negatives=negatives_tensor,
        )


class EvalDataset(_BaseDataset[EvalSequenceSample]):
    """Dataset adjusted for model evaluation (validation and testing)."""

    @override
    def __getitem__(self, idx: int) -> EvalSequenceSample:
        row = self._dataset.row(idx, named=True)
        context_tensor = torch.tensor(
            data=[session["items"] for session in row["context"]],
            dtype=torch.long,
        )
        target_tensor = torch.tensor(data=row["target"]["items"], dtype=torch.long)
        return EvalSequenceSample(context=context_tensor, target=target_tensor)
