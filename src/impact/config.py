from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any

from impact.dataset import DatasetSize, DatasetType, InteractionType

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for the Yambda dataset fetcher."""

    size: DatasetSize = DatasetSize.SIZE_50M
    """Yambda dataset variant to process (50M, 500M, or 5B)."""

    format: DatasetType = DatasetType.FLAT
    """Yambda dataset format type. Currently only FLAT is supported."""

    interaction_type: InteractionType = InteractionType.MULTI_EVENT
    """Type of interactions (events) to load from the Yambda dataset.

    MULTI_EVENT includes all event types (listen, like, etc.); other variants contain
    only a specific subset.
    """

    def __post_init__(self) -> None:
        """Verify that the selected dataset format is supported."""
        if self.format != DatasetType.FLAT:
            raise ValueError(f"Dataset format {self.format} not supported.")


@dataclass(frozen=True)
class SessionConfig:
    """Configuration for the sessionization stage (Algorithm 1 in the thesis)."""

    tracks_per_session: int = 10
    """Number of tracks that constitute a session. Denoted as K in the thesis.

    Sessions with less than K tracks are discarded. Sessions with more than K tracks
    are truncated.
    """


@dataclass(frozen=True)
class SequenceConfig:
    """Configuration for the sequence extraction stage (Algorithm 2 in the thesis)."""

    context_sessions: int = 20
    """Number of historical sessions used as context. Denoted as L in the thesis."""

    step: int = 5
    """Step size for the sliding window sequence extraction."""


@dataclass(frozen=True)
class GTSConfig:
    """Configuration for the temporal split (GTS) stage (Algorithm 3 in the thesis)."""

    test_days: float = 1.0
    """Number of days at the end of the timeline allocated for the test set."""

    val_days: float = 1.0
    """Number of days preceding test set allocated for the validation set."""

    gap_minutes: float = 30.0
    """Buffer time (in minutes) between dataset splits.

    The purpose of this parameter is to mimic production system deployment latency.
    """


@dataclass(frozen=True)
class ACTRConfig:
    """Configuration for the ACT-R (P-Pop) scorer."""

    decay: float = 0.5
    """Power-law time decay exponent. Denoted as d in the thesis.

    Higher values result in older listens being forgotten faster.
    """

    organic_weight: float = 1.0
    """Contribution weight for organic (user-initiated) listens."""

    autoplay_weight: float = 0.5
    """Contribution weight for autoplay (recsys-initiated) listens."""

    eps: float = 1e-8
    """Numerical floor added before applying log to prevent log(0)."""


@dataclass(frozen=True)
class SASRecConfig:
    """Configuration for the SASRecNBR model."""

    embedding_dim: int = 64
    """Dimension of item and positional embeddings."""

    num_heads: int = 2
    """Number of attention heads in each Transformer layer."""

    num_layers: int = 2
    """Number of Transformer encoder layers."""

    dropout_rate: float = 0.2
    """Dropout probability applied after embedding within each layer."""

    train_batch_size: int = 2048
    """Number of sequences per training batch."""

    eval_batch_size: int = 2048
    """Number of sequences per evaluation batch.

    For larger dataset variants, should be lower than training batch size due to
    full-catalog scoring memory requirements.
    """

    max_epochs: int = 100
    """Maximum number of training epochs to run."""

    learning_rate: float = 0.001
    """Learning rate for the optimizer."""

    val_k_cutoff: int = 10
    """Rank cutoff for NDCG@K and Recall@K metrics used during model training."""

    patience: int = 10
    """Number of epochs to wait for validation improvement before stopping."""


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for the late-fusion evaluation of SASRecNBR and ACT-R (IMPACT)."""

    batch_size: int = 512
    """Number of sequences per evaluation batch.

    Should be lower than training batch size due to full-catalog scoring memory
    requirements.
    """

    k_cutoffs: tuple[int, ...] = (5, 10, 40, 100)
    """Rank cutoffs for computing NDCG@K and Recall@K during evaluation."""

    alpha: float = 0.5
    """Relative contribution of SASRec and ACT-R in the final late-fusion score.

    The final score is computed as:

        alpha * SASRec + (1 - alpha) * ACT-R,

    where 1.0 corresponds to SASRec-only recommendations and 0.0 to ACT-R-only
    recommendations.
    """


@dataclass(frozen=True)
class SystemConfig:
    """System-level parameters for reproducibility and runtime behavior."""

    seed: int = 42
    """Global random seed for strict reproducibility."""

    num_workers: int = 4
    """Number of subprocesses to use for data loading.

    Set to 0 to load data in the main process, or higher for parallel loading
    (hardware-dependent).
    """


@dataclass(frozen=True)
class ImpactConfig:
    """Immutable configuration for the IMPACT Framework.

    Single source of truth for all parameters governing data processing, feature
    engineering, model training and evaluation. Immutability ensures reproducibility
    across runs. The entire configuration can be easily serialized to a dict or YAML
    file for exact reconstruction of any pipeline artifact.
    """

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """Configuration for the Yambda dataset fetcher."""

    session: SessionConfig = field(default_factory=SessionConfig)
    """Configuration for the sessionization stage (Algorithm 1 in the thesis)."""

    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    """Configuration for the sequence extraction stage (Algorithm 2 in the thesis)."""

    gts: GTSConfig = field(default_factory=GTSConfig)
    """Configuration for the temporal split (GTS) stage (Algorithm 3 in the thesis)."""

    actr: ACTRConfig = field(default_factory=ACTRConfig)
    """Configuration for the ACT-R (P-Pop) scorer."""

    sasrec: SASRecConfig = field(default_factory=SASRecConfig)
    """Configuration for the SASRecNBR model."""

    eval: EvalConfig = field(default_factory=EvalConfig)
    """Configuration for the late-fusion evaluation of SASRecNBR and ACT-R (IMPACT)."""

    system: SystemConfig = field(default_factory=SystemConfig)
    """System-level parameters for reproducibility and runtime behavior."""

    @staticmethod
    def _format_value(val: float, dot_replacement: str = "-") -> str:
        """Format float values to be filesystem-friendly (replace dots)."""
        if val.is_integer():
            return str(int(val))
        return str(val).replace(".", dot_replacement)

    @cached_property
    def dataset_id(self) -> str:
        """Cache key for the raw data fetching stage."""
        parts = [
            "dataset",
            self.dataset.size.value,
            self.dataset.format.value,
            self.dataset.interaction_type.value,
        ]
        return "_".join(parts)

    @cached_property
    def sessionization_id(self) -> str:
        """Cache key for the sessionization stage (Algorithm 1 in the thesis)."""
        parts = [
            "sessions",
            f"k{self.session.tracks_per_session}",
        ]
        return "_".join(parts)

    @cached_property
    def sequence_extraction_id(self) -> str:
        """Cache key for the sequence extraction stage (Algorithm 2 in the thesis)."""
        parts = [
            "sequences",
            f"l{self.sequence.context_sessions}",
            f"step{self.sequence.step}",
        ]
        return "_".join(parts)

    @cached_property
    def temporal_split_id(self) -> str:
        """Cache key for the temporal split (GTS) stage (Algorithm 3 in the thesis)."""
        parts = [
            "split",
            f"val{self._format_value(self.gts.val_days)}d",
            f"test{self._format_value(self.gts.test_days)}d",
            f"gap{self._format_value(self.gts.gap_minutes)}m",
        ]
        return "_".join(parts)

    @cached_property
    def config_hash(self) -> str:
        """SHA-256 hash representing the entire configuration.

        Converts the serialized ImpactConfig (dict) to a JSON string (with sorted keys)
        before computing the hash to guarantee reproducibility across different runs.
        """
        import json

        canonical_json = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @cached_property
    def display_id(self) -> str:
        """Short human-readable ID representing the experiment configuration.

        Format: {size}-A{alpha}-{embedding_dim}D-LR{learning_rate}-{hash[:7]}
        Example: 5B-A0_5-128D-LR0_001-a1b2c3d
        """
        parts = [
            self.dataset.size.value.upper(),
            f"A{self._format_value(self.eval.alpha, dot_replacement='_')}",
            f"{self.sasrec.embedding_dim}D",
            f"A{self._format_value(self.sasrec.learning_rate, dot_replacement='_')}",
            self.config_hash[:7],
        ]
        return "-".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the nested configuration into a dictionary.

        Automatically converts all Enums to their raw string/numeric values.
        """

        def enum_dict_factory(data: list[tuple[str, Any]]) -> dict[str, Any]:
            """Dict factory that extracts .value from Enum objects."""
            return {k: v.value if isinstance(v, Enum) else v for k, v in data}

        return asdict(self, dict_factory=enum_dict_factory)

    def write_yaml(self, output_path: Path) -> None:
        """Write the configuration to a YAML file.

        Args:
            output_path: Target path where the YAML file with the configuration snapshot
                should be saved.
        """
        import yaml

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                data=self.to_dict(),
                stream=f,
                indent=2,
                sort_keys=False,  # Prevents from sorting keys alphabetically
                allow_unicode=True,
            )
