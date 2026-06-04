"""Module defining the IMPACT Framework configuration."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from impact.dataset import DatasetSize, DatasetType, InteractionType


@dataclass(frozen=True)
class ACTRConfig:
    """Configuration for the Intention-Modulated Personalized ACT-R (P-Pop) scorer."""

    decay: float = 0.5
    """Power-law time decay exponent. Denoted as d in the thesis.

    Higher values result in older listens being forgotten faster.
    """
    organic_weight: float = 1.0
    """Contribution weight for organic (user-initiated) listens."""
    autoplay_weight: float = 0.5
    """Contribution weight for autoplay (recsys-initiated) listens."""
    eps: float = 1e-8
    """Numerical floor added before log to prevent log(0) for unseen catalog items."""


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
    batch_size: int = 2048
    """Number of sequences per training batch."""
    max_epochs: int = 100
    """Maximum number of training epochs to run."""
    learning_rate: float = 0.001
    """Learning rate for the optimizer."""
    k_cutoff: int = 10
    """Cutoff for NDCG@K and Recall@K metrics used during model training."""
    patience: int = 10
    """Number of epochs to wait for validation improvement before stopping."""


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for the late-fusion evaluation of SASRecNBR and ACT-R (IMPACT)."""

    batch_size: int = 512
    """Number of sequences per evaluation batch.

    Should be lower than `SASRecConfig.batch_size` due to full-catalog scoring.
    """
    k_cutoffs: tuple[int, ...] = (5, 10, 40, 100)
    """Rank cutoffs for computing NDCG@K and Recall@K during evaluation."""
    alpha: float = 0.5
    """Relative contribution of SASRec and ACT-R in the final late-fusion score.

    The final score is computed as:

        alpha * SASRec + (1 - alpha) * ACT-R

    where 1.0 corresponds to SASRec-only recommendations and 0.0 to ACT-R-only
    recommendations.
    """


@dataclass(frozen=True)
class ImpactConfig:
    """Immutable configuration for the IMPACT Framework.

    Single source of truth for all parameters governing data processing, feature
    engineering, model training and evaluation. Immutability ensures reproducibility
    across runs. The entire configuration can be easily serialized to a dict or YAML
    file for exact reconstruction of any pipeline artifact.
    """

    # === Dataset ===
    dataset_size: DatasetSize
    """Yambda dataset variant to process (50M, 500M, or 5B)."""
    dataset_type: DatasetType = DatasetType.FLAT
    """Dataset format type. Currently only FLAT is supported."""
    interaction_type: InteractionType = InteractionType.MULTI_EVENT
    """Type of interactions to load from the Yambda dataset.

    MULTI_EVENT includes all event types (listen, like, etc.); other variants contain
    only a specific subset.
    """

    # === Sessions ===
    inactivity_gap_seconds: int = 1200
    """Number of seconds of inactivity before a session is considered finished."""
    tracks_per_session: int = 10
    """Number of tracks that consitute a session. Denoted as K in the thesis.

    Sessions with less than K tracks are discarded. Sessions with more than K tracks are
    truncated.
    """
    min_listen_seconds: int = 30
    """Minimum playback duration (in seconds) to consider an event as a valid listen.

    Defaults to 30 (industry standard used by authors of other publications).
    """

    # === Sequences ===
    context_sessions: int = 20
    """Number of historical sessions used as context. Denoted as L in the thesis."""
    step: int = 5
    """Step size for the sliding window sequence generation."""

    # === Timesplit (GTS) ===
    test_days: float = 1.0
    """Number of days at the end of the timeline allocated for the test set."""
    val_days: float = 1.0
    """Number of days preceding test set allocated for the validation set."""
    gap_minutes: float = 30.0
    """Buffer time (in minutes) between dataset splits.

    The purpose of this parameter is to mimic production system deployment latency.
    """

    # === ACTR ===
    actr: ACTRConfig = field(default_factory=ACTRConfig)
    """Configuration for the Intention-Modulated Personalized ACT-R (P-Pop) scorer."""

    # === SASRecNBR ===
    sasrec: SASRecConfig = field(default_factory=SASRecConfig)
    """Configuration for the SASRecNBR model."""

    # === Evaluation ===
    eval: EvalConfig = field(default_factory=EvalConfig)
    """Configuration for the late-fusion evaluation of SASRecNBR and ACT-R (IMPACT)."""

    # === System / Hardware ===
    seed: int = 42
    """Global random seed for strict reproducibility."""
    num_workers: int = 4
    """Number of subprocesses to use for data loading.

    Set to 0 to load data in the main process, or higher for parallel loading
    (hardware-dependent).
    """

    def __post_init__(self) -> None:
        """Verify that the selected dataset type is supported."""
        if self.dataset_type != DatasetType.FLAT:
            raise ValueError(f"Dataset type {self.dataset_type} not supported.")

    @staticmethod
    def _format_float_value(val: float) -> str:
        """Format float values to be filesystem-friendly (no dots)."""
        return f"{int(val)}" if val.is_integer() else str(val).replace(".", "-")

    @property
    def dataset_id(self) -> str:
        """Cache key for the raw data fetching stage."""
        return "_".join([
            "dataset",
            self.dataset_size,
            self.dataset_type,
            self.interaction_type,
        ])  # fmt: skip

    @property
    def sessionization_id(self) -> str:
        """Cache key for the session extraction stage (Algorithm 1 in the thesis)."""
        return "_".join([
            "sessions",
            f"gap{self.inactivity_gap_seconds}",
            f"k{self.tracks_per_session}",
            f"min{self.min_listen_seconds}",
        ])  # fmt: skip

    @property
    def sequencing_id(self) -> str:
        """Cache key for the sequence extraction stage (Algorithm 2 in the thesis)."""
        return "_".join([
            "sequences",
            f"l{self.context_sessions}",
            f"step{self.step}",
        ])  # fmt: skip

    @property
    def timesplit_id(self) -> str:
        """Cache key for the temporal split (GTS) stage (Algorithm 3 in the thesis)."""
        val_str = self._format_float_value(self.val_days)
        test_str = self._format_float_value(self.test_days)
        gap_str = self._format_float_value(self.gap_minutes)
        return "_".join([
            "split",
            f"val{val_str}d",
            f"test{test_str}d",
            f"gap{gap_str}m",
        ])  # fmt: skip

    @property
    def run_id(self) -> str:
        """Deterministic key representing the full pipeline configuration."""
        return "__".join([
            self.dataset_id,
            self.sessionization_id,
            self.sequencing_id,
            self.timesplit_id,
        ])  # fmt: skip

    @property
    def config_hash(self) -> str:
        """SHA-256 hash of the full run_id."""
        return hashlib.sha256(self.run_id.encode()).hexdigest()

    @property
    def display_id(self) -> str:
        """Short human-readable ID, e.g. '5B-10K-20L-20S-a1b2c3d'.

        Human-readable ID includes the following components:
        - Dataset size, uppercased (50M, 500M, 5B),
        - Tracks per session (K),
        - Context sessions (L),
        - Step size (S).
        - Last 7 characters of the SHA-256 hash of the run_id (e.g. 'a1b2c3d').
        """
        return "-".join([
            self.dataset_size.value.upper(),
            f"{self.tracks_per_session}K",
            f"{self.context_sessions}L",
            f"{self.step}S",
            self.config_hash[:7],
        ])  # fmt: skip

    def to_dict(self, nested: bool = False) -> dict[str, Any]:
        """Serialize the IMPACT Framework configuration to a dictionary.

        Args:
            nested: If True, parameters are grouped by section (e.g. `dataset`,
            `sasrec`). If False, all parameters are returned as a flat mapping of
            field names to values.
        """
        if nested:
            return {
                "dataset": {
                    "dataset_size": self.dataset_size.value,
                    "dataset_type": self.dataset_type.value,
                    "interaction_type": self.interaction_type.value,
                },
                "sessions": {
                    "inactivity_gap_seconds": self.inactivity_gap_seconds,
                    "tracks_per_session": self.tracks_per_session,
                    "min_listen_seconds": self.min_listen_seconds,
                },
                "sequences": {
                    "context_sessions": self.context_sessions,
                    "step": self.step,
                },
                "timesplit": {
                    "test_days": self.test_days,
                    "val_days": self.val_days,
                    "gap_minutes": self.gap_minutes,
                },
                "actr": {
                    "decay": self.actr.decay,
                    "organic_weight": self.actr.organic_weight,
                    "autoplay_weight": self.actr.autoplay_weight,
                    "eps": self.actr.eps,
                },
                "sasrec": {
                    "embedding_dim": self.sasrec.embedding_dim,
                    "num_heads": self.sasrec.num_heads,
                    "num_layers": self.sasrec.num_layers,
                    "dropout_rate": self.sasrec.dropout_rate,
                    "batch_size": self.sasrec.batch_size,
                    "max_epochs": self.sasrec.max_epochs,
                    "learning_rate": self.sasrec.learning_rate,
                    "k_cutoff": self.sasrec.k_cutoff,
                    "patience": self.sasrec.patience,
                },
                "eval": {
                    "batch_size": self.eval.batch_size,
                    "k_cutoffs": self.eval.k_cutoffs,
                    "alpha": self.eval.alpha,
                },
                "system": {
                    "seed": self.seed,
                    "num_workers": self.num_workers,
                },
            }
        return {
            "dataset_size": self.dataset_size.value,
            "dataset_type": self.dataset_type.value,
            "interaction_type": self.interaction_type.value,
            "inactivity_gap_seconds": self.inactivity_gap_seconds,
            "tracks_per_session": self.tracks_per_session,
            "min_listen_seconds": self.min_listen_seconds,
            "context_sessions": self.context_sessions,
            "step": self.step,
            "test_days": self.test_days,
            "val_days": self.val_days,
            "gap_minutes": self.gap_minutes,
            "actr_decay": self.actr.decay,
            "actr_organic_weight": self.actr.organic_weight,
            "actr_autoplay_weight": self.actr.autoplay_weight,
            "actr_eps": self.actr.eps,
            "sasrec_embedding_dim": self.sasrec.embedding_dim,
            "sasrec_num_heads": self.sasrec.num_heads,
            "sasrec_num_layers": self.sasrec.num_layers,
            "sasrec_dropout_rate": self.sasrec.dropout_rate,
            "sasrec_batch_size": self.sasrec.batch_size,
            "sasrec_max_epochs": self.sasrec.max_epochs,
            "sasrec_learning_rate": self.sasrec.learning_rate,
            "sasrec_k_cutoff": self.sasrec.k_cutoff,
            "sasrec_patience": self.sasrec.patience,
            "eval_batch_size": self.eval.batch_size,
            "eval_k_cutoffs": self.eval.k_cutoffs,
            "eval_alpha": self.eval.alpha,
            "seed": self.seed,
            "num_workers": self.num_workers,
        }

    def write_yaml(self, output_path: Path) -> None:
        """Write the configuration to a YAML file.

        Args:
            output_path: Target path where the YAML file with the configuration snapshot
                should be saved.
        """
        import yaml

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                data=self.to_dict(),
                stream=f,
                default_flow_style=False,
                sort_keys=False,  # Prevents from sorting keys alphabetically
                allow_unicode=True,
            )
