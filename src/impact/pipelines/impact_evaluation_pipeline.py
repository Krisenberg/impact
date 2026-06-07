from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, final, get_args

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from impact.evaluation.metrics import compute_ndcg_at_k, compute_recall_at_k
from impact.models.actr import ACTR, ACTRHistoryIndex
from impact.models.sasrec import (
    EvalSequenceBatch,
    EvalSequenceDataset,
    EvalSequenceSample,
    SASRecNBR,
    get_eval_collate_fn,
)
from impact.utils import seed_worker

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

    from impact.config import ImpactConfig
    from impact.schemas import ItemIndex, TensorizedSplit

logger = structlog.stdlib.get_logger(__name__)

# Configured evaluation scopes
EvalScopeKey = Literal["global", "local_rep", "local_exp", "global_rep", "global_exp"]


def _metric_name(metric: str, scope: EvalScopeKey, k: int) -> str:
    """Return a canonical metric name for a given k cutoff and evaluation scope."""
    return f"{metric}_at_{k}" if scope == "global" else f"{metric}_{scope}_at_{k}"


@dataclass(frozen=True, slots=True)
class _TargetSplits:
    """Ground-truth target tensors masked to each evaluation scope.

    Derived from the raw target tensor by zeroing out (0 is the padding token) items
    that do not belong to a given evaluation scope.

    Attributes:
        global_: [B, K] unmasked ground-truth targets.
        local_rep: [B, K] targets that appeared in the context window.
        local_exp: [B, K] targets that did not appear in the context window.
        global_rep: [B, K] targets that appeared previously in the user's full history.
        global_exp: [B, K] targets that did not appear previously in the user's history.
    """

    global_: torch.Tensor
    local_rep: torch.Tensor
    local_exp: torch.Tensor
    global_rep: torch.Tensor
    global_exp: torch.Tensor


@final
class IMPACTEvaluationPipeline:
    """Late-fusion evaluation pipeline for the IMPACT framework.

    Combines a trained SASRecNBR model with a deterministic ACT-R scorer via a weighted
    combination controlled by `ImpactConfig.eval.alpha` (1.0 = SASRec only, 0.0 = ACT-R
    only). Both models score the full item catalog; their probability distributions are
    then fused and evaluated against held-out test sessions.

    Reported metrics are NDCG@k and Recall@k decomposed across five evaluation scopes:
    - global: all target items.
    - local_rep / local_exp: items repeated from or absent in the L context sessions
        (i.e. short-term novelty).
    - global_rep / global_exp: items repeated from or absent in the user's full
        interaction history (i.e. long-term novelty).

    This decomposition lets the caller analyze the repetition-exploration trade-off
    introduced by varying alpha.
    """

    def __init__(
        self,
        config: ImpactConfig,
        item_index: ItemIndex,
        events_lf: pl.LazyFrame,
    ) -> None:
        """Initialize the IMPACT Framework evaluation pipeline.

        Args:
            config: Complete IMPACT Framework configuration.
            item_index: Train-derived index mapping raw item (track) IDs to contiguous
                1-based indices.
            events_lf: LazyFrame of all interaction events; used to build the ACT-R
                history index.
        """
        self._config = config
        self._item_index = item_index
        self._events_lf = events_lf

    @torch.no_grad()
    def evaluate(
        self,
        split: TensorizedSplit,
        sasrec_state_dict_path: Path,
    ) -> dict[str, float]:
        """Evaluate the IMPACT model on the held-out test split.

        Loads the SASRecNBR checkpoint, builds the ACT-R history index for test users,
        and runs batched inference over the full item catalog. Final scores are
        a combination of SASRec and ACT-R probabilities (late-fusion) controlled by
        `ImpactConfig.eval.alpha` (1.0 = SASRec only, 0.0 = ACT-R only).

        Metrics are computed at every cutoff listed in `ImpactConfig.eval.k_cutoffs`
        and returned as a flat dictionary. The key schema is:

            `{metric}_{scope}_at_{k}`

        where `metric` is `ndcg` or `recall`, `scope` is either empty (for the overall
        metric) or is one of `local_rep`, `local_exp`, `global_rep`, `global_exp`,
        and `k` is the cutoff. Examples: ndcg_at_5, recall_global_exp_at_10.

        Args:
            split: Tensorized train/val/test temporal splits; only the test partition
                is used here.
            sasrec_state_dict_path: Path to the saved SASRecNBR checkpoint.

        Returns:
            Flat metric dictionary with `10 * len(ImpactConfig.eval.k_cutoffs)` entries.
        """
        logger.info("Starting evaluation pipeline")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha = self._config.eval.alpha

        sasrec = self._load_sasrec(sasrec_state_dict_path, device)

        actr_history_index = ACTRHistoryIndex(
            events_lf=self._events_lf,
            eval_uids=np.unique(split.test.uid).tolist(),
            item_index=self._item_index,
        )
        actr = ACTR(
            num_items=self._item_index.num_items,
            decay=self._config.actr.decay,
            organic_weight=self._config.actr.organic_weight,
            autoplay_weight=self._config.actr.autoplay_weight,
            eps=self._config.actr.eps,
        )

        metric_names: list[str] = [
            _metric_name(metric, scope, k)
            for k in self._config.eval.k_cutoffs
            for metric in ("ndcg", "recall")
            for scope in get_args(EvalScopeKey)
        ]
        batch_metric_sums: dict[str, float] = dict.fromkeys(metric_names, 0.0)
        total_counts: dict[str, int] = dict.fromkeys(get_args(EvalScopeKey), 0)

        test_loader = DataLoader[EvalSequenceSample](
            dataset=EvalSequenceDataset(split.test),
            batch_size=self._config.eval.batch_size,
            shuffle=False,
            collate_fn=get_eval_collate_fn(),  # No negative sampling
            num_workers=self._config.system.num_workers,
            persistent_workers=self._config.system.num_workers > 0,
            worker_init_fn=seed_worker,
            pin_memory=torch.cuda.is_available(),
        )

        iterator = cast(Iterator[EvalSequenceBatch], iter(test_loader))
        for batch in tqdm(iterator, desc="Evaluation", leave=False):
            batch = batch.to(device)

            sasrec_logits = sasrec.score_all_items(sasrec.encode(context=batch.context))

            # Compute global-repetition mask and optionally BLA scores in a single pass
            is_global_repeated, bla_scores = self._score_actr_batch(
                batch=batch,
                actr=actr,
                actr_history_index=actr_history_index,
                device=device,
                compute_bla=alpha != 1.0,
            )

            # Late fusion
            if alpha == 1.0:
                final_scores = sasrec_logits
            else:
                if bla_scores is None:
                    raise ValueError("ACT-R scores must be computed when alpha != 1.0")

                # Normalize both logits to the same scale and then combine (late fusion)
                sasrec_probs = torch.softmax(sasrec_logits, dim=-1)
                del sasrec_logits  # Free up memory

                actr_probs = torch.softmax(bla_scores, dim=-1)
                del bla_scores  # Free up memory

                # final_scores = alpha * sasrec_probs + (1 - alpha) * actr_probs
                final_scores = sasrec_probs.mul_(alpha).add_(
                    actr_probs, alpha=1.0 - alpha
                )
                del actr_probs

            target_splits = self._derive_target_splits(
                targets=batch.positive,
                context=batch.context,
                is_global_repeated=is_global_repeated,
            )
            batch_metrics, counts = self._compute_batch_metrics(
                final_scores=final_scores,
                target_splits=target_splits,
            )

            for metric, sum_ in batch_metrics.items():
                batch_metric_sums[metric] += sum_
            for scope, count in counts.items():
                total_counts[scope] += count

        logger.info("Aggregating metrics across batches")
        metrics: dict[str, float] = {
            _metric_name(metric, scope, k): (
                batch_metric_sums[_metric_name(metric, scope, k)]
                / max(total_counts[scope], 1)  # guard against empty split
            )
            for k in self._config.eval.k_cutoffs
            for metric in ("ndcg", "recall")
            for scope in get_args(EvalScopeKey)
        }

        logger.info("Evaluation complete")
        return metrics

    def _load_sasrec(self, state_dict_path: Path, device: torch.device) -> SASRecNBR:
        """Load SASRecNBR from a checkpoint and set the model to eval mode."""
        sasrec = SASRecNBR(
            num_items=self._item_index.num_items,
            context_sessions=self._config.sequence.context_sessions,
            embedding_dim=self._config.sasrec.embedding_dim,
            num_heads=self._config.sasrec.num_heads,
            num_layers=self._config.sasrec.num_layers,
            dropout=self._config.sasrec.dropout_rate,
        )

        logger.info("Loading best SASRecNBR weights", path=state_dict_path.as_posix())
        sasrec.load_state_dict(torch.load(state_dict_path, map_location=device))
        sasrec.to(device)
        sasrec.eval()

        return sasrec

    def _score_actr_batch(
        self,
        batch: EvalSequenceBatch,
        actr: ACTR,
        actr_history_index: ACTRHistoryIndex,
        device: torch.device,
        *,
        compute_bla: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Query ACT-R history index once per user and return batch-level artifacts.

        For each user in the batch, a single ACT-R history lookup yields:
        - a global repetition mask required for computing metrics at different
            evaluation scopes,
        - ACT-R BLA scores used for the final IMPACT evaluation.

        Collecting both in one pass avoids querying the history index twice, which can
        be a bottleneck for larger dataset variants.

        Args:
            batch: Current evaluation batch.
            actr: Instantiated ACTR model.
            actr_history_index: Pre-indexed event histories for all evaluation users.
            device: Target compute device for output tensors.
            compute_bla: When False (i.e. pure SASRec, alpha == 1.0), BLA computation
                is skipped and the second output element is None.

        Returns:
            Results of resolving ACT-R history for the users in the evaluation batch:
            - [B, K] boolean mask indicating whether each target item appeared in the
                user's interaction history before the evaluation cutoff.
            - [B, num_items + 1] float tensor containing ACT-R Base-Level Activation
                scores for all items in the catalog, or None when BLA computation is
                skipped (pure-SASRec mode, alpha == 1.0).
        """
        batch_size = batch.positive.size(0)
        batch_uids = cast(list[int], batch.uid.tolist())
        batch_timestamps = cast(list[int], batch.target_ts.tolist())

        is_global_repeated = torch.zeros_like(batch.positive, dtype=torch.bool)

        # [B, num_items + 1] empty matrix for ACT-R scores (if required)
        bla_scores: torch.Tensor | None = (
            torch.empty(
                batch_size,
                self._item_index.num_items + 1,
                dtype=torch.float32,
                device=device,
            )
            if compute_bla
            else None
        )

        # Calculate the BLA for each row in the batch
        for b_idx in range(batch_size):
            item_ids, time_deltas, is_organic = actr_history_index.query(
                uid=batch_uids[b_idx],
                t_cutoff=batch_timestamps[b_idx],
            )
            item_ids = item_ids.to(device)
            is_global_repeated[b_idx] = torch.isin(batch.positive[b_idx], item_ids)

            if bla_scores is not None:
                bla_scores[b_idx] = actr.score_all_items(
                    item_ids=item_ids,
                    time_deltas=time_deltas.to(device),
                    is_organic=is_organic.to(device),
                )

        return is_global_repeated, bla_scores

    def _compute_batch_metrics(
        self,
        final_scores: torch.Tensor,
        target_splits: _TargetSplits,
    ) -> tuple[dict[str, float], dict[EvalScopeKey, int]]:
        """Compute metric sums (unnormalized) for one batch across all k cutoffs.

        For each configured k and each evaluation scope, computes NDCG@k and Recall@k
        weighted by the number of valid samples in that scope (value * count). Division
        by cumulative counts is done in `evaluate()` so that final metric averages are
        correct across batches.

        Args:
            final_scores: [B, num_items + 1] final score tensor after late-fusion with
                either probabilities or logits (relative ordering is what matters).
            target_splits: Scope-masked target tensors, as returned by
                `_derive_target_splits`.

        Returns:
            metric_sums: Metric sums keyed by canonical metric name (e.g., "ndcg_at_5",
                "recall_global_exp_at_5").
            counts: Number of valid (non-empty) samples per `EvalScopeKey`.
        """
        counts: dict[EvalScopeKey, int] = {
            "global": target_splits.global_.size(0),
            "local_rep": int((target_splits.local_rep != 0).any(dim=-1).sum().item()),
            "local_exp": int((target_splits.local_exp != 0).any(dim=-1).sum().item()),
            "global_rep": int((target_splits.global_rep != 0).any(dim=-1).sum().item()),
            "global_exp": int((target_splits.global_exp != 0).any(dim=-1).sum().item()),
        }

        metric_scopes: list[tuple[EvalScopeKey, torch.Tensor, int]] = [
            ("global", target_splits.global_, counts["global"]),
            ("local_rep", target_splits.local_rep, counts["local_rep"]),
            ("local_exp", target_splits.local_exp, counts["local_exp"]),
            ("global_rep", target_splits.global_rep, counts["global_rep"]),
            ("global_exp", target_splits.global_exp, counts["global_exp"]),
        ]

        batch_metrics: dict[str, float] = {}
        for k in self._config.eval.k_cutoffs:
            for scope, targets, count in metric_scopes:
                batch_metrics[_metric_name("ndcg", scope, k)] = (
                    compute_ndcg_at_k(final_scores, targets, k) * count
                )
                batch_metrics[_metric_name("recall", scope, k)] = (
                    compute_recall_at_k(final_scores, targets, k) * count
                )

        return batch_metrics, counts

    def _derive_target_splits(
        self,
        targets: torch.Tensor,
        context: torch.Tensor,
        is_global_repeated: torch.Tensor,
    ) -> _TargetSplits:
        """Derive scope-masked target tensors for all evaluation scopes.

        Args:
            targets: [B, K] ground-truth target sessions (with 0 as the padding token).
            context: [B, L, K] context sessions for local scope evaluation.
            is_global_repeated: [B, K] boolean mask indicating if target items exist in
                the user's global history.

        Returns:
            _TargetSplits with one [B, K] masked target tensor per scope.
        """
        context_flat = context.flatten(start_dim=1)  # [B, L*K]
        # [B, L*K] replace padding with -1 to 'release' the padding token
        context_valid = context_flat.masked_fill(context_flat == 0, -1)

        # [B, K, 1] == [B, 1, L*K] --broadcast--> [B, K, L*K] -> [B, K]
        valid = targets != 0
        _is_local_rep = (targets.unsqueeze(2) == context_valid.unsqueeze(1)).any(dim=-1)

        # Local scope
        is_local_rep = _is_local_rep & valid
        is_local_exp = (~_is_local_rep) & valid

        # Global scope
        is_global_rep = is_global_repeated & valid
        is_global_exp = (~is_global_repeated) & valid

        return _TargetSplits(
            global_=targets,
            local_rep=targets.masked_fill(~is_local_rep, 0),
            local_exp=targets.masked_fill(~is_local_exp, 0),
            global_rep=targets.masked_fill(~is_global_rep, 0),
            global_exp=targets.masked_fill(~is_global_exp, 0),
        )
