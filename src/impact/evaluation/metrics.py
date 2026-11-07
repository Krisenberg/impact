"""Batched, fully vectorized ranking metrics for Next-Basket Recommendation.

Provides Recall@k and NDCG@k implementations that support both per-sample and
aggregated evaluation.

Ground-truth target tensors use 0 as the non-evaluable item token (item outside the
training index space). Per-sample metric functions return NaN for samples without any
valid targets. Samples containing no evaluable targets are excluded from the mean in the
aggregated evaluation. This is particularly important for rep-/exp-focused sub-metrics,
where a sample may contribute only to one target subset.
"""

import torch


def compute_per_sample_recall_at_k(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Calculate per-sample Recall@k.

    Args:
        logits: [B, num_items + 1] scores from the model with index 0 masked.
        targets: [B, K] ground-truth item indices; 0 denotes a non-evaluable item.
        k: Rank cutoff.

    Returns:
        Tensor of shape [B] containing Recall@k for each sample. Samples without at
        least one valid target receive NaN.
    """
    valid_mask = (targets != 0).any(dim=-1)  # [B]
    if not valid_mask.any():
        # No valid samples
        return torch.full((logits.shape[0],), torch.nan, device=logits.device)

    # Extract top-K predicted item indices
    _, topk_indices = torch.topk(logits, k=k, dim=-1)  # [B, k]

    # Per-row hit detection [B, k]
    hits = (
        (
            # [B, k, 1] == [B, 1, K] -> [B, k, K] (after broadcast)
            topk_indices.unsqueeze(2) == targets.unsqueeze(1)
        )
        .any(dim=-1)
        .float()
    )
    num_hits = hits.sum(dim=1)  # [B]

    # Compute valid target lengths (ignoring padding index 0)
    target_lengths = (targets != 0).sum(dim=-1).float().clamp(min=1)  # [B]

    # Recall is the number of hits divided by the number of ground truth items
    recalls = num_hits / target_lengths  # [B]
    recalls = recalls.masked_fill(~valid_mask, torch.nan)
    return recalls


def compute_recall_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Calculate mean Recall@k over all samples with at least one valid target.

    Args:
        logits: [B, num_items + 1] scores from the model with index 0 masked.
        targets: [B, K] ground-truth item indices; 0 denotes a non-evaluable item.
        k: Rank cutoff.

    Returns:
        Mean Recall@k across all valid samples.
    """
    return torch.nanmean(compute_per_sample_recall_at_k(logits, targets, k)).item()


def compute_per_sample_ndcg_at_k(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Calculate per-sample NDCG@k.

    Args:
        logits: [B, num_items + 1] scores from the model with index 0 masked.
        targets: [B, K] ground-truth item indices; 0 denotes a non-evaluable item.
        k: Rank cutoff.

    Returns:
        Tensor of shape [B] containing NDCG@k for each sample. Samples without at
        least one valid target receive NaN.
    """
    valid_mask = (targets != 0).any(dim=-1)  # [B]
    if not valid_mask.any():
        # No valid samples
        return torch.full((logits.shape[0],), torch.nan, device=logits.device)

    # Extract top-K predicted item indices
    _, topk_indices = torch.topk(logits, k=k, dim=-1)  # [B, k]

    # Calculate discounts: 1 / log2(rank + 1) [k]
    ranks = torch.arange(2, k + 2, device=logits.device, dtype=torch.float)
    discounts = 1.0 / torch.log2(ranks)

    # Per-row hit detection [B, k]
    hits = (
        (
            # [B, k, 1] == [B, 1, K] -> [B, k, K] (after broadcast)
            topk_indices.unsqueeze(2) == targets.unsqueeze(1)
        )
        .any(dim=-1)
        .float()
    )

    # Discounted Cumulative Gain (DCG)
    dcg = (hits * discounts).sum(dim=1)

    # Compute valid target lengths (ignoring padding index 0)
    target_lengths = (targets != 0).sum(dim=-1).float()  # [B]

    # Ideal DCG: best possible ranking given the number of relevant items
    ideal_ranks = torch.arange(1, k + 1, device=logits.device)  # [k]
    ideal_hits = (ideal_ranks.unsqueeze(0) <= target_lengths.unsqueeze(1)).float()
    idcg = (ideal_hits * discounts).sum(dim=1).clamp(min=1e-8)  # [B]

    ndcg = dcg / idcg  # [B]
    ndcg = ndcg.masked_fill(~valid_mask, torch.nan)
    return ndcg


def compute_ndcg_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Calculate mean NDCG@k over all samples with at least one valid target.

    Args:
        logits: [B, num_items + 1] scores from the model with index 0 masked.
        targets: [B, K] ground-truth item indices; 0 denotes a non-evaluable item.
        k: Rank cutoff.

    Returns:
        Mean NDCG@k across all valid samples.
    """
    return torch.nanmean(compute_per_sample_ndcg_at_k(logits, targets, k)).item()
