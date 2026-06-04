"""Batched, fully vectorized NDCG@K and Recall@K metrics for Next-Basket Recommendation.

All functions accept a targets tensor where 0 denotes a non-evaluable item (item outside
the training index space). Samples containing no evaluable targets are excluded from the
mean. This is particularly important for repetition-/exploration-focused sub-metrics,
where a sample may contribute only to one target subset.
"""

import torch


def compute_recall_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Calculate mean Recall@K over all samples with at least one valid target.

    Args:
        logits: [B, num_items + 1] scores from the model with index 0 masked.
        targets: [B, K] ground-truth item indices; 0 denotes a non-evaluable item.
        k: Rank cutoff.

    Returns:
        Mean Recall@K across all valid samples. Returns 0.0 if no valid sample exist.
    """
    valid_mask = (targets != 0).any(dim=-1)  # [B]
    if not valid_mask.any():
        return 0.0  # No valid samples

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
    return recalls[valid_mask].mean().item()


def compute_ndcg_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Calculate mean NDCG@K over all samples with at least one valid target.

    Args:
        logits: [B, num_items + 1] scores from the model with index 0 masked.
        targets: [B, K] ground-truth item indices; 0 denotes a non-evaluable item.
        k: Rank cutoff.

    Returns:
        Mean NDCG@K across all valid samples. Returns 0.0 if no valid sample exist.
    """
    valid_mask = (targets != 0).any(dim=-1)  # [B]
    if not valid_mask.any():
        return 0.0  # No valid samples

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
    return ndcg[valid_mask].mean().item()
