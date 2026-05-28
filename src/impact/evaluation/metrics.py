"""Module containing batched, fully vectorized evaluation metrics for NBR."""

import torch


def compute_recall_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Calculate Recall@K for Next-Basket Recommendation (vectorized).

    Args:
        logits: Logits from the model of shape [B, num_items + 1] with index 0 masked.
        targets: Ground truth items of shape [batch_size, K]. Zeros indicate padding.
        k: The cut-off rank for evaluation.

    Returns:
        The mean Recall@K across the batch.
    """
    # Extract top-K predicted item indices
    _, topk_indices = torch.topk(logits, k=k, dim=-1)

    # Compute hits
    hits = torch.isin(elements=topk_indices, test_elements=targets).float().sum(dim=1)

    # Compute valid target lengths (ignoring padding index 0)
    target_lengths = (targets != 0).sum(dim=-1).float().clamp(min=1)

    # Recall is the number of hits divided by the number of ground truth items
    return (hits / target_lengths).mean().item()


def compute_ndcg_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Calculate NDCG@K for Next-Basket Recommendation (vectorized).

    Args:
        logits: Logits from the model of shape [B, num_items + 1] with index 0 masked.
        targets: Ground truth items of shape [batch_size, K]. Zeros indicate padding.
        k: The cut-off rank for evaluation.

    Returns:
        The mean NDCG@K across the batch.
    """
    # Extract top-K predicted item indices
    _, topk_indices = torch.topk(logits, k=k, dim=-1)

    # Calculate discounts: 1 / log2(rank + 1)
    ranks = torch.arange(2, k + 2, device=logits.device, dtype=torch.float)
    discounts = 1.0 / torch.log2(ranks)

    # Discounted Cumulative Gain (DCG)
    hits = torch.isin(elements=topk_indices, test_elements=targets).float()
    dcg = (hits * discounts).sum(dim=1)

    # Compute valid target lengths (ignoring padding index 0)
    target_lengths = (targets != 0).sum(dim=-1).float().clamp(min=1)

    # Ideal Discounted Cumulative Gain (IDCG)
    ideal_ranks = torch.arange(1, k + 1, device=logits.device).unsqueeze(0)
    ideal_hits = (ideal_ranks <= target_lengths.unsqueeze(1)).float()
    idcg = (ideal_hits * discounts).sum(dim=1).clamp(min=1e-8)

    return (dcg / idcg).mean().item()
