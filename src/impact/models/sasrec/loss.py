"""Module defining training loss functions for SASRec."""

import torch
import torch.nn.functional as F  # noqa: N812


def bce_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Compute the pointwise Binary Cross-Entropy loss.

    Args:
        pos_scores: Logit scores for ground-truth items [B, K].
        neg_scores: Logit scores for randomly sampled negative items [B, K].

    Returns:
        A scalar tensor containing the mean BCE loss over the batch.
    """
    pos_loss = F.binary_cross_entropy_with_logits(
        input=pos_scores,
        target=torch.ones_like(pos_scores),
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        input=neg_scores,
        target=torch.zeros_like(neg_scores),
    )

    return pos_loss + neg_loss
