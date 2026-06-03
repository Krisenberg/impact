import torch
import torch.nn.functional as F  # noqa: N812


def bce_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Pointwise BCE loss over positive and negative item scores.

    Computes BCE(pos_scores, 1) + BCE(neg_scores, 0) where each term is averaged over
    the batch.

    Args:
        pos_scores: Positive item logits [B, K].
        neg_scores: Negative item logits [B, K].

    Returns:
        Scalar mean BCE loss.
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
