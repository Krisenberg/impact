"""Module implementing the Intention-Modulated ACT-R (P-Pop) memory activation model."""

from typing import final

import torch


@final
class ACTR:
    """Intention-Modulated Personalized ACT-R (P-Pop) model.

    The model operates as a plain Python class without trainable parameters. Tensors are
    processed dynamically on whichever device (CPU/GPU) the input arrays reside on.
    """

    def __init__(
        self,
        num_items: int,
        decay: float = 0.5,
        organic_weight: float = 1.0,
        autoplay_weight: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the ACT-R cognitive model.

        Args:
            num_items: Total number of unique items (excluding padding index 0).
            decay: Time decay parameter (d). Standard default is 0.5.
            organic_weight: Weight multiplier for organic (user-initiated) listens.
            autoplay_weight: Weight multiplier for autoplay (recsys-initiated) listens.
            eps: Tiny constant to prevent log(0) for unseen catalog items.
        """
        self.num_items = num_items
        self.decay = decay
        self.organic_weight = organic_weight
        self.autoplay_weight = autoplay_weight
        self.eps = eps

    def score_all_items(
        self,
        item_ids: torch.Tensor,
        time_deltas: torch.Tensor,
        is_organic: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Base-Level Activation (BLA) vector over the full item catalog.

        Index 0 represents the padding/unknown token and is always masked to -inf.

        Args:
            item_ids: 1D tensor containing 1-based mapped item indices for the user.
            time_deltas: 1D tensor containing the age of each interaction in seconds.
            is_organic: 1D boolean tensor marking whether each interaction was organic.

        Returns:
            A 1D tensor of shape [num_items + 1] containing BLA scores for all items.
        """
        # Map boolean is_organic flags to float weight values
        weights = torch.where(
            is_organic,
            torch.full_like(time_deltas, self.organic_weight),
            torch.full_like(time_deltas, self.autoplay_weight),
        )

        # Compute individual decay contributions
        contributions = weights * time_deltas.pow(-self.decay)

        # Initialize accumulator on the same device and dtype as input data
        accumulator = torch.zeros(
            self.num_items + 1,
            dtype=time_deltas.dtype,
            device=time_deltas.device,
        )

        # Sum up contributions indexed by item_ids (cumulative sum)
        _ = accumulator.scatter_add_(dim=0, index=item_ids, src=contributions)

        # Apply logarithm with eps floor to approximate -inf for unseen items
        # log(0 + 1e-8) is approx. -18.42, which behaves correctly during softmax
        bla = torch.log(accumulator + self.eps)

        # Ensure the padding token is never recommended
        bla[0] = -torch.inf

        return bla
