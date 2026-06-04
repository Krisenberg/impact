from typing import final

import torch


@final
class ACTR:
    """Intention-Modulated Personalized ACT-R (P-Pop) scorer.

    Scores every catalog item by its Base-Level Activation (BLA) - a component of the
    ACT-R cognitive framework that models human memory by rewarding items the user has
    listened to often and recently. Organic (user-initiated) and autoplay listens
    contribute with different weights, letting the model distinguish genuine interest
    from passive exposure.

    This is a plain Python class, without any trainable parameters. Tensors are
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
        """Initialize the ACT-R scorer.

        Args:
            num_items: Catalog size, excluding padding index 0.
            decay: Time decay parameter, denoted as d in the thesis. Controls how fast
                old listens are forgotten. Standard default is 0.5.
            organic_weight: Weight for organic (user-initiated) listens.
            autoplay_weight: Weight for autoplay (recsys-initiated) listens.
            eps: Floor added before log to avoid log(0) for unseen items.
        """
        self._num_items = num_items
        self._decay = decay
        self._organic_weight = organic_weight
        self._autoplay_weight = autoplay_weight
        self._eps = eps

    def score_all_items(
        self,
        item_ids: torch.Tensor,
        time_deltas: torch.Tensor,
        is_organic: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Base-Level Activation scores over the full catalog for one user.

        Args:
            item_ids: [E] 1-based item indices from the training index space.
            time_deltas: [E] interaction ages in seconds (strictly positive).
            is_organic: [E] True for organic listens, False for autoplay.

        Returns:
            [num_items + 1] BLA scores. Index 0 (padding) representing the unknown items
            is masked to -inf. Unseen items receive log(eps) that for small eps is
            approximating -inf under softmax.
        """
        weight_diff = self._organic_weight - self._autoplay_weight
        weights = is_organic.to(time_deltas.dtype) * weight_diff + self._autoplay_weight

        # Compute individual decay contributions
        contributions = weights * time_deltas.pow(-self._decay)

        # Initialize accumulator on the same device and dtype as input data
        accumulator = torch.zeros(
            self._num_items + 1,
            dtype=time_deltas.dtype,
            device=time_deltas.device,
        )

        # Sum up contributions indexed by item_ids (scatter-accumulate keyed by index)
        _ = accumulator.scatter_add_(dim=0, index=item_ids, src=contributions)
        # This is GPU-vectorizable equivalent of:
        # for item, value in zip(item_ids, contributions, strict=True):
        #     accumulator[item] += value

        # Add eps for numerical stability, unseen items receive a small activation score
        bla = torch.log(accumulator + self._eps)

        # Ensure the padding token is never recommended
        bla[0] = -torch.inf

        return bla
