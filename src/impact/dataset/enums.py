from enum import StrEnum


class DatasetType(StrEnum):
    """Dataset format type."""

    FLAT = "flat"
    SEQUENTIAL = "sequential"


class DatasetSize(StrEnum):
    """Dataset size."""

    SIZE_50M = "50m"
    SIZE_500M = "500m"
    SIZE_5B = "5b"


class InteractionType(StrEnum):
    """Interaction type."""

    LIKES = "likes"
    LISTENS = "listens"
    MULTI_EVENT = "multi_event"
    DISLIKES = "dislikes"
    UNLIKES = "unlikes"
    UNDISLIKES = "undislikes"
