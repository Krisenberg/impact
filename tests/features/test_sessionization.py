"""Unit tests for impact.features.sessionization.sessionize()."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from impact.features.sessionization import sessionize


@dataclass(frozen=True)
class ListenEvent:
    """Listen event builder. Default values pass the 30-second playback threshold."""

    uid: int
    timestamp: int
    item_id: int = 123
    is_organic: Literal[0, 1] = 0
    played_ratio_pct: int = 100
    track_length_seconds: int = 200
    event_type: Literal["listen"] = "listen"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a raw event dict."""
        return self.__dict__


@dataclass(frozen=True)
class LikeEvent:
    """Like event builder. Playback fields are always None."""

    uid: int
    timestamp: int
    item_id: int = 123
    is_organic: Literal[0, 1] = 0
    played_ratio_pct: None = None
    track_length_seconds: None = None
    event_type: Literal["like"] = "like"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a raw event dict for pipeline input."""
        return self.__dict__


class TestEventTypeFilter:
    """Test that non-'listen' events are filtered out."""

    def test_non_listen_events_are_removed(self) -> None:
        events = [
            ListenEvent(uid=100, timestamp=20),
            ListenEvent(uid=100, timestamp=60),
            LikeEvent(uid=100, timestamp=120),
        ]
        events_lf = pl.LazyFrame([e.to_dict() for e in events])
        # K=1 so a single-listen session survives
        df = sessionize(events_lf, tracks_per_session=1).collect()
        assert (df["event_type"] == "listen").all()

    def test_only_non_listen_events_returns_empty(self) -> None:
        events = [
            LikeEvent(uid=100, timestamp=120),
            LikeEvent(uid=100, timestamp=1000),
        ]
        events_lf = pl.LazyFrame([e.to_dict() for e in events])
        # K=1 so a single-listen session survives
        df = sessionize(events_lf, tracks_per_session=1).collect()
        assert df.is_empty()
