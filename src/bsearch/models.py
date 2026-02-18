from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Post:
    """A Bluesky post (own or liked)."""

    uri: str
    cid: str
    author_did: str
    author_handle: str
    text: str
    created_at: datetime
    source: str  # "own_post", "like", "backfill_post", "backfill_like"
    indexed_at: datetime = field(default_factory=datetime.now)
    id: int | None = None
    has_embedding: bool = False


@dataclass
class JetstreamEvent:
    """A parsed Jetstream WebSocket event."""

    did: str
    time_us: int
    collection: str
    operation: str  # "create", "delete"
    rkey: str
    record: dict | None = None
    uri: str | None = None
