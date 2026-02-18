from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration loaded from .env and defaults."""

    handle: str
    app_password: str
    did: str = ""
    db_path: Path = field(default_factory=lambda: Path("bsearch.db"))
    jetstream_url: str = "wss://jetstream2.us-east.bsky.network/subscribe"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    like_batch_interval: float = 2.0
    embedding_batch_interval: float = 10.0
    reconnect_cursor_safety_seconds: int = 5

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> Config:
        """Load configuration from a .env file."""
        if env_path is None:
            env_path = Path(".env")
        load_dotenv(env_path)

        handle = os.environ.get("BSEARCH_HANDLE") or os.environ.get("user", "")
        app_password = os.environ.get("BSEARCH_APP_PASSWORD") or os.environ.get(
            "password", ""
        )

        if not handle or not app_password:
            msg = (
                "Missing credentials. Set BSEARCH_HANDLE and BSEARCH_APP_PASSWORD "
                "in .env (or 'user' and 'password')."
            )
            raise ValueError(msg)

        db_path = Path(os.environ.get("BSEARCH_DB_PATH", "bsearch.db"))

        return cls(
            handle=handle,
            app_password=app_password,
            db_path=db_path,
        )
