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
    pds_url: str = "https://bsky.social"
    db_path: Path = field(default_factory=lambda: Path("bsearch.db"))
    jetstream_url: str = "wss://jetstream2.us-east.bsky.network/subscribe"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    like_batch_interval: float = 2.0
    embedding_batch_interval: float = 10.0
    reconnect_cursor_safety_seconds: int = 5

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> Config:
        """Load configuration from a .env file.

        Supports both standard KEY=VALUE format and YAML-style key: value format.
        """
        if env_path is None:
            env_path = Path(".env")

        # Try standard dotenv loading first
        load_dotenv(env_path)

        # Also parse key: value format as a fallback
        env_values = _parse_env_file(env_path)

        handle = (
            os.environ.get("BSEARCH_HANDLE")
            or os.environ.get("user")
            or env_values.get("user", "")
        )
        app_password = (
            os.environ.get("BSEARCH_APP_PASSWORD")
            or os.environ.get("password")
            or env_values.get("password", "")
        )

        if not handle or not app_password:
            msg = (
                "Missing credentials. Set BSEARCH_HANDLE and BSEARCH_APP_PASSWORD "
                "in .env (or 'user' and 'password')."
            )
            raise ValueError(msg)

        db_path = Path(
            os.environ.get("BSEARCH_DB_PATH") or env_values.get("db_path", "bsearch.db")
        )
        pds_url = os.environ.get("BSEARCH_PDS_URL") or env_values.get(
            "pds_url", "https://bsky.social"
        )

        return cls(
            handle=handle,
            app_password=app_password,
            db_path=db_path,
            pds_url=pds_url,
        )


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file that may use 'key: value' or 'key=value' format."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
        elif ": " in line:
            key, _, value = line.partition(": ")
        else:
            continue
        values[key.strip()] = value.strip()
    return values
