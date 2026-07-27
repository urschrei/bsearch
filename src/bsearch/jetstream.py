from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import websockets

if TYPE_CHECKING:
    from bsearch.config import Config
    from bsearch.models import JetstreamEvent

logger = logging.getLogger(__name__)

COLLECTIONS = [
    "app.bsky.feed.post",
    "app.bsky.feed.like",
]


def _notify(title: str, message: str) -> None:
    """Send a macOS notification via osascript."""
    import subprocess

    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message}" with title "{title}"',
            ],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        logger.debug("Failed to send notification", exc_info=True)


class JetstreamClient:
    """WebSocket client for Bluesky Jetstream with auto-reconnection."""

    def __init__(
        self,
        config: Config,
        on_event: Callable[[JetstreamEvent], None],
    ) -> None:
        self.config = config
        self.on_event = on_event
        self._running = False
        self._last_cursor: int | None = None

    def _build_url(self, cursor: int | None = None) -> str:
        params = {
            "wantedDids": self.config.did,
        }
        for collection in COLLECTIONS:
            params.setdefault("wantedCollections", [])
            if isinstance(params["wantedCollections"], list):
                params["wantedCollections"].append(collection)

        parts = []
        for key, value in params.items():
            if isinstance(value, list):
                for v in value:
                    parts.append((key, v))
            else:
                parts.append((key, value))

        if cursor is not None:
            parts.append(("cursor", str(cursor)))

        return f"{self.config.jetstream_url}?{urlencode(parts)}"

    async def run(self, initial_cursor: int | None = None) -> None:
        """Connect to Jetstream and process events, reconnecting on failure."""
        self._running = True
        self._last_cursor = initial_cursor

        while self._running:
            cursor = self._last_cursor
            if cursor is not None:
                # Subtract safety window on reconnect
                cursor -= self.config.reconnect_cursor_safety_seconds * 1_000_000

            url = self._build_url(cursor)
            logger.info("Connecting to Jetstream: %s", url)

            try:
                async with websockets.connect(url) as ws:
                    logger.info("Connected to Jetstream")
                    _notify("bsearch: connected", "Jetstream WebSocket connected.")
                    async for message in ws:
                        if not self._running:
                            break
                        self._process_message(message)
            except websockets.ConnectionClosed as e:
                logger.warning("Jetstream connection closed: %s", e)
                _notify("bsearch: disconnected", f"Connection closed: {e}")
            except Exception:
                logger.exception("Jetstream connection error")
                _notify("bsearch: error", "Jetstream connection error.")

            if self._running:
                logger.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    def stop(self) -> None:
        """Signal the client to stop."""
        self._running = False

    def _process_message(self, raw: str | bytes) -> None:
        """Parse a raw Jetstream message and dispatch it."""
        from bsearch.models import JetstreamEvent

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Jetstream message: %s", raw[:200])
            return

        if data.get("kind") != "commit":
            return

        commit = data.get("commit", {})
        collection = commit.get("collection", "")
        operation = commit.get("operation", "")

        if collection not in COLLECTIONS:
            return
        if operation != "create":
            return

        time_us = data.get("time_us")
        if time_us is not None:
            self._last_cursor = time_us

        event = JetstreamEvent(
            did=data.get("did", ""),
            time_us=time_us or 0,
            collection=collection,
            operation=operation,
            rkey=commit.get("rkey", ""),
            record=commit.get("record"),
            uri=f"at://{data.get('did', '')}/{collection}/{commit.get('rkey', '')}",
        )

        try:
            self.on_event(event)
        except Exception:
            logger.exception("Error processing event: %s", event.uri)
