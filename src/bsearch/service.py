from __future__ import annotations

import asyncio
import logging
import signal
from collections import deque
from datetime import datetime

from bsearch.config import Config
from bsearch.db import Database
from bsearch.embeddings import Embedder
from bsearch.jetstream import JetstreamClient
from bsearch.models import JetstreamEvent, Post
from bsearch.resolver import ATProtoResolver

logger = logging.getLogger(__name__)


class Service:
    """Async service orchestrator tying together Jetstream, resolver, and embeddings."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.db = Database(config.db_path)
        self.embedder = Embedder(config.embedding_model)
        self.resolver = ATProtoResolver(config)
        self._pending_like_uris: deque[str] = deque()
        self._shutdown_event = asyncio.Event()

    def _handle_event(self, event: JetstreamEvent) -> None:
        """Process a Jetstream event synchronously (called from the WS client)."""
        if event.collection == "app.bsky.feed.post":
            self._handle_own_post(event)
        elif event.collection == "app.bsky.feed.like":
            self._handle_like(event)

        # Update cursor
        if event.time_us:
            self.db.set_cursor(event.time_us)

    def _handle_own_post(self, event: JetstreamEvent) -> None:
        """Handle an own post event."""
        record = event.record or {}
        text = record.get("text", "")
        if not text:
            return

        created_at_str = record.get("createdAt")
        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            created_at = datetime.now()

        post = Post(
            uri=event.uri or "",
            cid="",
            author_did=event.did,
            author_handle=self.config.handle,
            text=text,
            created_at=created_at,
            source="own_post",
        )
        post_id = self.db.insert_post(post)
        if post_id is not None:
            logger.info("Indexed own post: %s", post.uri)

    def _handle_like(self, event: JetstreamEvent) -> None:
        """Handle a like event by queueing the subject URI for batch resolution."""
        record = event.record or {}
        subject = record.get("subject", {})
        uri = subject.get("uri")
        if uri:
            self._pending_like_uris.append(uri)
            logger.debug("Queued like for resolution: %s", uri)

    async def _resolve_likes_loop(self) -> None:
        """Periodically resolve queued like URIs."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.like_batch_interval)
            except asyncio.CancelledError:
                break

            if not self._pending_like_uris:
                continue

            # Drain the queue
            uris = []
            while self._pending_like_uris and len(uris) < 25:
                uris.append(self._pending_like_uris.popleft())

            try:
                posts = await self.resolver.resolve_post_uris(uris)
                for post in posts:
                    post_id = self.db.insert_post(post)
                    if post_id is not None:
                        logger.info("Indexed liked post: %s", post.uri)
            except Exception:
                logger.exception("Failed to resolve like batch")
                # Put URIs back in the queue
                self._pending_like_uris.extendleft(reversed(uris))

    async def _generate_embeddings_loop(self) -> None:
        """Periodically generate embeddings for new posts."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.embedding_batch_interval)
            except asyncio.CancelledError:
                break

            pending = self.db.get_posts_without_embeddings()
            if not pending:
                continue

            ids, texts = zip(*pending, strict=True)
            try:
                vectors = self.embedder.encode(list(texts))
                embeddings = list(zip(ids, vectors, strict=True))
                self.db.store_embeddings(embeddings)
                logger.info("Generated embeddings for %d posts", len(embeddings))
            except Exception:
                logger.exception("Failed to generate embeddings")

    async def run(self) -> None:
        """Run the service: Jetstream listener + like resolver + embedding generator."""
        await self.resolver.login()
        if not self.config.did:
            self.config.did = await self.resolver.resolve_handle(self.config.handle)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._signal_handler)

        jetstream = JetstreamClient(self.config, self._handle_event)
        initial_cursor = self.db.get_cursor()

        logger.info("Starting service for %s (%s)", self.config.handle, self.config.did)

        tasks = [
            asyncio.create_task(jetstream.run(initial_cursor)),
            asyncio.create_task(self._resolve_likes_loop()),
            asyncio.create_task(self._generate_embeddings_loop()),
        ]

        await self._shutdown_event.wait()
        logger.info("Shutting down...")

        jetstream.stop()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        self.db.close()
        logger.info("Service stopped")

    def _signal_handler(self) -> None:
        logger.info("Received shutdown signal")
        self._shutdown_event.set()
