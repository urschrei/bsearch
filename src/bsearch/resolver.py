from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from atproto import AsyncClient, models

if TYPE_CHECKING:
    from bsearch.config import Config
    from bsearch.models import Post

logger = logging.getLogger(__name__)


class ATProtoResolver:
    """AT Protocol API client for resolving posts and backfilling history."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = AsyncClient(base_url=config.pds_url)
        self._logged_in = False

    async def login(self) -> None:
        """Authenticate with the AT Protocol server."""
        await self.client.login(
            login=self.config.handle,
            password=self.config.app_password,
        )
        self._logged_in = True
        logger.info("Logged in as %s", self.config.handle)

    async def _ensure_logged_in(self) -> None:
        if not self._logged_in:
            await self.login()

    async def resolve_handle(self, handle: str) -> str:
        """Resolve a handle to a DID."""
        await self._ensure_logged_in()
        response = await self.client.resolve_handle(handle)
        return response.did

    async def resolve_post_uris(self, uris: list[str]) -> list[Post]:
        """Resolve a batch of post URIs (max 25 per call) to Post objects."""

        await self._ensure_logged_in()
        if not uris:
            return []

        posts = []
        # Process in chunks of 25 (API limit)
        for i in range(0, len(uris), 25):
            chunk = uris[i : i + 25]
            try:
                response = await self.client.get_posts(chunk)
                for post_view in response.posts:
                    post = _post_view_to_post(post_view, source="like")
                    if post is not None:
                        posts.append(post)
            except Exception:
                logger.exception("Failed to resolve post URIs: %s", chunk)
        return posts

    async def backfill_own_posts(self, limit: int | None = None) -> list[Post]:
        """Fetch historical posts by the authenticated user."""

        await self._ensure_logged_in()
        posts = []
        prev_cursor = None
        cursor = None
        count = 0

        while True:
            page_limit = min(100, limit - count) if limit else 100
            response = await self.client.get_author_feed(
                actor=self.config.did or self.config.handle,
                cursor=cursor,
                filter="posts_no_replies",
                limit=page_limit,
            )

            for feed_item in response.feed:
                # Skip reposts
                if feed_item.reason is not None:
                    continue
                post = _post_view_to_post(feed_item.post, source="backfill_post")
                if post is not None:
                    posts.append(post)
                    count += 1
                    if limit and count >= limit:
                        return posts

            prev_cursor = cursor
            cursor = response.cursor
            if not cursor or cursor == prev_cursor:
                break

        return posts

    async def backfill_likes(self, limit: int | None = None) -> list[Post]:
        """Fetch historically liked posts."""
        await self._ensure_logged_in()
        posts = []
        prev_cursor = None
        cursor = None
        count = 0

        while True:
            page_limit = min(100, limit - count) if limit else 100
            response = await self.client.app.bsky.feed.get_actor_likes(
                models.AppBskyFeedGetActorLikes.Params(
                    actor=self.config.did or self.config.handle,
                    cursor=cursor,
                    limit=page_limit,
                )
            )

            for feed_item in response.feed:
                post = _post_view_to_post(feed_item.post, source="backfill_like")
                if post is not None:
                    posts.append(post)
                    count += 1
                    if limit and count >= limit:
                        return posts

            prev_cursor = cursor
            cursor = response.cursor
            if not cursor or cursor == prev_cursor:
                break

        return posts


def _post_view_to_post(post_view, source: str) -> Post | None:
    """Convert an AT Protocol PostView to our Post model."""
    from bsearch.models import Post as PostModel

    record = post_view.record
    text = getattr(record, "text", None) or ""
    if not text:
        return None

    created_at_str = getattr(record, "created_at", None)
    if created_at_str:
        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            created_at = datetime.now()
    else:
        created_at = datetime.now()

    author = post_view.author
    return PostModel(
        uri=post_view.uri,
        cid=post_view.cid,
        author_did=author.did,
        author_handle=author.handle or "",
        text=text,
        created_at=created_at,
        source=source,
    )
