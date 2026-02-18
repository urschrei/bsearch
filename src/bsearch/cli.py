from __future__ import annotations

import asyncio
import logging
import sys

import click

from bsearch import __version__
from bsearch.config import Config


@click.group()
@click.version_option(version=__version__, prog_name="bsearch")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def cli(verbose: bool):
    """Bluesky post and like monitor with vector search."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@cli.command()
def init():
    """Resolve DID from handle and verify credentials."""

    async def _init():
        config = Config.from_env()
        click.echo(f"Handle:  {config.handle}")
        click.echo(f"PDS:     {config.pds_url}")

        from bsearch.resolver import ATProtoResolver

        resolver = ATProtoResolver(config)
        try:
            await resolver.login()
        except Exception as e:
            click.echo(f"Authentication failed: {e}", err=True)
            click.echo(
                "If your handle is on a custom PDS, set pds_url in .env "
                "(e.g. pds_url: https://pds.example.com)",
                err=True,
            )
            raise SystemExit(1) from e
        did = await resolver.resolve_handle(config.handle)
        config.did = did
        click.echo(f"DID:     {did}")
        click.echo("Authentication successful.")

    asyncio.run(_init())


@cli.command()
def serve():
    """Run the Jetstream listener (foreground)."""

    async def _serve():
        config = Config.from_env()
        from bsearch.service import Service

        service = Service(config)
        await service.run()

    asyncio.run(_serve())


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Number of results.")
@click.option(
    "--source",
    "-s",
    type=click.Choice(["own_post", "like", "backfill_post", "backfill_like"]),
    default=None,
    help="Filter by source type.",
)
def search(query: str, limit: int, source: str | None):
    """Semantic search across indexed posts."""
    config = Config.from_env()
    from bsearch.db import Database
    from bsearch.embeddings import Embedder

    db = Database(config.db_path)
    embedder = Embedder(config.embedding_model)

    query_embedding = embedder.encode_single(query)
    results = db.search(query_embedding, limit=limit, source_filter=source)

    if not results:
        click.echo("No results found.")
        db.close()
        return

    for i, r in enumerate(results, 1):
        click.echo(f"\n--- Result {i} (distance: {r['distance']:.4f}) ---")
        click.echo(f"Author:  {r['author_handle']}")
        click.echo(f"Date:    {r['created_at']}")
        click.echo(f"Source:  {r['source']}")
        click.echo(f"URI:     {r['uri']}")
        click.echo(f"Text:    {r['text']}")

    db.close()


@cli.command()
@click.option("--limit", "-n", default=None, type=int, help="Max posts to fetch.")
def backfill(limit: int | None):
    """Fetch historical posts and likes via the AT Protocol API."""

    async def _backfill():
        config = Config.from_env()
        from bsearch.db import Database
        from bsearch.embeddings import Embedder
        from bsearch.resolver import ATProtoResolver

        resolver = ATProtoResolver(config)
        await resolver.login()

        if not config.did:
            config.did = await resolver.resolve_handle(config.handle)

        db = Database(config.db_path)

        # Backfill own posts
        click.echo("Fetching own posts...")
        own_posts = await resolver.backfill_own_posts(limit=limit)
        own_count = 0
        for post in own_posts:
            if db.insert_post(post) is not None:
                own_count += 1
        click.echo(f"  Indexed {own_count} new posts (of {len(own_posts)} fetched).")

        # Backfill likes
        click.echo("Fetching liked posts...")
        liked_posts = await resolver.backfill_likes(limit=limit)
        like_count = 0
        for post in liked_posts:
            if db.insert_post(post) is not None:
                like_count += 1
        click.echo(f"  Indexed {like_count} new likes (of {len(liked_posts)} fetched).")

        # Generate embeddings
        pending = db.get_posts_without_embeddings(limit=10000)
        if pending:
            click.echo(f"Generating embeddings for {len(pending)} posts...")
            embedder = Embedder(config.embedding_model)
            ids, texts = zip(*pending, strict=True)
            vectors = embedder.encode(list(texts))
            embeddings = list(zip(ids, vectors, strict=True))
            db.store_embeddings(embeddings)
            click.echo(f"  Generated {len(embeddings)} embeddings.")

        db.close()
        click.echo("Backfill complete.")

    asyncio.run(_backfill())


@cli.command()
def status():
    """Show database statistics and cursor position."""
    config = Config.from_env()
    from bsearch.db import Database

    db = Database(config.db_path)
    stats = db.get_stats()

    click.echo(f"Database:    {config.db_path}")
    click.echo(f"Total posts: {stats['total_posts']}")
    click.echo(f"  With embeddings:    {stats['with_embeddings']}")
    click.echo(f"  Without embeddings: {stats['without_embeddings']}")

    if stats["by_source"]:
        click.echo("By source:")
        for source_name, count in sorted(stats["by_source"].items()):
            click.echo(f"  {source_name}: {count}")

    if stats["cursor"] is not None:
        from datetime import datetime, timezone

        cursor_dt = datetime.fromtimestamp(stats["cursor"] / 1_000_000, tz=timezone.utc)
        click.echo(f"Cursor:      {stats['cursor']} ({cursor_dt.isoformat()})")
    else:
        click.echo("Cursor:      not set")

    db.close()


@cli.command("install-service")
def install_service():
    """Generate and load a launchd plist for background operation."""
    from bsearch.launchd import install_plist

    install_plist()


@cli.command("uninstall-service")
def uninstall_service():
    """Unload and remove the launchd plist."""
    from bsearch.launchd import uninstall_plist

    uninstall_plist()


if __name__ == "__main__":
    cli()
