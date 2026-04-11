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
@click.pass_context
def cli(ctx, verbose: bool):
    """Bluesky post and like monitor with vector search."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    level = logging.DEBUG if verbose else logging.WARNING
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
@click.pass_context
def serve(ctx):
    """Run the Jetstream listener (foreground)."""
    if not ctx.obj.get("verbose"):
        logging.getLogger().setLevel(logging.INFO)

    async def _serve():
        config = Config.from_env()
        from bsearch.service import Service

        service = Service(config)
        await service.run()

    asyncio.run(_serve())


@cli.command()
@click.argument("query", default="")
@click.option("--limit", "-n", default=10, help="Number of results.")
@click.option(
    "--source",
    "-s",
    type=click.Choice(["own_post", "like", "backfill_post", "backfill_like"]),
    default=None,
    help="Filter by source type.",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["hybrid", "keyword", "semantic"]),
    default="hybrid",
    help="Search mode: hybrid (default), keyword (FTS only), or semantic (vector only).",
)
@click.option(
    "--handle",
    "-a",
    default=None,
    help="Filter by author handle (e.g. alice.bsky.social). "
    "With no query, lists all posts from this handle.",
)
def search(query: str, limit: int, source: str | None, mode: str, handle: str | None):
    """Search across indexed posts."""
    if not query and not handle:
        click.echo("Provide a query and/or --handle to search.", err=True)
        raise SystemExit(1)

    config = Config.from_env()
    from bsearch.db import Database

    db = Database(config.db_path)

    # No query text: list posts by handle
    if not query:
        assert handle is not None
        results = db.list_by_handle(handle, limit=limit, source_filter=source)
        if not results:
            click.echo("No results found.")
            db.close()
            return

        for i, r in enumerate(results, 1):
            web_url = _at_uri_to_web_url(r["uri"])
            click.echo(f"\n--- Result {i} ---")
            click.echo(f"Author:  {r['author_handle']}")
            click.echo(f"Date:    {r['created_at']}")
            click.echo(f"Source:  {r['source']}")
            click.echo(f"Link:    {web_url}")
            click.echo(f"Text:    {r['text']}")

        db.close()
        return

    query_embedding = None
    if mode in ("hybrid", "semantic"):
        from bsearch.embeddings import Embedder

        embedder = Embedder(config.embedding_model, quiet=True)
        query_embedding = embedder.encode_single(query)

    if mode == "keyword":
        results = db.search_fts(
            query, limit=limit, source_filter=source, handle_filter=handle
        )
    elif mode == "semantic":
        assert query_embedding is not None
        results = db.search(
            query_embedding, limit=limit, source_filter=source, handle_filter=handle
        )
    else:
        results = db.search_hybrid(
            query,
            query_embedding,
            limit=limit,
            source_filter=source,
            handle_filter=handle,
        )

    if not results:
        click.echo("No results found.")
        db.close()
        return

    for i, r in enumerate(results, 1):
        web_url = _at_uri_to_web_url(r["uri"])

        if mode == "hybrid" and "rrf_score" in r:
            score_info = f"score: {r['rrf_score']:.4f}, match: {r['match_type']}"
        elif mode == "keyword" and "bm25_rank" in r:
            score_info = f"bm25: {r['bm25_rank']:.4f}"
        elif "distance" in r:
            score_info = f"distance: {r['distance']:.4f}"
        else:
            score_info = ""

        click.echo(f"\n--- Result {i} ({score_info}) ---")
        click.echo(f"Author:  {r['author_handle']}")
        click.echo(f"Date:    {r['created_at']}")
        click.echo(f"Source:  {r['source']}")
        click.echo(f"Link:    {web_url}")
        click.echo(f"Text:    {r['text']}")

    db.close()


@cli.command()
@click.option("--limit", "-n", default=None, type=int, help="Max posts to fetch.")
@click.pass_context
def backfill(ctx, limit: int | None):
    """Fetch historical posts and likes via the AT Protocol API."""
    if not ctx.obj.get("verbose"):
        logging.getLogger().setLevel(logging.INFO)

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


@cli.command()
def vacuum():
    """Reclaim unused space and defragment the database."""
    import os

    config = Config.from_env()
    from bsearch.db import Database

    db_file = config.db_path
    size_before = os.path.getsize(db_file)

    db = Database(db_file)
    db.vacuum()
    db.close()

    size_after = os.path.getsize(db_file)
    saved = size_before - size_after

    click.echo(f"Database:  {db_file}")
    click.echo(f"Before:    {size_before:,} bytes")
    click.echo(f"After:     {size_after:,} bytes")
    click.echo(f"Reclaimed: {saved:,} bytes")


@cli.command("export-model")
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Output directory for ONNX model and tokenizer "
    "(default: ~/.cache/bsearch/<model-name>).",
)
def export_model(output_dir: str | None):
    """Export the embedding model to ONNX format for the Rust search binary."""
    from pathlib import Path

    config = Config.from_env()

    if output_dir is None:
        cache_dir = Path.home() / ".cache" / "bsearch" / config.embedding_model
    else:
        cache_dir = Path(output_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading model '{config.embedding_model}'...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.embedding_model)

    # Export to ONNX
    onnx_path = cache_dir / "model.onnx"
    click.echo(f"Exporting ONNX model to {onnx_path}...")

    import torch

    # Get the transformer module (first module in the SentenceTransformer pipeline)
    transformer = model[0]
    tokenizer = transformer.tokenizer
    # Move model to CPU for export; ONNX export does not support MPS
    bert_model = transformer.auto_model.cpu()

    # Create dummy input (keep on CPU to match model)
    dummy_text = "This is a dummy sentence for tracing."
    encoded = tokenizer(dummy_text, return_tensors="pt")

    # Export
    torch.onnx.export(
        bert_model,
        (encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
    )

    # Save tokenizer files to the cache directory so the Rust binary can load them
    click.echo(f"Saving tokenizer to {cache_dir}...")
    tokenizer.save_pretrained(str(cache_dir))

    click.echo(f"\nExport complete. Model directory: {cache_dir}")
    click.echo("The Rust binary will use this directory by default.")


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


def _at_uri_to_web_url(uri: str) -> str:
    """Convert an at:// URI to a clickable bsky.app URL."""
    # at://did:plc:abc/app.bsky.feed.post/rkey -> https://bsky.app/profile/did:plc:abc/post/rkey
    if not uri.startswith("at://"):
        return uri
    parts = uri.removeprefix("at://").split("/")
    if len(parts) >= 3 and parts[1] == "app.bsky.feed.post":
        return f"https://bsky.app/profile/{parts[0]}/post/{parts[2]}"
    return uri


if __name__ == "__main__":
    cli()
