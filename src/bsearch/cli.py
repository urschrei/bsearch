from __future__ import annotations

import asyncio
import logging
import sys
from datetime import UTC

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
        from datetime import datetime

        cursor_dt = datetime.fromtimestamp(stats["cursor"] / 1_000_000, tz=UTC)
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


@cli.command()
@click.option(
    "--batch-size",
    default=100,
    type=int,
    help="Number of posts to embed per batch.",
)
@click.pass_context
def reindex(ctx, batch_size: int):
    """Clear all embeddings and regenerate them from scratch."""
    if not ctx.obj.get("verbose"):
        logging.getLogger().setLevel(logging.INFO)

    config = Config.from_env()
    from bsearch.db import Database
    from bsearch.embeddings import Embedder

    db = Database(config.db_path)
    total = db.clear_embeddings()
    if total == 0:
        click.echo("No posts to re-embed.")
        db.close()
        return

    click.echo(f"Cleared embeddings. Re-embedding {total} posts...")
    embedder = Embedder(config.embedding_model)

    embedded = 0
    while True:
        pending = db.get_posts_without_embeddings(limit=batch_size)
        if not pending:
            break
        ids, texts = zip(*pending, strict=True)
        vectors = embedder.encode(list(texts))
        embeddings = list(zip(ids, vectors, strict=True))
        db.store_embeddings(embeddings)
        embedded += len(embeddings)
        click.echo(f"  {embedded}/{total} posts embedded")

    db.close()
    click.echo(f"Reindex complete. {embedded} embeddings generated.")


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


if __name__ == "__main__":
    cli()
