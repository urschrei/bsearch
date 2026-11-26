# bsearch

A local tool that monitors a Bluesky account via [Jetstream](https://github.com/bluesky-social/jetstream), capturing both your own posts and posts you've liked into a SQLite database with full-text and vector search. Designed to run as a background service on macOS.

- Indexes your own posts and the posts you like, including the alt text of their images and the text in them
- Searches by keyword, by meaning, or both, from the command line
- Optionally files the links in liked posts in an Instapaper folder
- Runs as a Rust daemon that uses about 20 MB when idle

## Quick start

```
uv sync
uv run bsearch export-model
cargo build --release
uv run bsearch install-service
./target/release/bsearch-search "your query"
```

Before `install-service`, create a `.env` file with your credentials. See
[Installing bsearch](docs/installation.md) for the full steps.

## Documentation

- [Installing bsearch](docs/installation.md): setup, loading your history and searching
- [Configuration](docs/configuration.md): `.env` keys, environment variables and storage locations
- [Commands](docs/commands.md): the Python CLI, the search binary and the daemon
- [Running the daemon](docs/running-the-daemon.md): the launchd service, logs and maintenance
- [How bsearch works](docs/how-it-works.md): Jetstream, memory use, Instapaper filing and text in images

## Licence

[The Blue Oak Model Licence 1.0](LICENSE.md)
