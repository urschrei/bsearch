import click

from bsearch import __version__


@click.group()
@click.version_option(version=__version__, prog_name="bsearch")
def cli():
    """Bluesky post and like monitor with vector search."""


if __name__ == "__main__":
    cli()
