from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import click

LABEL = "social.bsky.bsearch"
PLIST_DIR = Path.home() / "Library" / "LaunchAgents"
PLIST_PATH = PLIST_DIR / f"{LABEL}.plist"
LOG_DIR = Path.home() / "Library" / "Logs" / "bsearch"


def _find_bsearch_executable() -> str:
    """Find the bsearch-serve binary.

    The daemon is the Rust binary, not this Python package: it does the same
    work in roughly 20 MB rather than 2.5 GB, because it embeds via ONNX
    Runtime instead of loading PyTorch.
    """
    local_build = Path.cwd() / "target" / "release" / "bsearch-serve"
    if local_build.exists():
        return str(local_build)
    on_path = shutil.which("bsearch-serve")
    if on_path:
        return on_path
    msg = (
        "Cannot find the bsearch-serve binary. Build it first:\n"
        "    cargo build --release -p bsearch-serve"
    )
    raise FileNotFoundError(msg)


def _generate_plist(executable: str, working_dir: str) -> str:
    """Generate the launchd plist XML."""
    return dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
            "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{LABEL}</string>
            <key>ProgramArguments</key>
            <array>
                <string>{executable}</string>
            </array>
            <key>WorkingDirectory</key>
            <string>{working_dir}</string>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>StandardOutPath</key>
            <string>{LOG_DIR / "stdout.log"}</string>
            <key>StandardErrorPath</key>
            <string>{LOG_DIR / "stderr.log"}</string>
            <key>EnvironmentVariables</key>
            <dict>
                <key>PATH</key>
                <string>/usr/bin:/bin:/usr/sbin:/sbin</string>
            </dict>
        </dict>
        </plist>
    """)


def install_plist() -> None:
    """Generate and install the launchd plist."""
    try:
        executable = _find_bsearch_executable()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e

    working_dir = str(Path.cwd())

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLIST_DIR.mkdir(parents=True, exist_ok=True)

    plist_content = _generate_plist(executable, working_dir)
    PLIST_PATH.write_text(plist_content)
    click.echo(f"Wrote plist to {PLIST_PATH}")

    # Load the plist
    result = subprocess.run(
        ["launchctl", "load", str(PLIST_PATH)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Warning: launchctl load returned: {result.stderr}", err=True)
    else:
        click.echo(f"Service loaded: {LABEL}")
        click.echo(f"Logs: {LOG_DIR}")


def uninstall_plist() -> None:
    """Unload and remove the launchd plist."""
    if not PLIST_PATH.exists():
        click.echo(f"Plist not found: {PLIST_PATH}")
        return

    result = subprocess.run(
        ["launchctl", "unload", str(PLIST_PATH)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Warning: launchctl unload returned: {result.stderr}", err=True)

    PLIST_PATH.unlink()
    click.echo(f"Removed plist: {PLIST_PATH}")
    click.echo(f"Service unloaded: {LABEL}")
