from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import click

LABEL = "social.bsky.bsearch"
PLIST_DIR = Path.home() / "Library" / "LaunchAgents"
PLIST_PATH = PLIST_DIR / f"{LABEL}.plist"
LOG_DIR = Path.home() / "Library" / "Logs" / "bsearch"


def _find_bsearch_executable() -> str:
    """Find the bsearch executable path."""
    bsearch_path = shutil.which("bsearch")
    if bsearch_path:
        return bsearch_path
    # Fall back to the venv's bin directory
    venv_path = Path(sys.executable).parent / "bsearch"
    if venv_path.exists():
        return str(venv_path)
    msg = "Cannot find bsearch executable"
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
                <string>serve</string>
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
                <string>{os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin")}</string>
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
