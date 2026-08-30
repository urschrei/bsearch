from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import click

LABEL = "social.bsky.bsearch"
ROTATE_LABEL = f"{LABEL}.logrotate"
PLIST_DIR = Path.home() / "Library" / "LaunchAgents"
PLIST_PATH = PLIST_DIR / f"{LABEL}.plist"
ROTATE_PLIST_PATH = PLIST_DIR / f"{ROTATE_LABEL}.plist"
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


def _generate_rotate_plist(script: str) -> str:
    """Generate the launchd plist XML for the log rotation agent.

    The daemon holds append-mode descriptors to its log files for as long
    as it runs, so the script rotates by copy-then-truncate rather than
    rename. A calendar job missed while the machine sleeps runs once on
    wake.
    """
    return dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
            "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{ROTATE_LABEL}</string>
            <key>ProgramArguments</key>
            <array>
                <string>{script}</string>
            </array>
            <key>StartCalendarInterval</key>
            <dict>
                <key>Hour</key>
                <integer>3</integer>
                <key>Minute</key>
                <integer>15</integer>
            </dict>
        </dict>
        </plist>
    """)


def _load(path: Path, label: str) -> None:
    """Load a plist, reporting failure as a warning."""
    result = subprocess.run(
        ["launchctl", "load", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        click.echo(f"Warning: launchctl load returned: {result.stderr}", err=True)
    else:
        click.echo(f"Service loaded: {label}")


def install_plist() -> None:
    """Generate and install the launchd plists for the daemon and log rotation."""
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
    _load(PLIST_PATH, LABEL)
    click.echo(f"Logs: {LOG_DIR}")

    rotate_script = Path(working_dir) / "scripts" / "bsearch-logrotate"
    if not rotate_script.exists():
        click.echo(
            f"Warning: {rotate_script} not found; log rotation not installed",
            err=True,
        )
        return
    ROTATE_PLIST_PATH.write_text(_generate_rotate_plist(str(rotate_script)))
    click.echo(f"Wrote plist to {ROTATE_PLIST_PATH}")
    _load(ROTATE_PLIST_PATH, ROTATE_LABEL)


def uninstall_plist() -> None:
    """Unload and remove the launchd plists."""
    for path, label in ((PLIST_PATH, LABEL), (ROTATE_PLIST_PATH, ROTATE_LABEL)):
        if not path.exists():
            click.echo(f"Plist not found: {path}")
            continue

        result = subprocess.run(
            ["launchctl", "unload", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            click.echo(f"Warning: launchctl unload returned: {result.stderr}", err=True)

        path.unlink()
        click.echo(f"Removed plist: {path}")
        click.echo(f"Service unloaded: {label}")
