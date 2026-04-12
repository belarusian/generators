"""Shared environment loading for Compass.

Loads project-local `.env` first, then `~/.compass/.env` to fill any gaps.
This keeps direct library entry points consistent with the CLI behavior.
"""

from __future__ import annotations

from pathlib import Path


def load_compass_env() -> None:
    """Load local and shared Compass environment variables."""
    from dotenv import load_dotenv

    load_dotenv()  # local .env in cwd wins
    shared_env = Path.home() / ".compass" / ".env"
    if shared_env.exists():
        load_dotenv(shared_env)  # fill only missing values
