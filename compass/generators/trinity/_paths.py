"""Path helpers for locating the generators checkout and bundled artifacts."""

from __future__ import annotations

from pathlib import Path

def generators_repo_root() -> Path | None:
    """Directory that contains both ``compass/`` and top-level ``artifacts/``.

    Used so Trinity can resolve ``artifacts/screen.py`` and similar even when
    the user's workspace (e.g. neo-lab) is not the generators repo.
    """
    for d in Path(__file__).resolve().parents:
        if (d / "compass").is_dir() and (d / "artifacts").is_dir():
            return d
    return None


def bundled_artifact_module(artifact_type: str) -> Path | None:
    """Return ``artifacts/{artifact_type}.py`` under the generators repo if it exists.

    Any runnable module in that directory may be used as ``artifact_type`` when the
    workspace has no copy — tools work from any cwd.
    """
    root = generators_repo_root()
    if root is None:
        return None
    p = root / "artifacts" / f"{artifact_type}.py"
    return p if p.is_file() else None
