"""Runtime reality checks: filesystem and git, not user prompt patterns."""

import subprocess
import tempfile
from pathlib import Path

import pytest

from compass.generators._types import Err, Ok
from compass.generators.trinity._types import Spec, Step
from compass.generators.trinity._runtime import validate_runtime_reality


def test_missing_read_file_path_errors():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        spec = Spec(
            question="q",
            steps=(
                Step(
                    step_id="s1",
                    description="read",
                    artifact_type="read_file",
                    artifact_ref="nope.txt",
                    inputs={},
                    expected_fact="c",
                ),
            ),
            synthesis="s",
        )
        r = validate_runtime_reality(spec, ws)
        assert isinstance(r, Err), r
        assert "does not exist" in r.error or "not a file" in r.error


def test_inline_only_always_ok_without_git():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        spec = Spec(
            question="q",
            steps=(
                Step(
                    step_id="s1",
                    description="x",
                    artifact_type="inline_python",
                    artifact_ref="result = 1",
                    inputs={},
                    expected_fact="a",
                ),
            ),
            synthesis="s",
        )
        r = validate_runtime_reality(spec, ws)
        assert isinstance(r, Ok), r


def test_git_shell_without_repo_errors():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        (ws / "dummy").write_text("x")
        spec = Spec(
            question="q",
            steps=(
                Step(
                    step_id="s1",
                    description="log",
                    artifact_type="shell",
                    artifact_ref="git log -1 --oneline",
                    inputs={},
                    expected_fact="x",
                ),
            ),
            synthesis="s",
        )
        r = validate_runtime_reality(spec, ws)
        assert isinstance(r, Err), r
        assert "git" in r.error.lower()


def test_git_shell_ok_inside_initialized_repo():
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        r_init = subprocess.run(["git", "init"], cwd=ws, capture_output=True)
        if r_init.returncode != 0:
            pytest.skip(f"git init unavailable: {r_init.stderr!r}")
        subprocess.run(
            ["git", "config", "user.email", "t@t.t"],
            cwd=ws,
            capture_output=True,
            check=False,
        )
        subprocess.run(
            ["git", "config", "user.name", "t"],
            cwd=ws,
            capture_output=True,
            check=False,
        )
        (ws / "f").write_text("h")
        subprocess.run(["git", "add", "f"], cwd=ws, capture_output=True, check=False)
        subprocess.run(
            ["git", "commit", "-m", "m"],
            cwd=ws,
            capture_output=True,
            check=False,
        )
        spec = Spec(
            question="q",
            steps=(
                Step(
                    step_id="s1",
                    description="log",
                    artifact_type="shell",
                    artifact_ref="git log -1 --oneline",
                    inputs={},
                    expected_fact="x",
                ),
            ),
            synthesis="s",
        )
        r = validate_runtime_reality(spec, ws)
        assert isinstance(r, Ok), r
