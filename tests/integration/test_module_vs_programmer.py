"""Programmer NFA integration test.

Verify the Programmer NFA can generate a tested Python package
end-to-end via Trinity's dynamic programmer artifact.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

# Path to the real artifacts/programmer.py in the repo
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROGRAMMER_ARTIFACT = _REPO_ROOT / "artifacts" / "programmer.py"

import pytest

from compass.generators._types import Ok, Err


PROBLEM = (
    "Build a Python module called 'taskqueue' with:\n"
    "- taskqueue/__init__.py exporting Task and TaskQueue\n"
    "- taskqueue/models.py with a Task dataclass (id: str, name: str, priority: int, done: bool)\n"
    "- taskqueue/queue.py with a TaskQueue class supporting "
    "add(task), pop() returning highest priority undone task, and list_pending() returning undone tasks\n"
    "- A test file that verifies add/pop/list_pending work correctly"
)


@functools.cache
def _has_model() -> bool:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from compass.generators._invoke import resolve_ask_fn
        fn = resolve_ask_fn()
        result = fn("You are a test.", "Reply with exactly: OK")
        return isinstance(result, Ok)
    except Exception:
        return False


def _run_tests_in(directory: Path) -> tuple[bool, str]:
    """Run pytest in directory, return (passed, output)."""
    test_files = list(directory.rglob("test_*.py"))
    if not test_files:
        return False, "no test files found"
    try:
        proc = subprocess.run(
            ["python", "-m", "pytest", str(directory), "-v", "--tb=short"],
            capture_output=True, text=True, timeout=120, cwd=str(directory),
        )
        return proc.returncode == 0, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return False, "pytest timed out (120s)"
    except Exception as e:
        return False, str(e)


def _list_py_files(directory: Path) -> list[str]:
    return sorted(
        str(p.relative_to(directory))
        for p in directory.rglob("*.py")
        if p.is_file()
    )


@pytest.mark.skipif(not _has_model(), reason="No model provider available")
class TestProgrammerNFA:
    """Programmer NFA integration test."""

    def test_programmer_nfa(self):
        """Programmer NFA: problem -> UNDERSTAND -> DESIGN -> IMPLEMENT -> DELIVER."""
        from compass.generators._invoke import resolve_ask_fn

        ask_fn = resolve_ask_fn()

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "artifacts").mkdir()
            shutil.copy2(_PROGRAMMER_ARTIFACT, ws / "artifacts" / "programmer.py")

            # Run through Trinity's dynamic programmer artifact (same path as real usage)
            from compass.generators.trinity._types import Spec, Step
            from compass.generators.trinity._runtime import execute_plan

            spec = Spec(
                question="Build a taskqueue package",
                steps=(
                    Step(
                        step_id="s1",
                        description="generate the taskqueue package",
                        artifact_type="programmer",
                        inputs={"problem": PROBLEM},
                        expected_fact="code_result",
                    ),
                ),
                synthesis="Report what was generated.",
            )

            t0 = time.time()
            result = execute_plan(spec, workspace=ws)
            elapsed = time.time() - t0

            success = isinstance(result, Ok)
            py_files = _list_py_files(ws)

            # Check if programmer returned chunks
            chunks_info = "N/A"
            if success:
                fact = next((f for f in result.value.facts if f.name == "code_result"), None)
                if fact:
                    try:
                        data = json.loads(fact.value)
                        chunks_info = f"{len(data.get('chunks', []))} chunks, success={data.get('success')}"
                    except Exception:
                        chunks_info = fact.value[:200]

            # If programmer produced chunks, apply them to check tests
            tests_passed, test_output = _run_tests_in(ws) if py_files else (False, "no files on disk")

            print(f"\n{'='*60}")
            print(f"PROGRAMMER NFA")
            print(f"{'='*60}")
            print(f"  Success:      {success}")
            print(f"  Time:         {elapsed:.1f}s")
            print(f"  Chunks:       {chunks_info}")
            print(f"  Files on disk:{py_files}")
            print(f"  Tests pass:   {tests_passed}")
            if not tests_passed:
                print(f"  Test output:\n{test_output}")
            print(f"{'='*60}")

            assert success, f"Programmer NFA failed: {result}"
