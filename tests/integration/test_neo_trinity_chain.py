"""Integration test: Neo -> Trinity -> code generation chain.

Proves the full chain fires: Neo's ProgramAction dispatches to Trinity,
Trinity generates a plan and executes it, working code gets produced.
We don't care whether Trinity picks 'programmer' or 'module' --
only that the program gets written.
"""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path

import pytest

from compass.agents.neo.types import ProgramAction
from compass.agents.neo.dispatch import execute


@functools.cache
def _has_model() -> bool:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from compass.generators._invoke import resolve_ask_fn
        from compass.generators._types import Ok
        fn = resolve_ask_fn()
        result = fn("You are a test.", "Reply with exactly: OK")
        return isinstance(result, Ok)
    except Exception:
        return False


@pytest.mark.skipif(not _has_model(), reason="No model provider available")
class TestNeoTrinityChain:
    """Neo dispatches ProgramAction through Trinity, code gets produced."""

    def test_neo_programs_through_trinity(self):
        """Neo fires ProgramAction -> Trinity plans -> code is generated.

        The problem is substantial enough that Trinity should reach for
        'module' or 'programmer' rather than just inline_python.
        We verify the chain completes and files appear on disk.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            action = ProgramAction(
                problem=(
                    "Build a Python package called 'taskqueue' with:\n"
                    "- taskqueue/__init__.py exporting Task and TaskQueue\n"
                    "- taskqueue/models.py with a Task dataclass (id, name, priority, done)\n"
                    "- taskqueue/queue.py with a TaskQueue class that supports "
                    "add(task), pop() returning highest priority, and list_pending()\n"
                    "- tests/test_taskqueue.py that verifies add/pop/list_pending work correctly"
                ),
            )

            success, summary = execute(action, project_path=tmpdir)

            assert success, f"Neo -> Trinity chain failed: {summary}"
            print(f"\n--- Summary ---\n{summary}")

            # Something was produced -- files on disk
            ws = Path(tmpdir)
            all_files = [str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file()]
            py_files = [f for f in all_files if f.endswith(".py")]
            print(f"\n--- Files produced ({len(all_files)} total, {len(py_files)} .py) ---")
            for f in sorted(all_files):
                print(f"  {f}")

            assert len(py_files) >= 1, f"No Python files produced. Files: {all_files}"
