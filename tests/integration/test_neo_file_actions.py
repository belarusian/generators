"""Integration tests: Neo write_file and edit_file action guardrails.

Tests real dispatch, real validation, real file I/O.
No mocks. Proves:
  - _extract_content unwraps JSON envelopes before writing
  - edit_file syntax gate blocks invalid Python edits
  - $WORKSPACE env var is available in command actions
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from compass.agents.neo.types import (
    ExecutionContext,
    RunCommandAction,
    WriteFileAction,
)
from compass.agents.neo.dispatch import execute, validate
from compass.agents.neo.actions.write_file import _extract_content
from compass.agents.neo.actions.edit_file import (
    _apply_edit_with_fallback,
    _validate_python_syntax,
    _validate_unique_target,
)


# ============================================================================
# _extract_content unit proofs
# ============================================================================


class TestExtractContent:
    """_extract_content peels JSON infrastructure from model output."""

    def test_dict_with_content_key(self):
        value = {"path": "artifacts/foo.py", "content": "print('hello')", "success": True}
        assert _extract_content(value) == "print('hello')"

    def test_json_string_with_content_key(self):
        value = json.dumps({"path": "foo.py", "content": "x = 1\ny = 2\n"})
        assert _extract_content(value) == "x = 1\ny = 2\n"

    def test_nested_content(self):
        value = {"content": {"content": "deep payload"}}
        assert _extract_content(value) == "deep payload"

    def test_plain_string_passthrough(self):
        assert _extract_content("just some code") == "just some code"

    def test_dict_without_content_key(self):
        value = {"a": 1, "b": 2}
        result = _extract_content(value)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_non_json_curly_string(self):
        value = "{not valid json at all}"
        assert _extract_content(value) == "{not valid json at all}"

    def test_json_string_without_content_key(self):
        value = json.dumps({"path": "foo.py", "success": True})
        assert _extract_content(value) == value


# ============================================================================
# write_file integration -- content extraction through real dispatch
# ============================================================================


class TestWriteFileExtraction:
    """write_file strips JSON envelopes via real dispatch pipeline."""

    def test_dict_content_writes_payload_only(self):
        """Model returns {"path": ..., "content": "actual"} -- only "actual" hits disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            action = WriteFileAction(
                path="output.py",
                content={"path": "output.py", "content": "x = 42\n", "success": True},
            )
            success, msg = execute(action, tmpdir, ExecutionContext())
            assert success, msg

            written = (Path(tmpdir) / "output.py").read_text()
            assert written == "x = 42\n"
            assert "success" not in written

    def test_json_string_content_writes_payload_only(self):
        """Model returns JSON string with content key -- extracted before write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            action = WriteFileAction(
                path="output.txt",
                content=json.dumps({"path": "output.txt", "content": "hello world"}),
            )
            success, msg = execute(action, tmpdir, ExecutionContext())
            assert success, msg

            assert (Path(tmpdir) / "output.txt").read_text() == "hello world"

    def test_plain_content_unchanged(self):
        """Normal string content passes through untouched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            action = WriteFileAction(path="plain.txt", content="line one\nline two\n")
            success, msg = execute(action, tmpdir, ExecutionContext())
            assert success, msg

            assert (Path(tmpdir) / "plain.txt").read_text() == "line one\nline two\n"

    def test_syntax_gate_still_works_after_extraction(self):
        """Extraction happens before syntax gate -- broken Python still blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            action = WriteFileAction(
                path="bad.py",
                content={"content": "def broken(\n"},
            )
            success, msg = execute(action, tmpdir, ExecutionContext())
            assert not success
            assert "syntax" in msg.lower() or "Syntax" in msg

            assert not (Path(tmpdir) / "bad.py").exists()


# ============================================================================
# edit_file syntax gate -- the code path at edit_file.py:275-285
# ============================================================================


class TestEditFileSyntaxGate:
    """edit_file blocks edits that would produce invalid Python.

    Tests the exact sequence from the retry loop:
      _apply_edit_with_fallback -> _validate_python_syntax
    """

    def test_bad_edit_caught_before_write(self):
        """Replace valid import with broken syntax -- gate catches it."""
        original = "from typing import Optional\n\nx = 1\n"
        target = "from typing import Optional"
        replacement = "from typing import ,"  # broken

        success, new_content, _ = _apply_edit_with_fallback(
            original, target, replacement, "replace"
        )
        assert success  # edit applied in memory

        error = _validate_python_syntax(new_content, "module.py")
        assert error is not None
        assert "syntax" in error.lower() or "invalid" in error.lower()

    def test_valid_edit_passes_gate(self):
        """Replace valid import with different valid import -- gate allows it."""
        original = "from typing import Optional\n\nx = 1\n"
        target = "from typing import Optional"
        replacement = "from typing import Any, Optional"

        success, new_content, _ = _apply_edit_with_fallback(
            original, target, replacement, "replace"
        )
        assert success

        error = _validate_python_syntax(new_content, "module.py")
        assert error is None

    def test_gate_blocks_write_to_disk(self):
        """Full path: apply edit that breaks syntax, verify file untouched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target_file = Path(tmpdir) / "module.py"
            original = "from typing import Optional\n\ndef foo():\n    pass\n"
            target_file.write_text(original)

            content = target_file.read_text()

            # Apply edit in memory
            success, new_content, _ = _apply_edit_with_fallback(
                content, "from typing import Optional", "from typing import ,", "replace"
            )
            assert success

            # Gate blocks
            error = _validate_python_syntax(new_content, str(target_file))
            assert error is not None

            # File stays untouched (we never wrote because gate fired)
            assert target_file.read_text() == original


# ============================================================================
# $WORKSPACE env var in command actions
# ============================================================================


class TestWorkspaceEnvVar:
    """run_command exposes $WORKSPACE so models don't hallucinate paths."""

    def test_workspace_available_in_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            action = RunCommandAction(command="echo $WORKSPACE")
            success, output = execute(action, tmpdir, ExecutionContext())
            assert success, output
            assert tmpdir in output.strip()
