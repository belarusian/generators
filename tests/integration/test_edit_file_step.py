"""Integration test: edit_file step type.

Tests the targeted string replacement step -- the safer alternative
to write_file for single-line or small edits. Validates:
  - Exact match replacement
  - Uniqueness guard (multiple matches -> error)
  - Not-found guard (no match -> error)
  - Python syntax gate (bad edit -> refused)
  - Failed-dependency propagation (upstream error -> downstream skip)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from compass.generators._types import Ok, Err
from compass.generators.trinity._types import EditFileStep, Spec, Step, WriteFileStep
from compass.generators.trinity._runtime import execute_plan
from compass.generators.trinity.step_dispatch import (
    StepContext,
    _extract_content,
    _find_target_by_lines,
    execute_step,
)


def _make_ctx(workspace: Path, inputs: dict | None = None) -> StepContext:
    return StepContext(
        resolved_inputs=inputs or {},
        facts={},
        workspace=workspace,
    )


class TestEditFileStep:
    """edit_file applies targeted replacements without full-file rewrite."""

    def test_basic_replacement(self):
        """Single unique match replaced correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "example.py"
            target.write_text("x = 1\ny = 2\nz = 3\n")

            step = EditFileStep(
                step_id="s1",
                description="change y value",
                artifact_ref="example.py",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "y = 2",
                "new_string": "y = 42",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok), f"edit failed: {result}"
            assert target.read_text() == "x = 1\ny = 42\nz = 3\n"

    def test_multiline_replacement(self):
        """Replacement spanning multiple lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "config.txt"
            target.write_text("host = localhost\nport = 8080\ndebug = true\n")

            step = EditFileStep(
                step_id="s1",
                description="change host and port",
                artifact_ref="config.txt",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "host = localhost\nport = 8080",
                "new_string": "host = 0.0.0.0\nport = 9090",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok)
            assert target.read_text() == "host = 0.0.0.0\nport = 9090\ndebug = true\n"

    def test_not_found(self):
        """old_string absent from file -> Err."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "data.txt"
            target.write_text("alpha\nbeta\n")

            step = EditFileStep(
                step_id="s1",
                description="edit missing text",
                artifact_ref="data.txt",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "gamma",
                "new_string": "delta",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Err)
            assert "not found" in result.error

            # File untouched
            assert target.read_text() == "alpha\nbeta\n"

    def test_multiple_matches_rejected(self):
        """old_string matches more than once -> Err (ambiguous)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "repeat.txt"
            target.write_text("foo\nbar\nfoo\nbaz\n")

            step = EditFileStep(
                step_id="s1",
                description="edit ambiguous text",
                artifact_ref="repeat.txt",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "foo",
                "new_string": "qux",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Err)
            assert "2 times" in result.error

            # File untouched
            assert target.read_text() == "foo\nbar\nfoo\nbaz\n"

    def test_identical_strings_noop(self):
        """old_string == new_string -> Ok (no-op, conditional edit)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "same.txt"
            target.write_text("hello\n")

            step = EditFileStep(
                step_id="s1",
                description="conditional edit",
                artifact_ref="same.txt",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "hello",
                "new_string": "hello",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok)
            assert target.read_text() == "hello\n"

    def test_file_not_found(self):
        """Target file missing -> Err."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)

            step = EditFileStep(
                step_id="s1",
                description="edit missing file",
                artifact_ref="nope.py",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "x",
                "new_string": "y",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Err)
            assert "file not found" in result.error

    def test_syntax_gate_blocks_bad_python_edit(self):
        """Edit that would produce invalid Python -> refused, file untouched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "module.py"
            original = "from typing import Callable, Dict, List, Optional\n\nx = 1\n"
            target.write_text(original)

            step = EditFileStep(
                step_id="s1",
                description="break the import",
                artifact_ref="module.py",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "from typing import Callable, Dict, List, Optional",
                "new_string": "from typing import Callable, , Optional",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Err)
            assert "SyntaxError" in result.error

            # File untouched
            assert target.read_text() == original

    def test_syntax_gate_allows_valid_python_edit(self):
        """Edit that produces valid Python -> accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "module.py"
            target.write_text("from typing import Optional\n\nx = 1\n")

            step = EditFileStep(
                step_id="s1",
                description="add Any to import",
                artifact_ref="module.py",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "from typing import Optional",
                "new_string": "from typing import Any, Optional",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok)
            assert target.read_text() == "from typing import Any, Optional\n\nx = 1\n"

    def test_non_python_files_skip_syntax_gate(self):
        """Non-.py files are written without syntax checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "config.yaml"
            target.write_text("key: value\nother: stuff\n")

            step = EditFileStep(
                step_id="s1",
                description="edit yaml",
                artifact_ref="config.yaml",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "key: value",
                "new_string": "key: new_value",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok)
            assert target.read_text() == "key: new_value\nother: stuff\n"


class TestFailedDependencyPropagation:
    """Downstream steps skip when upstream fails (not run with error content)."""

    def test_failed_read_skips_downstream_write(self):
        """read_file fails -> write_file depending on it is skipped, not executed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            output = ws / "output.txt"

            spec = Spec(
                question="Read a missing file and write its content",
                steps=(
                    Step(
                        step_id="s1",
                        description="read nonexistent file",
                        artifact_type="read_file",
                        artifact_ref="does_not_exist.txt",
                        inputs={},
                        expected_fact="file_content",
                        extraction_expr="result",
                    ),
                    Step(
                        step_id="s2",
                        description="write the content",
                        artifact_type="write_file",
                        artifact_ref="",
                        inputs={
                            "path": "output.txt",
                            "content": {"$fact": "file_content"},
                        },
                        expected_fact="write_result",
                        extraction_expr="write_result",
                        depends_on=("s1",),
                    ),
                ),
                synthesis="Report result.",
            )

            result = execute_plan(spec, workspace=ws)

            # Plan should fail
            assert isinstance(result, Err)
            assert "failed dependencies" in result.error

            # output.txt must NOT exist -- write_file was never executed
            assert not output.exists(), (
                f"output.txt was written with: {output.read_text()!r}"
            )

    def test_failed_read_skips_downstream_edit(self):
        """read_file fails -> edit_file depending on it is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "target.py"
            original = "x = 1\n"
            target.write_text(original)

            spec = Spec(
                question="Read missing file, then edit target",
                steps=(
                    Step(
                        step_id="s1",
                        description="read nonexistent file",
                        artifact_type="read_file",
                        artifact_ref="missing.txt",
                        inputs={},
                        expected_fact="source_content",
                        extraction_expr="result",
                    ),
                    Step(
                        step_id="s2",
                        description="edit target using source content",
                        artifact_type="edit_file",
                        artifact_ref="target.py",
                        inputs={
                            "old_string": "x = 1",
                            "new_string": {"$fact": "source_content"},
                        },
                        expected_fact="edit_result",
                        extraction_expr="edit_result",
                        depends_on=("s1",),
                    ),
                ),
                synthesis="Report result.",
            )

            result = execute_plan(spec, workspace=ws)
            assert isinstance(result, Err)

            # target.py untouched
            assert target.read_text() == original

    def test_edit_in_plan_end_to_end(self):
        """read_file -> edit_file chain through execute_plan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "hello.py"
            target.write_text('msg = "hello"\nprint(msg)\n')

            spec = Spec(
                question="Change hello to goodbye",
                steps=(
                    Step(
                        step_id="s1",
                        description="edit the greeting",
                        artifact_type="edit_file",
                        artifact_ref="hello.py",
                        inputs={
                            "old_string": 'msg = "hello"',
                            "new_string": 'msg = "goodbye"',
                        },
                        expected_fact="edit_done",
                        extraction_expr="edit_done",
                    ),
                    Step(
                        step_id="s2",
                        description="verify the edit",
                        artifact_type="inline_python",
                        artifact_ref=(
                            "from pathlib import Path\n"
                            "content = Path(str(workspace) + '/hello.py').read_text()\n"
                            "result = 'goodbye' in content\n"
                        ),
                        inputs={},
                        expected_fact="verified",
                        extraction_expr="result",
                        depends_on=("s1",),
                    ),
                ),
                synthesis="Report whether the edit was verified.",
            )

            result = execute_plan(spec, workspace=ws)
            assert isinstance(result, Ok), f"execute_plan failed: {result}"
            assert target.read_text() == 'msg = "goodbye"\nprint(msg)\n'


class TestEditFileBanners:
    """edit_file with ### step_id:old_string ### / ### step_id:new_string ### banners."""

    def test_banner_sub_keys_inject_into_inputs(self):
        """Sub-banners fill old_string/new_string without escaping in constructor."""
        from compass.generators.trinity._runtime import _attach_banner_code
        from compass.generators.trinity._types import promote_spec

        spec = Spec(
            question="Fix an import",
            steps=(
                Step(
                    step_id="s1",
                    description="add Any to import",
                    artifact_type="edit_file",
                    artifact_ref="module.py",
                    inputs={},
                    expected_fact="edit_done",
                    extraction_expr="edit_done",
                ),
            ),
            synthesis="Done.",
        )

        # Simulate what parse_response_with_files returns for sub-banners
        sections = [
            ("s1:old_string", "from typing import Optional"),
            ("s1:new_string", "from typing import Any, Optional"),
        ]

        attached = _attach_banner_code(spec, sections)
        step = attached.steps[0]

        assert step.inputs["old_string"] == "from typing import Optional"
        assert step.inputs["new_string"] == "from typing import Any, Optional"
        # artifact_ref stays as the file path
        assert step.artifact_ref == "module.py"

    def test_banner_edit_end_to_end(self):
        """Full pipeline: sub-banners -> _attach_banner_code -> execute_plan."""
        from compass.generators.trinity._runtime import _attach_banner_code

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "app.py"
            target.write_text(
                "import os\n"
                "from typing import Optional\n"
                "\n"
                "def main():\n"
                "    pass\n"
            )

            spec = Spec(
                question="Add Any to import",
                steps=(
                    Step(
                        step_id="s1",
                        description="add Any to typing import",
                        artifact_type="edit_file",
                        artifact_ref="app.py",
                        inputs={},
                        expected_fact="edit_done",
                        extraction_expr="edit_done",
                    ),
                ),
                synthesis="Done.",
            )

            sections = [
                ("s1:old_string", "from typing import Optional"),
                ("s1:new_string", "from typing import Any, Optional"),
            ]
            spec = _attach_banner_code(spec, sections)

            result = execute_plan(spec, workspace=ws)
            assert isinstance(result, Ok), f"execute_plan failed: {result}"
            assert "from typing import Any, Optional" in target.read_text()
            assert "from typing import Optional\n" not in target.read_text()


class TestLineNumberPrefixStripping:
    """edit_file strips 'line N: ' prefixes the model copies from display output."""

    def test_strips_line_prefix_from_old_string(self):
        """Model copies 'line 19: from typing...' -- handler strips prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "example.py"
            target.write_text("x = 1\nfrom typing import Optional\nz = 3\n")

            step = EditFileStep(
                step_id="s1",
                description="edit import",
                artifact_ref="example.py",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "line 2: from typing import Optional",
                "new_string": "from typing import Any, Optional",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok), f"edit failed: {result}"
            assert target.read_text() == "x = 1\nfrom typing import Any, Optional\nz = 3\n"

    def test_multiline_edit_uses_banners_not_prefix_strip(self):
        """Multi-line edits should use banners, not rely on prefix stripping."""
        from compass.generators.trinity._runtime import _attach_banner_code

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "data.txt"
            target.write_text("alpha\nbeta\ngamma\n")

            spec = Spec(
                question="Edit two lines",
                steps=(
                    Step(
                        step_id="s1",
                        description="edit two lines",
                        artifact_type="edit_file",
                        artifact_ref="data.txt",
                        inputs={},
                        expected_fact="edit_result",
                        extraction_expr="edit_result",
                    ),
                ),
                synthesis="Done.",
            )

            sections = [
                ("s1:old_string", "alpha\nbeta"),
                ("s1:new_string", "ALPHA\nBETA"),
            ]
            spec = _attach_banner_code(spec, sections)
            result = execute_plan(spec, workspace=ws)
            assert isinstance(result, Ok), f"execute_plan failed: {result}"
            assert target.read_text() == "ALPHA\nBETA\ngamma\n"

    def test_no_strip_when_exact_match_exists(self):
        """If old_string matches exactly, don't try stripping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "quirky.txt"
            # File literally contains "line 5: hello"
            target.write_text("line 5: hello\nworld\n")

            step = EditFileStep(
                step_id="s1",
                description="edit literal",
                artifact_ref="quirky.txt",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "line 5: hello",
                "new_string": "line 5: goodbye",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok)
            # Should match the literal content, not strip the prefix
            assert target.read_text() == "line 5: goodbye\nworld\n"


class TestEditFileNoOp:
    """edit_file with identical old/new is a no-op success (conditional edit)."""

    def test_identical_is_noop_success(self):
        """old_string == new_string -> Ok with no_change flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "stable.txt"
            target.write_text("content\n")

            step = EditFileStep(
                step_id="s1",
                description="conditional edit",
                artifact_ref="stable.txt",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "content",
                "new_string": "content",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok)
            assert target.read_text() == "content\n"


class TestExtractContent:
    """_extract_content strips JSON infrastructure from model output."""

    def test_dict_with_content_key(self):
        """{"path": "...", "content": "actual stuff"} -> "actual stuff"."""
        value = {"path": "artifacts/foo.py", "content": "print('hello')", "success": True}
        assert _extract_content(value) == "print('hello')"

    def test_json_string_with_content_key(self):
        """JSON string containing content key -> extracts content."""
        import json
        value = json.dumps({"path": "foo.py", "content": "x = 1\ny = 2\n"})
        assert _extract_content(value) == "x = 1\ny = 2\n"

    def test_nested_content(self):
        """Content value is itself a dict with content -> recurse."""
        value = {"content": {"content": "deep payload"}}
        assert _extract_content(value) == "deep payload"

    def test_plain_string_passthrough(self):
        """Regular string -> returned as-is."""
        assert _extract_content("just some code") == "just some code"

    def test_string_assignment_unwrapped(self):
        """varname = '''...''' -> inner content extracted."""
        value = "content = '''print('hello')'''"
        assert _extract_content(value) == "print('hello')"

    def test_dict_without_content_key(self):
        """Dict with no content key -> JSON-serialized."""
        import json
        value = {"a": 1, "b": 2}
        result = _extract_content(value)
        assert json.loads(result) == {"a": 1, "b": 2}

    def test_non_json_curly_string(self):
        """String that looks like JSON but isn't -> passthrough."""
        value = "{not valid json at all}"
        assert _extract_content(value) == "{not valid json at all}"

    def test_json_string_without_content(self):
        """Valid JSON string but no content key -> passthrough."""
        import json
        value = json.dumps({"path": "foo.py", "success": True})
        assert _extract_content(value) == value

    def test_write_file_with_dict_content(self):
        """Integration: write_file receiving dict with content writes only content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = WriteFileStep(
                step_id="w1",
                description="write artifact",
                artifact_ref="",
                inputs={},
                expected_fact="write_result",
                extraction_expr="write_result",
            )
            ctx = _make_ctx(ws, {
                "path": "output.py",
                "content": {"path": "output.py", "content": "x = 42\n", "success": True},
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok), f"write failed: {result}"
            written = (ws / "output.py").read_text()
            assert written == "x = 42\n"
            assert "success" not in written
            assert "path" not in written

    def test_write_file_with_json_string_content(self):
        """Integration: write_file receiving JSON string with content extracts it."""
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = WriteFileStep(
                step_id="w1",
                description="write artifact",
                artifact_ref="",
                inputs={},
                expected_fact="write_result",
                extraction_expr="write_result",
            )
            ctx = _make_ctx(ws, {
                "path": "output.txt",
                "content": json.dumps({"path": "output.txt", "content": "hello world"}),
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok), f"write failed: {result}"
            assert (ws / "output.txt").read_text() == "hello world"


class TestFindTargetByLines:
    """Unit tests for _find_target_by_lines -- the fuzzy matcher ported from Neo."""

    def test_exact_content_returns_as_is(self):
        content = "def foo():\n    return 42\n"
        assert _find_target_by_lines("def foo():\n    return 42", content) == "def foo():\n    return 42"

    def test_wrong_indentation_still_matches(self):
        """Model outputs 'return 42' at top level, file has it at 4-space indent."""
        content = "def foo():\n    return 42\n    x = 1\n"
        found = _find_target_by_lines("return 42", content)
        assert found == "    return 42"

    def test_multiline_wrong_indent(self):
        """Model indents relative to content block, not file."""
        content = "class Foo:\n    def bar(self):\n        x = 1\n        y = 2\n"
        # Model outputs with 0 indent (content block effect)
        target = "def bar(self):\n    x = 1\n    y = 2"
        found = _find_target_by_lines(target, content)
        assert found == "    def bar(self):\n        x = 1\n        y = 2"

    def test_trailing_whitespace_in_target(self):
        content = "x = 1\ny = 2\n"
        found = _find_target_by_lines("x = 1   \ny = 2  ", content)
        assert found == "x = 1\ny = 2"

    def test_ambiguous_match_returns_none(self):
        """Two identical-when-stripped blocks -> None (ambiguous)."""
        content = "    x = 1\n    y = 2\n    z = 3\n    x = 1\n    y = 2\n"
        found = _find_target_by_lines("x = 1\ny = 2", content)
        assert found is None

    def test_no_match_returns_none(self):
        content = "a = 1\nb = 2\n"
        assert _find_target_by_lines("c = 3", content) is None

    def test_empty_target_returns_none(self):
        assert _find_target_by_lines("", "x = 1\n") is None
        assert _find_target_by_lines("   \n\n  ", "x = 1\n") is None

    def test_blank_lines_in_target_skipped(self):
        """Model adds extra blank lines between code -- stripped blank lines vanish."""
        content = "def foo():\n    x = 1\n    return x\n"
        target = "def foo():\n\n    x = 1\n\n    return x"
        found = _find_target_by_lines(target, content)
        assert found == "def foo():\n    x = 1\n    return x"


class TestEditFileWhitespaceTolerance:
    """Trinity should handle whitespace/indentation mismatches like Neo.

    These are the failure modes that cause 'old_string not found' in practice:
    1. Model adds trailing whitespace
    2. Model shifts indentation (content block effect)
    3. Both combined

    Before this fix, all of these returned Err. Now they match.
    """

    def test_trailing_whitespace_on_old_string(self):
        """Model's old_string has trailing spaces -- rstrip fallback catches it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "example.py"
            target.write_text("def verify_push(\n    remote: str,\n    ref: str,\n) -> dict:\n    pass\n")

            step = EditFileStep(
                step_id="s1",
                description="change signature",
                artifact_ref="example.py",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                # Note the trailing spaces -- this is what models do
                "old_string": "def verify_push(\n    remote: str,\n    ref: str,\n) -> dict:   ",
                "new_string": "def verify_push(\n    remote: str,\n    ref: str,\n    push_result,\n) -> dict:",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok), f"edit failed: {result}"
            assert "push_result" in target.read_text()

    def test_wrong_indentation_from_content_block(self):
        """Model outputs code at wrong indent level -- line-by-line match catches it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "module.py"
            target.write_text(
                "class Config:\n"
                "    def validate(self):\n"
                "        if not self.host:\n"
                "            raise ValueError('no host')\n"
                "        return True\n"
            )

            step = EditFileStep(
                step_id="s1",
                description="fix validation",
                artifact_ref="module.py",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                # Model's content block shifts everything left by 8 spaces
                "old_string": "if not self.host:\n    raise ValueError('no host')\nreturn True",
                "new_string": "        if not self.host:\n            raise ValueError('missing host')\n        return True",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok), f"edit failed: {result}"
            assert "missing host" in target.read_text()

    def test_multiline_mixed_indent_mismatch(self):
        """Model gets some lines right, some wrong -- line-by-line still matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "handler.py"
            target.write_text(
                "def process(data):\n"
                "    result = transform(data)\n"
                "    if result.ok:\n"
                "        return result.value\n"
                "    return None\n"
            )

            step = EditFileStep(
                step_id="s1",
                description="add logging",
                artifact_ref="handler.py",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                # Model nails first line indent, fumbles the rest
                "old_string": "    result = transform(data)\n    if result.ok:\n    return result.value",
                "new_string": "    result = transform(data)\n    log.info(result)\n    if result.ok:\n        return result.value",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok), f"edit failed: {result}"
            assert "log.info" in target.read_text()

    def test_ambiguous_fuzzy_match_still_rejected(self):
        """Fuzzy matching doesn't bypass uniqueness -- ambiguous stripped match -> Err."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "dup.py"
            # Same stripped content appears twice at different indent levels
            target.write_text(
                "if a:\n    x = 1\n    y = 2\nif b:\n    x = 1\n    y = 2\n"
            )

            step = EditFileStep(
                step_id="s1",
                description="edit ambiguous",
                artifact_ref="dup.py",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "x = 1\ny = 2",
                "new_string": "x = 99\ny = 99",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Err), "ambiguous match should still fail"

    def test_exact_match_preferred_over_fuzzy(self):
        """If exact match exists, fuzzy matching is never attempted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "precise.py"
            # "x = 1" only appears once as an exact substring
            target.write_text("x = 1\nfoo(x)\n")

            step = EditFileStep(
                step_id="s1",
                description="exact edit",
                artifact_ref="precise.py",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                "old_string": "x = 1",
                "new_string": "x = 42",
            })

            result = execute_step(step, ctx)
            assert isinstance(result, Ok)
            assert target.read_text() == "x = 42\nfoo(x)\n"

    def test_real_world_signature_edit(self):
        """Real failure case: model tried to edit push_verification.py signature.

        The model generated old_string with keyword-only args (*, pre_push_oid)
        but the actual file had positional args. The stripped lines don't match
        at all -- this correctly remains an Err (fuzzy can't fix semantic drift).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            target = ws / "push_verification.py"
            target.write_text(
                "def verify_push(\n"
                "    remote: str,\n"
                "    ref: str,\n"
                "    pre_push_oid: str,\n"
                "    push_result: dict,\n"
                ") -> dict:\n"
                "    pass\n"
            )

            step = EditFileStep(
                step_id="s3",
                description="fix verify_push",
                artifact_ref="push_verification.py",
                inputs={},
                expected_fact="edit_result",
                extraction_expr="edit_result",
            )
            ctx = _make_ctx(ws, {
                # Model hallucinated a completely different signature
                "old_string": "def verify_push(\n    *,\n    pre_push_oid: str,\n    push_result: dict,\n) -> dict:",
                "new_string": "def verify_push(\n    remote: str,\n    ref: str,\n    pre_push_oid: str,\n    push_result,\n) -> dict:",
            })

            result = execute_step(step, ctx)
            # This SHOULD fail -- the model hallucinated content that doesn't exist
            # even with fuzzy matching.  The stripped lines don't match.
            assert isinstance(result, Err)
            assert "not found" in result.error
