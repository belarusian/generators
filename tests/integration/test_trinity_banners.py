"""Integration test: Trinity implicit contracts.

Validates that the model understands:
  - inline_python steps omit artifact_ref in the constructor
  - Code goes after ### step_id ### banners
  - _attach_banner_code fills artifact_ref from the banner content
  - File reads use shell steps with cat, not auto/module

This tests the full model -> parse -> execute pipeline with a real model.
"""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path

import pytest

from compass.generators._types import DomainSection, GenerationContext, Ok, Err
from compass.generators._invoke import resolve_ask_fn
from compass.generators.trinity._types import Spec, Step
from compass.generators.trinity._runtime import (
    _attach_banner_code,
    execute_plan,
    invoke_model,
)


@functools.cache
def _has_model() -> bool:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        fn = resolve_ask_fn()
        result = fn("You are a test.", "Reply with exactly: OK")
        return isinstance(result, Ok)
    except Exception:
        return False


def _build_context() -> GenerationContext:
    return GenerationContext(
        user_prompt=(
            "Write a two-step plan:\n"
            "1. An inline_python step that computes 2+2 and assigns to result.\n"
            "2. A shell step that runs: echo \"done\"\n"
            "The inline_python code must go in a ### step_id ### banner."
        ),
        default_task="Answer a question by planning artifact applications.",
        domain_context=(
            DomainSection(
                heading="Plan Construction Principles",
                content=(
                    "- Use inline_python for computation.\n"
                    "- Use shell for CLI commands.\n"
                    "- inline_python: omit artifact_ref, put code in ### step_id ### banner.\n"
                    "- shell: put the command in artifact_ref directly.\n"
                ),
            ),
        ),
    )


@pytest.mark.skipif(not _has_model(), reason="No model provider available")
class TestTrinityBanners:
    """Banner code attaches to the correct field for inline_python steps."""

    def test_model_produces_banners_for_inline_python(self):
        """invoke_model -> parse -> attach: artifact_ref is populated from banner."""
        ctx = _build_context()
        result = invoke_model(ctx)

        assert isinstance(result, Ok), f"invoke_model failed: {result}"
        spec = result.value

        # Find the inline_python step
        inline_steps = [s for s in spec.steps if s.artifact_type == "inline_python"]
        assert len(inline_steps) >= 1, (
            f"Expected at least one inline_python step, got types: "
            f"{[s.artifact_type for s in spec.steps]}"
        )

        # The banner should have populated artifact_ref with actual Python code
        for step in inline_steps:
            assert step.artifact_ref is not None, (
                f"Step '{step.step_id}' has no artifact_ref -- "
                f"banner was not attached"
            )
            assert len(step.artifact_ref.strip()) > 0, (
                f"Step '{step.step_id}' has empty artifact_ref"
            )
            # It should be valid Python
            compile(step.artifact_ref, f"<{step.step_id}>", "exec")

    def test_shell_steps_have_command_in_artifact_ref(self):
        """Shell steps should have the command directly in artifact_ref, not banner."""
        ctx = _build_context()
        result = invoke_model(ctx)

        assert isinstance(result, Ok), f"invoke_model failed: {result}"
        spec = result.value

        shell_steps = [s for s in spec.steps if s.artifact_type == "shell"]
        if not shell_steps:
            pytest.skip("Model did not produce a shell step")

        for step in shell_steps:
            assert step.artifact_ref is not None, (
                f"Shell step '{step.step_id}' has no artifact_ref"
            )
            # Should look like a shell command, not Python
            ref = step.artifact_ref.strip()
            assert not ref.startswith("import "), (
                f"Shell step '{step.step_id}' has Python code in artifact_ref: {ref[:80]}"
            )


@pytest.mark.skipif(not _has_model(), reason="No model provider available")
class TestTrinityFileRead:
    """File reads should use read_file steps, not auto/module."""

    def test_file_read_uses_read_file(self):
        """Model should plan a read_file step when asked to read a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "sample.txt"
            target.write_text("line one\nline two\nline three\n")

            ctx = GenerationContext(
                user_prompt=f"Read the file {target} and report how many lines it has.",
                default_task="Answer a question by planning artifact applications.",
                domain_context=(
                    DomainSection(
                        heading="Plan Construction Principles",
                        content=(
                            "- Use read_file to read files.\n"
                            "- Use inline_python for computation.\n"
                            "- Use shell for CLI commands.\n"
                        ),
                    ),
                ),
            )

            result = invoke_model(ctx)
            assert isinstance(result, Ok), f"invoke_model failed: {result}"
            spec = result.value

            # The plan should NOT use auto/module for file reading
            for step in spec.steps:
                assert step.artifact_type != "auto", (
                    f"Step '{step.step_id}' uses artifact_type='auto' -- "
                    f"file reads should use 'read_file'"
                )

    def test_file_read_end_to_end(self):
        """Full generation_loop: invoke -> parse (whitelist) -> validate -> execute -> emit."""
        from compass.generators.trinity.generate import run

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "data.txt"
            target.write_text("alpha\nbeta\ngamma\n")

            ask_fn = resolve_ask_fn()
            result = run(
                prompt=(
                    f"Read the file {target} and count the lines. "
                    f"Report the count as a number."
                ),
                ask_fn=ask_fn,
                output=tmpdir,
                workspace=tmpdir,
                max_rounds=2,
                max_fixes=2,
            )

            assert isinstance(result, Ok), f"run() failed: {result}"

    def test_large_file_truncated(self):
        """Large files get head+tail treatment, not dumped in full."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "big.txt"
            lines = [f"content line {i}" for i in range(500)]
            target.write_text("\n".join(lines) + "\n")

            step = Step(
                step_id="s1",
                description="read large file",
                artifact_type="read_file",
                artifact_ref="big.txt",
                inputs={},
                expected_fact="file_content",
                extraction_expr="result",
            )

            from compass.generators.trinity.step_dispatch import _execute_read_file
            from compass.generators.trinity.fact_dispatch import display_fact

            result = _execute_read_file(step, {}, Path(tmpdir))
            assert isinstance(result, Ok), f"read failed: {result}"

            fact = result.value
            # Raw value: no line-number prefixes
            assert "content line 0" in fact.value
            assert "line 1:" not in fact.value

            # Display: line-numbered with pagination
            displayed = display_fact(fact)
            assert "line 1: content line 0" in displayed
            assert "line 120: content line 119" in displayed
            assert "lines omitted" in displayed
            assert "line 500: content line 499" in displayed
            # Should NOT contain middle lines
            assert "content line 200" not in displayed

    def test_offset_limit(self):
        """offset/limit inputs read specific line ranges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "lines.txt"
            lines = [f"line {i}" for i in range(100)]
            target.write_text("\n".join(lines) + "\n")

            step = Step(
                step_id="s1",
                description="read range",
                artifact_type="read_file",
                artifact_ref="lines.txt",
                inputs={},
                expected_fact="file_content",
                extraction_expr="result",
            )

            from compass.generators.trinity.step_dispatch import _execute_read_file
            from compass.generators.trinity.fact_dispatch import display_fact

            result = _execute_read_file(step, {"offset": 10, "limit": 5}, Path(tmpdir))
            assert isinstance(result, Ok), f"read failed: {result}"

            fact = result.value
            # Raw value: just the sliced lines, no prefixes
            assert "line 10" in fact.value
            assert "line 14" in fact.value
            assert "line 11:" not in fact.value

            # Display: line-numbered from original file position
            displayed = display_fact(fact)
            assert "line 11: line 10" in displayed
            assert "line 15: line 14" in displayed


class TestFactDispatch:
    """Fact dispatch separates raw value from display presentation."""

    def test_read_then_write_no_line_numbers(self):
        """read_file -> inline_python write: output file has no line-number prefixes.

        This is the exact bug that motivated fact dispatch: the model received
        line-numbered content via $fact and wrote it to disk with 'line N:' baked in.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            source = ws / "source.txt"
            source.write_text("alpha\nbeta\ngamma\n")

            spec = Spec(
                question="Copy source.txt to output.txt",
                steps=(
                    Step(
                        step_id="s1",
                        description="read the source file",
                        artifact_type="read_file",
                        artifact_ref="source.txt",
                        inputs={},
                        expected_fact="file_content",
                        extraction_expr="result",
                    ),
                    Step(
                        step_id="s2",
                        description="write content to output file",
                        artifact_type="inline_python",
                        artifact_ref=(
                            "from pathlib import Path\n"
                            "Path(str(workspace) + '/output.txt').write_text(content)\n"
                            "result = 'written'\n"
                        ),
                        inputs={"content": {"$fact": "file_content"}},
                        expected_fact="write_result",
                        extraction_expr="result",
                        depends_on=("s1",),
                    ),
                ),
                synthesis="Report whether the copy succeeded.",
            )

            exec_result = execute_plan(spec, workspace=ws)
            assert isinstance(exec_result, Ok), f"execute_plan failed: {exec_result}"

            output = (ws / "output.txt").read_text()
            # The written file must contain raw content -- no line-number prefixes
            assert "alpha" in output
            assert "beta" in output
            assert "gamma" in output
            assert "line 1:" not in output
            assert "line 2:" not in output

    def test_file_fact_display_vs_resolve(self):
        """FileFact.value is raw; display_fact adds line numbers; resolve_fact returns raw."""
        from compass.generators.trinity._types import FileFact
        from compass.generators.trinity.fact_dispatch import display_fact, resolve_fact

        raw = "def hello():\n    print('world')\n"
        fact = FileFact(
            step_id="s1", name="src", value=raw,
            fact_type="text", path="hello.py",
        )

        # resolve: raw content for downstream $fact references
        assert resolve_fact(fact) == raw
        assert "line 1:" not in resolve_fact(fact)

        # display: line-numbered for REPL / model context
        displayed = display_fact(fact)
        assert "line 1: def hello():" in displayed
        assert "line 2:     print('world')" in displayed

    def test_error_fact_isinstance(self):
        """ErrorFact is a Fact subclass with fact_type='error' by default."""
        from compass.generators.trinity._types import ErrorFact, Fact

        ef = ErrorFact(step_id="s1", name="err", value="boom")
        assert isinstance(ef, Fact)
        assert ef.fact_type == "error"
