"""Integration test: assert step type.

The assert step runs inline Python and checks the result for failure.
Convention: result must be {"ok": bool, "issues": [...]}.
If ok is false or issues is non-empty, the step fails -> Ouroboros can patch.

This is how Trinity verifies that prior steps produced correct output.
The type IS the contract (lesson 8): the model copies the docstring example.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from compass.generators._types import Ok, Err
from compass.generators.trinity._types import AssertStep, Spec, Step
from compass.generators.trinity._runtime import execute_plan
from compass.generators.trinity.step_dispatch import StepContext, execute_step


def _make_ctx(workspace: Path, inputs: dict | None = None) -> StepContext:
    return StepContext(
        resolved_inputs=inputs or {},
        facts={},
        workspace=workspace,
    )


class TestAssertStepConvention:
    """Assert steps fail the plan when ok=False or issues is non-empty."""

    def test_ok_true_no_issues_succeeds(self):
        """{"ok": True, "issues": []} -> Ok."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = AssertStep(
                step_id="v1",
                description="check something",
                artifact_ref='result = {"ok": True, "issues": []}',
                inputs={},
                expected_fact="verified",
                extraction_expr="result",
            )
            result = execute_step(step, _make_ctx(ws))
            assert isinstance(result, Ok), f"should pass: {result}"

    def test_ok_false_fails(self):
        """{"ok": False, "issues": ["bad"]} -> Err with the issue text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = AssertStep(
                step_id="v1",
                description="check something",
                artifact_ref='result = {"ok": False, "issues": ["definition not found"]}',
                inputs={},
                expected_fact="verified",
                extraction_expr="result",
            )
            result = execute_step(step, _make_ctx(ws))
            assert isinstance(result, Err)
            assert "definition not found" in result.error

    def test_issues_nonempty_fails_even_without_ok(self):
        """{"issues": ["x"]} with no ok key -> Err (issues present = failure)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = AssertStep(
                step_id="v1",
                description="check something",
                artifact_ref='result = {"issues": ["imports before function"]}',
                inputs={},
                expected_fact="verified",
                extraction_expr="result",
            )
            result = execute_step(step, _make_ctx(ws))
            assert isinstance(result, Err)
            assert "imports before function" in result.error

    def test_empty_issues_with_ok_true_passes(self):
        """{"ok": True, "issues": []} -> Ok (explicit pass)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = AssertStep(
                step_id="v1",
                description="verify",
                artifact_ref='result = {"ok": True, "issues": []}',
                inputs={},
                expected_fact="verified",
                extraction_expr="result",
            )
            result = execute_step(step, _make_ctx(ws))
            assert isinstance(result, Ok)

    def test_falsy_result_fails(self):
        """Non-dict falsy result (False, None, 0) -> Err."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = AssertStep(
                step_id="v1",
                description="check",
                artifact_ref="result = False",
                inputs={},
                expected_fact="verified",
                extraction_expr="result",
            )
            result = execute_step(step, _make_ctx(ws))
            assert isinstance(result, Err)
            assert "assertion failed" in result.error

    def test_truthy_non_dict_passes(self):
        """Non-dict truthy result (string, number) -> Ok."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = AssertStep(
                step_id="v1",
                description="check",
                artifact_ref='result = "all good"',
                inputs={},
                expected_fact="verified",
                extraction_expr="result",
            )
            result = execute_step(step, _make_ctx(ws))
            assert isinstance(result, Ok)

    def test_multiple_issues_joined(self):
        """Multiple issues are joined with semicolons in the error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = AssertStep(
                step_id="v1",
                description="check",
                artifact_ref='result = {"ok": False, "issues": ["no def", "bad import"]}',
                inputs={},
                expected_fact="verified",
                extraction_expr="result",
            )
            result = execute_step(step, _make_ctx(ws))
            assert isinstance(result, Err)
            assert "no def" in result.error
            assert "bad import" in result.error

    def test_execution_error_still_propagates(self):
        """If the code itself raises, that's still an Err (not masked)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = AssertStep(
                step_id="v1",
                description="broken code",
                artifact_ref="raise ValueError('kaboom')",
                inputs={},
                expected_fact="verified",
                extraction_expr="result",
            )
            result = execute_step(step, _make_ctx(ws))
            assert isinstance(result, Err)
            assert "kaboom" in result.error


class TestAssertStepInPlan:
    """Assert steps trigger plan failure -> Ouroboros can patch."""

    def test_failed_assert_fails_plan(self):
        """Plan with passing write + failing assert -> Err (plan fails)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)

            spec = Spec(
                question="Write a file and verify it",
                steps=(
                    Step(
                        step_id="s1",
                        description="write a file",
                        artifact_type="write_file",
                        artifact_ref="print('hello')\n",
                        inputs={"path": "output.py"},
                        expected_fact="write_result",
                        extraction_expr="write_result",
                    ),
                    Step(
                        step_id="s2",
                        description="verify the file has a main function",
                        artifact_type="assert",
                        inputs={},
                        expected_fact="verified",
                        extraction_expr="result",
                        depends_on=("s1",),
                    ),
                ),
                synthesis="Done.",
            )

            # s2 code checks for "def main" -- won't find it
            from compass.generators.trinity._runtime import _attach_banner_code
            sections = [
                ("s2", (
                    "from pathlib import Path\n"
                    "content = Path(str(workspace) + '/output.py').read_text()\n"
                    "ok = 'def main' in content\n"
                    "issues = [] if ok else ['main function not found']\n"
                    "result = {'ok': ok, 'issues': issues}\n"
                )),
            ]
            spec = _attach_banner_code(spec, sections)

            result = execute_plan(spec, workspace=ws)
            assert isinstance(result, Err), f"plan should fail: {result}"
            assert "main function not found" in result.error

    def test_passing_assert_succeeds_plan(self):
        """Plan with write + passing assert -> Ok."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)

            spec = Spec(
                question="Write and verify",
                steps=(
                    Step(
                        step_id="s1",
                        description="write a file",
                        artifact_type="write_file",
                        artifact_ref="def main():\n    print('hello')\n",
                        inputs={"path": "output.py"},
                        expected_fact="write_result",
                        extraction_expr="write_result",
                    ),
                    Step(
                        step_id="s2",
                        description="verify main exists",
                        artifact_type="assert",
                        inputs={},
                        expected_fact="verified",
                        extraction_expr="result",
                        depends_on=("s1",),
                    ),
                ),
                synthesis="Done.",
            )

            from compass.generators.trinity._runtime import _attach_banner_code
            sections = [
                ("s2", (
                    "from pathlib import Path\n"
                    "content = Path(str(workspace) + '/output.py').read_text()\n"
                    "ok = 'def main' in content\n"
                    "issues = [] if ok else ['main function not found']\n"
                    "result = {'ok': ok, 'issues': issues}\n"
                )),
            ]
            spec = _attach_banner_code(spec, sections)

            result = execute_plan(spec, workspace=ws)
            assert isinstance(result, Ok), f"plan should pass: {result}"
