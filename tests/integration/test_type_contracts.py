"""Integration tests: expected_type type contracts.

Steps declare expected_type to control the Python type of their fact value.
_make_typed_fact coerces at the boundary. resolve_fact returns raw_value.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from compass.generators._types import Ok, Err
from compass.generators.trinity._types import Fact, InlinePythonStep, Spec, Step
from compass.generators.trinity._runtime import execute_plan, _attach_banner_code
from compass.generators.trinity.step_dispatch import StepContext, execute_step
from compass.generators.trinity.fact_dispatch import resolve_fact


def _make_ctx(workspace: Path, inputs: dict | None = None, facts: dict | None = None) -> StepContext:
    return StepContext(
        resolved_inputs=inputs or {},
        facts=facts or {},
        workspace=workspace,
    )


class TestExpectedTypeCoercion:
    """expected_type controls the Python type of the fact value."""

    def test_str_coerces_dict_to_summary(self):
        """Dict with a 'summary' key and expected_type='str' -> summary string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = InlinePythonStep(
                step_id="s1",
                description="produce a dict",
                artifact_ref='result = {"summary": "3 files changed", "count": 3}',
                inputs={},
                expected_fact="output",
                expected_type="str",
                extraction_expr="result",
            )
            result = execute_step(step, _make_ctx(ws))
            assert isinstance(result, Ok), f"should succeed: {result}"
            fact = result.value
            assert fact.raw_value == "3 files changed"
            assert isinstance(fact.raw_value, str)

    def test_dict_parses_json_string(self):
        """JSON string with expected_type='dict' -> parsed dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = InlinePythonStep(
                step_id="s1",
                description="produce a JSON string",
                artifact_ref='import json; result = json.dumps({"key": "val"})',
                inputs={},
                expected_fact="output",
                expected_type="dict",
                extraction_expr="result",
            )
            result = execute_step(step, _make_ctx(ws))
            assert isinstance(result, Ok), f"should succeed: {result}"
            fact = result.value
            assert fact.raw_value == {"key": "val"}
            assert isinstance(fact.raw_value, dict)

    def test_no_expected_type_backward_compat(self):
        """Default expected_type='any' preserves existing behavior, raw_value set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = InlinePythonStep(
                step_id="s1",
                description="produce a number",
                artifact_ref="result = 42",
                inputs={},
                expected_fact="output",
                extraction_expr="result",
            )
            result = execute_step(step, _make_ctx(ws))
            assert isinstance(result, Ok)
            fact = result.value
            assert fact.raw_value == 42

    def test_type_mismatch_produces_err(self):
        """expected_type='int' on a non-numeric string -> Err."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = InlinePythonStep(
                step_id="s1",
                description="produce text",
                artifact_ref='result = "not a number"',
                inputs={},
                expected_fact="output",
                expected_type="int",
                extraction_expr="result",
            )
            result = execute_step(step, _make_ctx(ws))
            assert isinstance(result, Err)
            assert "type contract" in result.error

    def test_str_coerces_dict_without_summary_to_json(self):
        """Dict without 'summary' key and expected_type='str' -> JSON string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = InlinePythonStep(
                step_id="s1",
                description="produce a dict without summary",
                artifact_ref='result = {"a": 1, "b": 2}',
                inputs={},
                expected_fact="output",
                expected_type="str",
                extraction_expr="result",
            )
            result = execute_step(step, _make_ctx(ws))
            assert isinstance(result, Ok)
            fact = result.value
            assert isinstance(fact.raw_value, str)
            parsed = json.loads(fact.raw_value)
            assert parsed == {"a": 1, "b": 2}


class TestResolveFactWithRawValue:
    """resolve_fact uses raw_value when present."""

    def test_raw_value_returned_directly(self):
        """Fact with raw_value -> resolve_fact returns it, skips json.loads."""
        fact = Fact(
            step_id="s1",
            name="output",
            value='{"summary": "done"}',
            fact_type="text",
            raw_value="done",
        )
        assert resolve_fact(fact) == "done"

    def test_no_raw_value_falls_through(self):
        """Fact without raw_value -> json.loads as before."""
        fact = Fact(
            step_id="s1",
            name="output",
            value='{"key": "val"}',
            fact_type="json",
        )
        assert resolve_fact(fact) == {"key": "val"}


class TestTypeContractChain:
    """Two-step chain: produce dict, coerce to str, verify in assert."""

    def test_str_type_flows_through_chain(self):
        """Step 1 produces dict with expected_type='str'. Step 2 gets a string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)

            spec = Spec(
                question="Produce a summary and verify it's a string",
                steps=(
                    Step(
                        step_id="s1",
                        description="produce a dict that should become a string",
                        artifact_type="inline_python",
                        inputs={},
                        expected_fact="summary",
                        expected_type="str",
                        extraction_expr="result",
                    ),
                    Step(
                        step_id="s2",
                        description="verify we got a string",
                        artifact_type="assert",
                        inputs={"text": {"$fact": "summary"}},
                        expected_fact="verified",
                        extraction_expr="result",
                        depends_on=("s1",),
                    ),
                ),
                synthesis="Confirmed type contract works.",
            )

            sections = [
                ("s1", 'result = {"summary": "commit abc123", "details": [1, 2, 3]}'),
                ("s2", (
                    "ok = isinstance(text, str)\n"
                    "issues = [] if ok else [f'expected str, got {type(text).__name__}']\n"
                    "result = {'ok': ok, 'issues': issues}\n"
                )),
            ]
            spec = _attach_banner_code(spec, sections)

            result = execute_plan(spec, workspace=ws)
            assert isinstance(result, Ok), f"plan should pass: {result}"
