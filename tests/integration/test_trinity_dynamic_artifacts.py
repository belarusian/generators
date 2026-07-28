"""Integration test: Trinity dynamic artifact type resolution.

Validates that Trinity can:
  - Accept unknown artifact_types without rejecting at validation
  - Execute unknown types with banner code as inline_python fallback
  - Discover and execute saved artifact modules from an artifacts/ directory
  - Chain a discovered artifact into a multi-step plan
  - Generate a new artifact type, persist it, and reuse it (end-to-end)
"""

from __future__ import annotations

import functools
import tempfile
from pathlib import Path

import pytest

from compass.generators._types import Ok, Err
from compass.generators._invoke import resolve_ask_fn
from compass.generators.trinity._types import (
    Spec, Step, ProgrammerStep, validate_spec_instance, promote_step,
)
from compass.generators.trinity._runtime import execute_plan


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


class TestDynamicArtifactTypes:
    """Trinity resolves unknown artifact types dynamically."""

    def test_unknown_type_not_rejected_by_validator(self):
        """validate_spec_instance should not reject unknown artifact_types."""
        spec = Spec(
            question="Test unknown type",
            steps=(
                Step(
                    step_id="s1",
                    description="do something custom",
                    artifact_type="word_counter",
                    artifact_ref="result = '42'",
                    inputs={},
                    expected_fact="answer",
                    extraction_expr="result",
                ),
            ),
            synthesis="Report the answer.",
        )

        result = validate_spec_instance(spec)
        assert isinstance(result, Ok), (
            f"validate_spec_instance rejected unknown type 'word_counter': {result}"
        )

    def test_unknown_type_with_banner_executes_as_inline_python(self):
        """Unknown artifact_type with code in artifact_ref executes like inline_python."""
        spec = Spec(
            question="Count words",
            steps=(
                Step(
                    step_id="s1",
                    description="count words in text",
                    artifact_type="word_counter",
                    artifact_ref="text = 'hello world foo'\nresult = str(len(text.split()))",
                    inputs={},
                    expected_fact="word_count",
                    extraction_expr="result",
                ),
            ),
            synthesis="Report the word count.",
        )

        exec_result = execute_plan(spec)
        assert isinstance(exec_result, Ok), f"execute_plan failed: {exec_result}"

        facts = exec_result.value.facts
        fact = next((f for f in facts if f.name == "word_count"), None)
        assert fact is not None, f"No word_count fact. Facts: {[f.name for f in facts]}"
        assert fact.value == "3"

    def test_saved_artifact_module_discovered(self):
        """Artifact module in artifacts/ directory is discovered and executed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            artifacts_dir = ws / "artifacts"
            artifacts_dir.mkdir()
            (artifacts_dir / "line_counter.py").write_text(
                "from compass.generators._types import Ok\n"
                "from compass.generators.trinity._types import Fact\n"
                "\n"
                "\n"
                "def run(step, resolved_inputs, workspace):\n"
                "    text = resolved_inputs.get('text', '')\n"
                "    count = len(text.strip().splitlines())\n"
                "    return Ok(Fact(\n"
                "        step_id=step.step_id,\n"
                "        name=step.expected_fact,\n"
                "        value=str(count),\n"
                "        fact_type='numeric',\n"
                "    ))\n"
            )

            spec = Spec(
                question="Count lines",
                steps=(
                    Step(
                        step_id="s1",
                        description="count lines in text",
                        artifact_type="line_counter",
                        inputs={"text": "one\ntwo\nthree"},
                        expected_fact="line_count",
                        extraction_expr="result",
                    ),
                ),
                synthesis="Report the count.",
            )

            exec_result = execute_plan(spec, workspace=ws)
            assert isinstance(exec_result, Ok), f"execute_plan failed: {exec_result}"

            facts = exec_result.value.facts
            fact = next((f for f in facts if f.name == "line_count"), None)
            assert fact is not None, f"No line_count fact. Facts: {[f.name for f in facts]}"
            assert fact.value == "3"

    def test_discovered_artifact_chains_with_inline_python(self):
        """Discovered artifact feeds facts into a downstream inline_python step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            artifacts_dir = ws / "artifacts"
            artifacts_dir.mkdir()
            (artifacts_dir / "file_lister.py").write_text(
                "import os\n"
                "from compass.generators._types import Ok\n"
                "from compass.generators.trinity._types import Fact\n"
                "\n"
                "\n"
                "def run(step, resolved_inputs, workspace):\n"
                "    target = resolved_inputs.get('dir', str(workspace))\n"
                "    entries = os.listdir(target)\n"
                "    return Ok(Fact(\n"
                "        step_id=step.step_id,\n"
                "        name=step.expected_fact,\n"
                "        value='\\n'.join(sorted(entries)),\n"
                "        fact_type='text',\n"
                "    ))\n"
            )

            # Create some files to list
            (ws / "alpha.txt").write_text("a")
            (ws / "beta.txt").write_text("b")

            spec = Spec(
                question="List files and count them",
                steps=(
                    Step(
                        step_id="s1",
                        description="list files in workspace",
                        artifact_type="file_lister",
                        inputs={},
                        expected_fact="file_list",
                        extraction_expr="result",
                    ),
                    Step(
                        step_id="s2",
                        description="count the files",
                        artifact_type="inline_python",
                        artifact_ref="lines = file_list.strip().split('\\n')\nresult = str(len(lines))",
                        inputs={"file_list": {"$fact": "file_list"}},
                        expected_fact="file_count",
                        extraction_expr="result",
                        depends_on=("s1",),
                    ),
                ),
                synthesis="Report the file count.",
            )

            exec_result = execute_plan(spec, workspace=ws)
            assert isinstance(exec_result, Ok), f"execute_plan failed: {exec_result}"

            facts = exec_result.value.facts
            list_fact = next((f for f in facts if f.name == "file_list"), None)
            assert list_fact is not None
            assert "alpha.txt" in list_fact.value
            assert "beta.txt" in list_fact.value

            count_fact = next((f for f in facts if f.name == "file_count"), None)
            assert count_fact is not None
            # artifacts/ + alpha.txt + beta.txt = 3
            assert int(count_fact.value) >= 2


class TestWriteFilePersistence:
    """write_file + banner persists artifact modules to disk."""

    def test_write_file_creates_artifact_module(self):
        """write_file step writes banner content to the target path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)

            module_source = (
                "from compass.generators._types import Ok\n"
                "from compass.generators.trinity._types import Fact\n"
                "\n"
                "def run(step, resolved_inputs, workspace):\n"
                "    text = resolved_inputs.get('text', '')\n"
                "    count = len(text.split())\n"
                "    return Ok(Fact(\n"
                "        step_id=step.step_id,\n"
                "        name=step.expected_fact,\n"
                "        value=str(count),\n"
                "        fact_type='numeric',\n"
                "    ))\n"
            )

            spec = Spec(
                question="Persist a word counter",
                steps=(
                    Step(
                        step_id="s1",
                        description="persist word_counter as reusable artifact",
                        artifact_type="write_file",
                        artifact_ref=module_source,
                        inputs={"path": "artifacts/word_counter.py"},
                        expected_fact="saved_path",
                    ),
                ),
                synthesis="Confirm the artifact was saved.",
            )

            result = execute_plan(spec, workspace=ws)
            assert isinstance(result, Ok), f"execute_plan failed: {result}"

            # File was written
            saved = ws / "artifacts" / "word_counter.py"
            assert saved.exists(), f"artifact not written to {saved}"
            assert "def run(" in saved.read_text()

            # Fact contains the written path
            fact = next((f for f in result.value.facts if f.name == "saved_path"), None)
            assert fact is not None
            assert "word_counter.py" in fact.value

    def test_write_file_then_discover_and_run(self):
        """Full cycle: write_file persists a module, then a second step uses it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)

            module_source = (
                "from compass.generators._types import Ok\n"
                "from compass.generators.trinity._types import Fact\n"
                "\n"
                "def run(step, resolved_inputs, workspace):\n"
                "    text = resolved_inputs.get('text', '')\n"
                "    count = len(text.split())\n"
                "    return Ok(Fact(\n"
                "        step_id=step.step_id,\n"
                "        name=step.expected_fact,\n"
                "        value=str(count),\n"
                "        fact_type='numeric',\n"
                "    ))\n"
            )

            # Two-step plan: persist, then use
            spec = Spec(
                question="Build and use a word counter",
                steps=(
                    Step(
                        step_id="s1",
                        description="persist word_counter artifact",
                        artifact_type="write_file",
                        artifact_ref=module_source,
                        inputs={"path": "artifacts/word_counter.py"},
                        expected_fact="saved_path",
                    ),
                    Step(
                        step_id="s2",
                        description="count words using the saved artifact",
                        artifact_type="word_counter",
                        inputs={"text": "one two three four five"},
                        expected_fact="word_count",
                        extraction_expr="result",
                        depends_on=("s1",),
                    ),
                ),
                synthesis="Report the word count.",
            )

            result = execute_plan(spec, workspace=ws)
            assert isinstance(result, Ok), f"execute_plan failed: {result}"

            facts = result.value.facts
            count_fact = next((f for f in facts if f.name == "word_count"), None)
            assert count_fact is not None, f"No word_count fact. Facts: {[f.name for f in facts]}"
            assert count_fact.value == "5"


class TestProgrammerStep:
    """ProgrammerStep: Trinity can delegate to the Programmer NFA."""

    def test_programmer_step_promotes_from_base(self):
        """artifact_type='programmer' promotes to ProgrammerStep."""
        step = Step(
            step_id="s1",
            description="generate code",
            artifact_type="programmer",
            inputs={"problem": "Build a hello world"},
            expected_fact="code_result",
        )
        promoted = promote_step(step, {}, None)
        assert isinstance(promoted, ProgrammerStep)

    def test_programmer_step_validates_with_problem_input(self):
        """ProgrammerStep passes validation when problem is in inputs."""
        step = ProgrammerStep(
            step_id="s1",
            description="generate code",
            inputs={"problem": "Build a CSV parser"},
            expected_fact="code_result",
        )
        assert step.validate(0) == []

    def test_programmer_step_validates_with_artifact_ref(self):
        """ProgrammerStep passes validation when problem is in artifact_ref."""
        step = ProgrammerStep(
            step_id="s1",
            description="generate code",
            artifact_ref="Build a CSV parser",
            inputs={},
            expected_fact="code_result",
        )
        assert step.validate(0) == []

    def test_programmer_step_rejects_missing_problem(self):
        """ProgrammerStep fails validation without problem or artifact_ref."""
        step = ProgrammerStep(
            step_id="s1",
            description="generate code",
            inputs={},
            expected_fact="code_result",
        )
        errors = step.validate(0)
        assert len(errors) == 1
        assert "problem" in errors[0]

    def test_programmer_step_in_spec_validates(self):
        """A Spec containing a programmer step passes structural validation."""
        spec = Spec(
            question="Generate a module",
            steps=(
                Step(
                    step_id="s1",
                    description="use programmer to generate code",
                    artifact_type="programmer",
                    inputs={"problem": "Build a fibonacci module"},
                    expected_fact="code_chunks",
                ),
            ),
            synthesis="Report the generated code.",
        )
        result = validate_spec_instance(spec)
        assert isinstance(result, Ok), f"validation failed: {result}"

    def test_programmer_step_chains_with_inline_python(self):
        """Programmer step can feed facts into downstream inline_python."""
        spec = Spec(
            question="Generate and inspect code",
            steps=(
                Step(
                    step_id="s1",
                    description="use programmer to generate code",
                    artifact_type="programmer",
                    inputs={"problem": "Build a hello world module"},
                    expected_fact="code_chunks",
                ),
                Step(
                    step_id="s2",
                    description="count the chunks produced",
                    artifact_type="inline_python",
                    artifact_ref="import json\ndata = json.loads(code_chunks)\nresult = str(len(data.get('chunks', [])))",
                    inputs={"code_chunks": {"$fact": "code_chunks"}},
                    expected_fact="chunk_count",
                    extraction_expr="result",
                    depends_on=("s1",),
                ),
            ),
            synthesis="Report how many chunks were generated.",
        )
        result = validate_spec_instance(spec)
        assert isinstance(result, Ok), f"validation failed: {result}"


@functools.cache
def _has_oracle() -> bool:
    """Check if the Oracle can be constructed and reach a model."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from compass.llm.oracle import Oracle
        Oracle()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_oracle(), reason="No Oracle provider available")
class TestProgrammerStepExecution:
    """Trinity executes a ProgrammerStep end-to-end via the Programmer NFA."""

    def test_programmer_step_produces_chunks(self):
        """Execute a simple programmer step and verify it returns JSON chunks."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)

            spec = Spec(
                question="Generate a greeting function",
                steps=(
                    Step(
                        step_id="s1",
                        description="use programmer to generate a greeting module",
                        artifact_type="programmer",
                        inputs={"problem": "Write a Python function called greet(name) that returns 'Hello, {name}!'"},
                        expected_fact="code_result",
                    ),
                ),
                synthesis="Report the generated code.",
            )

            result = execute_plan(spec, workspace=ws)
            assert isinstance(result, Ok), f"execute_plan failed: {result}"

            facts = result.value.facts
            fact = next((f for f in facts if f.name == "code_result"), None)
            assert fact is not None, f"No code_result fact. Facts: {[f.name for f in facts]}"

            # Fact should be valid JSON with chunks
            data = json.loads(fact.value)
            assert data["success"] is True, f"Programmer reported failure: {data}"
            assert isinstance(data["chunks"], list)
            assert len(data["chunks"]) >= 1, "Programmer produced no chunks"

    def test_programmer_step_chains_to_inline_python(self):
        """Programmer produces chunks, inline_python inspects them."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)

            spec = Spec(
                question="Generate code and count chunks",
                steps=(
                    Step(
                        step_id="s1",
                        description="generate a greeting module",
                        artifact_type="programmer",
                        inputs={"problem": "Write a Python function called greet(name) that returns 'Hello, {name}!'"},
                        expected_fact="code_result",
                    ),
                    Step(
                        step_id="s2",
                        description="count the chunks produced",
                        artifact_type="inline_python",
                        artifact_ref="import json\ndata = json.loads(code_result) if isinstance(code_result, str) else code_result\nresult = str(len(data.get('chunks', [])))",
                        inputs={"code_result": {"$fact": "code_result"}},
                        expected_fact="chunk_count",
                        extraction_expr="result",
                        depends_on=("s1",),
                    ),
                ),
                synthesis="Report the chunk count.",
            )

            result = execute_plan(spec, workspace=ws)
            assert isinstance(result, Ok), f"execute_plan failed: {result}"

            facts = result.value.facts
            count_fact = next((f for f in facts if f.name == "chunk_count"), None)
            assert count_fact is not None, f"No chunk_count fact. Facts: {[f.name for f in facts]}"
            assert int(count_fact.value) >= 1, f"Expected at least 1 chunk, got {count_fact.value}"
