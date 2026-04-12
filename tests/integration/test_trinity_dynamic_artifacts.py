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
import shutil
import tempfile
from pathlib import Path

# Path to the real artifacts/programmer.py in the repo
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROGRAMMER_ARTIFACT = _REPO_ROOT / "artifacts" / "programmer.py"

import pytest

from compass.generators._types import Ok, Err
from compass.generators._invoke import resolve_ask_fn
from compass.generators.trinity._types import (
    Spec, Step, DynamicStep, validate_spec_instance, promote_step,
)
from compass.generators.trinity._runtime import (
    discover_artifacts,
    execute_plan,
    format_artifacts_for_context,
)
from compass.generators.trinity._paths import (
    bundled_artifact_module,
    generators_repo_root,
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


class TestArtifactRequiredInputs:
    """discover_artifacts extracts required_inputs from run() body."""

    def test_extracts_required_inputs_from_signature(self):
        """Kwonly params with no default -> required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "artifacts").mkdir()
            (ws / "artifacts" / "verifier.py").write_text(
                'from compass.generators._types import Ok, Err\n'
                'from compass.generators.trinity._types import Fact\n'
                '\n'
                'def run(step, resolved_inputs, workspace, *, auth_token, endpoint, timeout=30):\n'
                '    if not auth_token:\n'
                '        return Err("auth_token is required")\n'
                '    return Ok(Fact(step_id=step.step_id, name=step.expected_fact,\n'
                '                  value="ok", fact_type="text"))\n'
            )

            artifacts = discover_artifacts(ws)
            art = next(a for a in artifacts if "verifier" in a.path)
            # auth_token and endpoint have no default -> required
            # timeout has a default (30) -> optional
            assert "auth_token" in art.required_inputs
            assert "endpoint" in art.required_inputs
            assert "timeout" not in art.required_inputs

    def test_optional_inputs_not_required(self):
        """Kwonly params with defaults -> not required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "worker.py").write_text(
                'from compass.generators._types import Ok\n'
                'from compass.generators.trinity._types import Fact\n'
                '\n'
                'def run(step, resolved_inputs, workspace, *, payload=None, mode="fast"):\n'
                '    return Ok(Fact(step_id=step.step_id, name=step.expected_fact,\n'
                '                  value=str(payload), fact_type="text"))\n'
            )

            artifacts = discover_artifacts(ws)
            art = next(a for a in artifacts if "worker" in a.path)
            assert art.required_inputs == ()

    def test_required_inputs_shown_in_context(self):
        """format_artifacts_for_context includes required inputs line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "checker.py").write_text(
                'from compass.generators._types import Ok, Err\n'
                'from compass.generators.trinity._types import Fact\n'
                '\n'
                'def run(step, resolved_inputs, workspace, *, name, age):\n'
                '    if not name:\n'
                '        return Err("name required")\n'
                '    return Ok(Fact(step_id=step.step_id, name=step.expected_fact,\n'
                '                  value="ok", fact_type="text"))\n'
            )

            artifacts = discover_artifacts(ws)
            context = format_artifacts_for_context(artifacts)
            assert "required inputs: name, age" in context

    def test_no_required_inputs_line_when_empty(self):
        """Artifacts with no resolved_inputs.get() calls have no required inputs line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "simple.py").write_text(
                'from compass.generators._types import Ok\n'
                'from compass.generators.trinity._types import Fact\n'
                '\n'
                'def run(step, resolved_inputs, workspace):\n'
                '    return Ok(Fact(step_id=step.step_id, name=step.expected_fact,\n'
                '                  value="done", fact_type="text"))\n'
            )

            artifacts = discover_artifacts(ws)
            art = next(a for a in artifacts if "simple" in a.path)
            assert art.required_inputs == ()
            context = format_artifacts_for_context(artifacts)
            assert "required inputs" not in context


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
    """Programmer artifact: dynamic resolution via artifacts/programmer.py."""

    def test_programmer_promotes_to_dynamic_step(self):
        """artifact_type='programmer' promotes to DynamicStep when artifact exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "artifacts").mkdir()
            shutil.copy2(_PROGRAMMER_ARTIFACT, ws / "artifacts" / "programmer.py")

            step = Step(
                step_id="s1",
                description="generate code",
                artifact_type="programmer",
                inputs={"problem": "Build a hello world"},
                expected_fact="code_result",
            )
            promoted = promote_step(step, [], ws)
            assert isinstance(promoted, DynamicStep), f"Expected DynamicStep, got {type(promoted).__name__}"

    def test_unknown_artifact_type_stays_base_step_without_module(self):
        """No workspace + no bundled ``artifacts/{type}.py`` leaves a base Step.

        ``programmer`` is always bundled in the generators repo, so ``promote_step(..., None)``
        still resolves it to DynamicStep; use a fictitious type to test the unresolved path.
        """
        step = Step(
            step_id="s1",
            description="noop",
            artifact_type="nonexistent_trinity_artifact_type_xyz",
            inputs={},
            expected_fact="out",
        )
        promoted = promote_step(step, [], None)
        assert type(promoted) is Step, f"Expected base Step, got {type(promoted).__name__}"

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


def _load_artifact_module(filename: str):
    """Load ``artifacts/{filename}.py`` from the repo (not an import package)."""
    import importlib.util

    path = _REPO_ROOT / "artifacts" / filename
    spec = importlib.util.spec_from_file_location(f"artifact_{filename}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestBundledBridgeArtifacts:
    """Bundled ``artifacts/*.py`` from the generators repo (screen, action_invoker)."""

    def test_generators_repo_root_detects_checkout(self):
        root = generators_repo_root()
        assert root is not None, "generators_repo_root() should find this checkout"
        assert (root / "artifacts" / "screen.py").is_file()
        assert (root / "artifacts" / "action_invoker.py").is_file()

    def test_bundled_artifact_module_resolves_any_generators_artifact(self):
        assert bundled_artifact_module("screen") is not None
        assert bundled_artifact_module("action_invoker") is not None
        assert bundled_artifact_module("user_query") is not None
        assert bundled_artifact_module("programmer") is not None
        assert bundled_artifact_module("no_such_artifact_xyz") is None

    def test_promote_screen_without_workspace_copy_uses_bundled_file(self):
        """Empty workspace: ``artifact_type=screen`` still resolves to DynamicStep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = Step(
                step_id="s1",
                description="replay skill",
                artifact_type="screen",
                inputs={"skill": "nonexistent_skill_for_promotion_only"},
                expected_fact="out",
            )
            promoted = promote_step(step, [], ws)
            assert isinstance(promoted, DynamicStep), type(promoted).__name__
            assert promoted.module_path.endswith("artifacts/screen.py")
            assert Path(promoted.module_path).is_file()

    def test_promote_action_invoker_without_workspace_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            step = Step(
                step_id="a1",
                description="invoke neo action",
                artifact_type="action_invoker",
                inputs={"action": {"type": "ScreenshotAction", "region": "full"}},
                expected_fact="shot",
            )
            promoted = promote_step(step, [], ws)
            assert isinstance(promoted, DynamicStep), type(promoted).__name__
            assert promoted.module_path.endswith("artifacts/action_invoker.py")

    def test_discover_merges_bundled_when_workspace_has_no_artifacts(self):
        """Neo-lab-style workspace: lists all runnable generators repo artifacts/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            (ws / "README.md").write_text("x")
            found = discover_artifacts(ws)
            paths = {a.path for a in found}
            assert "artifacts/screen.py" in paths
            assert "artifacts/action_invoker.py" in paths
            assert "artifacts/user_query.py" in paths
            text = format_artifacts_for_context(found)
            assert "screen.py" in text
            assert "action_invoker.py" in text

    def test_action_invoker_run_returns_err_without_action(self):
        """``run()`` contract: missing inputs['action'] -> Err (no GUI)."""
        from compass.generators._types import Err as ErrT

        mod = _load_artifact_module("action_invoker.py")

        class FakeStep:
            step_id = "t1"
            expected_fact = "fact"

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Path(tmpdir)
            out = mod.run(FakeStep(), {}, ws)
            assert isinstance(out, ErrT), out
            assert "action" in out.error.lower()


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


def _make_workspace_with_programmer(tmpdir: str) -> Path:
    """Create a temp workspace with artifacts/programmer.py copied in."""
    ws = Path(tmpdir)
    (ws / "artifacts").mkdir()
    shutil.copy2(_PROGRAMMER_ARTIFACT, ws / "artifacts" / "programmer.py")
    return ws


@pytest.mark.skipif(not _has_oracle(), reason="No Oracle provider available")
class TestProgrammerStepExecution:
    """Trinity executes programmer artifact end-to-end via dynamic resolution."""

    def test_programmer_step_produces_chunks(self):
        """Execute a simple programmer step and verify it returns JSON chunks."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = _make_workspace_with_programmer(tmpdir)

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
        """Programmer is cycle-breaking: execute_plan returns Cycle after s1.

        The chaining to s2 happens across plan rounds (generation loop
        re-plans with s1's facts). Here we verify s1's fact is well-formed
        so s2 could consume it in a subsequent round.
        """
        import json
        from compass.generators._types import Cycle

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = _make_workspace_with_programmer(tmpdir)

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
            # Programmer is CYCLE_BREAKING -- runtime stops after s1
            assert isinstance(result, Cycle), f"Expected Cycle, got {type(result).__name__}: {result}"
            assert "code_result" in result.facts, f"Missing code_result fact. Facts: {list(result.facts)}"

            # Verify the fact is valid JSON that s2 could consume
            fact = result.facts["code_result"]
            data = json.loads(fact.value)
            assert data["success"] is True, f"Programmer reported failure: {data}"
            assert len(data.get("chunks", [])) >= 1, "Programmer produced no chunks"

            # Chunks are applied to disk before cycle break
            for chunk in data["chunks"]:
                target = Path(ws / chunk["target"])
                assert target.exists(), f"Chunk not written to disk: {chunk['target']}"
