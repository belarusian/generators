"""Integration tests: generators handle escaping-heavy prompts correctly.

Each test gives a generator a prompt that requires multi-line content with:
  - regex patterns (backslashes)
  - nested string literals with quotes
  - code that would need \\n escaping in JSON

Requires a live model. Skip if no provider is available.
"""

from __future__ import annotations

import ast
import functools
import logging
import pytest
from pathlib import Path

from compass.generators._types import (
    DomainSection,
    GenerationContext,
    Ok,
)
from compass.generators._invoke import resolve_ask_fn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@functools.cache
def _has_model() -> bool:
    """Check if any model provider is reachable (cached)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        fn = resolve_ask_fn()
        result = fn("You are a test.", "Reply with exactly: OK")
        return isinstance(result, Ok)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Escaping-heavy prompts per generator
# ---------------------------------------------------------------------------

# Trinity/Vision: inline_python with regex and nested strings
TRINITY_ESCAPING_PROMPT = (
    "Write inline Python code that does the following:\n"
    "1. Define a multi-line string containing a JSON template with "
    "nested double quotes and backslash-n literals\n"
    "2. Use a regex pattern to extract all email addresses from text "
    "(pattern should use \\w, \\., etc.)\n"
    "3. The code should define 'result' as a dict with keys "
    "'template' (the JSON string) and 'emails' (list of found emails)\n"
    "\n"
    "Test text: 'Contact alice@example.com or bob.smith@corp.co.uk for info'"
)

# Meta: generate a generator module whose code has regex and docstrings
META_ESCAPING_PROMPT = (
    "Generate a small generator module called 'regex_tool' that:\n"
    "1. Has a _runtime.py with a function that uses regex patterns like "
    "r'\\d{3}-\\d{3}-\\d{4}' and r'[\\w.+-]+@[\\w-]+\\.[\\w.]+'\n"
    "2. The code should include triple-quoted docstrings with example "
    "patterns containing backslashes\n"
    "3. Include a _types.py with a dataclass that has a field for "
    "the regex pattern string"
)

# Neo: plan with steps that need complex prompts
NEO_ESCAPING_PROMPT = (
    "Create a plan to build a log parser that:\n"
    "1. One step should have a prompt describing regex patterns for "
    "log lines like r'(\\d{4}-\\d{2}-\\d{2}) (\\w+): (.*)'\n"
    "2. Another step should describe generating code that handles "
    "file paths with backslashes like 'C:\\\\Users\\\\data\\\\logs'\n"
    "3. The prompts should include example log lines with special characters"
)

# Factory: notebook cells with regex code
FACTORY_ESCAPING_PROMPT = (
    "Generate a Jupyter notebook that demonstrates regex patterns:\n"
    "1. A code cell that defines patterns like r'\\b\\d{3}-\\d{4}\\b' "
    "and r'(?P<name>[\\w.]+)@(?P<domain>[\\w.]+)'\n"
    "2. A markdown cell with triple-backtick code blocks showing "
    "escaped patterns\n"
    "3. A code cell that uses re.sub() with replacement strings "
    "containing \\1, \\2 backreferences"
)


def _build_trinity_context() -> GenerationContext:
    return GenerationContext(
        user_prompt=TRINITY_ESCAPING_PROMPT,
        default_task="Answer a question by planning artifact applications.",
        domain_context=(
            DomainSection(
                heading="Plan Construction Principles",
                content=(
                    "- Use inline_python for computation.\n"
                    "- Each step must be self-contained.\n"
                    "- extraction_expr names the result variable.\n"
                ),
            ),
        ),
        available_packages="re, json, pathlib",
    )


def _build_meta_context() -> GenerationContext:
    return GenerationContext(
        user_prompt=META_ESCAPING_PROMPT,
        default_task="Generate a generator module.",
        domain_context=(
            DomainSection(
                heading="Generator Structure",
                content=(
                    "A generator module has these files:\n"
                    "- _types.py: dataclasses for the spec\n"
                    "- _runtime.py: invoke_model, validate, ouroboros\n"
                    "- _context.py: build context\n"
                    "- generate.py: CLI entry point\n"
                ),
            ),
        ),
        available_packages="re, json, pathlib, dataclasses",
    )


def _build_neo_context() -> GenerationContext:
    return GenerationContext(
        user_prompt=NEO_ESCAPING_PROMPT,
        default_task="Create a multi-step plan.",
        domain_context=(
            DomainSection(
                heading="Available Generators",
                content=(
                    "- code: generates Python source files\n"
                    "- file: generates arbitrary text files\n"
                ),
            ),
        ),
        available_packages="re, json, pathlib",
    )


def _build_factory_context() -> GenerationContext:
    return GenerationContext(
        user_prompt=FACTORY_ESCAPING_PROMPT,
        default_task="Generate a Jupyter notebook.",
        domain_context=(
            DomainSection(
                heading="Notebook Structure",
                content=(
                    "A notebook has cells with:\n"
                    "- cell_type: 'code' or 'markdown'\n"
                    "- content: the cell source\n"
                    "- index: cell position\n"
                ),
            ),
        ),
        available_packages="re, json, pathlib",
    )


# ---------------------------------------------------------------------------
# Integration tests -- hit the real model
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_model(), reason="No model provider available")
class TestTrinityEscaping:
    """Trinity handles escaping-heavy prompts correctly."""

    def test_invoke_with_escaping(self):
        """Full G -> V -> G' loop should produce a valid executed plan."""
        from compass.generators.trinity.generate import run
        import tempfile

        ask_fn = resolve_ask_fn()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run(
                prompt=TRINITY_ESCAPING_PROMPT,
                ask_fn=ask_fn,
                output=tmpdir,
                max_rounds=3,
                max_fixes=3,
            )

        assert isinstance(result, Ok), f"run() failed: {result}"

    def test_ouroboros_with_escaping(self):
        """ouroboros_fix should produce valid patch with escaping-heavy code."""
        from compass.generators.trinity._runtime import ouroboros_fix
        from compass.generators.trinity._types import Spec, Step

        ctx = _build_trinity_context()

        # Build a spec with a deliberately broken inline_python step
        broken_spec = Spec(
            question="Extract emails using regex",
            steps=(
                Step(
                    step_id="s1",
                    description="Extract emails with regex",
                    artifact_type="inline_python",
                    artifact_ref="import re\npattern = r'[\\w.+-]+@[\\w-]+\\.[\\w.]+\nemails = re.findall(pattern, text)\nresult = emails",
                    inputs={},
                    expected_fact="emails",
                    extraction_expr="result",
                ),
            ),
            synthesis="Report the found emails.",
        )
        error = (
            "step 's1': SyntaxError in inline code at line 2: "
            "unterminated string literal"
        )

        ask_fn = resolve_ask_fn()
        fixed = ouroboros_fix(broken_spec, error, ctx, ask_fn=ask_fn)

        # ouroboros may return None if the model can't fix it
        if fixed is not None:
            # If it returned a fix, it should have valid steps
            for step in fixed.steps:
                if step.artifact_type == "inline_python":
                    ast.parse(step.artifact_ref)



@pytest.mark.skipif(not _has_model(), reason="No model provider available")
class TestMetaEscaping:
    """Meta handles escaping-heavy prompts correctly."""

    def test_invoke_with_escaping(self):
        """Full G -> V -> G' loop should produce a valid generator module."""
        from compass.generators.meta.generate import run
        import tempfile

        ask_fn = resolve_ask_fn()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run(
                prompt=META_ESCAPING_PROMPT,
                ask_fn=ask_fn,
                output=tmpdir,
                max_rounds=2,
                max_fixes=2,
            )

        assert isinstance(result, Ok), f"run() failed: {result}"

    def test_ouroboros_with_escaping(self):
        """Meta ouroboros should handle escaping-heavy code in file content."""
        from compass.generators.meta._runtime import ouroboros_meta
        from compass.generators.meta._types import GeneratorModuleSpec, SourceFile

        ctx = _build_meta_context()
        broken_spec = GeneratorModuleSpec(
            name="regex_tool",
            purpose="A regex validation tool",
            domain="text processing",
            files=(
                SourceFile(
                    path="_types.py",
                    content="from dataclasses import dataclass\n\n@dataclass\nclass Spec:\n    pattern: str",
                    description="Types",
                ),
                SourceFile(
                    path="_runtime.py",
                    content="import re\ndef validate(text, pattern):\n    return re.match(pattern, text",
                    description="Runtime with broken code",
                ),
            ),
            test_prompt="Test the regex tool",
        )
        error = "SyntaxError in _runtime.py: unexpected EOF while parsing"

        ask_fn = resolve_ask_fn()
        fixed = ouroboros_meta(broken_spec, error, ctx, ask_fn=ask_fn)

        if fixed is not None:
            for f in fixed.files:
                assert len(f.content) > 0



@pytest.mark.skipif(not _has_model(), reason="No model provider available")
class TestNeoEscaping:
    """Neo handles escaping-heavy prompts correctly."""

    def test_invoke_with_escaping(self):
        """Neo invoke_model should produce valid plan spec."""
        from compass.generators.neo._runtime import invoke_model
        from compass.generators.neo._types import PlanConfig

        ctx = _build_neo_context()
        ask_fn = resolve_ask_fn()
        config = PlanConfig(ask_fn=ask_fn)

        result = invoke_model(ctx, config)
        assert isinstance(result, Ok), f"invoke_model failed: {result}"

        spec = result.value
        assert hasattr(spec, "steps"), f"No steps attr on spec: {type(spec)}"
        assert len(spec.steps) > 0, "Empty steps list"


    def test_ouroboros_with_escaping(self):
        """Neo ouroboros should handle escaping-heavy prompts in steps."""
        from compass.generators.neo._runtime import ouroboros
        from compass.generators.neo._types import PlanConfig, PlanSpec, Step

        ctx = _build_neo_context()
        broken_spec = PlanSpec(
            goal="Build a log parser with regex",
            reasoning="Need to parse structured log lines",
            steps=(
                Step(
                    description="Generate the regex parser module",
                    artifact_type="module",
                    prompt="Create a module with pattern r'(\\d{4}-\\d{2}-\\d{2}) (\\w+): (.*)",
                ),
            ),
        )
        error = "Step 0 failed: artifact_type 'module' not found in available generators"

        ask_fn = resolve_ask_fn()
        config = PlanConfig(ask_fn=ask_fn)
        result = ouroboros(broken_spec, error, ctx, config)

        if isinstance(result, Ok) and result.value is not None:
            assert len(result.value.steps) > 0


# ---------------------------------------------------------------------------
# Vision integration test -- requires VISION_MODEL env var
# ---------------------------------------------------------------------------


def _has_vision_model() -> bool:
    """Check if VISION_MODEL is set and reachable."""
    import os
    vision_model = os.environ.get("VISION_MODEL", "")
    if not vision_model:
        return False
    try:
        from compass.llm.providers import get_provider_by_id
        provider = get_provider_by_id(vision_model)
        return provider is not None
    except Exception:
        return False


@pytest.mark.skipif(not _has_vision_model(), reason="VISION_MODEL not set")
class TestVisionExecution:
    """Test that trinity can execute vision steps via _execute_vision_step."""

    def test_vision_reads_image(self):
        """Vision step should describe what's in the test image."""
        from compass.generators.trinity.step_dispatch import _execute_vision_step
        from compass.generators.trinity._types import Step

        image_path = Path(__file__).parent.parent / "images" / "image.png"
        assert image_path.exists(), f"Test image not found: {image_path}"

        step = Step(
            step_id="read_image",
            description="Read the text in the image",
            artifact_type="vision",
            artifact_ref=str(image_path),
            inputs={"prompt": "What text is visible in this image? Reply concisely."},
            expected_fact="image_text",
            extraction_expr="result",
        )

        result = _execute_vision_step(step, step.inputs)
        assert isinstance(result, Ok), f"Vision step failed: {result}"

        fact = result.value
        response = fact.value.lower()
        print(f"\n--- vision response ---\n{fact.value}\n---")

        assert "sasha" in response, (
            f"Model should have seen 'I am Sasha' in the image, got: {fact.value}"
        )


