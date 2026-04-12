"""Unit tests: ouroboros parse and patch normalization robustness.

Covers:
- parse_typed_response rejects duplicate keyword arguments (SyntaxError)
  instead of crashing the process.
- SpecPatch __post_init__ coercion: bare StepPatch / Step / str values
  are normalized to 1-tuples at construction time.
- TRINITY_PARSE_EXTRA_TYPES: Spec eval context includes ShellStep, SpecPatch, etc.
"""

from __future__ import annotations

import pytest

from compass.core.python_schema import parse_typed_response
from compass.generators._types import Ok
from compass.generators.trinity._types import (
    Spec,
    SpecPatch,
    StepPatch,
    Step,
    TRINITY_PARSE_EXTRA_TYPES,
    apply_spec_patch,
)


# ---------------------------------------------------------------------------
# parse_typed_response: SyntaxError becomes ValueError
# ---------------------------------------------------------------------------


class TestParseTypedResponseSyntaxErrors:

    def test_duplicate_keyword_raises_valueerror(self):
        """Model emits description=... twice -> ValueError, not SyntaxError."""
        code = (
            'SpecPatch(\n'
            '    step_patches=(\n'
            '        StepPatch(step_id="s3", fields={"artifact_type": "shell"}),\n'
            '    ),\n'
            '    description="first",\n'
            '    description="second",\n'
            ')'
        )
        with pytest.raises(ValueError, match="Constructor error"):
            parse_typed_response(
                code, SpecPatch, extra_types=TRINITY_PARSE_EXTRA_TYPES
            )

    def test_other_syntax_error_raises_valueerror(self):
        """Garbled output -> ValueError, not SyntaxError."""
        code = 'SpecPatch(step_patches= @#$ garbage'
        with pytest.raises(ValueError):
            parse_typed_response(
                code, SpecPatch, extra_types=TRINITY_PARSE_EXTRA_TYPES
            )

    def test_valid_spec_patch_still_parses(self):
        """Sanity: well-formed SpecPatch still works."""
        code = (
            'SpecPatch(\n'
            '    step_patches=(\n'
            '        StepPatch(step_id="s1", fields={"description": "fixed"}),\n'
            '    ),\n'
            ')'
        )
        result = parse_typed_response(
            code, SpecPatch, extra_types=TRINITY_PARSE_EXTRA_TYPES
        )
        assert isinstance(result, SpecPatch)
        assert len(result.step_patches) == 1
        assert result.step_patches[0].step_id == "s1"


# ---------------------------------------------------------------------------
# SpecPatch __post_init__ coercion (missing trailing comma)
# ---------------------------------------------------------------------------


class TestSpecPatchTupleCoercion:
    """Coercion happens in __post_init__, so parse -> already normalized."""

    def test_bare_step_patch_coerced_at_construction(self):
        """(StepPatch(...)) without trailing comma -> 1-tuple after parse."""
        code = (
            'SpecPatch(\n'
            '    step_patches=(StepPatch(step_id="s1", fields={"x": "y"})),\n'
            ')'
        )
        patch = parse_typed_response(
            code, SpecPatch, extra_types=TRINITY_PARSE_EXTRA_TYPES
        )
        assert isinstance(patch.step_patches, tuple)
        assert len(patch.step_patches) == 1
        assert patch.step_patches[0].step_id == "s1"

    def test_bare_step_in_add_steps_coerced_at_construction(self):
        """(Step(...)) without trailing comma -> 1-tuple after parse."""
        code = (
            'SpecPatch(\n'
            '    add_steps=(Step(\n'
            '        step_id="s_new", description="do thing",\n'
            '        artifact_type="inline_python",\n'
            '        expected_fact="out", extraction_expr="result")),\n'
            ')'
        )
        patch = parse_typed_response(
            code, SpecPatch, extra_types=TRINITY_PARSE_EXTRA_TYPES
        )
        assert isinstance(patch.add_steps, tuple)
        assert len(patch.add_steps) == 1
        assert patch.add_steps[0].step_id == "s_new"

    def test_bare_string_in_remove_steps_coerced_at_construction(self):
        """("s7") without trailing comma -> 1-tuple, not iterated as chars."""
        code = 'SpecPatch(remove_steps=("s7"))'
        patch = parse_typed_response(
            code, SpecPatch, extra_types=TRINITY_PARSE_EXTRA_TYPES
        )
        assert isinstance(patch.remove_steps, tuple)
        assert patch.remove_steps == ("s7",)

    def test_proper_tuples_untouched(self):
        """Trailing commas present -> no coercion needed."""
        code = (
            'SpecPatch(\n'
            '    step_patches=(\n'
            '        StepPatch(step_id="s1", fields={"x": "y"}),\n'
            '    ),\n'
            '    remove_steps=("s3",),\n'
            ')'
        )
        patch = parse_typed_response(
            code, SpecPatch, extra_types=TRINITY_PARSE_EXTRA_TYPES
        )
        assert isinstance(patch.step_patches, tuple)
        assert len(patch.step_patches) == 1
        assert isinstance(patch.remove_steps, tuple)
        assert patch.remove_steps == ("s3",)

    def test_direct_construction_bare_values(self):
        """Coercion works via direct constructor too, not just parse."""
        sp = StepPatch(step_id="s1", fields={"a": "b"})
        patch = SpecPatch(step_patches=sp, remove_steps="s2")
        assert patch.step_patches == (sp,)
        assert patch.remove_steps == ("s2",)


# ---------------------------------------------------------------------------
# Trinity parse namespace: Step subtypes + SpecPatch alongside Spec
# ---------------------------------------------------------------------------


class TestTrinityParseExtraTypes:

    def test_spec_with_shellstep_fails_without_extras(self):
        code = (
            "s = Spec(\n"
            '    question="q",\n'
            "    steps=(\n"
            '        ShellStep(\n'
            '            step_id="s1", description="run", artifact_ref="true",\n'
            '            inputs={}, expected_fact="out",\n'
            "        ),\n"
            "    ),\n"
            '    synthesis="syn",\n'
            ")\n"
        )
        with pytest.raises(ValueError, match="ShellStep"):
            parse_typed_response(code, Spec)

    def test_spec_with_shellstep_parses_with_trinity_extras(self):
        code = (
            "s = Spec(\n"
            '    question="q",\n'
            "    steps=(\n"
            '        ShellStep(\n'
            '            step_id="s1", description="run", artifact_ref="true",\n'
            '            inputs={}, expected_fact="out",\n'
            "        ),\n"
            "    ),\n"
            '    synthesis="syn",\n'
            ")\n"
        )
        spec = parse_typed_response(code, Spec, extra_types=TRINITY_PARSE_EXTRA_TYPES)
        assert isinstance(spec, Spec)
        assert spec.steps[0].artifact_type == "shell"

    def test_spec_block_may_construct_specpatch_with_trinity_extras(self):
        """Regression: full-Spec fallback must not raise name 'SpecPatch' is not defined."""
        code = (
            "_ = SpecPatch()\n"
            "s = Spec(\n"
            '    question="q",\n'
            "    steps=(\n"
            '        Step(\n'
            '            step_id="s1", description="d", artifact_type="shell",\n'
            '            artifact_ref="true", inputs={}, expected_fact="f",\n'
            "        ),\n"
            "    ),\n"
            '    synthesis="syn",\n'
            ")\n"
        )
        spec = parse_typed_response(code, Spec, extra_types=TRINITY_PARSE_EXTRA_TYPES)
        assert isinstance(spec, Spec)

    def test_step_accepts_code_keyword_as_inline_body_alias(self):
        """Models often emit code= instead of artifact_ref= for inline_python."""
        code = (
            "s = Spec(\n"
            '    question="q",\n'
            "    steps=(\n"
            "        InlinePythonStep(\n"
            '            step_id="s1", description="compute",\n'
            '            code="result = 42",\n'
            '            expected_fact="out",\n'
            "        ),\n"
            "    ),\n"
            '    synthesis="syn",\n'
            ")\n"
        )
        spec = parse_typed_response(code, Spec, extra_types=TRINITY_PARSE_EXTRA_TYPES)
        assert spec.steps[0].artifact_ref == "result = 42"
        assert spec.steps[0].code is None


class TestApplySpecPatchAddStepsUpsert:
    """add_steps with an existing step_id replaces the step (ouroboros repair)."""

    def test_add_steps_replaces_existing_step(self):
        spec = Spec(
            question="q",
            steps=(
                Step(
                    step_id="s1",
                    description="old",
                    artifact_type="inline_python",
                    artifact_ref="result = 1",
                    inputs={},
                    expected_fact="x",
                ),
            ),
            synthesis="syn",
        )
        replacement = Step(
            step_id="s1",
            description="new",
            artifact_type="inline_python",
            artifact_ref="result = 2",
            inputs={},
            expected_fact="x",
        )
        patch = SpecPatch(add_steps=(replacement,))
        out = apply_spec_patch(spec, patch)
        assert isinstance(out, Ok)
        assert out.value.steps[0].description == "new"
        assert out.value.steps[0].artifact_ref == "result = 2"
