"""Code generator types."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Optional

from compass.generators._types import (
    DomainSection,
    Err,
    FileSpec,
    GenerationContext,
    GenerationReport,
    Ok,
    Result,
)
from compass.generators._validation import (
    extract_first_cell_index,
    validate_python_sources,
    summarize_variable_flow,
)


# ============================================================================
# Code artifact types
# ============================================================================


@dataclass(frozen=True)
class CodeFile:
    """A single source file in the generated code artifact.

    Each file has a relative path, content, and description.
    Files are the deliverables -- the actual code the user wants.
    """

    path: str          # relative path, e.g. "src/bst.py", "main.py"
    content: str | None = None  # full source code (filled by content block)
    description: str = ""  # what this file does
    language: str = "python"  # language identifier


@dataclass(frozen=True)
class CodeTestCase:
    """A test case that validates the generated code.

    The model declares test cases alongside the code. Each test is
    a snippet of Python that exercises the generated code. Tests
    run after the main files, in a shared namespace.
    """

    name: str          # descriptive test name
    source: str | None = None  # Python code (filled by content block)
    description: str = ""  # what this test validates
    expected_output: str = ""  # optional expected stdout substring


@dataclass(frozen=True)
class CodeSpec:
    """Complete code artifact specification.

    Write a CodeSpec constructor, then raw code after ### banners:

    CodeSpec(
        title="BST Implementation",
        purpose="Binary search tree with insert and search",
        files=(
            CodeFile(path="bst.py", description="BST implementation"),
        ),
        tests=(
            CodeTestCase(name="test_insert", description="Test insert"),
        ),
        entry_point="bst.py",
    )

    ### bst.py ###
    class BST:
        def __init__(self):
            self.root = None
        def insert(self, val):
            ...

    ### test:test_insert ###
    tree = BST()
    tree.insert(5)
    assert tree.search(5)
    """

    title: str
    purpose: str
    files: tuple[CodeFile, ...]
    tests: tuple[CodeTestCase, ...] = ()
    entry_point: str = ""  # e.g. "main.py" -- the file to run
    dependencies: tuple[str, ...] = ()  # e.g. ("numpy", "requests")


@dataclass(frozen=True)
class FileResult:
    """Outcome of validating/executing a single file."""

    file: CodeFile
    output: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class TestResult:
    """Outcome of running a single test case."""

    test: CodeTestCase
    passed: bool = False
    output: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ExecutedCode:
    """A spec paired with execution results."""

    spec: CodeSpec
    file_results: tuple[FileResult, ...]
    test_results: tuple[TestResult, ...]


# ============================================================================
# Patch types (for ouroboros targeted edits)
# ============================================================================


@dataclass(frozen=True)
class CodeFileEdit:
    """A targeted edit within a file: find old text, replace with new."""

    old: str
    new: str


@dataclass(frozen=True)
class CodeFilePatch:
    """Patch for a single file. Either targeted edits or full replacement.

    If content is set, replaces the entire file content.
    If edits is set, applies search-and-replace edits.
    Exactly one of content or edits must be provided.
    """

    path: str
    content: str | None = None
    edits: tuple[CodeFileEdit, ...] = ()


@dataclass(frozen=True)
class CodePatch:
    """A partial update to a CodeSpec.

    Ouroboros returns this instead of the full spec. Each CodeFilePatch
    targets a single file with search-and-replace edits. The model
    only returns the changes, not the full file content.

    CodePatch(
        files=(
            CodeFilePatch(path="main.py", edits=(
                CodeFileEdit(old="text to find", new="replacement text"),
            )),
        ),
    )
    """

    files: tuple[CodeFilePatch, ...]


# ============================================================================
# Runtime config (code-specific)
# ============================================================================


@dataclass(frozen=True)
class CodeConfig:
    """Runtime configuration for the code generation loop."""

    output_dir: Path = Path("generated_code")
    max_rounds: int = 3
    max_fixes: int = 3
    model_id: str = ""  # empty = use ladder policy
    verbose: bool = False
    dry_run: bool = False
    focus: Optional[str] = None
    ask_fn: Optional[object] = None  # AskFn
    live: bool = False
    prompt: Optional[str] = None


# ============================================================================
# Structural validators
# ============================================================================


def _validate_code_file(raw: dict, index: int) -> Result[CodeFile, str]:
    """Validate a single raw file dict into a CodeFile."""
    errors: list[str] = []

    path = raw.get("path")
    if not isinstance(path, str) or not path.strip():
        return Err(f"files[{index}].path: must be a non-empty string")
    path = path.strip()

    if path.startswith("/"):
        return Err(f"files[{index}].path: must be relative (got '{path}')")
    if ".." in path.split("/"):
        return Err(f"files[{index}].path: must not contain '..'")

    content = raw.get("content")
    if not isinstance(content, str) or not content.strip():
        return Err(f"files[{index}].content: must be a non-empty string")

    description = raw.get("description", "")
    if not isinstance(description, str):
        return Err(f"files[{index}].description: must be a string")

    language = raw.get("language", "python")
    if not isinstance(language, str):
        return Err(f"files[{index}].language: must be a string")

    return Ok(CodeFile(
        path=path,
        content=content,
        description=description,
        language=language.strip().lower(),
    ))


def _validate_test_case(raw: dict, index: int) -> Result[CodeTestCase, str]:
    """Validate a single raw test dict into a CodeTestCase."""
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return Err(f"tests[{index}].name: must be a non-empty string")

    source = raw.get("source")
    if not isinstance(source, str) or not source.strip():
        return Err(f"tests[{index}].source: must be a non-empty string")

    description = raw.get("description", "")
    if not isinstance(description, str):
        return Err(f"tests[{index}].description: must be a string")

    expected_output = raw.get("expected_output", "")
    if not isinstance(expected_output, str):
        return Err(f"tests[{index}].expected_output: must be a string")

    return Ok(CodeTestCase(
        name=name.strip(),
        source=source,
        description=description,
        expected_output=expected_output,
    ))


def validate_spec(raw: dict) -> Result[CodeSpec, str]:
    """Validate a raw JSON dict into a CodeSpec. [STRUCTURAL]

    Checks shape, types, and basic constraints. Does NOT parse or
    execute the code -- that happens in later validation stages.
    """
    if not isinstance(raw, dict):
        return Err(f"expected dict, got {type(raw).__name__}")

    title = raw.get("title")
    if not isinstance(title, str) or not title.strip():
        return Err("title must be a non-empty string")

    purpose = raw.get("purpose")
    if not isinstance(purpose, str) or not purpose.strip():
        return Err("purpose must be a non-empty string")

    # Files (required, at least one)
    raw_files = raw.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        return Err("files must be a non-empty list")

    file_errors: list[str] = []
    validated_files: list[CodeFile] = []
    seen_paths: set[str] = set()

    for i, f in enumerate(raw_files):
        if not isinstance(f, dict):
            file_errors.append(f"files[{i}]: expected dict")
            continue
        match _validate_code_file(f, i):
            case Err(e):
                file_errors.append(e)
            case Ok(cf):
                if cf.path in seen_paths:
                    file_errors.append(f"files[{i}].path: duplicate path '{cf.path}'")
                else:
                    seen_paths.add(cf.path)
                    validated_files.append(cf)

    if file_errors:
        return Err("; ".join(file_errors))

    # Tests (optional)
    validated_tests: list[CodeTestCase] = []
    raw_tests = raw.get("tests")
    if raw_tests is not None:
        if not isinstance(raw_tests, list):
            return Err("tests must be a list")
        test_errors: list[str] = []
        for i, t in enumerate(raw_tests):
            if not isinstance(t, dict):
                test_errors.append(f"tests[{i}]: expected dict")
                continue
            match _validate_test_case(t, i):
                case Err(e):
                    test_errors.append(e)
                case Ok(tc):
                    validated_tests.append(tc)
        if test_errors:
            return Err("; ".join(test_errors))

    # Entry point (optional)
    entry_point = raw.get("entry_point", "")
    if not isinstance(entry_point, str):
        return Err("entry_point must be a string")
    if entry_point and entry_point not in seen_paths:
        return Err(
            f"entry_point '{entry_point}' not found in files "
            f"(available: {sorted(seen_paths)})"
        )

    # Dependencies (optional)
    dependencies: tuple[str, ...] = ()
    raw_deps = raw.get("dependencies")
    if raw_deps is not None:
        if not isinstance(raw_deps, list):
            return Err("dependencies must be a list")
        for i, d in enumerate(raw_deps):
            if not isinstance(d, str) or not d.strip():
                return Err(f"dependencies[{i}]: must be a non-empty string")
        dependencies = tuple(d.strip() for d in raw_deps)

    return Ok(CodeSpec(
        title=title.strip(),
        purpose=purpose.strip(),
        files=tuple(validated_files),
        tests=tuple(validated_tests),
        entry_point=entry_point.strip(),
        dependencies=dependencies,
    ))


def validate_spec_instance(spec: CodeSpec) -> Result[CodeSpec, str]:
    """Validate a CodeSpec instance (from Python-as-schema).

    Lighter than validate_spec (which validates raw dicts).
    Checks what the constructor can't enforce: non-empty content,
    no duplicate paths, entry_point references a real file.
    """
    errors: list[str] = []
    if not spec.title:
        errors.append("title must be non-empty")
    if not spec.purpose:
        errors.append("purpose must be non-empty")
    if not spec.files:
        errors.append("files must be non-empty")
    seen: set[str] = set()
    for i, f in enumerate(spec.files):
        if not f.content:
            errors.append(f"files[{i}] ({f.path}): content is empty -- write code after a ### {f.path} ### banner")
        if f.path in seen:
            errors.append(f"files[{i}]: duplicate path '{f.path}'")
        seen.add(f.path)
    for i, t in enumerate(spec.tests):
        if not t.source:
            errors.append(f"tests[{i}] ({t.name}): source is empty -- write code after a ### test:{t.name} ### banner")
    if spec.entry_point and spec.entry_point not in {f.path for f in spec.files}:
        errors.append(f"entry_point '{spec.entry_point}' not in files")
    if errors:
        return Err("; ".join(errors))
    return Ok(spec)


def validate_code_patch(raw: dict) -> Result[CodePatch, str]:
    """Validate a raw JSON dict into a CodePatch. [STRUCTURAL]"""
    if not isinstance(raw, dict):
        return Err(f"expected dict, got {type(raw).__name__}")

    raw_files = raw.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        return Err("files must be a non-empty list")

    errors: list[str] = []
    validated: list[CodeFilePatch] = []
    seen_paths: set[str] = set()

    for i, f in enumerate(raw_files):
        if not isinstance(f, dict):
            errors.append(f"files[{i}]: expected dict")
            continue

        path = f.get("path")
        if not isinstance(path, str) or not path.strip():
            errors.append(f"files[{i}].path: must be a non-empty string")
            continue
        path = path.strip()

        if path in seen_paths:
            errors.append(f"files[{i}].path: duplicate path '{path}'")
            continue
        seen_paths.add(path)

        content = f.get("content")
        raw_edits = f.get("edits")
        has_content = isinstance(content, str) and content.strip()
        has_edits = isinstance(raw_edits, list) and raw_edits

        if not has_content and not has_edits:
            errors.append(f"files[{i}]: must have 'content' or 'edits'")
            continue

        if has_content and has_edits:
            errors.append(f"files[{i}]: provide 'content' or 'edits', not both")
            continue

        if has_content:
            validated.append(CodeFilePatch(path=path, content=content))
            continue

        edits: list[CodeFileEdit] = []
        for j, e in enumerate(raw_edits):
            if not isinstance(e, dict):
                errors.append(f"files[{i}].edits[{j}]: expected dict")
                continue
            old = e.get("old")
            new = e.get("new")
            if not isinstance(old, str) or not old:
                errors.append(f"files[{i}].edits[{j}].old: must be a non-empty string")
                continue
            if not isinstance(new, str):
                errors.append(f"files[{i}].edits[{j}].new: must be a string")
                continue
            edits.append(CodeFileEdit(old=old, new=new))

        if edits:
            validated.append(CodeFilePatch(path=path, edits=tuple(edits)))

    if errors:
        return Err("; ".join(errors))

    return Ok(CodePatch(files=tuple(validated)))


def apply_code_patch(
    spec: CodeSpec,
    patch: CodePatch,
) -> Result[CodeSpec, str]:
    """Apply a CodePatch to a CodeSpec.

    Each CodeFilePatch targets a file by path. Each CodeFileEdit within it
    is a search-and-replace on the file content. If an old string
    is not found, the patch fails with a precise error.
    """
    file_map = {f.path: f for f in spec.files}
    errors: list[str] = []

    for fp in patch.files:
        if fp.path not in file_map:
            errors.append(f"{fp.path}: not in spec files")
            continue

        original = file_map[fp.path]

        if fp.content is not None:
            # Full replacement
            file_map[fp.path] = CodeFile(
                path=fp.path,
                content=fp.content,
                description=original.description,
                language=original.language,
            )
        else:
            # Targeted edits
            content = original.content
            for edit in fp.edits:
                if edit.old not in content:
                    errors.append(
                        f"{fp.path}: old text not found: {edit.old[:80]}..."
                    )
                    continue
                content = content.replace(edit.old, edit.new, 1)

            file_map[fp.path] = CodeFile(
                path=fp.path,
                content=content,
                description=original.description,
                language=original.language,
            )

    if errors:
        return Err("; ".join(errors))

    # Rebuild files tuple preserving original order
    merged = tuple(file_map[f.path] for f in spec.files)
    return Ok(replace(spec, files=merged))


def validate_python_files(spec: CodeSpec) -> Result[None, str]:
    """Cross-file Python validation: syntax, variable flow. [SEMANTIC]

    Only validates Python files. Non-Python files are skipped.
    Delegates to shared validate_python_sources.
    """
    python_files = [
        f for f in spec.files if f.language == "python"
    ]
    if not python_files:
        return Ok(None)

    python_sources = tuple(f.content for f in python_files)
    python_labels = [f.path for f in python_files]

    # Validate each file independently for syntax (cross-file flow
    # is less meaningful for standalone modules)
    errors: list[str] = []
    for i, (source, label) in enumerate(zip(python_sources, python_labels)):
        match validate_python_sources((source,), label=label):
            case Err(e):
                # Remap generic index to file path
                remapped = e.replace(f"{label}[0]", f"files[{i}] ({label})")
                errors.append(remapped)
            case Ok(_):
                pass

    if errors:
        return Err("; ".join(errors))
    return Ok(None)


def validate_test_syntax(spec: CodeSpec) -> Result[None, str]:
    """Validate test case syntax. [SEMANTIC]"""
    if not spec.tests:
        return Ok(None)

    test_sources = tuple(t.source for t in spec.tests)
    test_names = [t.name for t in spec.tests]

    errors: list[str] = []
    for i, (source, name) in enumerate(zip(test_sources, test_names)):
        match validate_python_sources((source,), label=name):
            case Err(e):
                remapped = e.replace(f"{name}[0]", f"tests[{i}] ({name})")
                errors.append(remapped)
            case Ok(_):
                pass

    if errors:
        return Err("; ".join(errors))
    return Ok(None)


def summarize_file_contents(spec: CodeSpec) -> str:
    """Produce a human/model-readable summary of the code artifact."""
    lines: list[str] = [f"Code artifact: {spec.title}"]
    lines.append(f"Purpose: {spec.purpose}")
    lines.append(f"Files: {len(spec.files)}")
    for i, f in enumerate(spec.files):
        lines.append(f"  files[{i}] ({f.path}): {f.description}")
    if spec.tests:
        lines.append(f"Tests: {len(spec.tests)}")
        for i, t in enumerate(spec.tests):
            lines.append(f"  tests[{i}] ({t.name}): {t.description}")
    if spec.entry_point:
        lines.append(f"Entry point: {spec.entry_point}")
    if spec.dependencies:
        lines.append(f"Dependencies: {', '.join(spec.dependencies)}")
    return "\n".join(lines)


# ============================================================================
# Ouroboros helpers
# ============================================================================

_FILE_INDEX_RE = re.compile(r"files\[(\d+)\]")
_TEST_INDEX_RE = re.compile(r"tests\[(\d+)\]")


def extract_first_file_index(error: str) -> int | None:
    """Extract the first files[N] index from an error string."""
    m = _FILE_INDEX_RE.search(error)
    return int(m.group(1)) if m else None


def extract_first_test_index(error: str) -> int | None:
    """Extract the first tests[N] index from an error string."""
    m = _TEST_INDEX_RE.search(error)
    return int(m.group(1)) if m else None


def replace_file(
    spec: CodeSpec, index: int, new_file: CodeFile,
) -> CodeSpec:
    """Return a new CodeSpec with one file replaced. Pure."""
    files = list(spec.files)
    files[index] = new_file
    return replace(spec, files=tuple(files))


# ============================================================================
# Versioned output paths
# ============================================================================

_VERSION_SUFFIX_RE = re.compile(r"_v(\d+)$")


def detect_version(path: Path) -> int:
    """Extract version from _vN directory suffix. Returns 0 if absent."""
    m = _VERSION_SUFFIX_RE.search(path.name)
    return int(m.group(1)) if m else 0


def versioned_dir(base: Path, version: int) -> Path:
    """Return directory path with _vN suffix."""
    name = base.name
    m = _VERSION_SUFFIX_RE.search(name)
    if m:
        name = name[:m.start()]
    return base.with_name(f"{name}_v{version}")
