"""IO boundaries for code generation.

All side effects live here: code execution, file I/O, ouroboros invocation.
Uses shared invoke/prompt machinery from compass.generators._invoke.

Ouroboros uses CodePatch (targeted edits) instead of full spec replacement.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from compass.generators._types import (
    AskFn,
    Err,
    GenerationContext,
    GenerationReport,
    Ok,
    Result,
)
from compass.generators._invoke import (
    build_system_prompt,
    resolve_ask_fn,
)
import inspect

from compass.core.python_schema import parse_typed_response, parse_response_with_files

from compass.generators.code._types import (
    CodeConfig,
    CodeFile,
    CodeFileEdit,
    CodeFilePatch,
    CodePatch,
    CodeSpec,
    CodeTestCase,
    ExecutedCode,
    FileResult,
    TestResult,
    apply_code_patch,
    extract_first_file_index,
    extract_first_test_index,
    summarize_file_contents,
)

logger = logging.getLogger(__name__)

_SPEC_TYPE_SOURCE = (
    inspect.getsource(CodeFile) + "\n\n"
    + inspect.getsource(CodeTestCase) + "\n\n"
    + inspect.getsource(CodeSpec)
)


# ---------------------------------------------------------------------------
# Model invocation (code-specific prompt)
# ---------------------------------------------------------------------------


def invoke_model(ctx: GenerationContext, config: CodeConfig) -> Result:
    """Call the model and return a CodeSpec. G_code.

    Python-as-schema: model writes CodeSpec constructor + ### banner ### sections.
    No JSON, no escaping.
    """
    ask_fn = resolve_ask_fn(config.model_id, config.ask_fn)

    system = build_system_prompt(
        ctx,
        _SPEC_TYPE_SOURCE,
        role="You are an expert software engineer who generates clean, well-structured Python code.",
        contract_preamble="Respond as shown in the CodeSpec docstring.",
    )

    user = _build_code_user_message(ctx, config)

    logger.debug("--- G_code SYSTEM ---\n%s\n--- END SYSTEM ---", system)
    logger.debug("--- G_code USER ---\n%s\n--- END USER ---", user)

    match ask_fn(system, user):
        case Err() as e:
            return e
        case Ok(raw_text):
            pass

    logger.debug("--- G_code RESPONSE ---\n%s\n--- END RESPONSE ---", raw_text)

    # Try banner format first
    try:
        spec, sections = parse_response_with_files(raw_text, CodeSpec)
        spec = _attach_banner_code(spec, sections)
        return Ok(spec)
    except ValueError:
        pass

    # Fallback: no banners (all fields inline)
    try:
        spec = parse_typed_response(raw_text, CodeSpec)
        return Ok(spec)
    except ValueError as e:
        return Err(str(e))


def _attach_banner_code(spec: CodeSpec, sections: list[tuple[str, str]]) -> CodeSpec:
    """Attach banner code sections to CodeSpec files and tests."""
    from dataclasses import replace

    code_map = dict(sections)
    new_files = []
    for f in spec.files:
        if f.path in code_map:
            new_files.append(replace(f, content=code_map[f.path]))
        else:
            new_files.append(f)

    new_tests = []
    for t in spec.tests:
        key = f"test:{t.name}"
        if key in code_map:
            new_tests.append(replace(t, source=code_map[key]))
        else:
            new_tests.append(t)

    return replace(spec, files=tuple(new_files), tests=tuple(new_tests))


def _build_code_user_message(
    ctx: GenerationContext, config: CodeConfig,
) -> str:
    """Build code-specific user message."""
    primary = (
        ctx.user_prompt if ctx.user_prompt is not None else
        ctx.default_task if ctx.default_task is not None else
        "Generate a Python code artifact."
    )
    parts = [primary]

    parts.extend([
        "",
        "Write a CodeSpec(...) constructor followed by ### banner ### code sections.",
        "Follow the shape of the CodeSpec docstring.",
        "No markdown fencing.",
    ])

    if ctx.available_packages:
        parts.extend(["", f"Available packages: {ctx.available_packages}"])

    if config.focus:
        parts.extend(["", f"FOCUS AREA: Emphasise '{config.focus}' patterns."])

    if ctx.feedback:
        parts.extend(["", "Your previous attempt had errors:", ""])
        for fb in ctx.feedback:
            parts.append(f"  {fb}")
        parts.extend(["", "Please fix these issues in your next attempt."])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Ouroboros -- targeted file editing via CodePatch
# ---------------------------------------------------------------------------


def ouroboros(
    spec: CodeSpec,
    error: str,
    ctx: GenerationContext,
    config: CodeConfig,
) -> Result:
    """Ouroboros: the model consumes its own output and produces a CodePatch.

    G' : (CodeSpec, Error) -> CodePatch -> CodeSpec

    The model returns a CodePatch with targeted edits, not the full spec.
    The patch is validated and applied to produce the corrected spec.
    """
    ask_fn = resolve_ask_fn(config.model_id, config.ask_fn)

    file_listing: list[str] = []
    for i, f in enumerate(spec.files):
        file_listing.append(f"### files[{i}] ({f.path}) - {f.description}")
        file_listing.append(f.content)

    test_listing: list[str] = []
    if spec.tests:
        for i, t in enumerate(spec.tests):
            test_listing.append(f"### tests[{i}] ({t.name}) - {t.description}")
            test_listing.append(t.source)
            if t.expected_output:
                test_listing.append(f"Expected output contains: {t.expected_output}")

    summary = summarize_file_contents(spec)

    system_parts = [
        "You are correcting code you previously generated.",
    ]
    if ctx.user_prompt:
        system_parts.append(f"Original request: {ctx.user_prompt}")

    if ctx.domain_context:
        system_parts.append("")
        system_parts.append("Domain context (from original generation):")
        for ds in ctx.domain_context:
            if ds.content and not ds.content.startswith("No "):
                system_parts.append(f"  - {ds.heading}")

    system_parts.extend([
        "",
        "You will receive your prior code artifact and an error.",
        "Write a CodePatch(...) expression with targeted edits to fix the error.",
        "Each edit is a search-and-replace: find 'old' text, replace with 'new'.",
        "Only include the files that need to change.",
        "",
        "CodePatch format (targeted edits -- preferred for surgical changes):",
        "CodePatch(",
        "    files=(",
        '        CodeFilePatch(path="main.py", edits=(',
        '            CodeFileEdit(old="exact text to find", new="replacement text"),',
        "        )),",
        "    ),",
        ")",
        "",
        "CodePatch format (full replacement):",
        "CodePatch(",
        "    files=(",
        '        CodeFilePatch(path="main.py", content="...full replacement..."),',
        "    ),",
        ")",
        "",
        "Rules:",
        "- Use edits for surgical changes. Use full content for full replacement.",
        "- 'old' must be an EXACT substring of the current file content.",
        "- 'old' should include enough surrounding context to be unique.",
        "- 'new' is the complete replacement for that substring.",
        "- Each file must have either 'edits' or 'content', not both.",
        "- Focus your fix on the error described. Minimise changes to other files.",
        "- All Python code must have valid syntax and execute without errors.",
    ])
    if ctx.available_packages:
        system_parts.append(f"- Available packages: {ctx.available_packages}")
    system_parts.extend([
        "",
        "Write ONLY the CodePatch(...) expression. No markdown fencing, no explanation.",
    ])

    user_parts = [
        f"# Your code artifact: {spec.title}",
        f"# {summary}",
        "",
        "## Files",
        "",
        "\n".join(file_listing),
    ]
    if test_listing:
        user_parts.extend([
            "",
            "## Tests",
            "",
            "\n".join(test_listing),
        ])
    user_parts.extend([
        "",
        "## Error",
        "",
        error,
        "",
        "Write a CodePatch(...) expression with targeted edits to fix this error.",
    ])

    match ask_fn("\n".join(system_parts), "\n".join(user_parts)):
        case Err() as e:
            return e
        case Ok(raw_text):
            pass

    try:
        patch = parse_typed_response(raw_text, CodePatch)
    except ValueError as e:
        return Err(f"Ouroboros parse error: {e}")

    total_edits = sum(len(fp.edits) for fp in patch.files)
    logger.info(
        "Ouroboros patch: %d file(s), %d edit(s): %s",
        len(patch.files), total_edits, [fp.path for fp in patch.files],
    )

    # Apply the patch to the original spec
    match apply_code_patch(spec, patch):
        case Err(e):
            return Err(f"Ouroboros patch apply failed: {e}")
        case Ok(corrected):
            return Ok(corrected)


def try_ouroboros(
    spec: CodeSpec,
    error: str,
    ctx: GenerationContext,
    config: CodeConfig,
) -> CodeSpec | None:
    """Attempt an ouroboros fix.

    Returns the corrected full spec if successful, None if ouroboros cannot help.
    Always attempts a fix since code errors are always targetable.
    """
    logger.info("Ouroboros: attempting fix for error: %s", error[:120])
    match ouroboros(spec, error, ctx, config):
        case Ok(corrected):
            logger.info(
                "Ouroboros: returned corrected artifact (%d files, %d tests)",
                len(corrected.files), len(corrected.tests),
            )
            return corrected
        case Err(e):
            logger.warning("Ouroboros: failed -- %s", e)
            return None


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------


def execute_code(spec: CodeSpec) -> Result:
    """Execute all Python files and then run tests.

    Proves the code actually runs. Files are written to a temp directory
    and executed. Tests run in the same namespace after all files.

    Returns Ok(ExecutedCode) or Err(error_description).
    """
    tmpdir = tempfile.mkdtemp(prefix="code_gen_")
    original_path = sys.path[:]

    try:
        # Write all files to temp directory
        for f in spec.files:
            fpath = Path(tmpdir) / f.path
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(f.content)

        # Add temp directory (and subdirectories) to sys.path
        sys.path.insert(0, tmpdir)
        # Also add any subdirectories that contain __init__.py or .py files
        for subdir in Path(tmpdir).rglob("*"):
            if subdir.is_dir() and str(subdir) not in sys.path:
                sys.path.insert(0, str(subdir))

        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            pass

        namespace: dict[str, Any] = {
            "__name__": "__main__",
            "__file_dir__": tmpdir,
        }

        file_results: list[FileResult] = []
        execution_errors: list[str] = []

        # Execute each Python file in order
        python_files = [f for f in spec.files if f.language == "python"]
        for i, f in enumerate(python_files):
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            try:
                with contextlib.redirect_stdout(stdout_buf), \
                     contextlib.redirect_stderr(stderr_buf):
                    exec(f.content, namespace)  # noqa: S102

                file_results.append(FileResult(
                    file=f,
                    output=stdout_buf.getvalue() or None,
                ))

            except Exception as exc:
                tb = traceback.extract_tb(exc.__traceback__)
                location = ""
                if tb:
                    last = tb[-1]
                    location = f" (line {last.lineno})"

                error_msg = (
                    f"files[{i}] ({f.path}){location}: "
                    f"{type(exc).__name__}: {exc}"
                )
                execution_errors.append(error_msg)
                file_results.append(FileResult(
                    file=f,
                    output=stdout_buf.getvalue() or None,
                    error=error_msg,
                ))
                # Stop on first file error
                break

        if execution_errors:
            return Err("; ".join(execution_errors))

        # Run tests
        test_results: list[TestResult] = []
        test_errors: list[str] = []

        for i, test in enumerate(spec.tests):
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()

            try:
                with contextlib.redirect_stdout(stdout_buf), \
                     contextlib.redirect_stderr(stderr_buf):
                    exec(test.source, namespace)  # noqa: S102

                output = stdout_buf.getvalue()

                # Check expected output if specified
                if test.expected_output and test.expected_output not in (output or ""):
                    error_msg = (
                        f"tests[{i}] ({test.name}): expected output to contain "
                        f"'{test.expected_output}' but got '{output}'"
                    )
                    test_errors.append(error_msg)
                    test_results.append(TestResult(
                        test=test,
                        passed=False,
                        output=output or None,
                        error=error_msg,
                    ))
                else:
                    test_results.append(TestResult(
                        test=test,
                        passed=True,
                        output=output or None,
                    ))

            except Exception as exc:
                tb = traceback.extract_tb(exc.__traceback__)
                location = ""
                if tb:
                    last = tb[-1]
                    location = f" (line {last.lineno})"

                error_msg = (
                    f"tests[{i}] ({test.name}){location}: "
                    f"{type(exc).__name__}: {exc}"
                )
                test_errors.append(error_msg)
                test_results.append(TestResult(
                    test=test,
                    passed=False,
                    output=stdout_buf.getvalue() or None,
                    error=error_msg,
                ))

        if test_errors:
            return Err("; ".join(test_errors))

        return Ok(ExecutedCode(
            spec=spec,
            file_results=tuple(file_results),
            test_results=tuple(test_results),
        ))

    finally:
        sys.path[:] = original_path
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Code I/O
# ---------------------------------------------------------------------------


def emit_code(
    executed: ExecutedCode,
    output_dir: Path,
    *,
    report: GenerationReport | None = None,
) -> Path:
    """Write the code artifact to the output directory.

    Creates the directory, writes all files, and emits a report sidecar.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write source files
    for f in executed.spec.files:
        fpath = output_dir / f.path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(f.content)
        logger.info("  Written: %s", fpath)

    # Write test files
    if executed.spec.tests:
        test_dir = output_dir / "tests"
        test_dir.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(executed.spec.tests):
            safe_name = t.name.replace(" ", "_").replace("/", "_").lower()
            test_path = test_dir / f"test_{safe_name}.py"
            test_path.write_text(t.source)
            logger.info("  Test written: %s", test_path)

    # Write manifest
    manifest = {
        "title": executed.spec.title,
        "purpose": executed.spec.purpose,
        "entry_point": executed.spec.entry_point,
        "dependencies": list(executed.spec.dependencies),
        "files": [
            {"path": f.path, "description": f.description, "language": f.language}
            for f in executed.spec.files
        ],
        "tests": [
            {
                "name": t.name,
                "description": t.description,
                "passed": tr.passed,
            }
            for t, tr in zip(executed.spec.tests, executed.test_results)
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("  Manifest: %s", manifest_path)

    # Write report sidecar
    if report:
        report_path = output_dir / "generation_report.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("  Report: %s", report_path)

    # Write README
    readme_lines = [
        f"# {executed.spec.title}",
        "",
        executed.spec.purpose,
        "",
    ]
    if executed.spec.entry_point:
        readme_lines.extend([
            "## Usage",
            "",
            f"```bash",
            f"python {executed.spec.entry_point}",
            "```",
            "",
        ])
    if executed.spec.files:
        readme_lines.extend(["## Files", ""])
        for f in executed.spec.files:
            readme_lines.append(f"- **{f.path}**: {f.description}")
        readme_lines.append("")
    if executed.spec.dependencies:
        readme_lines.extend([
            "## Dependencies",
            "",
            ", ".join(executed.spec.dependencies),
            "",
        ])
    if executed.spec.tests:
        readme_lines.extend(["## Tests", ""])
        for t, tr in zip(executed.spec.tests, executed.test_results):
            status = "PASS" if tr.passed else "FAIL"
            readme_lines.append(f"- [{status}] **{t.name}**: {t.description}")
        readme_lines.append("")

    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines))
    logger.info("  README: %s", readme_path)

    return output_dir


def load_code_artifact(path: Path) -> Result:
    """Load a code artifact from a directory (reads manifest.json)."""
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return Err(f"No manifest.json found in {path}")

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return Err(f"Failed to read manifest: {e}")

    files: list[CodeFile] = []
    for f_info in manifest.get("files", []):
        fpath = path / f_info["path"]
        if not fpath.exists():
            return Err(f"File not found: {fpath}")
        files.append(CodeFile(
            path=f_info["path"],
            content=fpath.read_text(),
            description=f_info.get("description", ""),
            language=f_info.get("language", "python"),
        ))

    if not files:
        return Err(f"No files found in {path}")

    return Ok(CodeSpec(
        title=manifest.get("title", path.name),
        purpose=manifest.get("purpose", "loaded from directory"),
        files=tuple(files),
        entry_point=manifest.get("entry_point", ""),
        dependencies=tuple(manifest.get("dependencies", [])),
    ))
