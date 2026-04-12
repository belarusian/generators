"""IO boundaries for the CLI tool generator.

V_cli: validate generated Python source (syntax + execution).
G'_cli: ouroboros -- narrow fix loop for source errors.
IO: emit CLI script to disk, load existing script.

Validation pipeline (cheapest first):
  1. Structural: validate_spec (dict shape) -- in _types.py
  2. Semantic: ast.parse(source) -- syntax check
  3. Semantic: check source contains expected patterns
     (argparse, subcommands, --verbose, --version)
  4. Executive: exec() the source in isolated namespace
     to catch runtime import errors and logic bugs

The ouroboros fix is NARROW:
  - Model receives its prior CliToolSpec(...) expression + specific error
  - Model returns a corrected CliToolSpec(...) expression
  - Only the source field changes (surgical)
  - The inner fix loop in generation_loop handles re-validation
"""

from __future__ import annotations

import ast
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

from compass.generators._types import (
    AskFn,
    DomainSection,
    Err,
    GenerationContext,
    Ok,
    Result,
)
from compass.generators._invoke import (
    build_system_prompt,
    build_user_message,
    resolve_ask_fn,
)
from compass.core.python_schema import parse_typed_response
from compass.generators._validation import validate_python_sources

from compass.generators.cli_tool._types import CliToolSpec, validate_spec

logger = logging.getLogger(__name__)

_TYPES_SOURCE = (Path(__file__).parent / "_types.py").read_text()


# ---------------------------------------------------------------------------
# Model invocation -- G_cli
# ---------------------------------------------------------------------------


def invoke_model(
    ctx: GenerationContext,
    model_id: str = "",
    ask_fn: AskFn | None = None,
) -> Result:
    """Call the model and return parsed CliToolSpec. G_cli."""
    fn = resolve_ask_fn(model_id, ask_fn)

    system = build_system_prompt(
        ctx,
        _TYPES_SOURCE,
        role=(
            "You are an expert Python developer specializing in CLI tools. "
            "You generate complete, production-quality argparse-based command-line "
            "applications with subcommands, help text, and proper error handling."
        ),
        contract_preamble=(
            "Write a CliToolSpec(...) Python constructor expression. "
            "See the type definition below for the schema."
        ),
    )

    user = _build_cli_user_message(ctx)

    match fn(system, user):
        case Err() as e:
            return e
        case Ok(raw_text):
            pass

    try:
        spec = parse_typed_response(raw_text, CliToolSpec)
        return Ok(spec)
    except ValueError as e:
        return Err(str(e))


def _build_cli_user_message(ctx: GenerationContext) -> str:
    """Build user message for CLI tool generation."""
    primary = (
        ctx.user_prompt if ctx.user_prompt is not None else
        "Generate a CLI tool."
    )
    parts = [primary]

    parts.extend([
        "",
        "Write a CliToolSpec(...) Python constructor expression.",
        "",
        "Example response shape:",
        "",
        "CliToolSpec(",
        '    name="mytool",',
        '    version="1.0.0",',
        '    description="My CLI tool",',
        "    subcommands=(",
        "        Subcommand(",
        '            name="run",',
        '            help_text="Run the tool",',
        "            arguments=(",
        '                Argument(name="--path", help_text="Target path"),',
        "            ),",
        '            handler_body="print(args.path)",',
        "        ),",
        "    ),",
        '    source="#!/usr/bin/env python3\\n...",',
        ")",
        "",
        "REQUIREMENTS for the generated CLI tool:",
        "- Use argparse with subparsers for subcommands",
        "- Include --verbose flag (global, before subcommand) that enables detailed output",
        "- Include --version flag that prints the version and exits",
        "- Each subcommand has its own handler function",
        "- Proper error handling: catch exceptions, print user-friendly messages, exit with code 1",
        "- Include if __name__ == '__main__': main() at the bottom",
        "- The main() function must return an int exit code (0=success, 1=error)",
        "- Use sys.exit() with the return value of main()",
        "",
        "CRITICAL -- Python source quality:",
        "- The 'source' field must be syntactically valid Python that passes ast.parse()",
        "- Use f-strings for string formatting, NOT string concatenation with +",
        "- Use triple-quoted strings for multi-line text",
        "- Do NOT use string concatenation inside function call arguments",
        "- All imports must be from the standard library",
        "- Use proper indentation (4 spaces)",
        "",
        "STRUCTURE of the source:",
        "  #!/usr/bin/env python3",
        '  """<tool description>"""',
        "  import argparse",
        "  import sys",
        "  # ... other stdlib imports as needed",
        "",
        "  __version__ = '<version>'",
        "",
        "  def handle_<subcmd>(args):",
        "      # handler body",
        "      ...",
        "",
        "  def main():",
        "      parser = argparse.ArgumentParser(description='...')",
        "      parser.add_argument('--verbose', action='store_true', help='Enable verbose output')",
        "      parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')",
        "      subparsers = parser.add_subparsers(dest='command', help='Available commands')",
        "      # ... add subcommands ...",
        "      args = parser.parse_args()",
        "      if args.command is None:",
        "          parser.print_help()",
        "          return 1",
        "      try:",
        "          # dispatch to handler",
        "          ...",
        "      except Exception as e:",
        "          if args.verbose:",
        "              import traceback; traceback.print_exc()",
        "          print(f'Error: {e}', file=sys.stderr)",
        "          return 1",
        "      return 0",
        "",
        "  if __name__ == '__main__':",
        "      sys.exit(main())",
        "",
        "The constructor must include:",
        "- name: valid Python identifier for the tool",
        "- version: semver string (e.g. '1.0.0')",
        "- description: tool-level help text",
        "- subcommands: tuple of Subcommand objects with name, help_text, arguments, handler_body",
        "- source: the full Python source code for the CLI tool",
    ])

    if ctx.available_packages:
        parts.extend(["", f"Available packages: {ctx.available_packages}"])

    if ctx.feedback:
        parts.extend(["", "Your previous attempt had errors:", ""])
        for fb in ctx.feedback:
            parts.append(f"  {fb}")
        parts.extend(["", "Please fix these issues in your next attempt."])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Validation pipeline -- V_cli (cheapest first)
# ---------------------------------------------------------------------------


def validate_source_syntax(spec: CliToolSpec) -> Result[None, str]:
    """V_cli layer 1 -- Semantic: ast.parse() the source.

    Pure, cheapest. Catches syntax errors before we try to exec.
    """
    try:
        ast.parse(spec.source)
    except SyntaxError as exc:
        loc = f" at line {exc.lineno}" if exc.lineno else ""
        return Err(f"SyntaxError in generated source{loc}: {exc.msg}")
    return Ok(None)


def validate_source_patterns(spec: CliToolSpec) -> Result[None, str]:
    """V_cli layer 2 -- Semantic: check source contains expected patterns.

    Verifies the source has argparse, subcommands, --verbose, --version.
    Still pure -- just string/AST inspection.
    """
    errors: list[str] = []
    source = spec.source

    if "argparse" not in source:
        errors.append("source must import argparse")

    if "add_subparsers" not in source:
        errors.append("source must use add_subparsers for subcommands")

    if "--verbose" not in source and "'verbose'" not in source and '"verbose"' not in source:
        errors.append("source must include --verbose flag")

    if "--version" not in source and "'version'" not in source and '"version"' not in source:
        errors.append("source must include --version flag")

    if "def main" not in source:
        errors.append("source must define a main() function")

    if "__name__" not in source:
        errors.append("source must have if __name__ == '__main__' guard")

    # Check each subcommand name appears in source
    for sc in spec.subcommands:
        if sc.name not in source:
            errors.append(f"subcommand '{sc.name}' not found in source")

    if errors:
        return Err("; ".join(errors))
    return Ok(None)


def validate_source_exec(spec: CliToolSpec) -> Result[None, str]:
    """V_cli layer 3 -- Executive: exec() the source in isolated namespace.

    Catches runtime errors: missing imports, name errors, logic bugs.
    We exec the module-level code but do NOT call main().
    """
    namespace: dict[str, Any] = {"__name__": "__test_module__", "__builtins__": __builtins__}
    try:
        code = compile(spec.source, f"<{spec.name}.py>", "exec")
        exec(code, namespace)
    except Exception as exc:
        return Err(f"Runtime error in generated source: {type(exc).__name__}: {exc}")

    # Verify main() exists and is callable
    if "main" not in namespace or not callable(namespace["main"]):
        return Err("Generated source must define a callable main() function")

    return Ok(None)


def validate_cli_tool(spec: CliToolSpec) -> Result[CliToolSpec, str]:
    """Full V_cli pipeline. Cheapest first:

    1. Semantic: ast.parse() the source
    2. Semantic: check expected patterns
    3. Executive: exec() the source

    Returns Ok(spec) on success (the spec IS the artifact).
    """
    # Layer 1: syntax
    match validate_source_syntax(spec):
        case Err(e):
            return Err(e)

    logger.info("V_cli layer 1: source parses cleanly")

    # Layer 2: patterns
    match validate_source_patterns(spec):
        case Err(e):
            return Err(e)

    logger.info("V_cli layer 2: expected patterns found")

    # Layer 3: exec
    match validate_source_exec(spec):
        case Err(e):
            return Err(e)

    logger.info("V_cli layer 3: source executes cleanly")

    return Ok(spec)


# ---------------------------------------------------------------------------
# Ouroboros -- G'_cli (narrow fix loop)
# ---------------------------------------------------------------------------


def ouroboros_cli(
    spec: CliToolSpec,
    error: str,
    ctx: GenerationContext,
    model_id: str = "",
    ask_fn: AskFn | None = None,
) -> CliToolSpec | None:
    """G'_cli: model sees its prior spec + error, returns corrected spec.

    The fix is narrow:
    - Model receives its prior spec and the specific error
    - Model returns a corrected CliToolSpec(...) expression
    - Typically only the 'source' field changes
    """
    fn = resolve_ask_fn(model_id, ask_fn)

    # Build a readable representation of the prior spec for context
    subcmd_reprs: list[str] = []
    for sc in spec.subcommands:
        arg_reprs = ", ".join(
            f'Argument(name={a.name!r}, help_text={a.help_text!r}, '
            f'required={a.required!r}, default={a.default!r}, arg_type={a.arg_type!r})'
            for a in sc.arguments
        )
        subcmd_reprs.append(
            f'        Subcommand(\n'
            f'            name={sc.name!r},\n'
            f'            help_text={sc.help_text!r},\n'
            f'            arguments=({arg_reprs}),\n'
            f'            handler_body={sc.handler_body!r},\n'
            f'        )'
        )
    spec_repr = (
        f'CliToolSpec(\n'
        f'    name={spec.name!r},\n'
        f'    version={spec.version!r},\n'
        f'    description={spec.description!r},\n'
        f'    subcommands=(\n'
        + ",\n".join(subcmd_reprs)
        + f'\n    ),\n'
        f'    source=...,  # full source shown below for reference\n'
        f')\n'
        f'\n'
        f'# Current source:\n'
        f'{spec.source}'
    )

    system_parts = [
        "You are fixing a CLI tool you previously generated.",
        f"Tool: {spec.name} v{spec.version} -- {spec.description}",
        "",
        "You will receive your prior spec and an error.",
        "Write a corrected CliToolSpec(...) expression.",
        "",
        "RULES:",
        "- Fix ONLY what is broken. Keep everything else identical.",
        "- The 'source' field must be syntactically valid Python (passes ast.parse())",
        "- Use f-strings, NOT string concatenation with +",
        "- Use triple-quoted strings for multi-line text",
        "- All imports must be from the standard library",
    ]

    user_parts = [
        "# Your prior spec",
        "",
        spec_repr[:8000],
        "",
        "# Error to fix",
        "",
        error[:4000],
        "",
        "Write a corrected CliToolSpec(...) expression. Fix only what is broken.",
    ]

    match fn("\n".join(system_parts), "\n".join(user_parts)):
        case Err(e):
            logger.warning("CLI ouroboros model error: %s", e)
            return None
        case Ok(raw_text):
            pass

    try:
        corrected = parse_typed_response(raw_text, CliToolSpec)
    except ValueError as e:
        logger.warning("CLI ouroboros parse error: %s", e)
        return None

    from compass.generators.cli_tool._types import validate_spec_instance
    match validate_spec_instance(corrected):
        case Err(e):
            logger.warning("CLI ouroboros spec validation: %s", e)
            return None
        case Ok(corrected):
            logger.info("CLI ouroboros produced corrected spec")
            return corrected


# ---------------------------------------------------------------------------
# Emit -- write CLI tool to disk
# ---------------------------------------------------------------------------


def emit_cli_tool(
    spec: CliToolSpec,
    artifact: CliToolSpec,
    rounds: int,
    fixes: int,
    version_hint: int,
    prompt: str | None,
    output_dir: Path | None = None,
) -> Result[Path, str]:
    """Write the CLI tool script and report to disk."""
    from compass.generators._types import GenerationReport

    if output_dir is None:
        output_dir = Path.cwd() / "generated_cli"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        script_path = output_dir / f"{spec.name}.py"
        script_path.write_text(spec.source)
        script_path.chmod(0o755)

        report = GenerationReport(
            rounds=rounds,
            ouroboros_fixes=fixes,
            outcome="success",
            user_prompt=prompt,
        )
        report_path = output_dir / f"{spec.name}.report.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2))

        logger.info("CLI tool written to %s", script_path)
        return Ok(script_path)
    except Exception as exc:
        return Err(f"Emit error: {exc}")


# ---------------------------------------------------------------------------
# Load -- read CLI tool from disk back into CliToolSpec
# ---------------------------------------------------------------------------


def load_cli_tool(path: Path) -> Result[CliToolSpec, str]:
    """Load a CLI tool script back into a CliToolSpec.

    Reconstructs the spec from the source file. Subcommand metadata
    is extracted from the AST where possible.
    """
    if not path.is_file():
        return Err(f"Not a file: {path}")

    source = path.read_text()
    name = path.stem

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return Err(f"SyntaxError in {path}: {exc}")

    # Extract version from __version__ assignment
    version = "0.0.0"
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__version__"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            version = node.value.value
            break

    # Extract module docstring
    description = ""
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        description = tree.body[0].value.value.strip()

    if not description:
        description = f"CLI tool: {name}"

    return Ok(CliToolSpec(
        name=name,
        version=version,
        description=description,
        subcommands=(),  # Cannot fully reconstruct from source alone
        source=source,
    ))


def serialize_spec(spec: CliToolSpec) -> str:
    """Serialize a CliToolSpec to a human-readable string."""
    parts = [
        f"# CLI Tool: {spec.name} v{spec.version}",
        f"# Description: {spec.description}",
        f"# Subcommands: {', '.join(sc.name for sc in spec.subcommands)}",
        "",
        "## Source",
        "",
        spec.source,
    ]
    return "\n".join(parts)
