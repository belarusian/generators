#!/usr/bin/env python3
"""CLI tool generator -- generates argparse-based command-line applications.

Algebra:
  G_cli   : invoke_model         -- Context -> Result[raw_dict]
  V1      : validate_spec        -- raw_dict -> Result[CliToolSpec]
  V2      : validate_cli_tool    -- CliToolSpec -> Result[CliToolSpec]
            Composed cheapest-first:
              1. Semantic: ast.parse(source)
              2. Semantic: check expected patterns (argparse, subcommands, flags)
              3. Executive: exec() the source in isolated namespace
  G'_cli  : ouroboros_cli        -- (spec, error, ctx) -> spec | None
            NARROW: returns corrected CliToolSpec, only source changes
  IO      : emit_cli_tool        -- (spec, ...) -> Result[Path]

Entry points:
  run(**kwargs) -> Result  -- programmatic, no sys.argv
  main() -> int            -- CLI wrapper

Usage:
    python -m compass.generators.cli_tool.generate \\
        --prompt "Build a file utility CLI with count, search, info subcommands"

    python -m compass.generators.cli_tool.generate \\
        --prompt "Build a git helper CLI" --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path

from compass.generators._types import (
    Err,
    GenerationContext,
    GenerationReport,
    Ok,
    Result,
)
from compass.generators._loop import generation_loop, refine_context, result_to_exit

from compass.generators.cli_tool._types import CliToolSpec, validate_spec, validate_spec_instance
from compass.generators.cli_tool._runtime import (
    emit_cli_tool,
    invoke_model,
    load_cli_tool,
    ouroboros_cli,
    serialize_spec,
    validate_cli_tool,
)
from compass.generators.cli_tool._context import build_cli_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The five functions for generation_loop
# ---------------------------------------------------------------------------


def _make_invoke(model_id: str, ask_fn=None):
    """G_cli: Context -> Result[raw_dict]"""
    def invoke(ctx: GenerationContext) -> Result:
        return invoke_model(ctx, model_id, ask_fn)
    return invoke


def _make_parse():
    """V1: CliToolSpec -> Result[CliToolSpec] (structural validation)"""
    def parse(spec: CliToolSpec) -> Result:
        return validate_spec_instance(spec)
    return parse


def _make_validate():
    """V2: CliToolSpec -> Result[CliToolSpec]

    Composed cheapest-first:
      1. ast.parse(source)
      2. pattern checks
      3. exec() in isolated namespace
    """
    def validate(spec: CliToolSpec) -> Result:
        return validate_cli_tool(spec)
    return validate


def _make_fix(model_id: str, ask_fn=None):
    """G'_cli: (spec, error, ctx) -> spec | None

    NARROW ouroboros: model returns corrected CliToolSpec.
    Typically only the source field changes.
    """
    def fix(spec: CliToolSpec, error: str, ctx: GenerationContext):
        return ouroboros_cli(spec, error, ctx, model_id, ask_fn)
    return fix


def _make_emit(output_dir: Path):
    """IO: write CLI tool script + report to disk."""
    def emit(spec, artifact, rounds, fixes, version_hint, prompt) -> Result:
        return emit_cli_tool(
            spec, artifact, rounds, fixes, version_hint, prompt,
            output_dir=output_dir,
        )
    return emit


# ---------------------------------------------------------------------------
# Programmatic entry point
# ---------------------------------------------------------------------------


def run(
    prompt: str | None = None,
    *,
    model_id: str = "",
    output: str | Path | None = None,
    max_rounds: int = 3,
    max_fixes: int = 3,
    verbose: bool = False,
    dry_run: bool = False,
    refine: tuple[str, str] | None = None,
    ask_fn=None,
) -> Result:
    """Programmatic entry point. No argparse, no sys.argv.

    Args:
        prompt: What CLI tool to build
        model_id: Model spec (e.g. anthropic:sonnet)
        output: Output directory (default: ./generated_cli/)
        max_rounds: Outer loop budget
        max_fixes: Inner loop budget per round
        verbose: Enable debug logging
        dry_run: Print prompts without calling the model
        refine: (artifact_path, claim) -- load existing tool, claim becomes prompt
        ask_fn: Optional pre-built AskFn (for testing)
    """
    if not prompt and not refine:
        return Err("prompt or refine is required")

    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )

    out = Path(output) if output else Path.cwd() / "generated_cli"

    # Build context with CLI domain knowledge
    ctx = build_cli_context(prompt)

    # Handle refinement: load existing tool into context
    if refine:
        artifact_path, claim = refine
        match refine_context(
            ctx, artifact_path, claim,
            load=load_cli_tool,
            serialize=serialize_spec,
        ):
            case Err() as e:
                return e
            case Ok(resolved):
                ctx = resolved
    else:
        ctx = ctx.with_prompt(prompt)

    if dry_run:
        from compass.generators._invoke import build_system_prompt
        types_source = (Path(__file__).parent / "_types.py").read_text()
        system = build_system_prompt(ctx, types_source)
        from compass.generators.cli_tool._runtime import _build_cli_user_message
        user = _build_cli_user_message(ctx)

        print("=" * 80)
        print("SYSTEM PROMPT")
        print("=" * 80)
        print(system)
        print()
        print("=" * 80)
        print("USER MESSAGE")
        print("=" * 80)
        print(user)
        print("=" * 80)
        return Ok(None)

    # Run the generation loop
    return generation_loop(
        ctx,
        invoke=_make_invoke(model_id, ask_fn),
        parse=_make_parse(),
        validate=_make_validate(),
        fix=_make_fix(model_id, ask_fn),
        emit=_make_emit(out),
        max_rounds=max_rounds,
        max_fixes=max_fixes,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point. Parses args, calls run(), returns exit code."""
    parser = argparse.ArgumentParser(
        description="CLI tool generator: generate argparse-based command-line applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m compass.generators.cli_tool.generate \\
                --prompt "Build a file utility CLI with count and search subcommands"

              python -m compass.generators.cli_tool.generate \\
                --refine generated_cli/fileutils.py \\
                --claim "Add a 'rename' subcommand"

              python -m compass.generators.cli_tool.generate \\
                --prompt "Build a git helper" --dry-run
        """),
    )
    parser.add_argument("--prompt", help="What CLI tool to build")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: ./generated_cli/)",
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-fixes", type=int, default=3)
    parser.add_argument(
        "--model-id", default="",
        help="Model spec (e.g. anthropic:sonnet)",
    )
    parser.add_argument("--refine", help="Path to existing CLI tool script to refine")
    parser.add_argument("--claim", help="Domain expert claim for refinement")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )

    refine = None
    if args.refine:
        refine = (args.refine, args.claim or "Improve this CLI tool")

    result = run(
        prompt=args.prompt,
        model_id=args.model_id,
        output=args.output_dir,
        max_rounds=args.max_rounds,
        max_fixes=args.max_fixes,
        verbose=args.verbose,
        dry_run=args.dry_run,
        refine=refine,
    )
    match result:
        case Ok(path):
            if path is not None:
                print(f"CLI tool generated: {path}")
        case Err(e):
            print(f"ERROR: {e}", file=sys.stderr)
    return result_to_exit(result)


if __name__ == "__main__":
    sys.exit(main())
