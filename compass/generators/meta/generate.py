#!/usr/bin/env python3
"""Meta-generator -- generates generator modules.

Algebra:
  G_meta  : invoke_model     -- Context -> Result[raw_dict]
  V1      : validate_spec_instance -- GeneratorModuleSpec -> Result[GeneratorModuleSpec]
  V2      : validate_generated_module -- GeneratorModuleSpec -> Result[True]
            Composed cheapest-first:
              1. Semantic: ast.parse() every .py file
              2. Executive (light): materialize + import in tmpdir
              3. Executive (full): install + run inner generator
  G'_meta : ouroboros_meta   -- (spec, error, ctx) -> spec | None
            NARROW: returns ModulePatch, not full rewrite
            Now uses ctx for domain context in fix prompts
  IO      : emit_module      -- (spec, ...) -> Result[Path]

The generation_loop from compass.generators._loop handles the two-tier
iteration:
  - Outer loop (max_rounds): wholesale generation
    invoke -> parse -> inner loop
    If inner loop fails, scrap spec, keep errors as feedback
  - Inner loop (max_fixes): ouroboros repair
    validate -> fix -> re-validate
    fix() returns patched spec (surgical edits only)
    If fix() returns None, break to outer loop

Entry points:
  run(**kwargs) -> Result  -- programmatic, no sys.argv
  main() -> int            -- CLI wrapper

Usage:
    python -m compass.generators.meta.generate \\
        --prompt "Build a generator for Python CLI tools" \\
        --model-id anthropic:opus

    python -m compass.generators.meta.generate \\
        --prompt "Build a notebook generator" --dry-run

    # Refine existing generator
    python -m compass.generators.meta.generate \\
        --refine compass/generators/neo \\
        --claim "Use module invocation in _run_generator" \\
        --model-id anthropic:opus
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

from compass.generators.meta._types import GeneratorModuleSpec, validate_spec_instance
from compass.generators.meta._runtime import (
    emit_module,
    invoke_model,
    load_module,
    ouroboros_meta,
    serialize_spec,
    validate_generated_module,
)
from compass.generators.meta._context import build_meta_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The five functions for generation_loop
# ---------------------------------------------------------------------------


def _make_invoke(model_id: str, ask_fn=None):
    """G_meta: Context -> Result[GeneratorModuleSpec]

    Python-as-schema: model writes GeneratorModuleMeta constructor
    + ### banner ### file sections.
    """
    def invoke(ctx: GenerationContext) -> Result:
        return invoke_model(ctx, model_id, ask_fn)
    return invoke


def _make_parse():
    """V1: GeneratorModuleSpec -> Result[GeneratorModuleSpec]

    Instance validation: non-empty content, valid identifier,
    required files present. Lighter than dict validation.
    """
    def parse(spec: GeneratorModuleSpec) -> Result:
        return validate_spec_instance(spec)
    return parse


def _make_validate(model_id: str, ask_fn=None):
    """V2: GeneratorModuleSpec -> Result[True]

    Composed cheapest-first:
      1. ast.parse() every .py file (semantic, pure)
      2. materialize + import in tmpdir (executive, light)
      3. install + run inner generator (executive, full)
    """
    def validate(spec: GeneratorModuleSpec) -> Result:
        return validate_generated_module(spec, model_id, ask_fn)
    return validate


def _make_fix(model_id: str, ask_fn=None):
    """G'_meta: (spec, error, ctx) -> spec | None

    NARROW ouroboros: model returns a ModulePatch with surgical edits.
    Never rewrites the whole module. If the patch cannot be applied
    or the model fails, returns None (outer loop retries from scratch).

    ctx is passed through to ouroboros_meta so the fix model has access
    to domain context (shared framework, exemplar, principles) and
    available_packages. This enables better surgical patches because
    the model knows what APIs exist and what patterns to follow.
    """
    def fix(spec: GeneratorModuleSpec, error: str, ctx: GenerationContext):
        return ouroboros_meta(spec, error, ctx, model_id, ask_fn)
    return fix


def _make_emit(output_dir: Path):
    """IO: write module to output directory + report sidecar."""
    def emit(spec, artifact, rounds, fixes, _version_hint, prompt_or_claim) -> Result:
        try:
            pkg_dir = emit_module(spec, output_dir)
            report = GenerationReport(
                rounds=rounds,
                ouroboros_fixes=fixes,
                outcome="success",
                user_prompt=prompt_or_claim,
            )
            report_path = pkg_dir / ".report.json"
            report_path.write_text(json.dumps(report.to_dict(), indent=2))
            return Ok(pkg_dir)
        except Exception as exc:
            return Err(f"Emit error: {exc}")
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

    In-process callers (Neo, other generators) call this directly.

    Args:
        prompt: What generator to build
        model_id: Model spec (e.g. anthropic:opus, anthropic:sonnet)
        output: Output directory (default: compass/generators/)
        max_rounds: Outer loop budget (wholesale generation)
        max_fixes: Inner loop budget (ouroboros repair per round)
        verbose: Enable debug logging
        dry_run: Print prompts without calling the model
        refine: (artifact_path, claim) -- load existing generator,
                claim becomes the prompt
        ask_fn: Optional pre-built AskFn (for testing)
    """
    if not prompt and not refine:
        return Err("prompt or refine is required")

    out = (
        Path(output) if output
        else Path(__file__).resolve().parent.parent
    )

    # Build context -- includes full shared framework
    logger.info("Building meta-generation context...")
    ctx = build_meta_context(prompt)

    # Handle refinement: load existing generator into context
    if refine:
        artifact_path, claim = refine
        match refine_context(
            ctx, artifact_path, claim,
            load=load_module,
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
        from compass.generators.meta._runtime import _META_TYPE_SOURCE
        system = build_system_prompt(
            ctx,
            _META_TYPE_SOURCE,
            role=(
                "You are an expert at building code generation pipelines. "
                "You generate Python generator modules that follow a specific architecture."
            ),
            contract_preamble=(
                "Respond as shown in the GeneratorModuleMeta docstring."
            ),
        )
        from compass.generators.meta._runtime import _build_meta_user_message
        user = _build_meta_user_message(ctx)

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
        print()
        print(f"Domain sections: {len(ctx.domain_context)}")
        for s in ctx.domain_context:
            print(f"  {s.heading}: {len(s.content)} chars")
        return Ok(None)

    # Run the generation loop
    # Outer loop: invoke -> parse -> inner loop
    # Inner loop: validate -> fix -> re-validate
    return generation_loop(
        ctx,
        invoke=_make_invoke(model_id, ask_fn),
        parse=_make_parse(),
        validate=_make_validate(model_id, ask_fn),
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
        description="Meta-generator: generate generator modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m compass.generators.meta.generate \\
                --prompt "Build a generator for Python CLI tools"

              python -m compass.generators.meta.generate \\
                --refine compass/generators/neo \\
                --claim "Fix module invocation" \\
                --model-id anthropic:opus

              python -m compass.generators.meta.generate \\
                --prompt "Build a notebook generator" --dry-run
        """),
    )
    parser.add_argument("--prompt", help="What generator to build")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: compass/generators/)",
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-fixes", type=int, default=3)
    parser.add_argument(
        "--model-id", default="",
        help="Model spec (e.g. anthropic:opus, anthropic:sonnet)",
    )
    parser.add_argument("--refine", help="Path to existing generator directory to refine")
    parser.add_argument("--claim", help="Domain expert claim for refinement")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    refine = None
    if args.refine:
        refine = (args.refine, args.claim or "Re-validate this generator module")

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
                print(f"Generator module created: {path}")
        case Err(e):
            print(f"ERROR: {e}", file=sys.stderr)
    return result_to_exit(result)


if __name__ == "__main__":
    sys.exit(main())
