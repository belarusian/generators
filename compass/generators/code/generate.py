#!/usr/bin/env python3
"""Code generator -- wires five functions into generation_loop.

G  : Prompt + DomainContext -> CodeSpec
V  : CodeSpec -> Ok | Err      (structural -> AST -> exec)
G' : (CodeSpec, Error) -> CodePatch -> CodeSpec

Usage:
    # One-shot from prompt
    python -m compass.generators.code \\
        --prompt "Create a stack data structure with push, pop, peek"

    # Interactive REPL
    python -m compass.generators.code --live

    # Dry run (print prompt, don't call model)
    python -m compass.generators.code \\
        --prompt "fibonacci" --dry-run

    # Refine existing artifact
    python -m compass.generators.code \\
        --refine generated_code/stack_v1 \\
        --claim "Add a size() method" \\
        --model-id anthropic:sonnet
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

from compass.generators._types import (
    Err,
    GenerationContext,
    GenerationReport,
    Ok,
    Result,
)
from compass.generators._loop import generation_loop, repl_loop, refine_context, result_to_exit
from compass.generators._invoke import build_system_prompt, build_user_message

from compass.generators.code._types import (
    CodeConfig,
    CodeSpec,
    ExecutedCode,
    validate_python_files,
    validate_spec,
    validate_spec_instance,
    validate_test_syntax,
    versioned_dir,
    summarize_file_contents,
)
from compass.generators.code._runtime import (
    emit_code,
    execute_code,
    invoke_model,
    load_code_artifact,
    try_ouroboros,
)
from compass.generators.code._context import build_code_context, build_generic_context

logger = logging.getLogger(__name__)

_VERSION_SUFFIX_RE = re.compile(r"_v(\d+)$")


# ---------------------------------------------------------------------------
# Context dispatch
# ---------------------------------------------------------------------------


def build_context(
    prompt: str | None = None,
    domain: str | None = None,
) -> GenerationContext:
    """Build generation context. With --domain, includes domain knowledge."""
    if domain == "generic":
        return build_generic_context(prompt)
    # Default: include software engineering best practices
    return build_code_context(prompt)


# ---------------------------------------------------------------------------
# Versioned output
# ---------------------------------------------------------------------------


def _next_version(base: Path) -> int:
    """Scan output directory for existing _vN directories and return N+1."""
    name = base.name
    m = _VERSION_SUFFIX_RE.search(name)
    if m:
        name = name[:m.start()]

    if not base.parent.exists():
        return 1

    max_v = -1
    if base.exists():
        max_v = 0

    for p in base.parent.iterdir():
        if not p.is_dir():
            continue
        vm = _VERSION_SUFFIX_RE.search(p.name)
        if vm and p.name[:vm.start()] == name:
            max_v = max(max_v, int(vm.group(1)))

    return max_v + 1


# ---------------------------------------------------------------------------
# The five functions for generation_loop
# ---------------------------------------------------------------------------


def _make_invoke(config: CodeConfig):
    """G: Context -> Result[raw_dict]"""
    def invoke(ctx: GenerationContext) -> Result:
        return invoke_model(ctx, config)
    return invoke


def _make_parse():
    """V1: CodeSpec -> Result[CodeSpec]

    Instance validation: non-empty content, no duplicate paths,
    entry_point references a real file.
    """
    def parse(spec: CodeSpec) -> Result:
        return validate_spec_instance(spec)
    return parse


def _make_validate():
    """V2: CodeSpec -> Result[ExecutedCode]

    Composed cheapest-first: semantic (AST) -> executive (exec).
    """
    def validate(spec: CodeSpec) -> Result:
        # Stage 1: Python syntax validation (AST)
        match validate_python_files(spec):
            case Err(e):
                summary = summarize_file_contents(spec)
                return Err(f"Python syntax error: {e}\n\n{summary}")
            case Ok(_):
                pass

        logger.info("Python file syntax validation passed")

        # Stage 2: Test syntax validation (AST)
        match validate_test_syntax(spec):
            case Err(e):
                return Err(f"Test syntax error: {e}")
            case Ok(_):
                pass

        logger.info("Test syntax validation passed")

        # Stage 3: Execute code and run tests
        match execute_code(spec):
            case Err(error):
                return Err(f"Execution error: {error}")
            case Ok(executed):
                pass

        logger.info(
            "All files executed and %d test(s) passed",
            len(executed.test_results),
        )
        return Ok(executed)

    return validate


def _make_fix(config: CodeConfig):
    """G': (CodeSpec, error, ctx) -> CodeSpec | None"""
    def fix(spec: CodeSpec, error: str, ctx: GenerationContext) -> CodeSpec | None:
        return try_ouroboros(spec, error, ctx, config)
    return fix


def _make_emit(config: CodeConfig):
    """IO: (spec, artifact, rounds, fixes, version_hint, prompt_or_claim) -> Result[Path]"""
    def emit(spec, artifact, rounds, fixes, _version_hint, prompt_or_claim) -> Result:
        version = _next_version(config.output_dir)
        out_dir = versioned_dir(config.output_dir, version)
        report = GenerationReport(
            version=version,
            rounds=rounds,
            ouroboros_fixes=fixes,
            outcome="success",
            validators=(
                "validate_spec",
                "validate_python_files",
                "validate_test_syntax",
                "execute_code",
            ),
            user_prompt=prompt_or_claim,
        )
        path = emit_code(artifact, out_dir, report=report)
        logger.info("Code artifact written to %s (v%d)", path, version)
        return Ok(path)

    return emit


# ---------------------------------------------------------------------------
# CLI
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
    focus: str | None = None,
    live: bool = False,
    domain: str | None = None,
    refine: tuple[str, str] | None = None,
    ask_fn=None,
) -> Result:
    """Programmatic entry point. No argparse, no sys.argv.

    In-process callers (Neo, meta-generator) call this directly.

    refine: (artifact_path, claim) -- load existing artifact into context,
            claim becomes the prompt. Same generation loop, richer context.
    """
    if not live and not prompt and not refine:
        return Err("prompt, live, or refine is required")

    output_dir = Path(output) if output else Path("generated_code")

    config = CodeConfig(
        output_dir=output_dir,
        max_rounds=max_rounds,
        max_fixes=max_fixes,
        model_id=model_id,
        verbose=verbose,
        dry_run=dry_run,
        focus=focus,
        ask_fn=ask_fn,
        live=live,
        prompt=prompt,
    )

    # Build context (pure dispatch)
    logger.info("Building generation context...")
    ctx = build_context(prompt, domain)

    # Refinement: load existing artifact onto context
    if refine:
        artifact_path, claim = refine
        match refine_context(
            ctx, artifact_path, claim,
            load=load_code_artifact,
            serialize=lambda spec: "\n\n".join(
                f"# {f.path}\n{f.content}" for f in spec.files
            ),
        ):
            case Err() as e:
                return e
            case Ok(resolved):
                ctx = resolved

    # Interactive REPL
    if config.live:
        return repl_loop(
            ctx,
            invoke=_make_invoke(config),
            parse=_make_parse(),
            validate=_make_validate(),
            fix=_make_fix(config),
            emit=_make_emit(config),
            max_rounds=config.max_rounds,
            max_fixes=config.max_fixes,
            load_artifact=load_code_artifact,
            banner="Code generator -- interactive mode",
        )

    # One-shot generation
    prompt_ctx = ctx.with_prompt(prompt) if prompt else ctx

    if config.dry_run:
        from compass.generators.code._runtime import _SPEC_TYPE_SOURCE, _build_code_user_message
        system = build_system_prompt(
            prompt_ctx, _SPEC_TYPE_SOURCE,
            role="You are an expert software engineer who generates clean, well-structured Python code.",
            contract_preamble="Respond as shown in the CodeSpec docstring.",
        )
        user = _build_code_user_message(prompt_ctx, config)

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

    return generation_loop(
        prompt_ctx,
        invoke=_make_invoke(config),
        parse=_make_parse(),
        validate=_make_validate(),
        fix=_make_fix(config),
        emit=_make_emit(config),
        max_rounds=config.max_rounds,
        max_fixes=config.max_fixes,
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate code artifacts using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m compass.generators.code.generate \\
                --prompt "Create a linked list with insert, delete, search"

              python -m compass.generators.code.generate --live

              python -m compass.generators.code.generate \\
                --prompt "fibonacci" --dry-run

              python -m compass.generators.code.generate \\
                --refine generated_code/stack_v1 \\
                --claim "Add a size() method"
        """),
    )
    parser.add_argument("--prompt", help="One-shot generation from a prompt.")
    parser.add_argument("--output", default=None, help="Output directory path")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-fixes", type=int, default=3)
    parser.add_argument(
        "--model-id", default="",
        help="Model spec (e.g. qwen3-coder:latest@local, anthropic:sonnet)",
    )
    parser.add_argument("--focus", help="Focus area for generation")
    parser.add_argument("--live", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt, don't call model")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--domain", default=None,
        help="Domain context ('generic' for no domain knowledge)",
    )
    parser.add_argument("--refine", help="Path to existing code artifact directory to refine")
    parser.add_argument("--claim", help="Domain expert claim for refinement")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    refine = (
        (args.refine, args.claim or "Re-validate this artifact")
        if args.refine
        else None
    )

    result = run(
        prompt=args.prompt,
        model_id=args.model_id,
        output=args.output,
        max_rounds=args.max_rounds,
        max_fixes=args.max_fixes,
        verbose=args.verbose,
        dry_run=args.dry_run,
        focus=args.focus,
        live=args.live,
        domain=args.domain,
        refine=refine,
    )
    match result:
        case Ok(path):
            if path is not None:
                print(f"Code artifact generated: {path}")
        case Err(e):
            print(f"ERROR: {e}", file=sys.stderr)
    return result_to_exit(result)


if __name__ == "__main__":
    sys.exit(main())
