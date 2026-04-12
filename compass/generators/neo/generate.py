#!/usr/bin/env python3
"""
Neo -- plan generator. Wires five functions into generation_loop.

G  : Prompt + DomainContext -> PlanSpec
V  : PlanSpec -> Ok | Err      (structural -> semantic -> executive)
G' : (PlanSpec, Error) -> PlanSpec

Usage:
    python -m compass.generators.neo.generate --prompt "build a fibonacci module with tests"
    python -m compass.generators.neo.generate --prompt "..." --dry-run
    python -m compass.generators.neo.generate --prompt "..." --verbose
"""

from __future__ import annotations

CYCLE_BREAKING = True  # Trinity halts and re-plans after calling this generator

import argparse
import json
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
from compass.generators._loop import generation_loop, result_to_exit
from compass.generators._invoke import build_system_prompt, build_user_message

from compass.generators.neo._types import (
    ExecutedPlan,
    PlanConfig,
    PlanSpec,
    validate_artifact_types,
    validate_spec,
    validate_spec_instance,
    versioned_path,
)
from compass.generators.neo._runtime import (
    emit_plan,
    execute_plan,
    invoke_model,
    try_ouroboros,
)
from compass.generators.neo._context import build_plan_context

logger = logging.getLogger(__name__)

_VERSION_SUFFIX_RE = re.compile(r"_v(\d+)$")


# ---------------------------------------------------------------------------
# Versioned output
# ---------------------------------------------------------------------------


def _next_version(base: Path) -> int:
    """Scan output directory for existing _vN files and return N+1."""
    stem = base.stem
    m = _VERSION_SUFFIX_RE.search(stem)
    if m:
        stem = stem[:m.start()]

    if not base.parent.exists():
        return 1

    max_v = -1
    if base.exists():
        max_v = 0

    for p in base.parent.iterdir():
        if p.suffix != base.suffix:
            continue
        vm = _VERSION_SUFFIX_RE.search(p.stem)
        if vm and p.stem[:vm.start()] == stem:
            max_v = max(max_v, int(vm.group(1)))

    return max_v + 1


# ---------------------------------------------------------------------------
# The five functions for generation_loop
# ---------------------------------------------------------------------------


def _make_invoke(config: PlanConfig):
    """G: Context -> Result[raw_dict]"""
    def invoke(ctx: GenerationContext) -> Result:
        return invoke_model(ctx, config)
    return invoke


def _make_parse():
    """V1: PlanSpec -> Result[PlanSpec] (structural validation)"""
    def parse(spec: PlanSpec) -> Result:
        return validate_spec_instance(spec)
    return parse


def _make_validate(config: PlanConfig):
    """V2: PlanSpec -> Result[ExecutedPlan]

    Composed cheapest-first: semantic (type check) -> executive (run steps).
    """
    def validate(spec: PlanSpec) -> Result:
        # Semantic: check artifact types (warn only)
        match validate_artifact_types(spec):
            case Ok(warnings):
                for w in warnings:
                    logger.warning("Semantic: %s", w)
            case Err(e):
                return Err(f"Semantic validation error: {e}")

        logger.info("Semantic validation passed (%d steps)", len(spec.steps))

        # Executive: actually run the plan
        match execute_plan(spec, config):
            case Err(e):
                return Err(f"Execution error: {e}")
            case Ok(executed):
                pass

        logger.info("All steps executed successfully")
        return Ok(executed)

    return validate


def _make_fix(config: PlanConfig):
    """G': (PlanSpec, error, ctx) -> PlanSpec | None"""
    def fix(spec: PlanSpec, error: str, ctx: GenerationContext) -> PlanSpec | None:
        return try_ouroboros(spec, error, ctx, config)
    return fix


def _make_emit(config: PlanConfig):
    """IO: (spec, artifact, rounds, fixes, version_hint, prompt_or_claim) -> Result[Path]"""
    def emit(spec, artifact, rounds, fixes, _version_hint, prompt_or_claim) -> Result:
        version = _next_version(config.output_path)
        out_path = versioned_path(config.output_path, version)
        report = GenerationReport(
            version=version,
            rounds=rounds,
            ouroboros_fixes=fixes,
            outcome="success",
            validators=(
                "validate_spec",
                "validate_artifact_types",
                "execute_plan",
            ),
            user_prompt=prompt_or_claim,
        )
        path = emit_plan(artifact, out_path, report=report)
        logger.info("Plan written to %s (v%d)", path, version)
        return Ok(path)

    return emit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(step, resolved_inputs: dict, workspace) -> Result:
    """Trinity artifact entry point: run(step, resolved_inputs, workspace).

    Bridges Trinity's step dispatch contract to Neo's plan generator.
    Extracts prompt from resolved_inputs or step, delegates to generate().
    """
    from compass.generators.trinity._types import Fact

    prompt = resolved_inputs.get("prompt", "") or getattr(step, "artifact_ref", "") or ""
    if not prompt:
        return Err("Neo requires a 'prompt' input")

    model_id = resolved_inputs.get("model_id", "")
    output = resolved_inputs.get("output", None)

    result = generate(
        prompt,
        model_id=model_id,
        output=output,
    )

    match result:
        case Ok(path):
            plan_content = None
            if path:
                try:
                    plan_content = json.loads(Path(str(path)).read_text())
                except Exception:
                    pass
            value = {"path": str(path) if path else None}
            if plan_content:
                value["plan"] = plan_content
            return Ok(Fact(
                step_id=getattr(step, "step_id", "neo"),
                name=getattr(step, "expected_fact", "neo_plan"),
                value=json.dumps(value),
                fact_type="json",
            ))
        case Err(e):
            return Err(f"Neo generation failed: {e}")


def generate(
    prompt: str,
    *,
    model_id: str = "",
    output: str | None = None,
    max_rounds: int = 3,
    max_fixes: int = 3,
    verbose: bool = False,
    dry_run: bool = False,
) -> Result:
    """Programmatic entry point. No argparse, no sys.argv.

    In-process callers call this directly and get typed Result back.
    """
    output_path = Path(output) if output else Path("plans/neo_plan.json")

    config = PlanConfig(
        output_path=output_path,
        max_rounds=max_rounds,
        max_fixes=max_fixes,
        model_id=model_id,
        verbose=verbose,
        dry_run=dry_run,
        prompt=prompt,
    )

    # Build context
    logger.info("Building plan generation context...")
    ctx = build_plan_context(prompt)

    # Dry run
    if config.dry_run:
        _TYPES_MODULE = (Path(__file__).parent / "_types.py").read_text()
        system = build_system_prompt(
            ctx, _TYPES_MODULE,
            role="You are Neo, an expert plan generator.",
        )
        user = build_user_message(ctx)

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

    # One-shot generation
    prompt_ctx = ctx.with_prompt(prompt)

    return generation_loop(
        prompt_ctx,
        invoke=_make_invoke(config),
        parse=_make_parse(),
        validate=_make_validate(config),
        fix=_make_fix(config),
        emit=_make_emit(config),
        max_rounds=config.max_rounds,
        max_fixes=config.max_fixes,
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Neo -- plan generator (Compass)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m compass.generators.neo.generate \\
                --prompt "build a fibonacci module with tests"

              python -m compass.generators.neo.generate \\
                --prompt "fibonacci" --dry-run
        """),
    )
    parser.add_argument("--prompt", required=True, help="Generation prompt.")
    parser.add_argument("--output", default=None, help="Output plan path")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-fixes", type=int, default=3)
    parser.add_argument(
        "--model-id", default="",
        help="Model spec (e.g. qwen3-coder:latest@local)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompt, don't call model")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    result = generate(
        prompt=args.prompt,
        model_id=args.model_id,
        output=args.output,
        max_rounds=args.max_rounds,
        max_fixes=args.max_fixes,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    match result:
        case Ok(path):
            if path is not None:
                print(f"Plan generated: {path}")
        case Err(e):
            print(f"ERROR: {e}", file=sys.stderr)
    return result_to_exit(result)


if __name__ == "__main__":
    sys.exit(main())
