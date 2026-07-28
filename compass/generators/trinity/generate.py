#!/usr/bin/env python3
"""Trinity -- artifact orchestrator that produces structured facts.

Trinity takes a question or hypothesis, discovers available Python
programs in the workspace, plans artifact applications by inspecting
their signatures, executes them with inferred arguments, and collects
the results as structured facts.

Algebra:
    G_trinity   : Context -> Result[raw_dict]           (invoke_model)
    V1_trinity  : raw_dict -> Result[Spec]              (validate_spec)
    V2_trinity  : Spec -> Result[ExecutionResult]       (validate_plan)
    G'_trinity  : (Spec, error, ctx) -> Spec | None     (ouroboros_fix via SpecPatch)
    IO_trinity  : (Spec, result, ...) -> Result[Path]   (emit_result)

Validation composition (cheapest first):
    1. Structural: validate_spec -- pure dict -> Spec
    2. Semantic: validate_semantics -- ast.parse inline code
    3. Executive: execute_plan -- actually run steps, collect facts

Ouroboros patching:
    G'_trinity returns a SpecPatch (targeted step-level edits) instead of
    the full Spec. The patch is applied via apply_spec_patch(). Falls back
    to full Spec replacement if patching fails.

    ctx is passed through to ouroboros_fix so the fix model has access to
    domain context (discovered artifacts, plan principles) enabling better
    surgical patches.

Artifact dispatch is dynamic:
    - discover_artifacts() scans workspace for Python files
    - _inspect_python_file() extracts entry points and signatures
    - _map_inputs_to_params() maps step inputs to function parameters
    - No hard-coded artifact type list -- any Python file with run()/main()
      is automatically available

Usage:
    # One-shot
    python -m compass.generators.trinity.generate \\
        --prompt \"What is fibonacci(10)?\" \\
        --model-id anthropic:sonnet

    # Interactive REPL (multi-turn, session persistence, Ctrl+C pause)
    python -m compass.generators.trinity.generate --live

    # Dry run
    python -m compass.generators.trinity.generate \\
        --prompt \"Compare sorting algorithms\" --dry-run

    # Refine existing plan
    python -m compass.generators.trinity.generate \\
        --refine trinity_output \\
        --claim \"Add a step to verify the result\" \\
        --model-id anthropic:opus
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

from compass.generators._types import (
    Err,
    GenerationContext,
    Ok,
    Result,
)
from compass.generators._loop import generation_loop, refine_context, result_to_exit

from compass.generators.trinity._types import Spec, ExecutionResult, validate_spec, validate_spec_instance, serialize_spec
from compass.generators.trinity._runtime import (
    emit_result,
    invoke_model,
    load_result,
    ouroboros_fix,
    validate_plan,
)
from compass.generators.trinity._context import build_trinity_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The five functions for generation_loop
# ---------------------------------------------------------------------------


class ModelRef:
    """Mutable model reference so /model can switch mid-session.

    Closures capture this object, not the string. When the REPL
    changes model_id, the next generation uses the new model.
    """

    __slots__ = ("model_id",)

    def __init__(self, model_id: str = ""):
        self.model_id = model_id


def _make_invoke(model_ref: ModelRef, ask_fn=None, on_progress=None):
    """G_trinity: Context -> Result[raw_dict]"""
    def invoke(ctx: GenerationContext) -> Result:
        if on_progress:
            on_progress("invoke_start")
        result = invoke_model(ctx, model_ref.model_id, ask_fn)
        if on_progress:
            match result:
                case Ok(spec):
                    on_progress("invoke_done", spec=spec)
                case Err(e):
                    on_progress("invoke_error", error=str(e))
        return result
    return invoke


def _make_parse(on_progress=None):
    """V1_trinity: Spec -> Result[Spec] (instance validation after parse_typed_response)"""
    def parse(spec: Spec) -> Result:
        result = validate_spec_instance(spec)
        if on_progress:
            match result:
                case Ok(s):
                    on_progress("parse_done", spec=s)
                case Err(e):
                    on_progress("parse_error", error=str(e))
        return result
    return parse


def _make_validate(model_ref: ModelRef, ask_fn=None, workspace: Path | None = None, on_progress=None):
    """V2_trinity: Spec -> Result[ExecutionResult]

    Composed cheapest-first:
      1. Semantic: inline Python syntax, artifact refs
      2. Executive: run the plan, collect facts
    """
    def validate(spec: Spec) -> Result:
        return validate_plan(spec, model_ref.model_id, ask_fn, workspace, on_step=on_progress)
    return validate


def _make_fix(model_ref: ModelRef, ask_fn=None, on_progress=None):
    """G'_trinity: (Spec, error, ctx) -> Spec | None

    Uses SpecPatch for targeted step-level edits.
    Falls back to full Spec replacement if patching fails.
    ctx is passed through so the fix model has domain context.
    """
    def fix(spec: Spec, error: str, ctx: GenerationContext):
        if on_progress:
            on_progress("fix_start", error=error)
        result = ouroboros_fix(spec, error, ctx, model_ref.model_id, ask_fn)
        if on_progress:
            on_progress("fix_done", fixed=result is not None)
        return result
    return fix


def _make_emit(output_dir: Path | None = None):
    """IO_trinity: write plan + results to disk."""
    def emit(spec, artifact, rounds, fixes, version, prompt) -> Result:
        return emit_result(
            spec, artifact, rounds, fixes, version, prompt,
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
    workspace: str | Path | None = None,
    refine: tuple[str, str] | None = None,
    ask_fn=None,
    live: bool = False,
    history: bool = False,
) -> Result:
    """Programmatic entry point. No argparse, no sys.argv.

    In-process callers (Neo, other orchestrators) call this directly.

    Args:
        prompt: The question or hypothesis to investigate.
        model_id: Model spec (e.g. 'anthropic:sonnet').
        output: Output directory for results.
        max_rounds: Maximum wholesale generation rounds.
        max_fixes: Maximum ouroboros fix attempts per round.
        verbose: Enable debug logging.
        dry_run: Print prompts without executing.
        workspace: Directory to search for artifacts.
        refine: (artifact_path, claim) to refine an existing plan.
        ask_fn: Optional pre-built ask function (for testing).
        live: Interactive REPL mode with session persistence.
    """
    if not live and not prompt and not refine:
        return Err("prompt, live, or refine is required")

    out = Path(output) if output else None
    ws = Path(workspace) if workspace else None

    # Build context
    logger.info("Building Trinity context...")
    ctx = build_trinity_context(prompt, ws)

    # Handle refinement: load existing plan into context
    if refine:
        artifact_path, claim = refine
        match refine_context(
            ctx, artifact_path, claim,
            load=load_result,
            serialize=serialize_spec,
        ):
            case Err() as e:
                return e
            case Ok(resolved):
                ctx = resolved
    elif prompt:
        ctx = ctx.with_prompt(prompt)

    if dry_run:
        from compass.generators._invoke import build_system_prompt
        from compass.generators.trinity._runtime import _SPEC_TYPE_SOURCE, _build_user_message
        system = build_system_prompt(
            ctx,
            _SPEC_TYPE_SOURCE,
            contract_preamble="Respond as shown in the Spec docstring.",
        )
        user = _build_user_message(ctx)

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

    # Interactive REPL
    if live:
        from compass.generators.trinity._repl import trinity_repl, make_progress_callback
        model_ref = ModelRef(model_id)
        on_progress = make_progress_callback(workspace=ws)
        return trinity_repl(
            ctx,
            invoke=_make_invoke(model_ref, ask_fn, on_progress),
            parse=_make_parse(on_progress),
            validate=_make_validate(model_ref, ask_fn, ws, on_progress),
            fix=_make_fix(model_ref, ask_fn, on_progress),
            emit=_make_emit(out),
            max_rounds=max_rounds,
            max_fixes=max_fixes,
            model_ref=model_ref,
            history=history,
        )

    # One-shot generation
    model_ref = ModelRef(model_id)
    return generation_loop(
        ctx,
        invoke=_make_invoke(model_ref, ask_fn),
        parse=_make_parse(),
        validate=_make_validate(model_ref, ask_fn, ws),
        fix=_make_fix(model_ref, ask_fn),
        emit=_make_emit(out),
        max_rounds=max_rounds,
        max_fixes=max_fixes,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Trinity: answer questions by orchestrating artifact execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m compass.generators.trinity.generate \\
                --prompt \"What is fibonacci(10)?\" \\
                --model-id anthropic:sonnet

              python -m compass.generators.trinity.generate --live

              python -m compass.generators.trinity.generate \\
                --refine trinity_output \\
                --claim \"Add verification step\" \\
                --model-id anthropic:opus
        """),
    )
    parser.add_argument("--prompt", help="Question or hypothesis to investigate")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: trinity_output/)",
    )
    parser.add_argument("--workspace", default=None, help="Workspace to search for artifacts")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-fixes", type=int, default=3)
    parser.add_argument(
        "--model-id", default="",
        help="Model spec (e.g. anthropic:sonnet, anthropic:opus)",
    )
    parser.add_argument("--refine", help="Path to existing Trinity result to refine")
    parser.add_argument("--claim", help="Claim for refinement")
    parser.add_argument("--live", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--history", action="store_true", help="Include full previous answer in context")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    refine = (
        (args.refine, args.claim or "Re-validate this plan")
        if args.refine else None
    )

    result = run(
        prompt=args.prompt,
        model_id=args.model_id,
        output=args.output_dir,
        workspace=args.workspace,
        max_rounds=args.max_rounds,
        max_fixes=args.max_fixes,
        verbose=args.verbose,
        dry_run=args.dry_run,
        refine=refine,
        ask_fn=None,
        live=args.live,
        history=args.history,
    )
    match result:
        case Ok(path):
            if path is not None:
                print(f"Trinity result written to: {path}")
        case Err(e):
            print(f"ERROR: {e}", file=sys.stderr)
    return result_to_exit(result)


if __name__ == "__main__":
    sys.exit(main())
