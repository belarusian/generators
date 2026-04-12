"""
Parameterized generation loop.

The loop itself is invariant -- it does not know what artifact it is
generating. The five function arguments define the generator:

    invoke  : G   -- Context -> Result[raw]
    parse   : V1  -- raw -> Result[Spec]
    validate: V2  -- Spec -> Result[Artifact]
    fix     : G'  -- (Spec, error) -> Spec | None
    emit    : IO  -- (Spec, Artifact) -> Result[Path]

G -> V -> G', repeat until V(artifact) = Ok or budget exhausted.

Every generator has two entry points:

    run(**kwargs) -> Result[Path, str]
        Programmatic. No argparse, no sys.argv. In-process callers
        (Neo, meta-generator) call this directly and get typed results.

    main() -> int
        CLI. Parses args, calls run(), prints the outcome, and returns
        result_to_exit(result). Shell entry:
            if __name__ == "__main__":
                sys.exit(main())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from ._types import Cycle, DomainSection, Err, GenerationContext, Ok, Result

logger = logging.getLogger(__name__)


def result_to_exit(result: Result) -> int:
    """Boundary adapter: Result -> exit code.

    0 for Ok, 1 for Err. Used by main() in every generator.
    """
    return 0 if isinstance(result, Ok) else 1


# Type aliases for the five functions.
# Kept as plain Callable -- the loop is parametric over Spec and Artifact.
InvokeFn = Callable[[GenerationContext], Result]
ParseFn = Callable[[Any], Result]
ValidateFn = Callable[[Any], Result]
FixFn = Callable[[Any, str, GenerationContext], Optional[Any]]
EmitFn = Callable[[Any, Any, int, int, int, Optional[str]], Result]


LoadFn = Callable[[Path], Result]
SerializeFn = Callable[[Any], str]


def refine_context(
    ctx: GenerationContext,
    artifact_path: str,
    claim: str,
    *,
    load: LoadFn,
    serialize: SerializeFn,
) -> Result:
    """Load an existing artifact onto context for refinement.

    Returns Ok(ctx) with artifact as domain section and claim as prompt,
    or Err if load fails.
    """
    match load(Path(artifact_path)):
        case Err() as e:
            return e
        case Ok(spec):
            pass
    from ._types import DomainSection
    ctx = ctx.with_domain(DomainSection(
        heading="Existing Artifact (modify according to claim)",
        content=serialize(spec),
    ))
    return Ok(ctx.with_prompt(claim))


def generation_loop(
    ctx: GenerationContext,
    *,
    invoke: InvokeFn,
    parse: ParseFn,
    validate: ValidateFn,
    fix: FixFn,
    emit: EmitFn,
    max_rounds: int = 3,
    max_fixes: int = 3,
) -> Result[Path, str]:
    """Two-tier generation loop: wholesale rounds x ouroboros fixes.

    Outer loop (max_rounds): wholesale generation. The model produces
    a new spec from context. If ouroboros cannot fix it, the spec is
    discarded and the errors are preserved in context -- the learnings
    travel forward even when the page is scrapped.

    Inner loop (max_fixes): ouroboros. The model receives its own
    prior output + the error and returns a corrected version. Same type
    in, same type out. The spec is never discarded within this loop.

    Total budget: max_rounds x max_fixes iterations.
    """
    ouroboros_fixes = 0
    last_error = ""

    for round_num in range(max_rounds):
        logger.info("=== Round %d/%d ===", round_num + 1, max_rounds)

        # -- Wholesale: invoke model and parse into Spec -------------------

        match invoke(ctx):
            case Err(e):
                msg = f"[Round {round_num}] Model error: {e}"
                logger.warning(msg)
                ctx = ctx.with_feedback(msg)
                continue
            case Ok(raw):
                pass

        match parse(raw):
            case Err(e):
                # Log what the model actually returned so we can diagnose
                if isinstance(raw, dict):
                    keys = sorted(raw.keys())
                    sample = {
                        k: (repr(v)[:80] + "..." if len(repr(v)) > 80 else repr(v))
                        for k, v in raw.items()
                        if not isinstance(v, (list, dict))
                    }
                    logger.warning(
                        "[Round %d] Parse failed: %s\n"
                        "  skeleton keys: %s\n"
                        "  scalar fields: %s",
                        round_num, e, keys, sample,
                    )
                else:
                    logger.warning(
                        "[Round %d] Parse failed: %s (raw type: %s)",
                        round_num, e, type(raw).__name__,
                    )
                msg = f"[Round {round_num}] Type error: {e}"
                ctx = ctx.with_feedback(msg)
                continue
            case Ok(spec):
                pass

        logger.info("Spec parsed successfully")

        # -- Ouroboros: validate + fix in place ----------------------------

        cycle_occurred = False

        for fix_attempt in range(max_fixes + 1):
            match validate(spec):
                case Ok(artifact):
                    match emit(spec, artifact, round_num + 1, ouroboros_fixes, 0, ctx.user_prompt):
                        case Ok(path):
                            return Ok(path)
                        case Err(e):
                            last_error = f"Emit error: {e}"
                            break

                case Cycle() as c:
                    # Partial success. Re-invoke planner with facts.
                    logger.info(
                        "[Round %d] Cycle break -- %d facts collected, "
                        "re-invoking planner",
                        round_num, len(c.facts),
                    )
                    ctx = ctx.with_domain(DomainSection(
                        heading="Completed Facts (continue from here)",
                        content=c.message,
                    ))
                    cycle_occurred = True
                    break

                case Err(error):
                    last_error = error if isinstance(error, str) else str(error)
                    logger.warning(
                        "[Round %d, fix %d/%d] %s",
                        round_num, fix_attempt, max_fixes, last_error,
                    )

                    if fix_attempt < max_fixes:
                        patched = fix(spec, last_error, ctx)
                        if patched is not None:
                            spec = patched
                            ouroboros_fixes += 1
                            logger.info(
                                "Ouroboros fix %d applied, re-validating",
                                ouroboros_fixes,
                            )
                            continue

                    # Ouroboros exhausted or can't target -- scrap the page
                    break

        if cycle_occurred:
            continue  # re-invoke planner with enriched context

        # Scrap the page, keep the learnings
        msg = f"[Round {round_num}] {last_error}"
        logger.info("Scrapping spec, preserving errors for next round")
        ctx = ctx.with_feedback(msg)

    return Err(
        f"Generation failed after {max_rounds} rounds: {last_error}"
    )


def repl_loop(
    ctx: GenerationContext,
    *,
    invoke: InvokeFn,
    parse: ParseFn,
    validate: ValidateFn,
    fix: FixFn,
    emit: EmitFn,
    max_rounds: int = 3,
    max_fixes: int = 3,
    load_artifact: Callable[[Path], Result] | None = None,
    banner: str = "Generator -- interactive mode",
) -> Result:
    """Interactive REPL: read prompts, generate artifacts, repeat.

    Session context persists across prompts -- each turn builds on
    the last artifact if load_artifact is provided.
    """
    import sys

    try:
        import readline  # noqa: F401 -- enables line editing
    except ImportError:
        pass

    print()
    print(banner)
    print("Type a prompt, press Enter twice to submit.")
    print("  exit / quit / :q -- exit")
    print()

    while True:
        text = _read_multiline(">>> ")
        if text is None:
            print()
            return Ok(None)

        text = text.strip()
        if not text:
            continue

        prompt_ctx = ctx.with_prompt(text)

        match generation_loop(
            prompt_ctx,
            invoke=invoke,
            parse=parse,
            validate=validate,
            fix=fix,
            emit=emit,
            max_rounds=max_rounds,
            max_fixes=max_fixes,
        ):
            case Ok(path):
                print(f"Artifact generated: {path}")
                if load_artifact is not None:
                    match load_artifact(path):
                        case Ok(_):
                            pass
                        case Err(_):
                            pass
            case Err(e):
                print(f"Generation failed: {e}", file=sys.stderr)

        print()


def _read_multiline(prompt: str = ">>> ") -> str | None:
    """Read multi-line input. Blank line submits. Ctrl-D / 'exit' quits."""
    lines: list[str] = []
    try:
        first = input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None

    stripped = first.strip()
    if stripped.lower() in ("exit", "quit", ":q"):
        return None

    lines.append(first)

    while True:
        try:
            line = input("... ")
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip() == "":
            break
        lines.append(line)

    return "\n".join(lines)
