#!/usr/bin/env python3
"""Notebook generator -- generates Jupyter notebook artifacts.

G_notebook  : Query -> NotebookSpec
V_notebook  : syntax check + exec all code cells
G_notebook' : (NotebookSpec, Error) -> NotebookPatch -> NotebookSpec

Usage:
    python -m compass.generators.notebook.generate \\
        --prompt "Create a notebook exploring numpy random distributions" \\
        --output-dir ./notebooks

    python -m compass.generators.notebook.generate \\
        --prompt "Plot a sine wave" --dry-run
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

from compass.generators.notebook._types import NotebookSpec, validate_spec_instance
from compass.generators.notebook._runtime import (
    emit_notebook,
    invoke_model,
    load_notebook,
    ouroboros_notebook,
    validate_notebook,
)
from compass.generators.notebook._context import build_notebook_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The five functions for generation_loop
# ---------------------------------------------------------------------------


def _make_invoke(model_id: str, ask_fn=None):
    """G_notebook: Context -> Result[NotebookSpec]

    Python-as-schema: model writes constructor + content blocks.
    Returns typed NotebookSpec on success.
    """
    def invoke(ctx: GenerationContext) -> Result:
        return invoke_model(ctx, model_id, ask_fn)
    return invoke


def _make_parse():
    """V1: NotebookSpec -> Result[NotebookSpec]

    Instance validation: non-empty content, valid cell types,
    at least one code cell. Lighter than dict validation.
    """
    def parse(spec: NotebookSpec) -> Result:
        return validate_spec_instance(spec)
    return parse


def _make_validate():
    """V2: NotebookSpec -> Result[NotebookSpec]

    Composed cheapest-first: syntax -> exec.
    """
    def validate(spec: NotebookSpec) -> Result:
        return validate_notebook(spec)
    return validate


def _make_fix(model_id: str, ask_fn=None):
    """G'_notebook: (spec, error, ctx) -> spec | None

    Uses patch-based ouroboros: model returns NotebookPatch,
    which is applied to the original spec.
    """
    def fix(spec: NotebookSpec, error: str, ctx: GenerationContext):
        return ouroboros_notebook(spec, error, ctx, model_id, ask_fn)
    return fix


def _make_emit(output_dir: Path):
    """IO: write notebook to output directory."""
    def emit(
        spec: NotebookSpec,
        artifact: NotebookSpec,
        rounds: int,
        fixes: int,
        version_hint: int,
        prompt_or_claim,
    ) -> Result:
        try:
            nb_path = emit_notebook(spec, output_dir, version=version_hint)
            # Write report sidecar
            report = GenerationReport(
                version=version_hint,
                rounds=rounds,
                ouroboros_fixes=fixes,
                outcome="success",
                claim=spec.claim,
                user_prompt=prompt_or_claim if isinstance(prompt_or_claim, str) else None,
            )
            report_path = nb_path.with_suffix(".report.json")
            report_path.write_text(json.dumps(report.to_dict(), indent=2))
            return Ok(nb_path)
        except Exception as exc:
            return Err(f"Emit error: {exc}")
    return emit


# ---------------------------------------------------------------------------
# Entry points
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
) -> Result:
    """Programmatic entry point. No argparse, no sys.argv.

    In-process callers (Neo, meta-generator) call this directly.

    refine: (artifact_path, claim) -- load existing notebook into context,
            claim becomes the prompt. Same generation loop, richer context.
    """
    if not prompt and not refine:
        return Err("prompt or refine is required")
    out = Path(output) if output else Path.cwd() / "notebooks"

    logger.info("Building notebook generation context...")
    ctx = build_notebook_context(prompt)

    if refine:
        artifact_path, claim = refine
        match refine_context(
            ctx, artifact_path, claim,
            load=load_notebook,
            serialize=lambda spec: "\n\n".join(
                f"# Cell {i} ({c.cell_type})\n{c.source}"
                for i, c in enumerate(spec.cells)
            ),
        ):
            case Err() as e:
                return e
            case Ok(resolved):
                ctx = resolved
    else:
        ctx = ctx.with_prompt(prompt)

    if dry_run:
        from compass.generators._invoke import build_system_prompt, build_user_message
        _types_src = (Path(__file__).parent / "_types.py").read_text()
        system = build_system_prompt(
            ctx,
            _types_src,
            role=(
                "You are an expert Jupyter notebook author. "
                "You generate well-structured, executable notebooks "
                "with clear explanations and correct Python code."
            ),
            contract_preamble=(
                "Respond with a Python expression constructing NotebookSpec.\n"
                "See the NotebookSpec docstring for the exact response format."
            ),
        )
        user = build_user_message(
            ctx,
            suffix_lines=(
                "Write a NotebookSpec(...) expression. See its docstring for the response format.",
                "No markdown fencing.",
            ),
        )
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
        ctx,
        invoke=_make_invoke(model_id),
        parse=_make_parse(),
        validate=_make_validate(),
        fix=_make_fix(model_id),
        emit=_make_emit(out),
        max_rounds=max_rounds,
        max_fixes=max_fixes,
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Notebook generator: generate Jupyter notebooks from prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m compass.generators.notebook.generate \\
                --prompt "Explore numpy random distributions"

              python -m compass.generators.notebook.generate \\
                --prompt "Plot a sine wave" \\
                --output-dir ./notebooks \\
                --model-id anthropic:sonnet
        """),
    )
    parser.add_argument("--prompt", help="What notebook to generate")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: ./notebooks)",
    )
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-fixes", type=int, default=3)
    parser.add_argument(
        "--model-id", default="",
        help="Model spec (e.g. anthropic:sonnet, anthropic:opus)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refine", help="Path to existing notebook to refine")
    parser.add_argument("--claim", help="Domain expert claim for refinement")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    refine = (args.refine, args.claim or "Re-validate this notebook") if args.refine else None

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
                print(f"Notebook generated: {path}")
        case Err(e):
            print(f"ERROR: {e}", file=sys.stderr)

    return result_to_exit(result)


if __name__ == "__main__":
    sys.exit(main())
