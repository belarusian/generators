"""Context builder for Trinity."""

from __future__ import annotations

import os
from pathlib import Path

from compass.generators._types import DomainSection, GenerationContext

def _discover_available_packages() -> str:
    """List installed Python packages."""
    try:
        from importlib.metadata import distributions
        pkgs = sorted(
            {d.metadata["Name"] for d in distributions() if d.metadata["Name"]},
            key=str.lower,
        )
        return ", ".join(pkgs)
    except Exception:
        return ""


_PLAN_PRINCIPLES = """\
Plan construction principles for Trinity:

- Each step should be atomic: one artifact, one fact.
- Steps form a DAG via depends_on. No cycles.
- Prefer inline_python for computation -- it's the most reliable.
- Use 'auto' for discovered artifacts -- Trinity will inspect their
  signatures and map your inputs to parameters automatically.
- Use 'module' only when you know the exact dotted module path.
- Use 'shell' only for system commands.
- extraction_expr is evaluated in the step's namespace after execution.
  For inline_python, it's a variable name from the executed code.
  For auto/module, 'result' holds the return value of the entry function.
- inputs can reference prior facts via {"$fact": "fact_name"}.
- The synthesis field should describe how facts combine to answer the question.
- Plans should be minimal: fewest steps that answer the question.
- Each step's artifact_ref (for inline_python) must be complete, runnable code.
  Do not assume imports or variables from other steps unless passed via inputs.
- When using discovered artifacts, match your input keys to the parameter
  names shown in the artifact listing. Trinity maps by name.
"""


def build_trinity_context(
    prompt: str | None = None,
    workspace: Path | None = None,
) -> GenerationContext:
    """Build context for Trinity.

    The model needs to understand the plan contract and see what
    artifacts are available to construct an execution plan.
    Artifact discovery uses signature inspection, not hard-coded lists.
    """
    sections: list[DomainSection] = []

    # Discover artifacts with full signature information
    from compass.generators.trinity._runtime import discover_artifacts, format_artifacts_for_context

    artifacts = discover_artifacts(workspace)
    artifact_info = format_artifacts_for_context(artifacts)
    if artifact_info and not artifact_info.startswith("No "):
        sections.append(DomainSection(
            heading="Discovered Artifacts",
            content=artifact_info,
        ))

    # Plan principles
    sections.append(DomainSection(
        heading="Plan Construction Principles",
        content=_PLAN_PRINCIPLES,
    ))

    return GenerationContext(
        domain_context=tuple(sections),
        available_packages=_discover_available_packages(),
        user_prompt=prompt,
    )
