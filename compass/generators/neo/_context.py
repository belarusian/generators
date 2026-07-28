"""
Domain context builders for the plan generator.

Each builder is a pure function: (prompt) -> GenerationContext.
"""

from __future__ import annotations

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


def _discover_available_generators() -> str:
    """List available generator modules in compass/generators/."""
    import compass.generators
    root = Path(compass.generators.__path__[0])
    generators: list[str] = []
    try:
        for child in sorted(root.iterdir()):
            if (child.is_dir()
                    and child.name != "neo"
                    and (child / "generate.py").exists()):
                generators.append(child.name)
    except OSError:
        pass
    return ", ".join(generators) if generators else "notebook"


def build_plan_context(
    prompt: str | None = None,
) -> GenerationContext:
    """Build generation context for plan generation."""
    available_gens = _discover_available_generators()

    return GenerationContext(
        domain_context=(
            DomainSection(
                "Available Generators",
                f"The following generators are available and can be used as "
                f"artifact_type values: {available_gens}. "
                f"For any artifact_type not in this list, the system will "
                f"attempt to meta-generate a new generator. "
                f"Prefer using 'notebook' as the artifact_type for Python "
                f"code generation -- it is the most mature generator.",
            ),
            DomainSection(
                "Plan Structure",
                "A plan is a DAG of steps. Each step produces one artifact. "
                "Steps execute in topological order respecting depends_on. "
                "Each step's prompt must be self-contained -- the generator "
                "receiving it has no context from other steps unless you "
                "explicitly include it in the prompt. "
                "Keep plans minimal: prefer 1-3 well-defined steps over many "
                "small fragmented ones.",
            ),
        ),
        available_packages=_discover_available_packages(),
        user_prompt=prompt,
        default_task="Generate a plan to build the requested software.",
    )
