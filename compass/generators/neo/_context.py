"""
Domain context builders for the plan generator.

Each builder is a pure function: (prompt) -> GenerationContext.
"""

from __future__ import annotations

import os
from pathlib import Path

from compass.generators._types import DomainSection, GenerationContext

# neo-lab -- configurable companion workspace for taught skills and states
_NEO_LAB = Path(
    os.environ.get(
        "COMPASS_NEO_LAB",
        str(Path.home() / ".compass" / "neo-lab"),
    )
)


def _neo_lab_context(query: str | None = None) -> str:
    """Retrieve relevant skills and states from neo-lab.

    If the skill index is available and query is provided, uses RAG to
    return only relevant skills + their dependency subgraph.
    Falls back to flat list if embedding backend is unavailable.
    """
    import sys
    neo_str = str(_NEO_LAB)
    if neo_str not in sys.path:
        sys.path.insert(0, neo_str)

    try:
        from neo.skill_index import get_index
        idx = get_index()

        if query:
            # RAG: retrieve by relevance + walk dependencies
            context = idx.context_for(query, top_k=5)
            if context:
                return context

        # No query or no matches -- list all
        return idx.list_all()
    except Exception:
        # Fallback: flat list without embeddings
        return _flat_list_fallback()


def _flat_list_fallback() -> str:
    """List all skills and states without embeddings."""
    lines = []

    skills_dir = _NEO_LAB / "skills"
    if skills_dir.exists():
        try:
            import yaml
            for p in sorted(skills_dir.glob("*.yaml")):
                with open(p) as f:
                    skill = yaml.safe_load(f) or {}
                name = skill.get("name", p.stem)
                n_steps = len(skill.get("steps", []))
                lines.append(f"  {name} ({n_steps} steps)")
        except Exception:
            names = sorted(p.stem for p in skills_dir.glob("*.yaml"))
            lines.extend(f"  {n}" for n in names)

    graph_path = _NEO_LAB / "states.yaml"
    if graph_path.exists():
        try:
            import yaml
            with open(graph_path) as f:
                graph = yaml.safe_load(f) or {}

            states = graph.get("states", {})
            if states:
                lines.append("")
                lines.append("States:")
                for name, s in states.items():
                    markers = s.get("markers", [])
                    visual = s.get("visual", [])
                    parts = []
                    if markers:
                        parts.append(f"text: {markers}")
                    if visual:
                        parts.append(f"visual: {visual}")
                    lines.append(f"  {name}: {' | '.join(parts)}")

            transitions = graph.get("transitions", [])
            if transitions:
                lines.append("")
                lines.append("Transitions:")
                for t in transitions:
                    lines.append(f"  {t['from']} --[{t['skill']}]--> {t['to']}")
        except Exception:
            pass

    return "\n".join(lines)


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

    sections = [
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
    ]

    # neo-lab: taught skills and screen states (RAG when available)
    neo_text = _neo_lab_context(query=prompt)
    if neo_text:
        sections.append(DomainSection(
            "Available Skills and States (neo-lab)",
            "Skills taught via interactive demonstration, available via "
            "artifact_type 'screen' (SkillAction) or 'state' "
            "(StateCheckAction). Each skill is a validated sequence "
            "(G . V . G . V). States are screen signatures (OCR + DINO).\n\n"
            f"{neo_text}",
        ))

    return GenerationContext(
        domain_context=tuple(sections),
        available_packages=_discover_available_packages(),
        user_prompt=prompt,
        default_task="Generate a plan to build the requested software.",
    )
