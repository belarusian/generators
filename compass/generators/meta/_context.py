"""Context builder for the meta-generator."""

from __future__ import annotations

from pathlib import Path

from compass.generators._types import DomainSection, GenerationContext


def _read_file(path: Path) -> str:
    """Read a file, return empty string on failure."""
    try:
        return path.read_text()
    except OSError:
        return ""


def _discover_available_packages() -> str:
    """List installed Python packages for the model's awareness."""
    try:
        from importlib.metadata import distributions
        pkgs = sorted(
            {d.metadata["Name"] for d in distributions() if d.metadata["Name"]},
            key=str.lower,
        )
        return ", ".join(pkgs)
    except Exception:
        return ""


def _build_framework_context(gen_root: Path) -> DomainSection:
    """Read ALL shared framework files and bundle them as one section.

    This is the critical piece: the model must see _loop.py to know
    how generation_loop works, _invoke.py to know how to call the model,
    _validation.py to know what validation helpers exist, and _types.py
    to know the shared types.
    """
    framework_files = [
        ("_types.py", "Shared foundation types: Result, Ok, Err, GenerationContext, DomainSection, FileSpec, GenerationReport"),
        ("_loop.py", "Parameterized generation loop: generation_loop(), refine_context(), repl_loop(), result_to_exit()"),
        ("_invoke.py", "Model invocation: resolve_ask_fn(), build_system_prompt(), build_user_message()"),
        ("_validation.py", "Python source validation: validate_python_sources(), collect_definitions(), collect_references()"),
    ]

    parts: list[str] = []
    for fname, description in framework_files:
        content = _read_file(gen_root / fname)
        if content:
            parts.append(f"### {fname}")
            parts.append(f"")
            parts.append(f"_{description}_")
            parts.append(f"")
            parts.append(content)
            parts.append("")

    return DomainSection(
        heading="Shared Framework",
        content="\n".join(parts),
    )


def _build_exemplar_context(gen_root: Path) -> DomainSection:
    """Read the meta-generator's own source as an exemplar.

    The model studies this to understand the pattern it must follow.
    """
    meta_root = gen_root / "meta"
    exemplar_files = ["_types.py", "_runtime.py", "_context.py", "generate.py"]

    parts: list[str] = []
    for fname in exemplar_files:
        content = _read_file(meta_root / fname)
        if content:
            parts.append(f"### {fname}")
            parts.append("")
            parts.append(content)
            parts.append("")

    return DomainSection(
        heading="Exemplar Generator (meta)",
        content="\n".join(parts),
    )


def build_meta_context(
    prompt: str | None = None,
    root: Path | None = None,
) -> GenerationContext:
    """Build context for the meta-generator."""
    if root is None:
        # Navigate from this file to the project root
        # __file__ = compass/generators/meta/_context.py
        # root    = compass/generators/meta/../../.. = project root
        root = Path(__file__).resolve().parent.parent.parent.parent

    gen_root = root / "compass" / "generators"

    framework_section = _build_framework_context(gen_root)
    exemplar_section = _build_exemplar_context(gen_root)

    # Filter out empty sections
    sections = tuple(
        s for s in (framework_section, exemplar_section)
        if s.content and not s.content.startswith("No ")
    )

    return GenerationContext(
        domain_context=sections,
        available_packages=_discover_available_packages(),
        user_prompt=prompt,
    )
