"""Domain context builders for the code generator.

Each domain is a pure function: (prompt) -> GenerationContext.
The generation loop is domain-independent. These functions populate
the context with domain-specific knowledge.
"""

from __future__ import annotations

from pathlib import Path

from compass.generators._types import DomainSection, GenerationContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_available_packages() -> str:
    """List installed Python packages."""
    from importlib.metadata import distributions

    pkgs = sorted(
        {d.metadata["Name"] for d in distributions() if d.metadata["Name"]},
        key=str.lower,
    )
    return ", ".join(pkgs)


# ---------------------------------------------------------------------------
# Software engineering domain
# ---------------------------------------------------------------------------


def build_code_context(
    prompt: str | None = None,
) -> GenerationContext:
    """Build generation context with software engineering best practices."""
    return GenerationContext(
        domain_context=(
            DomainSection(
                "Software Engineering Best Practices",
                (
                    "Follow these principles when generating code:\n"
                    "\n"
                    "1. **Clean Code**: Meaningful names, small functions, single responsibility\n"
                    "2. **Type Safety**: Use type hints on all function signatures\n"
                    "3. **Documentation**: Docstrings on all public functions and classes\n"
                    "4. **Error Handling**: Use exceptions appropriately, never silently swallow errors\n"
                    "5. **Testing**: Include tests that cover happy path and edge cases\n"
                    "6. **Immutability**: Prefer immutable data structures where practical\n"
                    "7. **Composition**: Build complex behavior from simple, composable functions\n"
                    "8. **DRY**: Don't repeat yourself -- extract common patterns\n"
                    "9. **KISS**: Keep it simple -- avoid unnecessary complexity\n"
                    "10. **Separation of Concerns**: Keep I/O at the boundaries, logic in the core\n"
                ),
            ),
            DomainSection(
                "Python Conventions",
                (
                    "- Follow PEP 8 style guidelines\n"
                    "- Use dataclasses for data containers\n"
                    "- Use pathlib.Path instead of os.path\n"
                    "- Use f-strings for string formatting\n"
                    "- Use context managers for resource management\n"
                    "- Prefer list/dict/set comprehensions over manual loops\n"
                    "- Use __all__ to control public API\n"
                    "- Use if __name__ == '__main__' guard for scripts\n"
                ),
            ),
        ),
        available_packages=_discover_available_packages(),
        user_prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Generic (no domain context)
# ---------------------------------------------------------------------------


def build_generic_context(
    prompt: str | None = None,
) -> GenerationContext:
    """Build generation context with no domain-specific knowledge."""
    return GenerationContext(
        available_packages=_discover_available_packages(),
        user_prompt=prompt,
    )
