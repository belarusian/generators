"""
Shared foundation types for all generators.

FP discipline: this module imports only stdlib. No compass, no providers.
All data is frozen/immutable. Validation functions return Result[T, E].

These types are the invariant core -- the generation loop, the context,
the result monad, the domain sections. Everything that changes between
generators (Spec shape, validators, exec strategy) lives in the generator
module itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Generic, Optional, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E")


# ============================================================================
# Result type
# ============================================================================


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True)
class Err(Generic[E]):
    error: E


@dataclass(frozen=True)
class Cycle:
    """Partial success: non-deterministic step completed, re-plan needed.

    Facts from completed steps are carried here. The generation loop
    adds them to context and re-invokes the planner.
    """
    facts: dict = field(default_factory=dict)
    message: str = ""


Result = Union[Ok[T], Err[E]]


# ============================================================================
# Domain context
# ============================================================================


@dataclass(frozen=True)
class DomainSection:
    """A named section of domain knowledge for the model's system prompt."""

    heading: str
    content: str


@dataclass(frozen=True)
class FileSpec:
    """A file declared by the model as part of the solution.

    The artifact is the semantic tree -- cells/sections explain and prove.
    Files are the leaves -- the actual deliverables (modules, configs,
    tests). The model declares both; the system validates and extracts.
    """

    path: str          # relative path, e.g. "src/app.py"
    content: str
    description: str


# ============================================================================
# Generation context
# ============================================================================


@dataclass(frozen=True)
class GenerationContext:
    """Immutable accumulator -- everything the model knows.

    Generic across all generators. Spec-specific fields (like prior_spec)
    belong in the generator's own Config type.
    """

    domain_context: tuple[DomainSection, ...] = ()
    available_packages: str = ""
    feedback: tuple[str, ...] = ()
    user_prompt: Optional[str] = None
    default_task: Optional[str] = None

    def with_domain(self, section: DomainSection) -> GenerationContext:
        return replace(self, domain_context=(*self.domain_context, section))

    def with_feedback(self, msg: str) -> GenerationContext:
        return replace(self, feedback=(*self.feedback, msg))

    def with_prompt(self, prompt: str) -> GenerationContext:
        """Return a fresh context with the given prompt and no feedback."""
        return replace(self, user_prompt=prompt, feedback=())


# ============================================================================
# AskFn -- the injected model invocation contract
# ============================================================================

# (system_prompt, user_message) -> Ok(response_text) | Err(error_msg)
AskFn = Callable[[str, str], Result]


# ============================================================================
# Generation report
# ============================================================================


@dataclass(frozen=True)
class GenerationReport:
    """Provenance record for a generated artifact.

    The artifact is the thing. The report is its history.
    Emitted as a JSON sidecar alongside the artifact.
    """

    version: int = 0
    rounds: int = 0
    ouroboros_fixes: int = 0
    outcome: str = "success"  # success | failed | stuck
    claim: Optional[str] = None
    validators: tuple[str, ...] = ()
    user_prompt: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialise for JSON output."""
        d: dict = {
            "version": self.version,
            "rounds": self.rounds,
            "ouroboros_fixes": self.ouroboros_fixes,
            "outcome": self.outcome,
            "validators": list(self.validators),
        }
        if self.claim:
            d["claim"] = self.claim
        if self.user_prompt:
            d["user_prompt"] = self.user_prompt
        return d
