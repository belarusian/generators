"""
Programmer Trace - Structured tracking of NFA state transitions.

Captures the decision history so CRITIC_EVALUATE and parent callers
can understand what happened and why, not just "iteration count = 3".
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from compass.agents.programmer.context import ProgrammerState


class TransitionReason(Enum):
    """Why a state transition occurred."""
    # Normal flow
    SUCCESS = "success"

    # Scribe-related
    SCRIBE_APPROVED = "scribe_approved"
    SCRIBE_REJECTED = "scribe_rejected"
    SCRIBE_NEEDS_PATTERN = "scribe_needs_pattern"
    SCRIBE_MAX_ITERATIONS = "scribe_max_iterations"

    # Critic-related
    CRITIC_APPROVED = "critic_approved"
    CRITIC_REVISE = "critic_revise"
    CRITIC_RETRY = "critic_retry"
    CRITIC_GIVE_UP = "critic_give_up"
    CRITIC_MAX_ITERATIONS = "critic_max_iterations"

    # Errors
    PARSE_ERROR = "parse_error"
    ORACLE_ERROR = "oracle_error"
    VALIDATION_ERROR = "validation_error"

    # Delivery
    DELIVER_SUCCESS = "deliver_success"
    DELIVER_FAILED = "deliver_failed"


@dataclass
class ProgrammerTransition:
    """Single state transition with context."""
    from_state: str  # ProgrammerState.name
    to_state: str    # ProgrammerState.name
    reason: TransitionReason
    feedback: Optional[str] = None  # Scribe/Critic feedback
    error: Optional[str] = None     # Error message if applicable
    chunks_affected: List[str] = field(default_factory=list)  # Chunk IDs modified
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __str__(self) -> str:
        arrow = f"{self.from_state} -> {self.to_state}"
        detail = self.feedback[:50] if self.feedback else self.reason.value
        return f"{arrow} ({detail})"


@dataclass
class ProgrammerTrace:
    """Full execution trace for Programmer NFA."""
    transitions: List[ProgrammerTransition] = field(default_factory=list)

    def add(
        self,
        from_state: "ProgrammerState",
        to_state: "ProgrammerState",
        reason: TransitionReason,
        feedback: Optional[str] = None,
        error: Optional[str] = None,
        chunks_affected: Optional[List[str]] = None,
    ) -> None:
        """Record a state transition."""
        self.transitions.append(ProgrammerTransition(
            from_state=from_state.name,
            to_state=to_state.name,
            reason=reason,
            feedback=feedback,
            error=error,
            chunks_affected=chunks_affected or [],
        ))

    @property
    def scribe_rejections(self) -> int:
        """Count of Scribe rejection cycles."""
        return sum(1 for t in self.transitions if t.reason == TransitionReason.SCRIBE_REJECTED)

    @property
    def scribe_iterations(self) -> int:
        """Total Scribe review iterations."""
        return sum(1 for t in self.transitions
                   if t.to_state == "SCRIBE_REVIEW" or t.from_state == "SCRIBE_REVIEW")

    @property
    def critic_revisions(self) -> int:
        """Count of Critic-requested revisions."""
        return sum(1 for t in self.transitions if t.reason == TransitionReason.CRITIC_REVISE)

    @property
    def errors(self) -> List[str]:
        """All error messages encountered."""
        return [t.error for t in self.transitions if t.error]

    @property
    def feedback_history(self) -> List[str]:
        """All feedback received (from Scribe and Critic)."""
        return [t.feedback for t in self.transitions if t.feedback]

    @property
    def unique_feedback(self) -> List[str]:
        """Deduplicated feedback - reveals stuck patterns."""
        seen = set()
        unique = []
        for fb in self.feedback_history:
            normalized = fb.strip().lower()[:100]  # Normalize for comparison
            if normalized not in seen:
                seen.add(normalized)
                unique.append(fb)
        return unique

    @property
    def is_stuck_pattern(self) -> bool:
        """Detect if same feedback repeated 2+ times (stuck in loop)."""
        feedback = self.feedback_history
        if len(feedback) < 2:
            return False
        # Check if last 2 feedbacks are similar
        last = feedback[-1].strip().lower()[:100] if feedback[-1] else ""
        prev = feedback[-2].strip().lower()[:100] if feedback[-2] else ""
        return last == prev and last != ""

    def state_path(self) -> str:
        """Compact state transition path."""
        if not self.transitions:
            return "(no transitions)"
        states = [self.transitions[0].from_state]
        for t in self.transitions:
            states.append(t.to_state)
        return " -> ".join(states)

    def summary_for_critic(self) -> str:
        """Actionable summary for CRITIC_EVALUATE decision."""
        if not self.transitions:
            return "(no execution history)"

        lines = ["--- PROGRAMMER EXECUTION TRACE ---"]

        # State path
        lines.append(f"Path: {self.state_path()}")

        # Key metrics
        lines.append(f"Scribe rejections: {self.scribe_rejections}")
        lines.append(f"Critic revisions: {self.critic_revisions}")

        # Stuck pattern detection
        if self.is_stuck_pattern:
            lines.append("WARNING: Same feedback repeated - may be stuck in loop")

        # Feedback history (deduplicated) - show full content for accurate decisions
        unique_fb = self.unique_feedback
        if unique_fb:
            lines.append(f"Feedback received ({len(unique_fb)} unique):")
            for fb in unique_fb[-3:]:  # Last 3 unique
                lines.append(f"  - {fb}")

        # Errors - show full content
        errors = self.errors
        if errors:
            lines.append(f"Errors ({len(errors)}):")
            for err in errors[-2:]:  # Last 2
                lines.append(f"  - {err}")

        return "\n".join(lines)

    def summary_for_parent(self) -> str:
        """Summary for parent Actor/Critic when Programmer is used as tool."""
        if not self.transitions:
            return "Programmer: no execution data"

        # Count key events
        scribe_rej = self.scribe_rejections
        critic_rev = self.critic_revisions
        total = len(self.transitions)

        parts = [f"Programmer trace: {total} transitions"]
        if scribe_rej:
            parts.append(f"{scribe_rej} Scribe rejections")
        if critic_rev:
            parts.append(f"{critic_rev} Critic revisions")
        if self.is_stuck_pattern:
            parts.append("STUCK PATTERN DETECTED")

        return ", ".join(parts)
