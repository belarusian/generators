"""
Programmer Types - Dataclass schemas for Programmer NFA responses.

The Type IS the schema. Model writes Python constructors, we eval.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


# --- UNDERSTAND + DESIGN: free-form (oracle.speak(), no response type) ---

# --- IMPLEMENT state ---

class ChunkOperation(Enum):
    """Type of chunk operation."""
    CREATE = "create"
    REPLACE = "replace"
    APPEND = "append"
    INSERT = "insert"


@dataclass
class Chunk:
    """A code chunk to be applied."""
    id: str
    content: str
    target: str  # File path
    operation: ChunkOperation
    insert_after: Optional[str] = None  # For insert operation
    dependencies: Optional[List[str]] = None
    reasoning: Optional[str] = None


@dataclass
class ImplementResponse:
    """
    Programmer's implementation as chunks.

    Model writes:
        ImplementResponse(
            chunks=[
                Chunk(id="1", content="class Cache:...", target="cache.py", operation=ChunkOperation.CREATE)
            ]
        )
    """
    chunks: Optional[List[Chunk]] = None


# --- SCRIBE_REVIEW state ---

class IssueSeverity(Enum):
    """Severity of a Scribe issue."""
    ERROR = "error"
    WARNING = "warning"
    SUGGESTION = "suggestion"


@dataclass
class ScribeIssue:
    """An issue found by Scribe."""
    chunk_id: str
    severity: IssueSeverity
    description: str
    suggestion: Optional[str] = None


class ScribeAction(Enum):
    """Scribe's decision."""
    APPROVE = "approve"
    FETCH_PATTERN = "fetch_pattern"
    FEEDBACK = "feedback"


@dataclass
class ScribeReviewResponse:
    """
    Scribe's review of the solution.

    Model writes:
        ScribeReviewResponse(
            action=ScribeAction.APPROVE,
            explanation="Solution follows patterns correctly"
        )
    """
    action: ScribeAction
    explanation: str
    query: Optional[str] = None  # For fetch_pattern
    feedback: Optional[str] = None  # For feedback action
    issues: Optional[List[ScribeIssue]] = None


# --- CRITIC states (Programmer's internal critic, not neo's) ---

class CriticReviewAction(Enum):
    """Programmer Critic's review decision."""
    APPROVE = "approve"
    REVISE = "revise"


@dataclass
class ProgrammerCriticReviewResponse:
    """
    Programmer's internal Critic review.

    Model writes:
        ProgrammerCriticReviewResponse(
            action=CriticReviewAction.APPROVE,
            explanation="Solution meets all requirements"
        )
    """
    action: CriticReviewAction
    explanation: str
    feedback: Optional[str] = None
    missing_requirements: Optional[List[str]] = None

class CriticEvaluateAction(Enum):
    """Programmer Critic's evaluate decision."""
    RETRY = "retry"
    FAIL = "fail"


# --- Parent Critic (program.py's outer review loop) ---

class ParentCriticAction(Enum):
    """Parent Critic's decision after reviewing Programmer's applied work."""
    DONE = "done"
    REVERT = "revert"
    REPLAN = "replan"


@dataclass
class ParentCriticOutput:
    """
    Parent Critic's evaluation of Programmer's applied chunks.

    Model writes:
        ParentCriticOutput(
            action=ParentCriticAction.DONE,
            explanation="Solution looks correct"
        )
    """
    action: ParentCriticAction
    explanation: str
    feedback: Optional[str] = None


class RetryFromState(Enum):
    """State to retry from."""
    UNDERSTAND = "understand"
    DESIGN = "design"
    IMPLEMENT = "implement"


@dataclass
class ProgrammerCriticEvaluateResponse:
    """
    Programmer's internal Critic evaluation after failure.

    Model writes:
        ProgrammerCriticEvaluateResponse(
            action=CriticEvaluateAction.RETRY,
            explanation="Can fix with different approach",
            retry_from=RetryFromState.DESIGN,
            feedback="Try using composition instead of inheritance"
        )
    """
    action: CriticEvaluateAction
    explanation: str
    retry_from: Optional[RetryFromState] = None
    feedback: Optional[str] = None
