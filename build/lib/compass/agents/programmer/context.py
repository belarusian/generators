"""
Programmer Context - State enum and context dataclasses.

Defines the bounded contexts for Programmer (Actor) and Scribe (Critic).
Each sees only what it needs - this is the key to preventing
"paralysis by analysis" from information overload.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Any, Tuple, TYPE_CHECKING

from compass.core.context import OracleAccess

if TYPE_CHECKING:
    from compass.agents.programmer.trace import ProgrammerTrace


class ProgrammerState(Enum):
    """
    States for the Programmer NFA.

    Programmer (Actor) states: UNDERSTAND, DESIGN, IMPLEMENT
    Scribe states: SCRIBE_REVIEW, SCRIBE_FETCH, SCRIBE_FEEDBACK
    Critic states: CRITIC_REVIEW (holistic), CRITIC_EVALUATE (tactical)
    """
    # Programmer (Actor) - abstract solution generation
    UNDERSTAND = auto()      # Parse problem, identify constraints
    DESIGN = auto()          # Create solution architecture (single doc)
    IMPLEMENT = auto()       # Generate code chunks

    # Scribe - system-constrained validation (grounds to codebase)
    SCRIBE_REVIEW = auto()   # Evaluate against system constraints
    SCRIBE_FETCH = auto()    # Request code patterns from system
    SCRIBE_FEEDBACK = auto() # Provide amendments back to Programmer

    # Critic - requirements validation (grounds to problem)
    CRITIC_REVIEW = auto()   # After Scribe approves: do chunks solve problem?
    CRITIC_EVALUATE = auto() # After failure: retry or give up?

    # Terminal
    DELIVER = auto()         # Final delivery (Critic approved)
    DONE = auto()
    FAILED = auto()


@dataclass
class ScribeView:
    """
    What Scribe (Critic) sees - system-constrained context.

    Scribe is isolated from the original problem - only sees the solution
    and system constraints. This prevents contamination of the critique.

    Scribe CAN:
    - Request code patterns (via callback)
    - See existing file structure
    - Validate against coding standards

    Scribe CANNOT:
    - See the original problem statement
    - Modify the solution directly (only provide feedback)
    """
    solution_chunks: List[Dict]                    # From Programmer
    request_pattern: Callable[[str], str]          # Callback to fetch code patterns
    file_structure: Dict[str, str]                 # Available files (names only)
    coding_standards: List[str]                    # Project conventions

    # Scribe's critique
    issues: List[Dict] = field(default_factory=list)
    approved_chunks: List[str] = field(default_factory=list)
    feedback_for_programmer: Optional[str] = None
    pattern_query: Optional[str] = None
    fetched_patterns: Dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        """Human-readable description for debugging."""
        n_issues = len(self.issues)
        n_approved = len(self.approved_chunks)
        return f"Scribe: {n_approved} approved, {n_issues} issues"


@dataclass
class ProgrammerContext:
    """
    Bounded context for Programmer NFA.

    Note the separation:
    - Programmer (Actor) sees: problem + solution space
    - Scribe (Critic) sees: solution + system constraints (via ScribeView)

    Neither sees:
    - Actor's action history (top-level concern)
    - Full codebase (only what Scribe requests)
    """
    oracle: OracleAccess  # OracleAccess protocol
    problem: str                                   # The problem statement
    constraints: List[str] = field(default_factory=list)  # Explicit constraints
    parent_feedback: Optional[str] = None          # Feedback from parent Critic (for retries)
    show_prompts: bool = True                      # Show debug prompts (False for parallel deep branches)

    # Callbacks for system interaction (set by caller)
    fetch_pattern: Optional[Callable[[str], str]] = None  # Get code patterns
    get_file_structure: Optional[Callable[[], Dict[str, str]]] = None
    get_coding_standards: Optional[Callable[[], List[str]]] = None
    apply_chunks: Optional[Callable[[List[Dict]], Tuple[bool, str]]] = None  # Apply chunks to filesystem

    # Programmer's work (Actor phase)
    understanding: Optional[str] = None            # Parsed problem understanding
    design: Optional[str] = None                   # Solution architecture
    solution_doc: Optional[str] = None             # The single-document solution
    chunks: List[Dict] = field(default_factory=list)  # Chunked deliverables

    # Scribe's view - populated during SCRIBE_* states
    scribe_view: Optional[ScribeView] = None
    scribe_iterations: int = 0
    max_scribe_iterations: int = 3                 # Max feedback loops before forced delivery

    # Critic state - requirements validation (CRITIC_REVIEW)
    critic_feedback: Optional[str] = None          # Feedback for revision
    critic_review_iterations: int = 0
    max_critic_review_iterations: int = 2          # Max "revise" loops before forced done

    # Critic state - failure recovery (CRITIC_EVALUATE)
    critic_evaluate_retries: int = 0
    max_critic_evaluate_retries: int = 3           # Max retry attempts before giving up
    last_error: Optional[str] = None               # For CRITIC_EVALUATE context

    # Execution trace - structured history of state transitions
    trace: Optional["ProgrammerTrace"] = None

    def describe(self) -> str:
        """Human-readable description for debugging (NFAContext protocol)."""
        phase = "programming" if not self.scribe_view else "scribe review"
        n_chunks = len(self.chunks)
        return f"Programmer ({phase}): {self.problem[:50]}... ({n_chunks} chunks)"


@dataclass
class ProgrammerResult:
    """What Programmer returns when invoked as an action."""
    success: bool
    solution_doc: str = ""
    chunks: List[Dict] = field(default_factory=list)  # [{id, content, target, operation}]
    reasoning: str = ""
    scribe_issues: List[Dict] = field(default_factory=list)  # Any unresolved issues
    error: Optional[str] = None
    iterations: int = 0
    trace: Optional["ProgrammerTrace"] = None  # Execution trace for debugging/analysis

    def describe(self) -> str:
        """Human-readable summary."""
        if self.success:
            return f"Success: {len(self.chunks)} chunks"
        else:
            return f"Failed: {self.error}"
