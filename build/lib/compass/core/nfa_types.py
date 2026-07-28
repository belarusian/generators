from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Optional, Dict, Tuple, List
from enum import Enum

# Context type variable (parameterized over the NFA)
C = Any  # or TypeVar('C')

@dataclass(frozen=True)
class State:
    id: str
    action: Optional[Callable[[C], Tuple[str, C]]] = None  # (next_state_id, new_context)

@dataclass(frozen=True)
class Transition:
    src: str          # source state ID
    dst: str          # destination state ID
    guard: Callable[[C], bool] = lambda x: True  # predicate on context

TransitionFn = Callable[[C], Tuple[str, C]]

@dataclass(frozen=True)
class NFAConfig:
    states: Dict[str, State]
    transitions: Dict[str, TransitionFn[C]]
    initial_state: str
    history: Tuple['NFAConfig', ...] = ()  # for traceability / structural sharing

@dataclass(frozen=True)
class CriticResult:
    success: bool = True
    loop_detected: bool = False
    stuck_state: Optional[str] = None
    error_messages: List[str] = ()
    should_rewrite: bool = False

@dataclass
class NestedFiniteAutomaton:
    """Self-evolving NFA -- holds structure, history, and optional rewriter."""
    states: Dict[str, State]
    transitions: Dict[str, list]  # state_id -> [Transition, ...]
    initial_state: str
    history: Tuple['NestedFiniteAutomaton', ...] = ()
    rewriter: Optional[Callable] = None

    def rewrite(self, plan):
        """Apply a rewrite plan. Delegates to nfa_evolution.rewrite()."""
        from .nfa_evolution import rewrite
        return rewrite(self, plan)


class RewriteOp(Enum):
    ADD_STATE = "add_state"
    REMOVE_STATE = "remove_state"
    INSERT_BEFORE = "insert_before"
    INSERT_AFTER = "insert_after"
    # ... could include TRANSFORM_TRANSITION, etc.

@dataclass(frozen=True)
class RewritePlan:
    op: RewriteOp
    state_id: Optional[str] = None
    insert_ref: Optional[str] = None  # e.g., state to insert before/after
    meta: Dict[str, Any] = None  # e.g., {"state": State(...), "transition": Transition(...)}

    def __post_init__(self):
        if self.meta is None:
            object.__setattr__(self, 'meta', {})