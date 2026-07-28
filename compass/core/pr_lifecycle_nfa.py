"""Generators NFA for GitHub PR lifecycle orchestration.

This module provides NFA-based orchestration for GitHub PR lifecycle events:
- PR opened -> review requested -> reviews submitted -> merged/closed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from compass.core.nfa_types import State, Transition, TransitionFn, NestedFiniteAutomaton


@dataclass
class PRContext:
    """Context for PR lifecycle NFA."""
    pr_number: int
    pr_state: str  # 'open', 'closed', 'merged'
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    comments: List[Dict[str, Any]] = field(default_factory=list)


class PRLifecycleNFA:
    """NFA for GitHub PR lifecycle orchestration."""
    
    def __init__(self):
        self.states = {
            'pr_opened': State(id='pr_opened'),
            'review_requested': State(id='review_requested'),
            'reviews_submitted': State(id='reviews_submitted'),
            'pr_merged': State(id='pr_merged'),
            'pr_closed': State(id='pr_closed'),
        }
        
        self.transitions: Dict[str, List[Transition]] = {
            'pr_opened': [
                Transition(src='pr_opened', dst='review_requested', guard=lambda ctx: ctx.pr_state == 'open'),
            ],
            'review_requested': [
                Transition(src='review_requested', dst='reviews_submitted', guard=lambda ctx: len(ctx.reviews) > 0),
                Transition(src='review_requested', dst='pr_closed', guard=lambda ctx: ctx.pr_state == 'closed'),
            ],
            'reviews_submitted': [
                Transition(src='reviews_submitted', dst='pr_merged', guard=lambda ctx: any(r.get('state') == 'APPROVED' for r in ctx.reviews)),
                Transition(src='reviews_submitted', dst='pr_closed', guard=lambda ctx: ctx.pr_state == 'closed'),
            ],
        }
        
        self.initial_state = 'pr_opened'
        
    def create_nfa(self) -> NestedFiniteAutomaton:
        """Create a NestedFiniteAutomaton for PR lifecycle."""
        return NestedFiniteAutomaton(
            states=self.states,
            transitions=self.transitions,
            initial_state=self.initial_state,
        )
        
    def get_terminal_states(self) -> set[str]:
        """Get terminal states for PR lifecycle."""
        return {'pr_merged', 'pr_closed'}
