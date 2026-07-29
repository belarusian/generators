"""NFA-based workflow for issue triage state machine.

This module provides NFA-based workflow for issue triage state machine in generators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from compass.core.nfa_types import State, Transition, NestedFiniteAutomaton


@dataclass
class TriageContext:
    """Context for issue triage NFA."""
    issue_id: str
    issue_text: str
    triage_status: str = 'pending'
    priority: Optional[int] = None
    labels: List[str] = field(default_factory=list)


class IssueTriageNFA:
    """NFA for issue triage state machine."""
    
    def __init__(self):
        self.states = {
            'triage_pending': State(id='triage_pending'),
            'triage_analyzing': State(id='triage_analyzing'),
            'triage_completed': State(id='triage_completed'),
            'triage_failed': State(id='triage_failed'),
        }
        
        self.transitions: Dict[str, List[Transition]] = {
            'triage_pending': [
                Transition(src='triage_pending', dst='triage_analyzing', guard=lambda ctx: True),
            ],
            'triage_analyzing': [
                Transition(src='triage_analyzing', dst='triage_completed', guard=lambda ctx: ctx.priority is not None),
                Transition(src='triage_analyzing', dst='triage_failed', guard=lambda ctx: ctx.priority is None),
            ],
        }
        
        self.initial_state = 'triage_pending'
        
    def create_nfa(self) -> NestedFiniteAutomaton:
        """Create a NestedFiniteAutomaton for issue triage."""
        return NestedFiniteAutomaton(
            states=self.states,
            transitions=self.transitions,
            initial_state=self.initial_state,
        )
        
    def get_terminal_states(self) -> set[str]:
        """Get terminal states for issue triage."""
        return {'triage_completed', 'triage_failed'}
