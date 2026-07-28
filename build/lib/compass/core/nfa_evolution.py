"""Immutable NFA evolution infrastructure.

This module provides the core logic for self-evolving NFAs with:
- Immutable graph updates with structural sharing
- Declarative rewrite plans
- Safe, pure rewrites
- History tracking for backtracking
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable, Optional, List, Dict, Any, Tuple
from enum import Enum
import copy

from .nfa_types import State, Transition, NFAConfig, CriticResult, NestedFiniteAutomaton
from .critics import run_critics


class RewriteOp(Enum):
    ADD_STATE = "add_state"
    REMOVE_STATE = "remove_state"
    ADD_TRANSITION = "add_transition"
    REMOVE_TRANSITION = "remove_transition"
    UPDATE_GUARD = "update_guard"
    INSERT_BEFORE = "insert_before"
    INSERT_AFTER = "insert_after"


@dataclass
class RewritePlan:
    """Declarative rewrite request — immutable, serializable, critic-verifiable."""
    op: RewriteOp
    state_id: Optional[str] = None
    src: Optional[str] = None
    dst: Optional[str] = None
    guard: Optional[Callable[[Any], bool]] = None
    insert_ref: Optional[str] = None  # for insert_before/after
    meta: Dict[str, Any] = None

    def __post_init__(self):
        if self.meta is None:
            object.__setattr__(self, 'meta', {})


def evolve(nfa: NestedFiniteAutomaton, goal: Any, critics: List[Callable[[NestedFiniteAutomaton, Any], CriticResult]]) -> NestedFiniteAutomaton:
    """
    Self-evolution loop:
      1. Run critics to diagnose degradation / opportunity
      2. If rewrite needed, ask Oracle for rewrite plan (via config.rewriter)
      3. Apply rewrite safely (via .rewrite())
      4. Return new NFA, preserving history
    """
    # Step 1: Diagnosis
    diagnosis = run_critics(nfa, goal, critics)
    if not diagnosis.should_rewrite:
        return nfa  # no evolution needed

    # Step 2: Oracle-guided rewrite (if rewriter available)
    if nfa.rewriter:
        rewrite_plan = nfa.rewriter(nfa, goal, diagnosis)
    else:
        # fallback: default planner (e.g., heuristic-based)
        rewrite_plan = _default_rewrite_planner(nfa, goal, diagnosis)

    # Step 3: Apply rewrite
    new_nfa = nfa.rewrite(rewrite_plan)

    # Step 4: Append to history (preserves old NFA)
    return replace(new_nfa, history=nfa.history + (nfa,))


def rewrite(nfa: NestedFiniteAutomaton, plan: RewritePlan) -> NestedFiniteAutomaton:
    """
    Apply a *single* rewrite operation safely and immutably.
    Returns a new NFA with modified structure.
    """
    # Clone for mutation
    new_states = dict(nfa.states)
    new_transitions = {k: list(v) for k, v in nfa.transitions.items()}

    # Apply rewrite op
    match plan.op:
        case RewriteOp.ADD_STATE:
            if plan.state_id not in new_states:
                new_states[plan.state_id] = plan.meta.get('state', State(plan.state_id))
        case RewriteOp.REMOVE_STATE:
            if plan.state_id in new_states:
                del new_states[plan.state_id]
                # Remove all transitions involving it
                new_transitions = {
                    s: [t for t in ts if t.dst != plan.state_id and t.src != plan.state_id]
                    for s, ts in new_transitions.items()
                }
        case RewriteOp.ADD_TRANSITION:
            src = plan.src or nfa.initial_state
            guard = plan.guard or (lambda _: True)
            new_trans = Transition(src=src, dst=plan.dst, guard=guard)
            if src not in new_transitions:
                new_transitions[src] = []
            new_transitions[src].append(new_trans)
        case RewriteOp.REMOVE_TRANSITION:
            if plan.src in new_transitions:
                new_transitions[plan.src] = [
                    t for t in new_transitions[plan.src]
                    if t.dst != plan.dst
                ]
        case RewriteOp.UPDATE_GUARD:
            if plan.src in new_transitions:
                for t in new_transitions[plan.src]:
                    if t.dst == plan.dst:
                        # Replace guard (note: guard must be pure & hashable-safe)
                        object.__setattr__(t, 'guard', plan.guard)
        case RewriteOp.INSERT_BEFORE:
            # Insert new state `plan.state_id` before `plan.insert_ref`
            if plan.insert_ref and plan.state_id:
                # Create new intermediate state
                new_states[plan.state_id] = State(plan.state_id)
                # Rewrite: old transition into insert_ref → now into plan.state_id
                # then plan.state_id → insert_ref
                for src, transitions in list(new_transitions.items()):
                    new_transitions[src] = []
                    for t in transitions:
                        if t.dst == plan.insert_ref:
                            new_transitions[src].append(
                                Transition(src=t.src, dst=plan.state_id, guard=t.guard)
                            )
                            # add bridge
                            if plan.state_id not in new_transitions:
                                new_transitions[plan.state_id] = []
                            new_transitions[plan.state_id].append(
                                Transition(src=plan.state_id, dst=plan.insert_ref, guard=lambda x: True)
                            )
                        else:
                            new_transitions[src].append(t)
        case RewriteOp.INSERT_AFTER:
            # Symmetric to INSERT_BEFORE
            if plan.insert_ref and plan.state_id:
                new_states[plan.state_id] = State(plan.state_id)
                for src, transitions in list(new_transitions.items()):
                    new_transitions[src] = []
                    for t in transitions:
                        if t.src == plan.insert_ref:
                            new_transitions[src].append(
                                Transition(src=t.src, dst=plan.state_id, guard=t.guard)
                            )
                            if plan.state_id not in new_transitions:
                                new_transitions[plan.state_id] = []
                            new_transitions[plan.state_id].append(
                                Transition(src=plan.state_id, dst=t.dst, guard=lambda x: True)
                            )
                        else:
                            new_transitions[src].append(t)

    # Return new NFA
    return replace(
        nfa,
        states=new_states,
        transitions=new_transitions,
        history=nfa.history + (nfa,)
    )


# === Default Planner ===
def _default_rewrite_planner(
    nfa: NestedFiniteAutomaton,
    goal: Any,
    diagnosis: CriticResult
) -> RewritePlan:
    """Heuristic-based planner when no Oracle is available."""
    # Simple example: if stuck in loop, insert fallback
    if diagnosis.loop_detected:
        return RewritePlan(
            op=RewriteOp.INSERT_BEFORE,
            state_id="__fallback__",
            insert_ref=diagnosis.stuck_state,
            meta={"state": State("__fallback__", action=lambda x: ("fallback", x))}
        )
    return RewritePlan(op=RewriteOp.ADD_STATE, state_id="__retry__", meta={"state": State("__retry__")})