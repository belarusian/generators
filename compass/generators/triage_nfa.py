"""
NFA-based Workflow for Issue Triage State Machine.

This module provides an NFA-based workflow for issue triage state machine in
generators. The issue_agent bash call is the terminal state in the generators
workflow, with TOML parsing via generators trusted code layer.
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

from compass.core.nfa import NFARunner, NFAResult


def parse_triage_result(content: str) -> Dict[str, Any]:
    """Parse triage result from TOML key=value format."""
    result: Dict[str, Any] = {}
    for line in content.split('\n'):
        line = line.strip()
        if not line or '=' not in line:
            continue
        key, value = line.split('=', 1)
        result[key.lower().strip()] = value.strip()
    return result


class TriageState(Enum):
    """States for the issue triage NFA."""
    INIT = "init"
    ANALYZE_CONTEXT = "analyze_context"
    CALL_ISSUE_AGENT = "call_issue_agent"
    PARSE_TRIAGE_RESULT = "parse_triage_result"
    DONE = "done"
    FAILED = "failed"


@dataclass
class TriageContext:
    """Context for issue triage NFA."""
    issue_id: str
    issue_title: str
    issue_body: str
    sender: str
    triage_toml_output: Optional[str] = None
    triage_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def analyze_context_transition(ctx: TriageContext) -> tuple[TriageState, TriageContext]:
    """Transition from INIT to ANALYZE_CONTEXT or CALL_ISSUE_AGENT."""
    # In production, this would analyze the issue context via DB tools
    return (TriageState.CALL_ISSUE_AGENT, ctx)


def call_issue_agent_transition(ctx: TriageContext) -> Tuple[TriageState, TriageContext]:
    """Transition that calls issue_agent bash script as terminal state."""
    # In production, this would invoke the issue_agent bash script:
    # issue_agent --subject "{title}" --body "{body}" --output {work_dir}/triage_result.json
    # For now, simulate the TOML output
    toml_output = """PRIORITY=2
LABELS=bug,api
COMPLEXITY=small
DISCARD=false
REPO=belarusian/compsci.boutique
"""
    ctx.triage_toml_output = toml_output
    return (TriageState.PARSE_TRIAGE_RESULT, ctx)


def parse_triage_result_transition(ctx: TriageContext) -> Tuple[TriageState, TriageContext]:
    """Transition that parses TOML triage result via trusted code layer."""
    if not ctx.triage_toml_output:
        ctx.error = "No triage result available"
        return (TriageState.FAILED, ctx)
    
    try:
        # Parse TOML output via trusted code layer (zero regex)
        parsed_result = parse_triage_result(ctx.triage_toml_output)
        ctx.triage_result = parsed_result
    except Exception as e:
        ctx.error = f"Failed to parse triage result: {e}"
        return (TriageState.FAILED, ctx)
    
    return (TriageState.DONE, ctx)


def create_triage_nfa() -> NFARunner[TriageState, TriageContext]:
    """
    Create the issue triage NFA state machine.
    
    Returns:
        NFARunner configured for issue triage workflow
    """
    transitions = {
        TriageState.INIT: analyze_context_transition,
        TriageState.ANALYZE_CONTEXT: analyze_context_transition,
        TriageState.CALL_ISSUE_AGENT: call_issue_agent_transition,
        TriageState.PARSE_TRIAGE_RESULT: parse_triage_result_transition,
    }
    
    return NFARunner(
        transitions=transitions,
        initial_state=TriageState.INIT,
        terminal_states={TriageState.DONE, TriageState.FAILED},
        success_states={TriageState.DONE},
    )
