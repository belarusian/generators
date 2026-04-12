"""
Schemas for code mode.

JSON Schemas for LLM response validation.
Action types are defined as dataclasses in types.py.
Hints and display names are colocated with action handlers via singledispatch in actions/*.py.
"""

# Driver approval schemas - used by ClaudeDriver to get structured decisions
# These are used when Claude is in the driver seat (via /claude command)

PLAN_APPROVAL_SCHEMA = {
    "type": "object",
    "description": "Claude's decision on whether to approve, modify, or reject a plan",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "modified"],
            "description": "approved=proceed as-is, rejected=replan from scratch, modified=use modified_steps instead"
        },
        "reason": {
            "type": "string",
            "description": "Brief explanation of the decision"
        },
        "modified_steps": {
            "type": "array",
            "description": "If decision=modified, the corrected steps to use instead",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "string"},
                },
                "required": ["step"]
            }
        },
    },
    "required": ["decision"],
}

ACTION_APPROVAL_SCHEMA = {
    "type": "object",
    "description": "Claude's decision on whether to approve, modify, or reject an action",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "modified"],
            "description": "approved=execute as-is, rejected=skip/fail, modified=use modified_action instead"
        },
        "reason": {
            "type": "string",
            "description": "Brief explanation of the decision"
        },
        "modified_action": {
            "type": "object",
            "description": "If decision=modified, the corrected action to execute instead"
        },
    },
    "required": ["decision"],
}

# Schema for when Claude makes the Critic's decision (escalation from ask_claude)
CLAUDE_CRITIC_DECISION_SCHEMA = {
    "type": "object",
    "description": "Claude's decision when Critic escalates via ask_claude",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["replan", "done"],
            "description": "replan=revise the plan, done=stop and return to user"
        },
        "explanation": {
            "type": "string",
            "description": "Reasoning for the decision"
        },
        "feedback": {
            "type": "string",
            "description": "Specific feedback for Planner if action=replan"
        },
    },
    "required": ["action", "explanation"],
}