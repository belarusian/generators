"""Neo agent -- autonomous agent that reads files, runs commands, edits code.

Plan guide: inputs should include {"request": "the task"} or {"prompt": "the question"}.
Falls back to step description if no explicit request.
"""

CYCLE_BREAKING = True


def run(step, resolved_inputs, workspace):
    """Delegate to the Neo agent."""
    from compass.agents.neo.user_query import run as _run
    return _run(step, resolved_inputs, workspace)
