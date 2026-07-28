"""
grep action - singledispatch handlers.

Registers handlers for GrepAction.

Regex pattern search using ripgrep or grep.
"""

import json
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

from compass.core.content import preview_head_tail

from compass.agents.neo.types import GrepAction, ActionTarget, ExecutionContext, Reflector
from compass.agents.neo.dispatch import display, validate, execute, extract_learnings, action_key, hint, display_name, content_field
from compass.agents.neo.memory import Learning


# =============================================================================
# Singledispatch handlers
# =============================================================================

@content_field.register(GrepAction)
def _(action): return "pattern"


@display.register(GrepAction)
def _(action: GrepAction) -> ActionTarget:
    """Get display info for grep."""
    path_info = f" in {action.path}" if action.path else ""
    return ActionTarget(
        target=action.pattern,
        display=f"grep: {action.pattern}{path_info}",
        content=None,
    )


@validate.register(GrepAction)
def _(
    action: GrepAction,
    project_path: str = ".",
    files_read: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate grep action.

    Returns (is_valid, error_message).

    Required fields:
    - pattern: Regular expression pattern to search for

    Optional fields:
    - path: Directory or file to search in. Default: project root.
    - timeout: Seconds before timeout. Default 30.

    Grep finds exact text matches using regex patterns. Use grep when:
    - You know the exact text to find (function name, variable, string)
    - You need pattern matching (def.*config, import.*json)
    - Search didn't find what you need

    Examples:
        {"action_type": "grep", "pattern": "def get_session_context"}
        {"action_type": "grep", "pattern": "import.*json", "path": "src/"}

    For semantic/conceptual search, use search action instead.
    """
    if not action.pattern:
        return False, "Missing required field: pattern"

    return True, None


@execute.register(GrepAction)
def _(action: GrepAction, project_path: str, ctx: ExecutionContext = None) -> Tuple[bool, str]:
    """
    Execute grep action using grep/ripgrep.

    Returns (success, message).

    Prefers ripgrep if available, falls back to grep.
    """
    pattern = action.pattern or ""
    path = action.path or ""
    timeout = action.timeout or 30

    # Resolve path - default to project root
    if not path:
        path = project_path
    elif not os.path.isabs(path):
        path = os.path.join(project_path, path)

    # Prefer ripgrep if available, fall back to grep
    rg = shutil.which("rg")
    grep_bin = shutil.which("grep")

    fixed = action.fixed or False

    if rg:
        cmd = [rg, "--line-number", "--no-heading", "--color=never"]
        if fixed:
            cmd.append("-F")
        cmd.extend(["-e", pattern, path])
    elif grep_bin:
        flag = "-F" if fixed else "-E"
        cmd = [grep_bin, "-rn", flag, pattern, path]
    else:
        return False, "Neither ripgrep (rg) nor grep found in PATH"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=path if os.path.isdir(path) else os.path.dirname(path),
        )

        output = result.stdout
        if result.stderr and not output:
            output = result.stderr

        if not output.strip():
            return True, f"No matches for pattern: {pattern}"

        match_count = output.count('\n')
        return True, f"Found {match_count} matches:\n{output}"

    except subprocess.TimeoutExpired:
        return False, f"Grep timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Grep failed: {e}"


@extract_learnings.register(GrepAction)
def _(
    action: GrepAction,
    success: bool,
    result: str,
    reflect: Reflector,
) -> List[Learning]:
    """
    Extract learnings from grep results.

    LLM reflects and chooses learning type - no heuristics.
    """
    from dataclasses import asdict

    action_data = asdict(action) if hasattr(action, '__dataclass_fields__') else action

    prompt = f"""Action: grep (regex search)
Input: {json.dumps(action_data)}
Success: {success}
Result:
{preview_head_tail(result, max_lines=64)}

What did we learn from this? Consider:
- Where things are located in the codebase
- Whether the pattern found what we expected
- Any corrections to our understanding"""

    return [reflect(prompt)]


@action_key.register(GrepAction)
def _(action: GrepAction) -> tuple:
    """Hashable key for grep comparison."""
    return ("grep", action.pattern, action.path)


@hint.register(GrepAction)
def _(action: GrepAction) -> str:
    """Hint for Critic when grep fails."""
    return "Regex search. Check pattern syntax."


@display_name.register(GrepAction)
def _(action: GrepAction) -> str:
    """Human-friendly name for UI."""
    return "Grep"
