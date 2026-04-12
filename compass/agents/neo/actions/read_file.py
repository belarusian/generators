"""
ReadFileAction handlers - singledispatch registration.

Type-based dispatch for read_file action: display, validate, execute, extract_learnings.
"""

import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from compass.core.content import preview_head_tail

from compass.agents.neo.types import ReadFileAction
from compass.agents.neo.dispatch import display, validate, execute, extract_learnings, action_key, hint, display_name, content_field

if TYPE_CHECKING:
    from compass.agents.neo.types import ActionTarget, ExecutionContext
    from compass.agents.neo.memory import Learning


@content_field.register(ReadFileAction)
def _(action) -> str:
    return "path"


DEFAULT_READ_LIMIT = 202  # Adaptive: files <= this are read in full


# --- Display ---

@display.register(ReadFileAction)
def _(action: ReadFileAction) -> "ActionTarget":
    """Get display info for ReadFileAction."""
    from compass.agents.neo.types import ActionTarget

    path = action.path
    offset_info = f" (from line {action.offset})" if action.offset else ""
    limit_info = f" ({action.limit} lines)" if action.limit else ""
    display_str = f"read {path}{offset_info}{limit_info}"

    return ActionTarget(target=path, display=display_str, content=None)


# --- Validation ---

@validate.register(ReadFileAction)
def _(
    action: ReadFileAction,
    project_path: str = ".",
    files_read: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate read_file action.

    Returns (is_valid, error_message).

    Required fields:
    - path: File path to read (relative to project root, or absolute)

    Optional fields:
    - offset: Start reading at this line (0-based). Default 0.
    - limit: Read this many lines. Default 0 (all lines).

    Results appear in FILES READ section of context. If a file is already
    in FILES READ, you don't need to read it again - proceed with your task.

    If read_file fails, check:
    - Path exists (use search/grep to find correct path)
    - File is not binary
    """
    path = action.path

    if not path:
        return False, "Missing required field: path"



    return True, None


# --- Execution ---

@execute.register(ReadFileAction)
def _(action: ReadFileAction, project_path: str, ctx: "ExecutionContext" = None) -> Tuple[bool, str]:
    """
    Execute read_file action.

    Returns (success, message).

    Args:
        action: ReadFileAction
        project_path: Project root directory
        ctx: ExecutionContext (unused, for uniform signature)
    """
    path = action.path
    offset = action.offset or 0
    limit = action.limit or 0

    # Resolve relative path
    if path and not os.path.isabs(path):
        path = os.path.join(project_path, path)

    try:
        with open(path) as f:
            lines = f.readlines()

        total_lines = len(lines)

        # Validate offset - can't start past end of file
        if offset >= total_lines:
            return True, f"[End of file - only {total_lines} lines total]"

        # Adaptive read: full for small files, head+tail for large ones
        if limit > 0:
            # Explicit limit requested -- honour it
            start_line = offset
            selected = lines[start_line:start_line + limit]
        elif offset > 0:
            # Explicit offset -- read from there, cap at DEFAULT_READ_LIMIT
            start_line = offset
            selected = lines[start_line:start_line + DEFAULT_READ_LIMIT]
        elif total_lines <= DEFAULT_READ_LIMIT:
            # Small file -- read in full
            start_line = 0
            selected = lines
        else:
            # Large file -- head + tail (imports + structure + entry points)
            head_n = 120
            tail_n = 80
            head = lines[:head_n]
            tail = lines[-tail_n:]
            gap = total_lines - head_n - tail_n
            separator = [f"    ... [{gap} lines omitted — use offset/limit to read specific sections] ...\n"]
            start_line = 0
            selected = head + separator + tail

        # Add line numbers for easier patching
        # Format: "line 42: code here" - explicit prefix, colon separator
        numbered = []
        if total_lines > DEFAULT_READ_LIMIT and limit == 0 and offset == 0:
            # Head+tail mode: number head normally, skip separator, number tail from file end
            head_n = 120
            tail_n = 80
            for i, line in enumerate(selected):
                if i < head_n:
                    numbered.append(f"line {i + 1}: {line.rstrip()}")
                elif i == head_n:
                    numbered.append(line.rstrip())  # separator line, no number
                else:
                    tail_idx = i - head_n - 1  # index within tail
                    file_line = total_lines - tail_n + tail_idx + 1
                    numbered.append(f"line {file_line}: {line.rstrip()}")
        else:
            for i, line in enumerate(selected):
                line_num = start_line + i + 1  # 1-based
                numbered.append(f"line {line_num}: {line.rstrip()}")

        content = "\n".join(numbered)
        if offset > 0 or limit > 0:
            header = f"[Lines {start_line+1}-{min(start_line+len(selected), total_lines)} of {total_lines}]\n"
            content = header + content
        elif total_lines > DEFAULT_READ_LIMIT:
            header = f"[Lines 1-{head_n} + {total_lines-tail_n+1}-{total_lines} of {total_lines}]\n"
            content = header + content
        return True, content
    except FileNotFoundError:
        return False, f"File not found: {path}"
    except Exception as e:
        return False, f"Failed to read {path}: {e}"


# --- Learning Extraction ---

@extract_learnings.register(ReadFileAction)
def _(
    action: ReadFileAction,
    success: bool,
    result: str,
    reflect,
) -> List["Learning"]:
    """Extract learnings from read_file action."""
    path = action.path

    prompt = f"""Action: read_file
Path: {path}
Success: {success}
Result:
{preview_head_tail(result, max_lines=64)}

What did we learn from this?"""

    return [reflect(prompt)]


# --- Action Key ---

@action_key.register(ReadFileAction)
def _(action: ReadFileAction) -> tuple:
    """Hashable key for ReadFileAction comparison."""
    return ("read_file", action.path, action.offset, action.limit)


@hint.register(ReadFileAction)
def _(action: ReadFileAction) -> str:
    """Hint for Critic when read_file fails."""
    return "Read file contents. Check path exists."


@display_name.register(ReadFileAction)
def _(action: ReadFileAction) -> str:
    """Human-friendly name for UI."""
    return "Read"