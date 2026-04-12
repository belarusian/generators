"""
Content Truncation - Clean formatting for long content.

Pure functions for truncating long content with readable formatting.
No expansion machinery - model uses ReadFileAction if she needs more.

Usage:
    truncated = truncate_lines(content, max_lines=77, label="file")
    # Returns formatted block with header, line numbers, footer
"""

from typing import Optional


# --- Pure truncation functions ---

def format_content_block(
    content: str,
    max_lines: int = 77,
    label: str = "file",
    path: Optional[str] = None,
    offset: int = 0,
) -> str:
    """Format content as a readable block with line numbers.

    Renders like a text file - contiguous, numbered, with header/footer.
    The center is the muzzle - she orients by reading top-to-bottom.
    If she needs more, she uses ReadFileAction(offset=...).

    Format:
        [path - lines 1-77 of 250]
        line 1: content here
        line 2: more content
        ...
        line 77: last shown
        ================

    Args:
        content: Full content
        max_lines: Lines to show (contiguous from offset)
        label: Label for header if no path
        path: File path for header
        offset: Starting line (0-based)

    Returns:
        Formatted block with header, numbered lines, footer
    """
    lines = content.split('\n')
    total = len(lines)

    # Apply offset and limit
    start = offset
    end = min(offset + max_lines, total)
    selected = lines[start:end]

    # Header
    display_name = path or label
    if total <= max_lines and offset == 0:
        header = f"[{display_name} - {total} lines]"
    else:
        header = f"[{display_name} - lines {start + 1}-{end} of {total}]"

    # Number lines (like ReadFileAction)
    numbered = [f"line {start + i + 1}: {line}" for i, line in enumerate(selected)]

    # Footer
    footer = "================"

    return header + "\n" + "\n".join(numbered) + "\n" + footer


def truncate_lines(
    content: str,
    max_lines: int = 77,
    label: str = "content",
) -> str:
    """Truncate to contiguous head block with line numbers.

    Shows beginning as readable block. If she needs more,
    she uses ReadFileAction(offset=...).

    Args:
        content: Full content to truncate
        max_lines: Maximum lines to show
        label: Label for header

    Returns:
        Formatted block with header, numbered lines, footer
    """
    return format_content_block(content, max_lines=max_lines, label=label)


def truncate_chars(
    content: str,
    max_chars: int = 2000,
) -> str:
    """Truncate by character count, break at word boundary.

    Args:
        content: Full content to truncate
        max_chars: Maximum characters

    Returns:
        Truncated content with footer
    """
    if len(content) <= max_chars:
        return content

    # Break at word boundary
    preview = content[:max_chars].rsplit(' ', 1)[0]
    return f"{preview}...\n================"


def truncate_with_skeleton(
    content: str,
    max_preview_lines: int = 50,
    label: str = "file",
) -> str:
    """Truncate code file showing head as formatted block.

    Args:
        content: Full content
        max_preview_lines: Lines to show
        label: Label for header

    Returns:
        Formatted block
    """
    return format_content_block(content, max_lines=max_preview_lines, label=label)


def preview_head_tail(
    content: str,
    max_lines: int = 64,
    label: str = "content",
) -> str:
    """Preview as contiguous head block.

    Previously showed head+tail with middle cut out, but that disrupts
    orientation - the center is the muzzle. Now shows contiguous head.

    If she needs more, she uses ReadFileAction(offset=...).
    """
    return format_content_block(content, max_lines=max_lines, label=label)


def preview_head(
    content: str,
    max_lines: int = 10,
    label: str = "content",
) -> str:
    """Preview showing head as formatted block."""
    return format_content_block(content, max_lines=max_lines, label=label)


def preview_tail(
    content: str,
    max_lines: int = 10,
    label: str = "content",
) -> str:
    """Preview showing tail as formatted block."""
    lines = content.split('\n')
    total = len(lines)
    offset = max(0, total - max_lines)
    return format_content_block(content, max_lines=max_lines, label=label, offset=offset)
