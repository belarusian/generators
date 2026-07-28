"""
Input utilities for compass.

Functions for reading user input with line continuation,
bracketed paste detection, and parsing utilities.

Uses prompt_toolkit for proper terminal handling like Claude Code.
"""

import hashlib
import tempfile
from datetime import datetime
from typing import Optional, Protocol, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style
import os


class MemoryWithAttachments(Protocol):
    """Protocol for memory objects that support attachments."""
    def add_attachment(self, content: str) -> int: ...


def grab_clipboard_image() -> Optional[Tuple[str, str]]:
    """
    Check clipboard for image and save to temp file.

    Returns:
        Tuple of (temp_file_path, media_type) if image found, None otherwise.
    """
    try:
        from PIL import ImageGrab

        img = ImageGrab.grabclipboard()
        if img is None:
            return None

        # Save to temp file
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = os.path.join(temp_dir, f"compass_paste_{timestamp}.png")
        img.save(temp_path, "PNG")

        return (temp_path, "image/png")
    except ImportError:
        # Pillow not installed or ImageGrab not available (Linux without display)
        return None
    except Exception:
        # Clipboard doesn't have image or other error
        return None


# Module state
_last_input_was_paste = False
_session: Optional[PromptSession] = None

# Style for the prompt
_style = Style.from_dict({
    'prompt': '#00aa00 bold',  # Green prompt
})


def _get_session() -> PromptSession:
    """Get or create the prompt session with history."""
    global _session
    if _session is None:
        # Store history in ~/.compass_history
        history_file = os.path.expanduser("~/.compass_history")
        _session = PromptSession(
            history=FileHistory(history_file),
            style=_style,
            enable_history_search=True,  # Ctrl+R to search history
        )
    return _session


def read_line(prompt: str = "") -> Optional[str]:
    """Get input with the compass prompt. Use \\ at end of line to continue.

    Args:
        prompt: Optional prompt text to display after '>'

    Returns:
        User input with line continuations joined, or None on Ctrl+C/EOF
    """
    print()
    try:
        lines = []
        current_prompt = f"> {prompt}"
        while True:
            line = input(current_prompt)
            if line.rstrip().endswith("\\"):
                # Line continuation - strip the backslash and continue
                lines.append(line.rstrip()[:-1])
                current_prompt = "  "  # Indent continuation lines
            else:
                lines.append(line)
                break
        return "\n".join(lines).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def read_with_paste(prompt: str = "", memory: Optional[MemoryWithAttachments] = None) -> Optional[str]:
    """
    Get input with bracketed paste mode detection using prompt_toolkit.

    Features:
    - Bracketed paste mode: cleanly detects pasted vs typed content
    - Command history: up/down arrows, persisted to ~/.compass_history
    - History search: Ctrl+R to search previous commands
    - Line editing: Ctrl+A/E/K/U etc.
    - Line continuation: use \\ at end of line

    Args:
        prompt: Optional prompt text to display after '>'
        memory: Optional memory object (unused, kept for API compat)

    Returns:
        User input or None on Ctrl+C/EOF.
    """
    global _last_input_was_paste

    print()

    # Track paste state
    paste_detected = False
    paste_content = []

    # Key bindings for paste detection and multiline input
    bindings = KeyBindings()

    @bindings.add('c-j')  # Ctrl+J inserts newline
    def insert_newline(event):
        """Insert newline without submitting."""
        event.current_buffer.insert_text('\n')

    @bindings.add('c-i')  # Ctrl+I pastes image from clipboard
    def paste_image(event):
        """Check clipboard for image and insert reference."""
        nonlocal paste_detected, paste_content
        result = grab_clipboard_image()
        if result:
            path, _ = result
            paste_detected = True
            paste_content.append(f"@{path}")
            event.current_buffer.insert_text(f"[Image from clipboard]")
            print(f"\n  Saved clipboard image to: {path}")
        else:
            print("\n  No image in clipboard")

    @bindings.add(Keys.BracketedPaste)
    def handle_paste(event):
        """Insert pasted content directly into the buffer."""
        nonlocal paste_detected, paste_content
        paste_detected = True
        # Normalize line endings (Windows \r\n -> \n, old Mac \r -> \n)
        normalized = event.data.replace('\r\n', '\n').replace('\r', '\n')
        paste_content.append(normalized)
        event.current_buffer.insert_text(normalized)

    try:
        session = _get_session()
        lines = []
        current_prompt = f"> {prompt}" if prompt else "> "

        while True:
            # Reset paste tracking
            paste_detected = False
            paste_content = []

            result = session.prompt(
                current_prompt,
                key_bindings=bindings,
            )

            if paste_detected:
                _last_input_was_paste = True
                return result.strip()

            # Regular input - check for line continuation
            if result.rstrip().endswith("\\"):
                lines.append(result.rstrip()[:-1])
                current_prompt = "  "
            else:
                lines.append(result)
                break

        _last_input_was_paste = False
        return "\n".join(lines).strip()

    except (EOFError, KeyboardInterrupt):
        print()
        return None


def was_last_input_paste() -> bool:
    """Check if the last input was from a paste operation."""
    return _last_input_was_paste


def parse_int(text: Optional[str], default: int) -> int:
    """Extract an integer from free-form input.

    Pure function: extracts digits from text and parses as int.

    Args:
        text: Input text that may contain an integer
        default: Value to return if no digits found

    Returns:
        Extracted integer or default
    """
    if not text:
        return default
    digits = "".join(filter(str.isdigit, text))
    try:
        return int(digits) if digits else default
    except Exception:
        return default


def create_seed(origin: str, intention: Optional[str] = None) -> int:
    """Create a deterministic seed from origin and intention.

    Pure function: same inputs always produce same output.

    Args:
        origin: Origin string (e.g., location name)
        intention: Optional intention/purpose string

    Returns:
        64-bit integer seed
    """
    now = datetime.now()
    seed_material = now.isoformat()
    if origin:
        seed_material += origin
    if intention:
        seed_material += intention
    seed_material += str(now.year * now.month * now.day)
    hash_obj = hashlib.sha256(seed_material.encode())
    return int(hash_obj.hexdigest()[:16], 16)


# --- Backward compatibility aliases ---
# These maintain the old names for existing code

get_input = read_with_paste  # captures multiline pastes
get_input_with_paste_detection = read_with_paste
parse_int_from_text = parse_int
