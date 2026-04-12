"""
UI utilities for Oracle interaction.

Clean, Claude Code-style output formatting.
"""

import os
import re
import shutil
import sys
import time
import textwrap
import threading

# Auto-approve state
_auto_approve = False

# Regex patterns for text sanitization
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

def _sanitize(text: str) -> str:
    """Remove ANSI escape sequences and control characters from text.

    Keeps only printable characters, newlines, and tabs.
    """
    # Strip ANSI escape sequences
    text = _ANSI_RE.sub("", text)
    # Strip control characters
    text = _CTRL_RE.sub("", text)
    return text


class Colors:
    """ANSI color codes for terminal output."""
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    RESET = "\033[0m"

    _enabled = True

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        cls._enabled = False

    @classmethod
    def _c(cls, code):
        return code if cls._enabled else ""

    @classmethod
    def red(cls, text):
        return f"{cls._c(cls.RED)}{text}{cls._c(cls.RESET)}"

    @classmethod
    def green(cls, text):
        return f"{cls._c(cls.GREEN)}{text}{cls._c(cls.RESET)}"

    @classmethod
    def yellow(cls, text):
        return f"{cls._c(cls.YELLOW)}{text}{cls._c(cls.RESET)}"

    @classmethod
    def blue(cls, text):
        return f"{cls._c(cls.BLUE)}{text}{cls._c(cls.RESET)}"

    @classmethod
    def magenta(cls, text):
        return f"{cls._c(cls.MAGENTA)}{text}{cls._c(cls.RESET)}"

    @classmethod
    def dim(cls, text):
        return f"{cls._c(cls.DIM)}{text}{cls._c(cls.RESET)}"

    @classmethod
    def bold(cls, text):
        return f"{cls._c(cls.BOLD)}{text}{cls._c(cls.RESET)}"

    @classmethod
    def cyan(cls, text):
        return f"{cls._c(cls.CYAN)}{text}{cls._c(cls.RESET)}"

    @classmethod
    def italic(cls, text):
        return f"{cls._c(cls.ITALIC)}{text}{cls._c(cls.RESET)}"


# Check if stdout is a TTY
if not sys.stdout.isatty():
    Colors.disable()


class Spinner:
    """Animated spinner for long-running operations."""

    # Braille spinner frames (smooth animation)
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str = "Thinking"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = None

    def _animate(self):
        """Animation loop running in background thread."""
        if not sys.stdout.isatty():
            # No animation when stdout isn't a terminal (pytest, pipes, etc.)
            self._stop_event.wait()
            return
        idx = 0
        while not self._stop_event.is_set():
            frame = self.FRAMES[idx % len(self.FRAMES)]
            # \r moves cursor to start of line, overwrites
            text = f"\r{Colors.dim('•')} {Colors.cyan(frame)} {Colors.italic(self.message)}"
            sys.stdout.write(text)
            sys.stdout.flush()
            idx += 1
            self._stop_event.wait(0.08)  # ~12 FPS
        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()

    def start(self):
        """Start the spinner animation."""
        if self._thread is not None:
            return  # Already running
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the spinner animation."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=0.5)
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# Global spinner instance for convenience
_spinner = None


def start_spinner(message: str = "Thinking"):
    """Start a global spinner.

    In DEBUG mode, shows static indicator instead (streaming thoughts follow).
    Suppressed entirely in non-interactive mode (pytest, pipes, etc.).
    """
    global _spinner
    if not sys.stdout.isatty() or 'pytest' in sys.modules:
        return  # No spinner in non-interactive mode
    # In DEBUG mode, show static indicator - streaming thoughts take over
    if os.getenv("DEBUG"):
        print(f"\n{Colors.dim('•')} {Colors.italic(message)}{Colors.dim('...')}")
        return
    if _spinner is not None:
        _spinner.stop()
    _spinner = Spinner(message)
    _spinner.start()


def stop_spinner():
    """Stop the global spinner."""
    global _spinner
    if _spinner is not None:
        _spinner.stop()
        _spinner = None


def _term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def _wrap_text(text: str, width: int) -> list:
    if width <= 0:
        return [text]
    return textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False) or [""]


def show_thinking(message: str = "Thinking"):
    """Show a thinking/processing indicator."""
    print(f"\n{Colors.dim('•')} {Colors.italic(message)}{Colors.dim('...')}")


# Track streaming thought state
_thinking_stream_started = False
_thinking_color = None  # None=dim gray, or Colors.BLUE/CYAN/YELLOW


def set_thinking_color(color: str = None):
    """Set color for thinking stream. None=dim gray, or 'planner'/'actor'/'critic'."""
    global _thinking_color
    if color == "planner":
        _thinking_color = Colors.BLUE
    elif color == "actor":
        _thinking_color = Colors.CYAN
    elif color == "critic":
        _thinking_color = Colors.YELLOW
    else:
        _thinking_color = None


def start_thinking_stream():
    """Start a new thinking stream (resets state)."""
    global _thinking_stream_started
    stop_spinner()  # Stop spinner before streaming thinking
    _thinking_stream_started = False
    print()  # Newline before thinking


def show_thinking_stream(chunk: str):
    """Print thinking chunk inline (real-time streaming)."""
    global _thinking_stream_started, _thinking_color
    if 'pytest' in sys.modules:
        return  # Suppress streaming output during tests
    # Add • prefix on first chunk
    if not _thinking_stream_started:
        color_code = _thinking_color if _thinking_color and Colors._enabled else ""
        print(f"{Colors.dim('•')} {color_code}{Colors.ITALIC if Colors._enabled else ''}", end="", flush=True)
        _thinking_stream_started = True
    # Print in color (or dim if no color set), no newline, flush immediately
    if _thinking_color and Colors._enabled:
        print(_sanitize(chunk), end="", flush=True)
    else:
        print(Colors.dim(_sanitize(chunk)), end="", flush=True)


def make_thinking_streamer(role: str):
    """Create a thinking stream callback with captured color (thread-safe).

    Use this instead of show_thinking_stream when streaming from background threads.
    The color is captured at creation time, not read from global state.
    """
    color_map = {
        "planner": Colors.BLUE,
        "actor": Colors.CYAN,
        "critic": Colors.YELLOW,
    }
    color = color_map.get(role)
    started = [False]  # Mutable container for closure

    def streamer(chunk: str):
        if not started[0]:
            color_code = color if color and Colors._enabled else ""
            print(f"{Colors.dim('•')} {color_code}{Colors.ITALIC if Colors._enabled else ''}", end="", flush=True)
            started[0] = True
        if color and Colors._enabled:
            print(_sanitize(chunk), end="", flush=True)
        else:
            print(Colors.dim(_sanitize(chunk)), end="", flush=True)

    return streamer


def end_thinking_stream():
    """End thinking stream with reset and newline."""
    global _thinking_stream_started
    if _thinking_stream_started and Colors._enabled:
        print(Colors.RESET, end="")  # Close italic
    _thinking_stream_started = False
    print()


def show_step(step_num: int, total: int, description: str, elapsed: float = None):
    """Show step progress with Unicode bar."""
    bar_width = 10
    if total > 0:
        filled = int((step_num / total) * bar_width)
    else:
        filled = 0
    bar = "▓" * filled + "░" * (bar_width - filled)
    width = _term_width()
    time_str = f" {elapsed:.1f}s" if elapsed is not None else ""
    # Wrap step description to 2 lines
    prefix_len = 35 + len(time_str)  # "• Step X/Y [▓▓▓▓░░░░░░] Xs "
    desc_lines = _wrap_text(_sanitize(description), max(20, width - prefix_len))[:2]
    print(f"\n{Colors.dim('•')} Step {step_num}/{total} [{Colors.cyan(bar)}]{time_str} {desc_lines[0]}")
    for cont in desc_lines[1:]:
        print(f"  {' ' * (prefix_len - 2)}{_sanitize(cont)}")


def show_action(name: str, target: str, reasoning: str = None):
    """Show an action being executed. Name is the display name (e.g. 'Read', 'Write')."""
    # Sanitize inputs to remove ANSI escape sequences and control characters
    name = _sanitize(name)
    target = _sanitize(target)
    if reasoning:
        reasoning = _sanitize(reasoning)
    width = _term_width()
    inline = f"{name}({target})"
    if len(inline) > max(20, width - 6):
        print(f"• {Colors.cyan(name)}")
        for line in _wrap_text(target, max(20, width - 8)):
            print(f"  {Colors.dim('└')} {line}")
    else:
        print(f"• {Colors.cyan(name)}({target})")
    if reasoning:
        # Extra dim: DIM + ITALIC for reasoning (clearly secondary)
        for i, line in enumerate(_wrap_text(reasoning, max(20, width - 6))):
            prefix = Colors.dim('└') if i == 0 else Colors.dim('│')
            dimmed = Colors.italic(Colors.dim(line))
            print(f"  {prefix} {dimmed}")


def show_result(success: bool, summary: str, inline: bool = False):
    """Show action result."""
    marker = Colors.green("✓") if success else Colors.red("✗")
    width = _term_width()
    lines = _sanitize(summary).splitlines() or [""]
    first = _truncate(lines[0], max(20, width - 8))
    prefix = " " if inline else "  "
    print(f"{prefix}[{marker}] {first}")
    for line in lines[1:]:
        for chunk in _wrap_text(_sanitize(line), max(20, width - 8)):
            print(f"{prefix}    {chunk}")


def show_plan(plan: dict, verbose: bool = False):
    """Display a plan for review (clean indented format)."""
    summary = plan.get("summary", "No summary")
    steps = plan.get("steps", [])
    files = plan.get("files_affected", [])
    risks = plan.get("risks", [])

    width = _term_width()
    content_width = min(width - 6, 76)

    # Show auto-approve indicator if active
    auto_indicator = f"{Colors.yellow('[AUTO]')} " if _auto_approve else ""

    # Header line
    print(f"\n{Colors.dim('─')} {auto_indicator}{Colors.cyan('Plan')} {Colors.dim('─' * (content_width - 6 - len(auto_indicator)))}")

    # Summary
    summary_lines = _wrap_text(_sanitize(summary), content_width)
    for line in summary_lines:
        print(f"  {line}")
    print()

    # Steps
    if steps:
        print(f"  {Colors.bold('Steps:')}")
        for i, step in enumerate(steps, 1):
            # Handle both string steps and dict steps {step: "...", description: "..."}
            if isinstance(step, dict):
                step_text = step.get("step") or step.get("description") or str(step)
            else:
                step_text = step
            step_lines = _wrap_text(_sanitize(step_text), content_width - 5)
            print(f"    {Colors.green(str(i))}. {step_lines[0]}")
            for cont in step_lines[1:]:
                print(f"       {_sanitize(cont)}")
        print()

    # Files
    if files:
        print(f"  {Colors.bold('Files:')}")
        for f in files:
            print(f"    {Colors.dim('•')} {_sanitize(f)}")
        print()

    # Risks
    if risks:
        print(f"  {Colors.yellow('Risks:')}")
        for risk in risks:
            risk_lines = _wrap_text(_sanitize(risk), content_width - 5)
            print(f"    {Colors.yellow('⚠')} {Colors.dim(risk_lines[0])}")
            for cont in risk_lines[1:]:
                print(f"       {Colors.dim(_sanitize(cont))}")


def show_diff(path: str, patches: list, original_lines: list = None):
    """Show diff-style output for patches."""
    width = _term_width()
    inline = f"Edit({path})"
    if len(inline) > max(20, width - 6):
        print(f"\n• {Colors.cyan('Edit')}")
        for line in _wrap_text(_sanitize(path), max(20, width - 8)):
            print(f"  {Colors.dim('L')} {line}")
    else:
        print(f"\n• {Colors.cyan('Edit')}({path})")
    print(f"  {Colors.dim('L')} {len(patches)} change(s)")

    if not original_lines:
        return

    for patch in patches[:5]:  # Limit displayed patches
        line_num = patch.get("line", 0)
        delete = patch.get("delete", 0)
        insert = patch.get("insert", "")

        print()
        # Show removed lines
        for i in range(delete):
            idx = line_num - 1 + i
            if idx < len(original_lines):
                old_line = _sanitize(original_lines[idx].rstrip()[:80])
                print(f"    {line_num + i:4}  {Colors.red('-')} {Colors.red(old_line)}")

        # Show added lines
        if insert:
            for i, new_line in enumerate(insert.split('\n')[:10]):
                print(f"    {line_num + i:4}  {Colors.green('+')} {Colors.green(_sanitize(new_line[:80]))}")

    if len(patches) > 5:
        print(f"\n  ... and {len(patches) - 5} more changes")


def colorize_diff(text: str) -> str:
    """Colorize unified diff output (git-style).

    - Lines starting with '-' (removals) -> red
    - Lines starting with '+' (additions) -> green
    - Lines starting with '@@' (hunks) -> cyan
    - File headers (--- / +++) -> bold
    """
    lines = []
    for line in text.splitlines():
        if line.startswith('---') or line.startswith('+++'):
            lines.append(Colors.bold(line))
        elif line.startswith('@@'):
            lines.append(Colors.cyan(line))
        elif line.startswith('-'):
            lines.append(Colors.red(line))
        elif line.startswith('+'):
            lines.append(Colors.green(line))
        else:
            lines.append(line)
    return '\n'.join(lines)


def print_diff(text: str):
    """Print colorized diff output."""
    print(colorize_diff(_sanitize(text)))


def show_thought(text: str):
    """Show model reasoning/thoughts in italic."""
    text = _sanitize(text)
    width = _term_width()
    print()
    lines = _wrap_text(text, max(20, width - 4))
    for i, line in enumerate(lines):
        prefix = Colors.dim('•') if i == 0 else ' '
        print(f"{prefix} {Colors.italic(Colors.dim(line))}")


def show_separator(label: str = None):
    """Show a separator line, optionally with a label."""
    width = _term_width()
    if label:
        label = _sanitize(label)
        padding = (width - len(label) - 4) // 2
        line = "─" * max(1, padding)
        print(f"\n{Colors.dim(line)} {label} {Colors.dim(line)}")
    else:
        print(f"\n{Colors.dim('─' * min(width, 60))}")


def show_duration(seconds: float):
    """Show a separator line with duration."""
    if seconds < 60:
        label = f"Worked for {seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        label = f"Worked for {mins}m {secs}s"
    show_separator(_sanitize(label))


def show_answer(data):
    """Show the final answer after execution.

    Args:
        data: String (legacy), dict, or AnswerResponse dataclass
    """
    print()
    print(f"• {Colors.cyan('Answer')}")
    print()

    # Handle legacy string, dict, or AnswerResponse dataclass
    if isinstance(data, str):
        text = data
        references = []
        next_steps = []
    elif isinstance(data, dict):
        text = data.get("answer", "")
        references = data.get("references", [])
        next_steps = data.get("next_steps", [])
    else:
        # AnswerResponse dataclass - use attribute access
        text = data.answer or ""
        references = data.references or []
        next_steps = data.next_steps or []

    # Main answer text
    for line in (text or "").splitlines() or [""]:
        print(f"  {_sanitize(line)}")

    # Show references if any
    if references:
        print()
        print(f"  {Colors.dim('References:')}")
        for ref in references:
            # Handle dict or AnswerReference dataclass
            if isinstance(ref, dict):
                file = ref.get("file", "")
                line_num = ref.get("line", "")
                note = ref.get("note", "")
            else:
                file = ref.file
                line_num = ref.line
                note = ref.note or ""
            loc = f"{file}:{line_num}" if line_num else file
            if note:
                print(f"    {Colors.dim('-')} {_sanitize(loc)} {Colors.dim(f'({_sanitize(note)})')}")
            else:
                print(f"    {Colors.dim('-')} {_sanitize(loc)}")

    # Show next steps if any - styled as oracle's vision of future paths
    if next_steps:
        print()
        print(f"  {Colors.dim(Colors.italic(_sanitize('~ What comes next ~')))}")
        for step in next_steps:
            print(f"    {Colors.dim(Colors.italic(_sanitize(step)))}")


def set_auto_approve(enabled: bool):
    """Set auto-approve mode."""
    global _auto_approve
    _auto_approve = enabled


def is_auto_approve() -> bool:
    """Check if auto-approve is enabled."""
    return _auto_approve


def approval_prompt() -> str:
    """Show approval prompt and get response (inline)."""
    import select
    import sys
    global _auto_approve

    print()

    # Auto-approve mode
    if _auto_approve:
        print(_sanitize(f"  {Colors.yellow('[AUTO]')} approving... (Ctrl+C to stop)"))
        time.sleep(0.3)
        return 'a'

    # Flush any buffered input (e.g., from paste) before prompting
    try:
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
    except (OSError, ValueError):
        pass

    try:
        # Inline prompt with colored options
        # [!] toggles auto-approve mode
        prompt = f"  [{Colors.green('a')}]pprove  [{Colors.red('r')}]eject  [{Colors.yellow('m')}]odify  [{Colors.blue('d')}]etails  [{Colors.dim('!')}]auto  [{Colors.dim('q')}]uit > "
        response = input(prompt).strip().lower()
        return response
    except (EOFError, KeyboardInterrupt):
        # Ctrl+C disables auto-approve and returns to prompt
        if _auto_approve:
            _auto_approve = False
            print(f"\n  {Colors.dim('Auto-approve disabled.')}")
            return approval_prompt()  # Re-prompt
        print()
        return "q"
