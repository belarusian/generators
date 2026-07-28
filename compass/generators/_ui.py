"""Terminal UI utilities for generators.

Colors, spinner, progress display. Ported from oracle- and slimmed
down for the generation loop.
"""

from __future__ import annotations

import shutil
import sys
import textwrap
import threading


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
    def cyan(cls, text):
        return f"{cls._c(cls.CYAN)}{text}{cls._c(cls.RESET)}"

    @classmethod
    def dim(cls, text):
        return f"{cls._c(cls.DIM)}{text}{cls._c(cls.RESET)}"

    @classmethod
    def bold(cls, text):
        return f"{cls._c(cls.BOLD)}{text}{cls._c(cls.RESET)}"

    @classmethod
    def magenta(cls, text):
        return f"{cls._c(cls.MAGENTA)}{text}{cls._c(cls.RESET)}"

    @classmethod
    def italic(cls, text):
        return f"{cls._c(cls.ITALIC)}{text}{cls._c(cls.RESET)}"


if not sys.stdout.isatty():
    Colors.disable()


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------


class Spinner:
    """Animated braille spinner for long-running operations."""

    FRAMES = "|/-\\"

    def __init__(self, message: str = "Thinking"):
        self.message = message
        self._stop = threading.Event()
        self._thread = None

    def _animate(self):
        if not sys.stdout.isatty():
            self._stop.wait()
            return
        idx = 0
        while not self._stop.is_set():
            frame = self.FRAMES[idx % len(self.FRAMES)]
            sys.stdout.write(f"\r  {Colors.cyan(frame)} {Colors.italic(self.message)}")
            sys.stdout.flush()
            idx += 1
            self._stop.wait(0.1)
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


_spinner = None


def start_spinner(message: str = "Thinking"):
    """Start a global spinner (suppressed in non-TTY / pytest)."""
    global _spinner
    if not sys.stdout.isatty() or "pytest" in sys.modules:
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


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def show_step(label: str, detail: str = "", elapsed: float | None = None):
    """Show a step in progress."""
    time_str = f" {elapsed:.1f}s" if elapsed is not None else ""
    if detail:
        print(f"  {Colors.dim('*')} {label}{time_str} {Colors.dim(detail)}")
    else:
        print(f"  {Colors.dim('*')} {label}{time_str}")


def show_result(success: bool, summary: str):
    """Show a result line."""
    marker = Colors.green("+") if success else Colors.red("x")
    width = _term_width()
    first_line = summary.split("\n")[0]
    print(f"  [{marker}] {_truncate(first_line, width - 8)}")


def show_answer(text: str):
    """Show a final answer."""
    print()
    for line in text.splitlines():
        print(f"  {line}")
    print()


def show_facts(facts: list[dict]):
    """Show a list of facts."""
    for f in facts:
        marker = Colors.green("+") if f.get("fact_type") != "error" else Colors.red("x")
        name = f.get("name", "?")
        val = str(f.get("value", ""))
        if len(val) > 120:
            val = val[:117] + "..."
        print(f"  [{marker}] {name}: {val}")


def show_separator(label: str | None = None):
    """Show a separator line."""
    width = _term_width()
    if label:
        pad = (width - len(label) - 2) // 2
        left = "\u2500" * max(1, pad)
        right = "\u2500" * max(1, width - pad - len(label) - 2)
        print(f"\n{Colors.dim(left)} {label} {Colors.dim(right)}")
    else:
        print(f"\n{Colors.dim(chr(0x2500) * width)}")


def show_duration(seconds: float):
    """Show elapsed time."""
    if seconds < 60:
        show_separator(f"Worked for {seconds:.1f}s")
    else:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        show_separator(f"Worked for {mins}m {secs}s")
