"""Thread-safe status line for showing progress during execution."""

import sys
import threading
from typing import Optional
from contextlib import contextmanager


class StatusLine:
    """Thread-safe status line that can coexist with streaming output."""

    _instance: Optional["StatusLine"] = None

    def __init__(self):
        self._message: Optional[str] = None
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "StatusLine":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set(self, message: str) -> None:
        """Set status message."""
        with self._lock:
            self._message = message
            self._redraw()

    def clear(self) -> None:
        """Clear status message."""
        with self._lock:
            if self._message:
                # Clear the line
                sys.stdout.write("\033[2K\r")
                sys.stdout.flush()
            self._message = None

    def _redraw(self) -> None:
        """Redraw status line (call while holding lock)."""
        if self._message:
            # Save cursor, newline, clear line, print status, restore cursor
            sys.stdout.write(f"\033[s\n\033[2K  {self._message}\033[u")
            sys.stdout.flush()

    def get_message(self) -> Optional[str]:
        """Get current message (for testing)."""
        with self._lock:
            return self._message

    @contextmanager
    def status(self, message: str):
        """Context manager for temporary status."""
        self.set(message)
        try:
            yield
        finally:
            self.clear()


# Global convenience functions


def show_status(message: str) -> None:
    """Show a status message."""
    StatusLine.get().set(message)


def clear_status() -> None:
    """Clear the status message."""
    StatusLine.get().clear()


@contextmanager
def status_context(message: str):
    """Context manager for temporary status."""
    with StatusLine.get().status(message):
        yield


# Integration helper for thinking stream


def redraw_after_chunk() -> None:
    """
    Call after printing a thinking chunk to redraw status.

    Can be integrated into show_thinking_stream in ui.py:
    After printing chunk, call status.redraw_after_chunk()
    """
    line = StatusLine.get()
    with line._lock:
        if line._message:
            line._redraw()
