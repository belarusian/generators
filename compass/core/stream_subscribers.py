"""
Concrete stream subscribers for NFA visualization.

Subscribers that consume stream events and route them to different outputs:
- TerminalSubscriber: Live CLI output (current behavior)
- JSONLSubscriber: Appends to JSONL file for replay/analytics
- CollectingSubscriber: Collects events in memory for testing/replay
- WebSocketSubscriber: Sends to web UI (stub for future)
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from compass.core.stream_types import StreamEvent, StreamEventType


class TerminalSubscriber:
    """
    Renders stream events to terminal.

    Provides live CLI output - tokens stream as they arrive,
    state changes are shown with visual indicators.

    Designed to replace scattered print() calls with centralized rendering.
    """

    def __init__(self, show_tokens: bool = True, show_states: bool = True, show_thinking: bool = True):
        self._show_tokens = show_tokens
        self._show_states = show_states
        self._show_thinking = show_thinking
        self._in_thinking = False

    def on_event(self, event: StreamEvent) -> None:
        """Handle a stream event."""
        handler = {
            StreamEventType.NFA_START: self._on_nfa_start,
            StreamEventType.NFA_END: self._on_nfa_end,
            StreamEventType.STATE_ENTER: self._on_state_enter,
            StreamEventType.STATE_EXIT: self._on_state_exit,
            StreamEventType.LLM_START: self._on_llm_start,
            StreamEventType.LLM_THINKING: self._on_llm_thinking,
            StreamEventType.LLM_END: self._on_llm_end,
            StreamEventType.TRANSITION: self._on_transition,
        }.get(event.type)

        if handler:
            handler(event)

    def on_token(self, token: str, nfa_path: tuple, state: str, instance_id: Optional[int] = None) -> None:
        """Fast path for token streaming."""
        if self._show_tokens:
            print(token, end="", flush=True)

    def _on_nfa_start(self, event: StreamEvent) -> None:
        """NFA execution started."""
        if self._show_states:
            nfa_name = event.nfa_path[-1] if event.nfa_path else "NFA"
            instance_info = f" [#{event.instance_id}]" if event.instance_id is not None else ""
            # Subtle indicator
            pass  # Could add: print(f"\n--- {nfa_name}{instance_info} ---", file=sys.stderr)

    def _on_nfa_end(self, event: StreamEvent) -> None:
        """NFA execution completed."""
        pass  # Clean exit

    def _on_state_enter(self, event: StreamEvent) -> None:
        """Entered a new state."""
        if self._show_states and event.state:
            # Visual indicator for state transitions
            pass  # Could add visual feedback

    def _on_state_exit(self, event: StreamEvent) -> None:
        """Exiting current state."""
        pass

    def _on_llm_start(self, event: StreamEvent) -> None:
        """LLM call started."""
        pass

    def _on_llm_thinking(self, event: StreamEvent) -> None:
        """Extended thinking chunk received."""
        if self._show_thinking and event.data:
            chunk = event.data.get("chunk", "")
            if chunk:
                print(chunk, end="", flush=True)

    def _on_llm_end(self, event: StreamEvent) -> None:
        """LLM call completed."""
        if self._in_thinking:
            print()  # Newline after thinking stream
            self._in_thinking = False

    def _on_transition(self, event: StreamEvent) -> None:
        """Complete transition recorded."""
        pass


class JSONLSubscriber:
    """
    Appends stream events to JSONL file.

    Each event is written as a single JSON line for easy parsing and replay.
    File is opened in append mode - safe for concurrent writes.

    Usage:
        subscriber = JSONLSubscriber(Path(".compass/streams/session.jsonl"))
        router.subscribe(subscriber)
    """

    def __init__(self, path: Path, include_tokens: bool = True):
        self._path = path
        self._include_tokens = include_tokens
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

    def on_event(self, event: StreamEvent) -> None:
        """Append event to JSONL file."""
        with open(self._path, "a") as f:
            f.write(event.to_json() + "\n")

    def on_token(self, token: str, nfa_path: tuple, state: str, instance_id: Optional[int] = None) -> None:
        """
        Append token event to JSONL file.

        Tokens are high-frequency, so this creates a StreamEvent on the fly.
        Set include_tokens=False if you don't want token-level logging.
        """
        if self._include_tokens:
            event = StreamEvent(
                type=StreamEventType.LLM_TOKEN,
                nfa_path=nfa_path,
                instance_id=instance_id,
                state=state,
                data={"token": token},
            )
            with open(self._path, "a") as f:
                f.write(event.to_json() + "\n")


class CollectingSubscriber:
    """
    Collects stream events in memory.

    Useful for testing and replay scenarios. Immutable-style API
    where get_events() returns all collected events.

    Usage:
        collector = CollectingSubscriber()
        router.subscribe(collector)
        # ... run NFA ...
        events = collector.get_events()
    """

    def __init__(self):
        self._events: list[StreamEvent] = []
        self._tokens: list[tuple[str, tuple, str, Optional[int]]] = []

    def on_event(self, event: StreamEvent) -> None:
        """Collect event."""
        self._events.append(event)

    def on_token(self, token: str, nfa_path: tuple, state: str, instance_id: Optional[int] = None) -> None:
        """Collect token."""
        self._tokens.append((token, nfa_path, state, instance_id))

    def get_events(self) -> list[StreamEvent]:
        """Return collected events."""
        return list(self._events)

    def get_tokens(self) -> str:
        """Return collected tokens as string."""
        return "".join(t[0] for t in self._tokens)

    def clear(self) -> None:
        """Clear collected events and tokens."""
        self._events.clear()
        self._tokens.clear()


class WebSocketSubscriber:
    """
    Sends stream events to web UI via websocket.

    Stub implementation - will be fleshed out when we add web visualization.

    Expected message format:
        {"type": "token", "token": "...", "state": "...", "branch": "..."}
        {"type": "event", "event_type": "STATE_ENTER", "state": "...", ...}
    """

    def __init__(self, ws):
        """
        Initialize with websocket connection.

        Args:
            ws: WebSocket-like object with send() method
        """
        self._ws = ws

    def on_event(self, event: StreamEvent) -> None:
        """Send event over websocket."""
        try:
            self._ws.send({
                "type": "event",
                "event_type": event.type.name,
                "state": event.state,
                "instance_id": event.instance_id,
                "nfa_path": list(event.nfa_path),
                "data": event.data,
            })
        except Exception:
            pass  # Connection may be closed

    def on_token(self, token: str, nfa_path: tuple, state: str, instance_id: Optional[int] = None) -> None:
        """Send token over websocket."""
        try:
            self._ws.send({
                "type": "token",
                "token": token,
                "state": state,
                "instance_id": instance_id,
            })
        except Exception:
            pass  # Connection may be closed


def create_session_jsonl_path(session_dir: Path = None) -> Path:
    """
    Create a JSONL path for stream logging.

    If session_dir is provided, creates streams/ subdirectory within it.
    Otherwise falls back to .compass/streams/ for ad-hoc runs.

    Args:
        session_dir: Optional session directory (e.g., ~/.compass/code/20260122-151554/)

    Returns:
        Path like {session_dir}/streams/neo.jsonl or .compass/streams/session_*.jsonl
    """
    if session_dir:
        streams_dir = Path(session_dir) / "streams"
        streams_dir.mkdir(parents=True, exist_ok=True)
        return streams_dir / "neo.jsonl"
    else:
        # Fallback for ad-hoc runs without a session
        base_dir = Path(".compass/streams")
        base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base_dir / f"session_{timestamp}.jsonl"


def create_instance_jsonl_path(instance_id: int, session_dir: Path = None) -> Path:
    """
    Create a JSONL path for a specific programmer instance.

    Each parallel programmer NFA gets its own file (programmer_0.jsonl, etc.).
    If session_dir is provided, puts logs in {session_dir}/streams/.

    Args:
        instance_id: Numeric instance ID (0, 1, 2, ...)
        session_dir: Optional session directory
    """
    if session_dir:
        streams_dir = Path(session_dir) / "streams"
        streams_dir.mkdir(parents=True, exist_ok=True)
        return streams_dir / f"programmer_{instance_id}.jsonl"
    else:
        base_dir = Path(".compass/streams")
        base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base_dir / f"programmer_{instance_id}_{timestamp}.jsonl"


# Backwards compat alias (deprecated)
def create_branch_jsonl_path(branch: str, session_dir: Path = None) -> Path:
    """DEPRECATED: Use create_instance_jsonl_path instead."""
    # Map old branch names to instance 0 for legacy code
    return create_instance_jsonl_path(0, session_dir)


def stream_logging_enabled() -> bool:
    """Check if stream logging is enabled via environment variable."""
    return os.environ.get("COMPASS_STREAM_LOG", "").lower() in ("1", "true", "yes")
