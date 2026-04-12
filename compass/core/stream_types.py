"""
Stream event types for NFA visualization.

Pure data types capturing NFA execution events - state transitions, LLM tokens,
and actions. Designed for routing to multiple subscribers (terminal, websocket, JSONL).

Key insight: The LLM stream IS the transition. Tokens flow during state transitions,
not just at boundaries. We capture the stream, not just final results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Optional
import json


class StreamEventType(Enum):
    """Types of stream events for NFA visualization."""

    # NFA lifecycle
    NFA_START = auto()       # NFA execution begins
    NFA_END = auto()         # NFA execution completes
    STATE_ENTER = auto()     # Entered a new state
    STATE_EXIT = auto()      # Exiting current state
    TRANSITION = auto()      # Complete transition record

    # LLM streaming (the thinking IS the transition)
    LLM_START = auto()       # Prompt sent to LLM
    LLM_TOKEN = auto()       # Single token (streaming output)
    LLM_THINKING = auto()    # Extended thinking chunk
    LLM_END = auto()         # LLM response complete

    # Actions (observable side effects)
    ACTION_START = auto()    # Beginning an action (file write, etc.)
    ACTION_END = auto()      # Action completed


@dataclass(frozen=True)
class StreamEvent:
    """
    Immutable stream event.

    Captures what happened during NFA execution without coupling to rendering.
    The nfa_path allows tracking nested NFAs (e.g., ("neo", "programmer")).
    The instance_id identifies parallel NFA instances (e.g., programmer_0, programmer_1).
    """
    type: StreamEventType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    nfa_path: tuple[str, ...] = ()            # Path for nested NFAs
    instance_id: Optional[int] = None         # Numeric ID for parallel NFA instances
    state: Optional[str] = None               # Current state name
    data: Optional[Dict[str, Any]] = None     # Event-specific payload

    def to_json(self) -> str:
        """Serialize to JSON for JSONL logging."""
        return json.dumps({
            "type": self.type.name,
            "ts": self.timestamp,
            "nfa_path": list(self.nfa_path),
            "instance_id": self.instance_id,
            "state": self.state,
            "data": self.data,
        })

    @classmethod
    def from_json(cls, json_str: str) -> "StreamEvent":
        """Deserialize from JSON."""
        d = json.loads(json_str)
        return cls(
            type=StreamEventType[d["type"]],
            timestamp=d["ts"],
            nfa_path=tuple(d.get("nfa_path", ())),
            instance_id=d.get("instance_id"),
            state=d.get("state"),
            data=d.get("data"),
        )


@dataclass(frozen=True)
class Transition:
    """
    Complete record of a state transition.

    Captures the full context of what happened during a transition:
    - The states involved (from/to)
    - The LLM interaction (input/output/thinking)
    - The parsed result and reason
    - Timing information
    """
    from_state: str
    to_state: str
    input_text: str           # The prompt sent to LLM
    output_text: str          # Full LLM response
    thinking: str             # Extended thinking (if available)
    parsed: Any               # Typed result after parsing
    reason: str               # Why this transition occurred
    duration_ms: int          # How long the transition took
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "from_state": self.from_state,
            "to_state": self.to_state,
            "input_len": len(self.input_text),
            "output_len": len(self.output_text),
            "thinking_len": len(self.thinking),
            "parsed": str(self.parsed)[:200],  # Truncate for logging
            "reason": self.reason,
            "duration_ms": self.duration_ms,
            "ts": self.timestamp,
        })


@dataclass(frozen=True)
class StreamEventStream:
    """
    Immutable stream of events.

    Functional append returns new stream - no mutation.
    Mirrors UIEventStream pattern for consistency.
    """
    events: tuple[StreamEvent, ...] = ()

    def append(self, event: StreamEvent) -> "StreamEventStream":
        """Return new stream with event appended."""
        return StreamEventStream(events=self.events + (event,))

    @property
    def tokens(self) -> str:
        """Join all LLM_TOKEN data into single string."""
        return "".join(
            e.data.get("token", "") for e in self.events
            if e.type == StreamEventType.LLM_TOKEN and e.data
        )

    @property
    def thinking(self) -> str:
        """Join all LLM_THINKING data into single string."""
        return "".join(
            e.data.get("chunk", "") for e in self.events
            if e.type == StreamEventType.LLM_THINKING and e.data
        )

    def filter_by_state(self, state: str) -> "StreamEventStream":
        """Return events filtered to specific state."""
        return StreamEventStream(
            events=tuple(e for e in self.events if e.state == state)
        )

    def filter_by_instance(self, instance_id: int) -> "StreamEventStream":
        """Return events filtered to specific NFA instance."""
        return StreamEventStream(
            events=tuple(e for e in self.events if e.instance_id == instance_id)
        )
