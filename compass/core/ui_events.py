"""
UI event types for decoupled rendering.

Pure data types for UI events - no side effects.
The UI layer consumes these events to render appropriately.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class UIEventType(Enum):
    """Types of UI events."""
    SPINNER_START = "spinner_start"
    SPINNER_STOP = "spinner_stop"
    ACTION_SHOW = "action_show"
    RESULT_SHOW = "result_show"
    THINKING_START = "thinking_start"
    THINKING_CHUNK = "thinking_chunk"
    THINKING_END = "thinking_end"
    THINKING_COLOR = "thinking_color"
    MESSAGE = "message"
    PROMPT_DEBUG = "prompt_debug"


@dataclass(frozen=True)
class UIEvent:
    """
    Immutable UI event.

    Captures what happened without coupling to how it's rendered.
    """
    event_type: UIEventType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message: Optional[str] = None
    action_type: Optional[str] = None
    target: Optional[str] = None
    reasoning: Optional[str] = None
    success: Optional[bool] = None
    color: Optional[str] = None
    chunk: Optional[str] = None


@dataclass(frozen=True)
class UIEventStream:
    """
    Immutable stream of UI events.

    Functional append returns new stream - no mutation.
    """
    events: tuple[UIEvent, ...] = ()

    def append(self, event: UIEvent) -> "UIEventStream":
        """Return new stream with event appended."""
        return UIEventStream(events=self.events + (event,))

    @property
    def thinking_chunks(self) -> str:
        """Join all THINKING_CHUNK chunks into single string."""
        return "".join(
            e.chunk for e in self.events
            if e.event_type == UIEventType.THINKING_CHUNK and e.chunk
        )
