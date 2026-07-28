"""
UI Adapter abstraction for decoupling UI operations from execution logic.

Provides:
- UIAdapter ABC: Abstract interface for UI operations
- ImmediateUIAdapter: Direct passthrough to compass.cli.ui
- CollectingUIAdapter: Collects events for deferred/batched rendering
- replay_events: Replay UIEventStream to an adapter
"""

from abc import ABC, abstractmethod
from typing import Optional

from compass.core.ui_events import UIEvent, UIEventType, UIEventStream


class UIAdapter(ABC):
    """Abstract base class for UI adapters.

    Defines the interface for all UI operations that execution logic
    may need. Implementations can render immediately, collect for later,
    or discard entirely (for testing).
    """

    @abstractmethod
    def show_action(self, action_type: str, target: str, reasoning: str = None) -> None:
        """Show an action being executed."""
        ...

    @abstractmethod
    def show_result(self, success: bool, message: str) -> None:
        """Show the result of an action."""
        ...

    @abstractmethod
    def start_spinner(self, message: str) -> None:
        """Start a spinner/loading indicator."""
        ...

    @abstractmethod
    def stop_spinner(self) -> None:
        """Stop the current spinner."""
        ...

    @abstractmethod
    def start_thinking(self) -> None:
        """Begin a thinking/reasoning stream."""
        ...

    @abstractmethod
    def show_thinking_chunk(self, chunk: str) -> None:
        """Display a chunk of thinking/reasoning text."""
        ...

    @abstractmethod
    def end_thinking(self) -> None:
        """End the thinking/reasoning stream."""
        ...

    @abstractmethod
    def set_thinking_color(self, role: Optional[str]) -> None:
        """Set the color for thinking output based on role."""
        ...

    @abstractmethod
    def message(self, text: str) -> None:
        """Display a simple message."""
        ...


class ImmediateUIAdapter(UIAdapter):
    """UI adapter that immediately renders to compass.cli.ui.

    Each method imports and calls the corresponding function from
    compass.cli.ui, providing direct terminal output.
    """

    def show_action(self, action_type: str, target: str, reasoning: str = None) -> None:
        from compass.cli import ui
        ui.show_action(action_type, target, reasoning)

    def show_result(self, success: bool, message: str) -> None:
        from compass.cli import ui
        ui.show_result(success, message)

    def start_spinner(self, message: str) -> None:
        from compass.cli import ui
        ui.start_spinner(message)

    def stop_spinner(self) -> None:
        from compass.cli import ui
        ui.stop_spinner()

    def start_thinking(self) -> None:
        from compass.cli import ui
        ui.start_thinking_stream()

    def show_thinking_chunk(self, chunk: str) -> None:
        from compass.cli import ui
        ui.show_thinking_stream(chunk)

    def end_thinking(self) -> None:
        from compass.cli import ui
        ui.end_thinking_stream()

    def set_thinking_color(self, role: Optional[str]) -> None:
        from compass.cli import ui
        ui.set_thinking_color(role)

    def message(self, text: str) -> None:
        print(text)


class CollectingUIAdapter(UIAdapter):
    """UI adapter that collects events for deferred rendering.

    Instead of rendering immediately, records all UI operations
    as UIEvent objects. Can later replay via replay_events().
    """

    def __init__(self):
        self._stream: UIEventStream = UIEventStream()
        self._current_color: Optional[str] = None

    def show_action(self, action_type: str, target: str, reasoning: str = None) -> None:
        self._stream = self._stream.append(UIEvent(
            event_type=UIEventType.ACTION_SHOW,
            action_type=action_type,
            target=target,
            reasoning=reasoning,
        ))

    def show_result(self, success: bool, message: str) -> None:
        self._stream = self._stream.append(UIEvent(
            event_type=UIEventType.RESULT_SHOW,
            success=success,
            message=message,
        ))

    def start_spinner(self, message: str) -> None:
        self._stream = self._stream.append(UIEvent(
            event_type=UIEventType.SPINNER_START,
            message=message,
        ))

    def stop_spinner(self) -> None:
        self._stream = self._stream.append(UIEvent(
            event_type=UIEventType.SPINNER_STOP,
        ))

    def start_thinking(self) -> None:
        self._stream = self._stream.append(UIEvent(
            event_type=UIEventType.THINKING_START,
        ))

    def show_thinking_chunk(self, chunk: str) -> None:
        self._stream = self._stream.append(UIEvent(
            event_type=UIEventType.THINKING_CHUNK,
            chunk=chunk,
        ))

    def end_thinking(self) -> None:
        self._stream = self._stream.append(UIEvent(
            event_type=UIEventType.THINKING_END,
        ))

    def set_thinking_color(self, role: Optional[str]) -> None:
        self._current_color = role
        self._stream = self._stream.append(UIEvent(
            event_type=UIEventType.THINKING_COLOR,
            color=role,
        ))

    def message(self, text: str) -> None:
        self._stream = self._stream.append(UIEvent(
            event_type=UIEventType.MESSAGE,
            message=text,
        ))

    def get_events(self) -> UIEventStream:
        """Get all collected events as immutable stream."""
        return self._stream


def replay_events(events: UIEventStream, target: UIAdapter) -> None:
    """
    Replay UI events to a target adapter.

    Pure function: takes event data, renders to adapter.
    """
    for event in events.events:
        if event.event_type == UIEventType.ACTION_SHOW:
            target.show_action(event.action_type, event.target, event.reasoning)
        elif event.event_type == UIEventType.RESULT_SHOW:
            target.show_result(event.success, event.message)
        elif event.event_type == UIEventType.SPINNER_START:
            target.start_spinner(event.message)
        elif event.event_type == UIEventType.SPINNER_STOP:
            target.stop_spinner()
        elif event.event_type == UIEventType.THINKING_START:
            target.start_thinking()
        elif event.event_type == UIEventType.THINKING_CHUNK:
            target.show_thinking_chunk(event.chunk)
        elif event.event_type == UIEventType.THINKING_END:
            target.end_thinking()
        elif event.event_type == UIEventType.THINKING_COLOR:
            target.set_thinking_color(event.color)
        elif event.event_type == UIEventType.MESSAGE:
            target.message(event.message)
