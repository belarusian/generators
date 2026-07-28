"""
Stream router for NFA visualization.

Pub/sub system that routes stream events to multiple subscribers.
Each subscriber can handle events differently (terminal, websocket, JSONL).

Key design:
- StreamSubscriber is a Protocol (structural typing)
- StreamRouter manages subscription lifecycle
- Thread-safe for parallel instance execution
- Context manager support for automatic cleanup
- Session-scoped instance IDs for parallel NFA tracking
"""

from contextlib import contextmanager
from threading import Lock
from typing import Callable, List, Optional, Protocol, runtime_checkable

from compass.core.stream_types import StreamEvent, StreamEventType


class InstanceIdAllocator:
    """
    Thread-safe allocator for NFA instance IDs.

    Session-scoped: create one per session, pass to program action.
    Each call to allocate() returns the next ID (0, 1, 2, ...).

    Usage:
        allocator = InstanceIdAllocator()
        id_0 = allocator.allocate()  # 0
        id_1 = allocator.allocate()  # 1
    """
    def __init__(self):
        self._counter = 0
        self._lock = Lock()

    def allocate(self) -> int:
        """Allocate the next instance ID. Thread-safe."""
        with self._lock:
            id = self._counter
            self._counter += 1
            return id

    @property
    def count(self) -> int:
        """Number of instances allocated."""
        with self._lock:
            return self._counter


@runtime_checkable
class StreamSubscriber(Protocol):
    """
    Protocol for stream event subscribers.

    Subscribers can implement either or both methods:
    - on_event: For structured events (state changes, transitions)
    - on_token: Fast path for streaming tokens (high frequency)
    """

    def on_event(self, event: StreamEvent) -> None:
        """Handle a stream event."""
        ...

    def on_token(self, token: str, nfa_path: tuple, state: str, instance_id: Optional[int] = None) -> None:
        """
        Fast path for token streaming.

        Separate from on_event for performance - token events are high frequency.
        Implementations may choose to buffer or immediately flush.
        """
        ...


class StreamRouter:
    """
    Routes stream events to subscribers.

    Thread-safe pub/sub for NFA stream events. Supports:
    - Multiple subscribers (terminal, websocket, JSONL)
    - Fast path for token streaming
    - Context tracking (current NFA path, state, instance_id)
    - Thread-safe subscription management

    Usage:
        router = StreamRouter()
        router.subscribe(TerminalSubscriber())
        router.subscribe(JSONLSubscriber(path))

        with router.nfa_context("programmer"):
            with router.state_context("UNDERSTAND"):
                router.emit_token("The user wants...")
    """

    def __init__(self, instance_id: Optional[int] = None):
        self._subscribers: List[StreamSubscriber] = []
        self._lock = Lock()
        self._nfa_path: tuple[str, ...] = ()
        self._current_state: Optional[str] = None
        self._instance_id = instance_id

    def subscribe(self, subscriber: StreamSubscriber) -> None:
        """Add a subscriber. Thread-safe."""
        with self._lock:
            self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: StreamSubscriber) -> None:
        """Remove a subscriber. Thread-safe."""
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def emit(self, event: StreamEvent) -> None:
        """Emit an event to all subscribers."""
        # Enrich event with current context
        enriched = StreamEvent(
            type=event.type,
            timestamp=event.timestamp,
            nfa_path=event.nfa_path or self._nfa_path,
            instance_id=event.instance_id if event.instance_id is not None else self._instance_id,
            state=event.state or self._current_state,
            data=event.data,
        )
        with self._lock:
            subscribers = list(self._subscribers)
        for sub in subscribers:
            try:
                sub.on_event(enriched)
            except Exception:
                pass  # Don't let subscriber errors break the pipeline

    def emit_token(self, token: str) -> None:
        """
        Fast path for token streaming.

        Bypasses full event construction for performance.
        """
        with self._lock:
            subscribers = list(self._subscribers)
        for sub in subscribers:
            try:
                sub.on_token(token, self._nfa_path, self._current_state, self._instance_id)
            except Exception:
                pass

    def emit_thinking(self, chunk: str) -> None:
        """Emit an extended thinking chunk."""
        self.emit(StreamEvent(
            type=StreamEventType.LLM_THINKING,
            data={"chunk": chunk},
        ))

    @contextmanager
    def nfa_context(self, name: str):
        """
        Context manager for tracking NFA nesting.

        Usage:
            with router.nfa_context("programmer"):
                # Events emitted here have nfa_path=("programmer",)
                with router.nfa_context("critic"):
                    # Events here have nfa_path=("programmer", "critic")
        """
        old_path = self._nfa_path
        self._nfa_path = old_path + (name,)
        self.emit(StreamEvent(type=StreamEventType.NFA_START))
        try:
            yield
        finally:
            self.emit(StreamEvent(type=StreamEventType.NFA_END))
            self._nfa_path = old_path

    @contextmanager
    def state_context(self, state: str):
        """
        Context manager for tracking current state.

        Emits STATE_ENTER on entry, STATE_EXIT on exit.
        """
        old_state = self._current_state
        self._current_state = state
        self.emit(StreamEvent(type=StreamEventType.STATE_ENTER))
        try:
            yield
        finally:
            self.emit(StreamEvent(type=StreamEventType.STATE_EXIT))
            self._current_state = old_state

    def set_state(self, state: str) -> None:
        """Set current state without context manager."""
        self._current_state = state

    def set_instance_id(self, instance_id: int) -> None:
        """Set instance ID."""
        self._instance_id = instance_id

    @property
    def nfa_path(self) -> tuple[str, ...]:
        """Current NFA path."""
        return self._nfa_path

    @property
    def current_state(self) -> Optional[str]:
        """Current state."""
        return self._current_state

    @property
    def instance_id(self) -> Optional[int]:
        """Instance ID for parallel NFA tracking."""
        return self._instance_id


def create_thinking_callback(router: StreamRouter) -> Callable[[str], None]:
    """
    Create a thinking callback for Oracle.

    Returns a callback that routes thinking chunks through the router.
    This is the bridge between Oracle's on_thinking and our streaming system.
    """
    def on_thinking(chunk: str) -> None:
        router.emit_thinking(chunk)
    return on_thinking


def create_token_callback(router: StreamRouter) -> Callable[[str], None]:
    """
    Create a token callback for streaming LLM output.

    Returns a callback that routes tokens through the router.
    """
    def on_token(token: str) -> None:
        router.emit_token(token)
    return on_token


class StreamingOracle:
    """
    Oracle wrapper that routes streaming events through a StreamRouter.

    Implements OracleAccess protocol, wrapping an existing Oracle and
    automatically injecting on_thinking callbacks to route extended
    thinking through the streaming system.

    This is the bridge between Oracle's streaming and NFA visualization:
    - Wraps any Oracle instance
    - Injects on_thinking callback into all ask() calls
    - Routes thinking chunks through the StreamRouter
    - Passes through other Oracle methods unchanged

    Usage:
        router = StreamRouter(instance_id=0)
        streaming_oracle = StreamingOracle(oracle, router)

        # All ask() calls now route thinking through router
        result = streaming_oracle.ask(prompt, ResponseType)
    """

    def __init__(self, oracle, router: StreamRouter):
        """
        Create a streaming oracle wrapper.

        Args:
            oracle: The underlying Oracle instance to wrap
            router: StreamRouter for routing events
        """
        self._oracle = oracle
        self._stream_router = router  # Avoid shadowing oracle._router
        self._on_thinking = create_thinking_callback(router)

    def ask(
        self,
        prompt: str,
        response_type=None,
        max_tokens: int = 2000,
        max_retries: int = 2,
        task: str = "ask",
        **kwargs,
    ):
        """
        Ask with automatic thinking stream routing.

        Wraps Oracle.ask() and injects on_thinking callback to route
        extended thinking through the StreamRouter.
        """
        # Emit LLM_START event
        self._stream_router.emit(StreamEvent(
            type=StreamEventType.LLM_START,
            data={"task": task, "prompt_len": len(prompt)},
        ))

        try:
            # Inject on_thinking callback (unless caller explicitly provided one)
            if "on_thinking" not in kwargs:
                kwargs["on_thinking"] = self._on_thinking

            result = self._oracle.ask(
                prompt,
                response_type=response_type,
                max_tokens=max_tokens,
                max_retries=max_retries,
                task=task,
                **kwargs,
            )

            # Emit LLM_END event with response text
            # Handle different result types: RawResponse has .text, dataclasses need serialization
            if result is None:
                response_text = None
            elif hasattr(result, 'text'):
                response_text = result.text
            elif hasattr(result, '__dataclass_fields__'):
                # Dataclass - serialize to readable format
                import json
                from dataclasses import asdict
                try:
                    # Convert enums to their values for readability
                    d = asdict(result)
                    for k, v in d.items():
                        if hasattr(v, 'value'):
                            d[k] = v.value
                    response_text = json.dumps(d, indent=2, default=str)
                except Exception:
                    response_text = str(result)
            else:
                response_text = str(result)

            self._stream_router.emit(StreamEvent(
                type=StreamEventType.LLM_END,
                data={
                    "task": task,
                    "success": True,
                    "response": response_text,
                },
            ))

            return result

        except Exception as e:
            # Emit LLM_END with error
            self._stream_router.emit(StreamEvent(
                type=StreamEventType.LLM_END,
                data={"task": task, "success": False, "error": str(e)},
            ))
            raise

    def converse_raw(self, messages, max_tokens: int = 4000, task: str = None, **kwargs):
        """
        Multi-turn conversation with thinking stream routing.

        Wraps Oracle.converse_raw() for raw output mode (IMPLEMENT state).
        """
        self._stream_router.emit(StreamEvent(
            type=StreamEventType.LLM_START,
            data={"task": task or "converse", "messages_len": len(messages)},
        ))

        try:
            result = self._oracle.converse_raw(messages, max_tokens=max_tokens, task=task, **kwargs)

            # Emit LLM_END event with response text
            response_text = getattr(result, 'text', None) or str(result) if result else None
            self._stream_router.emit(StreamEvent(
                type=StreamEventType.LLM_END,
                data={
                    "task": task or "converse",
                    "success": True,
                    "response": response_text,
                },
            ))

            return result

        except Exception as e:
            self._stream_router.emit(StreamEvent(
                type=StreamEventType.LLM_END,
                data={"task": task or "converse", "success": False, "error": str(e)},
            ))
            raise

    def __getattr__(self, name):
        """
        Pass through other methods to underlying oracle.

        This allows StreamingOracle to act as a drop-in replacement
        while only intercepting ask() and converse_raw().
        """
        return getattr(self._oracle, name)
