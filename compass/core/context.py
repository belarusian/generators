"""
Context Protocols - Interfaces for NFA contexts and Oracle access.

These protocols define the contracts that all NFA contexts and
Oracle interfaces must satisfy. They enable:

1. Bounded context: Each NFA defines its own context type
2. Abstracted Oracle access: NFAs don't depend on Oracle internals
3. Introspection: Contexts can describe themselves for debugging
"""

from typing import Any, Dict, Optional, Protocol, Type, TypeVar, Union, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class NFAContext(Protocol):
    """
    Protocol for NFA contexts.

    Every NFA context must implement this protocol to enable
    consistent debugging, logging, and introspection.

    The key insight: each NFA has a BOUNDED context that only
    contains what that NFA needs to see. This prevents
    "paralysis by analysis" from information overload.
    """

    def describe(self) -> str:
        """
        Return a human-readable description of the current context state.

        Used for debugging and logging. Should be concise but informative.

        Example:
            "Programmer (scribe review): Add CSV parser... (3 chunks)"
        """
        ...


@runtime_checkable
class OracleAccess(Protocol):
    """
    Protocol for Oracle access - the minimal interface NFAs need.

    This abstracts away the full Oracle implementation, giving each
    NFA only what it needs:
    - ask(): Type-validated LLM calls (Python expressions, JSON schemas, or raw)

    NFAs should depend on this protocol, not the full Oracle class.
    This enables:
    - Testing with mock oracles
    - Different providers per NFA
    - Clean dependency boundaries
    """

    def ask(
        self,
        prompt: str,
        response_type: Optional[Union[Type[T], Dict[str, Any]]] = None,
        max_tokens: int = 2000,
        max_retries: int = 2,
        task: str = "ask",
        **kwargs,
    ) -> Union[T, Dict[str, Any], Any]:
        """
        Make a type-validated LLM call.

        Three modes based on response_type:
        - None: Raw output - model writes freely, returns RawResponse(text, thinking)
        - Type: Python path - model writes constructor, we eval and validate
        - Dict: JSON path - model writes JSON, we parse and validate

        Args:
            prompt: The prompt to send to the LLM
            response_type: None (raw), Type (Python), or Dict schema (JSON)
            max_tokens: Maximum tokens in response
            max_retries: How many correction attempts
            task: Task identifier for logging/debugging
            **kwargs: Additional options (iteration, think_level, provider, etc.)

        Returns:
            RawResponse (if None), instance of response_type (if Type), or Dict (if schema)

        Raises:
            OracleSchemaError: If response doesn't match schema
        """
        ...


class ContextView:
    """
    Base class for creating bounded views of a larger context.

    Used when a Critic needs to see only part of what the Actor sees.
    For example, the Scribe (Critic) in Programmer NFA only sees:
    - The solution chunks (not the original problem)
    - System constraints (not the design reasoning)

    This enforces information hiding at the type level.
    """

    def __init__(self, parent_context: Any, allowed_fields: set):
        """
        Create a bounded view of a parent context.

        Args:
            parent_context: The full context to create a view of
            allowed_fields: Set of field names the view can access
        """
        self._parent = parent_context
        self._allowed = allowed_fields

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)

        if name not in self._allowed:
            raise AttributeError(
                f"ContextView does not expose '{name}'. "
                f"Allowed fields: {self._allowed}"
            )

        return getattr(self._parent, name)

    def describe(self) -> str:
        """Describe this bounded view."""
        return f"View of {type(self._parent).__name__} ({len(self._allowed)} fields)"


def create_bounded_view(
    context: Any,
    allowed_fields: set,
    view_class: Optional[type] = None,
) -> Any:
    """
    Create a bounded view of a context.

    Helper function to create ContextView instances or custom view classes.

    Args:
        context: The full context
        allowed_fields: Fields the view can access
        view_class: Optional custom view class (must accept parent_context, allowed_fields)

    Returns:
        A bounded view that only exposes allowed_fields
    """
    if view_class:
        return view_class(context, allowed_fields)
    return ContextView(context, allowed_fields)
