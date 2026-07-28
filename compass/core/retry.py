"""
Common Retry Loop - Shared retry-with-feedback infrastructure.

This module provides the retry loop pattern used by Oracle.ask,
FileEditor, ShellBuilder, and other agent tools. The pattern is:

1. Call LLM with messages
2. Parse and validate response
3. If invalid, append feedback to messages and retry
4. After max retries, return failure with context

The key insight: messages lists compose. Each retry appends to the
conversation, preserving thinking as a separate field for proper
multi-turn behavior.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union

T = TypeVar('T')


@dataclass
class AskResult:
    """Result from a single LLM call."""
    text: str
    thinking: str = ""
    truncated: bool = False


@dataclass
class RetryResult(Generic[T]):
    """
    Result from the retry loop.

    Attributes:
        success: True if validation passed
        value: The parsed/validated result
        error: Last error message if failed
        attempts: Number of attempts made
        last_response: Last response text (for escalation context)
        last_thinking: Last thinking (for escalation context)
    """
    success: bool
    value: Optional[T] = None
    error: Optional[str] = None
    attempts: int = 0
    last_response: str = ""
    last_thinking: str = ""


def retry_with_messages(
    ask_once: Callable[[List[Dict[str, Any]]], AskResult],
    initial_messages: List[Dict[str, Any]],
    parse: Callable[[str], T],
    validate: Callable[[T], Optional[str]],
    append_feedback: Callable[[List[Dict], str, str, str], List[Dict]],
    max_retries: int = 3,
    on_retry_failure: Optional[Callable[[int, str, str], None]] = None,
) -> RetryResult[T]:
    """
    Retry loop with message composition.

    The core retry infrastructure. Composes with any ask_once function.
    Messages lists compose naturally - each retry appends assistant
    response (with thinking) and user feedback.

    Args:
        ask_once: Single LLM call, takes messages, returns AskResult
        initial_messages: Starting messages [{"role": "user", "content": ...}]
        parse: Parse response text to value (raises on failure)
        validate: Validate value, returns error string or None
        append_feedback: (messages, response, error, thinking) -> new_messages
        max_retries: How many retry attempts
        on_retry_failure: Optional callback for parse/validation failures.
            Args: (attempt_index, failure_type, error_message), where
            failure_type is "parse" or "validation".

    Returns:
        RetryResult with success/failure and context for escalation

    Example:
        result = retry_with_messages(
            ask_once=lambda msgs: oracle.ask_once(msgs),
            initial_messages=[{"role": "user", "content": prompt}],
            parse=json.loads,
            validate=lambda x: None if "name" in x else "missing name",
            append_feedback=append_retry_feedback,
            max_retries=3,
        )
    """
    messages = initial_messages
    last_thinking = ""
    last_response = ""

    for attempt in range(max_retries + 1):
        result = ask_once(messages)
        last_response = result.text
        if result.thinking:
            last_thinking = result.thinking

        truncation_hint = " (response was TRUNCATED - too many tokens on thinking, be concise)" if result.truncated else ""

        # Try parse
        try:
            value = parse(result.text)
        except Exception as e:
            error = f"{type(e).__name__}: {e}{truncation_hint}"
            if on_retry_failure:
                on_retry_failure(attempt, "parse", error)
            if attempt < max_retries:
                messages = append_feedback(messages, result.text, error, result.thinking)
                continue
            return RetryResult(
                success=False, error=error, attempts=attempt + 1,
                last_response=last_response, last_thinking=last_thinking,
            )

        # Validate
        validation_error = validate(value)
        if validation_error:
            error = f"{validation_error}{truncation_hint}"
            if on_retry_failure:
                on_retry_failure(attempt, "validation", error)
            if attempt < max_retries:
                messages = append_feedback(messages, result.text, error, result.thinking)
                continue
            return RetryResult(
                success=False, error=error, attempts=attempt + 1,
                last_response=last_response, last_thinking=last_thinking,
            )

        # Success
        return RetryResult(success=True, value=value, attempts=attempt + 1)

    return RetryResult(
        success=False, error="Unexpected retry loop exit",
        last_response=last_response, last_thinking=last_thinking,
    )


def simple_retry(
    fn: Callable[[], T],
    validate: Callable[[T], bool],
    max_retries: int = 3,
    on_error: Optional[Callable[[Exception], None]] = None,
) -> Tuple[bool, Optional[T]]:
    """Simple retry without LLM feedback."""
    for _ in range(max_retries):
        try:
            result = fn()
            if validate(result):
                return True, result
        except Exception as e:
            if on_error:
                on_error(e)
            continue
    return False, None
