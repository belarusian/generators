"""
Pure functions for Oracle.ask() - decomposed for FP composition.

This module extracts the pure logic from Oracle.ask():
- Temperature/seed computation (retry strategy)
- Message building (provider-specific formats)
- Response parsing and validation

These functions have no side effects and can be composed/tested independently.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from compass.llm.providers import ThinkLevel


@dataclass(frozen=True)
class RetryParams:
    """
    Immutable retry parameters computed from iteration/attempt.

    Two axes with opposite directions:
    - iteration (Critic retries): INCREASE temp for creativity
    - attempt (JSON retries): DECREASE temp for structure
    """
    temperature: Optional[float]
    seed: Optional[int]
    think_level: "ThinkLevel"


@dataclass(frozen=True)
class AskRequest:
    """
    Immutable request for Oracle.ask().

    Pure data - no provider references, no side effects.
    """
    prompt: str
    schema: Dict[str, Any]
    images: Tuple[Any, ...] = ()  # Tuple for immutability
    max_tokens: int = 2000
    max_retries: int = 3
    task: str = "ask"
    iteration: int = 0


@dataclass
class AskResult:
    """
    Result from an ask attempt.

    Either success with data, or failure with error info.
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_response: str = ""
    thinking: str = ""  # For escalation context
    truncated: bool = False


def compute_retry_params(
    iteration: int,
    attempt: int,
    explicit_think: Optional["ThinkLevel"] = None,
    floor_think: Optional["ThinkLevel"] = None,
) -> RetryParams:
    """
    Pure function: compute temperature, seed, think level from retry state.

    Strategy:
    - iteration (problem-solving, from Critic): INCREASE base temp
    - attempt (JSON parsing failures): DECREASE temp
    - Any retry gets random seed for perturbation

    Args:
        iteration: Problem-solving iteration (0=first, higher=Critic retries)
        attempt: JSON retry attempt within iteration
        explicit_think: Explicit think level override (or None for auto)
        floor_think: Minimum think level from /think setting (or None for OFF)

    Returns:
        RetryParams with computed values

    Examples:
        iteration=0, attempt=0, floor=None  -> think=OFF (default)
        iteration=0, attempt=0, floor=MED   -> think=MEDIUM (floor)
        iteration=0, attempt=1, floor=None  -> think=MEDIUM (escalation)
        iteration=1, attempt=0, floor=HIGH  -> think=HIGH (floor > escalation)
    """
    from compass.llm.providers import ThinkLevel

    # Base temp: 0.7 + (iteration * 0.1), capped at [0.3, 1.0]
    base_temp = max(0.3, min(0.7 + (iteration * 0.1), 1.0))

    # Retry adjustment: -0.1 per JSON retry
    if attempt > 0:
        temperature = base_temp - (attempt * 0.1)
    elif iteration != 0:
        temperature = base_temp
    else:
        temperature = None  # Use provider default

    # Random seed on any retry (positive or negative iteration)
    seed = random.randint(1, 1000000) if (attempt > 0 or iteration != 0) else None

    # Think level: explicit or escalate from floor
    # Explicit overrides everything. Otherwise, escalate but never below floor.
    if explicit_think is not None:
        think_level = explicit_think
    else:
        rank = {ThinkLevel.OFF: 0, ThinkLevel.LOW: 1, ThinkLevel.MEDIUM: 2, ThinkLevel.HIGH: 3}
        floor = floor_think or ThinkLevel.OFF
        escalation = iteration + (1 if attempt >= 1 else 0)
        escalated = (
            ThinkLevel.HIGH if escalation >= 3 else
            ThinkLevel.MEDIUM if escalation >= 2 else
            ThinkLevel.LOW if escalation >= 1 else
            ThinkLevel.OFF
        )
        think_level = escalated if rank[escalated] >= rank[floor] else floor

    return RetryParams(
        temperature=temperature,
        seed=seed,
        think_level=think_level,
    )


def build_schema_prompt(prompt: str, schema: Dict[str, Any]) -> str:
    """
    Pure function: append schema to prompt.

    Args:
        prompt: The base prompt
        schema: JSON schema for expected response

    Returns:
        Prompt with schema appended
    """
    schema_json = json.dumps(schema, indent=2)
    return f"""{prompt}

Respond with valid JSON matching this schema:
{schema_json}"""


def build_messages_ollama(
    prompt: str,
    images: Tuple[Any, ...] = (),
) -> List[Dict[str, Any]]:
    """
    Pure function: build Ollama-format messages.

    Ollama uses images as separate base64 array.
    """
    if images:
        return [{
            "role": "user",
            "content": f"[{len(images)} image(s) attached]\n\n{prompt}",
            "images": [img.data for img in images],
        }]
    return [{"role": "user", "content": prompt}]


def build_messages_anthropic(
    prompt: str,
    images: Tuple[Any, ...] = (),
) -> List[Dict[str, Any]]:
    """
    Pure function: build Anthropic-format messages.

    Anthropic uses multimodal content array.
    """
    if images:
        content_parts = []
        for img in images:
            content_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": img.data,
                }
            })
        content_parts.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content_parts}]
    return [{"role": "user", "content": prompt}]


def build_messages(
    prompt: str,
    provider_type: str,
    images: Tuple[Any, ...] = (),
) -> List[Dict[str, Any]]:
    """
    Pure function: build provider-specific messages.

    Args:
        prompt: The full prompt (with schema)
        provider_type: "ollama" or "anthropic"
        images: Tuple of image objects

    Returns:
        List of message dicts for the provider
    """
    if "ollama" in provider_type.lower():
        return build_messages_ollama(prompt, images)
    return build_messages_anthropic(prompt, images)


def append_retry_feedback(
    messages: List[Dict[str, Any]],
    response: str,
    error: str,
    thinking: str = "",
    feedback_suffix: str = "Please correct and respond with valid JSON.",
    feedback_template: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Pure function: append retry feedback with proper thinking propagation.

    Adds assistant message (with thinking as separate field) and user feedback.
    Returns new list, does not mutate input.

    Args:
        messages: Current conversation messages
        response: The assistant's response text
        error: The error message to include
        thinking: Assistant's thinking (kept as separate field per Ollama docs)
        feedback_suffix: What to ask for (e.g., "valid JSON", "valid MyType(...)")
        feedback_template: Optional full template with {error} placeholder. If provided,
                          overrides the default "Error: {error}. {feedback_suffix}" format.

    Returns:
        New messages list with feedback appended
    """
    new_messages = list(messages)

    # Include thinking in assistant message per Ollama docs
    assistant_msg = {"role": "assistant", "content": response}
    if thinking:
        assistant_msg["thinking"] = thinking

    new_messages.append(assistant_msg)

    feedback_content = (
        feedback_template.format(error=error) if feedback_template
        else f"Error: {error}. {feedback_suffix}"
    )
    new_messages.append({"role": "user", "content": feedback_content})

    return new_messages


def append_error_feedback(
    messages: List[Dict[str, Any]],
    response: str,
    error: str,
    thinking: str = "",
) -> List[Dict[str, Any]]:
    """
    Pure function: append error feedback for JSON retry.

    Convenience wrapper around append_retry_feedback for JSON mode.
    """
    return append_retry_feedback(
        messages, response, error, thinking,
        feedback_suffix="Please correct and respond with valid JSON.",
    )


def append_truncation_feedback(
    messages: List[Dict[str, Any]],
    response: str,
    thinking: str = "",
) -> List[Dict[str, Any]]:
    """
    Pure function: append truncation feedback for retry.

    Returns new list, does not mutate input.
    """
    new_messages = list(messages)

    last_bit = response[-500:] if len(response) > 500 else response

    assistant_msg = {"role": "assistant", "content": response}
    if thinking:
        assistant_msg["thinking"] = thinking

    new_messages.append(assistant_msg)
    new_messages.append({
        "role": "user",
        "content": f"Your response was cut off (too long). Last part: ...{last_bit}\n\nPlease provide a CONCISE response - just the JSON, minimal explanation.",
    })

    return new_messages
