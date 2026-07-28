"""
Oracle module.
The voice of the compass - speaking through any LLM that can learn.

The Oracle communicates through schema validation with error feedback.
This makes it model-agnostic: Claude, Ollama, or any model that outputs
JSON and learns from correction works with our system.
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union

T = TypeVar("T")

from compass.llm.providers import Provider, get_provider, ThinkLevel, ProviderResponse
from compass.cli.ui import start_thinking_stream, show_thinking_stream, end_thinking_stream


def _debug(*args):
    """Print debug info if COMPASS_DEBUG or DEBUG is set."""
    if os.getenv("COMPASS_DEBUG") or os.getenv("DEBUG"):
        print("[DEBUG]", *args)


def _get_think_floor() -> Optional[ThinkLevel]:
    """Read the /think floor level. None if no floor set."""
    try:
        from compass.cli.commands import get_think_level_override
        level_str = get_think_level_override()
        if not level_str:
            return None
        return {
            "low": ThinkLevel.LOW,
            "medium": ThinkLevel.MEDIUM,
            "high": ThinkLevel.HIGH,
        }.get(level_str)
    except ImportError:
        return None


__all__ = [
    # Core class
    "Oracle",
    # Response types
    "RawResponse",
    # Schema validation
    "OracleSchemaError",
    "validate_response",
    # JSON helpers - reusable by ClaudeBridge, tests, etc.
    "parse_json_response",
    "ask_with_schema",
    # Schemas
    "JOURNEY_SCHEMA",
]


@dataclass
class RawResponse:
    """
    Raw response from oracle.ask() when response_type=None.

    Contains both text and thinking so caller can use thinking
    for retry context if needed.
    """
    text: str
    thinking: str = ""
    done_reason: str = ""


class OracleSchemaError(Exception):
    """Raised when Oracle response doesn't match expected schema."""

    def __init__(self, message: str, raw_response: str = None):
        super().__init__(message)
        self.raw_response = raw_response  # The raw text that failed to parse


def _convert_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert JSON Schema format to internal format.

    JSON Schema: {"type": "object", "properties": {...}, "required": [...]}
    Internal:    {"field": "type", "field?": "type"}
    """
    if schema.get("type") != "object" or "properties" not in schema:
        return schema  # Not JSON Schema format

    properties = schema["properties"]
    required = set(schema.get("required", []))
    result = {}

    # Map JSON Schema types to internal types
    type_map = {
        "string": "string",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }

    for field_name, field_schema in properties.items():
        if isinstance(field_schema, dict):
            json_type = field_schema.get("type", "string")
            internal_type = type_map.get(json_type, json_type)

            # Build internal field spec
            if "enum" in field_schema:
                field_spec = {"type": internal_type, "enum": field_schema["enum"]}
            elif "items" in field_schema or "properties" in field_schema:
                # Nested schema - convert recursively
                field_spec = {"type": internal_type}
                if "items" in field_schema:
                    field_spec["items"] = _convert_json_schema(field_schema["items"])
                if "properties" in field_schema:
                    field_spec["properties"] = _convert_json_schema(field_schema)
            else:
                field_spec = internal_type
        else:
            field_spec = field_schema

        # Add ? suffix for optional fields
        key = field_name if field_name in required else f"{field_name}?"
        result[key] = field_spec

    return result


def _is_json_schema(schema: Dict[str, Any]) -> bool:
    """Detect if schema is JSON Schema format."""
    return (
        isinstance(schema.get("type"), str)
        and schema.get("type") == "object"
        and "properties" in schema
    )


def validate_response(data: Any, schema: Dict[str, Any]) -> None:
    """
    Validate data against a schema.

    Schema formats supported:
    - JSON Schema: {"type": "object", "properties": {...}, "required": [...]}
    - Simple: {"field": "type"}
    - Extended: {"field": {"type": "string", "enum": ["a", "b"]}}
    - Nested lists: {"field": {"type": "list", "items": {"properties": {...}}}}
    - Nested objects: {"field": {"type": "dict", "properties": {...}}}

    Supported types: "string", "int", "float", "bool", "list", "dict"
    Optional fields end with "?": {"notes?": "string"}

    Constraints:
    - enum: list of allowed values (for any type)
    - pattern: regex pattern (for strings)
    - items: schema for list items (for lists)
    - properties: nested schema for objects (for dicts)

    Raises OracleSchemaError if validation fails.
    """
    # Convert JSON Schema to internal format if needed
    if _is_json_schema(schema):
        schema = _convert_json_schema(schema)

    if not isinstance(data, dict):
        raise OracleSchemaError(f"Expected object, got {type(data).__name__}")

    for key, field_spec in schema.items():
        # Handle optional fields
        optional = key.endswith("?")
        field_name = key.rstrip("?")

        if field_name not in data:
            if optional:
                continue
            raise OracleSchemaError(f"Missing required field: '{field_name}'")

        value = data[field_name]

        # Allow null for optional fields
        if value is None and optional:
            continue

        # Parse field spec - can be string ("type") or dict ({"type": ..., "enum": ...})
        if isinstance(field_spec, str):
            expected_type = field_spec
            constraints = {}
        elif isinstance(field_spec, dict):
            expected_type = field_spec.get("type", "string")
            constraints = field_spec
        else:
            raise OracleSchemaError(f"Invalid schema for field '{field_name}'")

        # Type checking
        type_map = {
            "string": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
        }

        if expected_type not in type_map:
            raise OracleSchemaError(f"Unknown type '{expected_type}' for field '{field_name}'")

        expected_python_type = type_map[expected_type]
        if not isinstance(value, expected_python_type):
            raise OracleSchemaError(
                f"Field '{field_name}' expected {expected_type}, got {type(value).__name__}"
            )

        # Enum constraint
        if "enum" in constraints:
            allowed = constraints["enum"]
            if value not in allowed:
                raise OracleSchemaError(
                    f"Field '{field_name}' must be one of {allowed}, got '{value}'"
                )

        # Pattern constraint (for strings)
        if "pattern" in constraints and isinstance(value, str):
            import re
            pattern = constraints["pattern"]
            if not re.match(pattern, value):
                raise OracleSchemaError(
                    f"Field '{field_name}' must match pattern '{pattern}'"
                )

        # Items constraint (for lists) - validate each item
        if "items" in constraints and isinstance(value, list):
            items_schema = constraints["items"]
            for i, item in enumerate(value):
                _validate_item(item, items_schema, f"{field_name}[{i}]")

        # Properties constraint (for nested dicts)
        if "properties" in constraints and isinstance(value, dict):
            try:
                validate_response(value, constraints["properties"])
            except OracleSchemaError as e:
                raise OracleSchemaError(f"In '{field_name}': {e}")


def _validate_item(value: Any, schema: Any, context: str) -> None:
    """Validate a single item against a schema (for list items)."""
    # Discriminated union - schema varies based on a discriminator field
    if isinstance(schema, dict) and "discriminator" in schema:
        if not isinstance(value, dict):
            raise OracleSchemaError(f"{context}: expected object, got {type(value).__name__}")

        discriminator = schema["discriminator"]
        schemas = schema.get("schemas", {})

        if discriminator not in value:
            raise OracleSchemaError(f"{context}: missing discriminator field '{discriminator}'")

        disc_value = value[discriminator]
        if disc_value not in schemas:
            valid_types = list(schemas.keys())
            raise OracleSchemaError(f"{context}: '{discriminator}' must be one of {valid_types}, got '{disc_value}'")

        # Validate against the specific schema for this type
        type_schema = schemas[disc_value]
        try:
            _validate_item(value, type_schema, context)
        except OracleSchemaError as e:
            raise OracleSchemaError(f"{context} ({disc_value}): {e}")
        return

    # If schema has "properties", validate as nested object
    if isinstance(schema, dict) and "properties" in schema:
        if not isinstance(value, dict):
            raise OracleSchemaError(f"{context}: expected object, got {type(value).__name__}")
        try:
            # If JSON Schema format, pass full schema so conversion works
            # Otherwise pass just properties (internal format)
            if _is_json_schema(schema):
                validate_response(value, schema)
            else:
                validate_response(value, schema["properties"])
        except OracleSchemaError as e:
            raise OracleSchemaError(f"{context}: {e}")
        return

    # Simple type check
    if isinstance(schema, str):
        type_map = {
            "string": str, "int": int, "float": (int, float),
            "bool": bool, "list": list, "dict": dict,
        }
        if schema in type_map and not isinstance(value, type_map[schema]):
            raise OracleSchemaError(f"{context}: expected {schema}, got {type(value).__name__}")


def parse_json_response(text: str) -> Any:
    """
    Parse JSON from model response.

    Handles:
    - Markdown code blocks (```json ... ```)
    - Prose before/after JSON (finds first { and matching })

    Args:
        text: Raw response text from model

    Returns:
        Parsed JSON object

    Raises:
        json.JSONDecodeError: If no valid JSON found
    """
    text = text.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
        text = text.strip()

    # Try direct parse first
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON object in the noise
    if parsed is None:
        start = text.find("{")
        if start == -1:
            raise json.JSONDecodeError("No JSON object found", text, 0)

        # Find matching closing brace
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    parsed = json.loads(text[start:i+1])
                    break

        if parsed is None:
            raise json.JSONDecodeError("No matching } found", text, start)

    return parsed


def ask_with_schema(
    complete_fn,
    prompt: str,
    schema: Dict[str, Any],
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Retry loop for schema-validated JSON responses.

    This is the core pattern used by both Oracle and ClaudeBridge:
    1. Build prompt with schema
    2. Call model via complete_fn
    3. Parse JSON, validate against schema
    4. If error, append error to messages and retry
    5. After max_retries + 1 attempts, raise OracleSchemaError

    Args:
        complete_fn: Callable(messages: List[Dict]) -> str
                     Takes conversation messages, returns response text
        prompt: The prompt to send
        schema: Expected JSON schema
        max_retries: How many correction attempts before giving up

    Returns:
        Parsed and validated response dict

    Raises:
        OracleSchemaError: If valid response not obtained after retries
    """
    schema_json = json.dumps(schema, indent=2)
    full_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{schema_json}"""

    messages = [{"role": "user", "content": full_prompt}]

    for attempt in range(max_retries + 1):
        response_text = complete_fn(messages)

        # Try to parse JSON
        try:
            parsed = parse_json_response(response_text)
        except json.JSONDecodeError as e:
            error = f"Invalid JSON: {e}"
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"Error: {error}. Please respond with valid JSON matching the schema."})
                continue
            raise OracleSchemaError(f"Failed to parse JSON after {max_retries + 1} attempts: {error}")

        # Validate against schema
        try:
            validate_response(parsed, schema)
            return parsed
        except OracleSchemaError as e:
            error = str(e)
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"Error: {error}. Please correct and respond with valid JSON."})
                continue
            raise OracleSchemaError(f"Schema validation failed after {max_retries + 1} attempts: {error}")

    raise OracleSchemaError("Unexpected error in ask_with_schema()")


# The schema the Guide needs to plan a journey
JOURNEY_SCHEMA = {
    "travelers": {
        "type": "string",
        "description": "Who is traveling - names, ages, relationships",
        "required": True,
    },
    "origin": {
        "type": "string",
        "description": "Where the journey begins - city, town, or address",
        "required": True,
    },
    "days": {
        "type": "int",
        "description": "How many days the journey will last",
        "default": 4,
        "min": 2,
        "max": 14,
    },
    "budget": {
        "type": "int",
        "description": "Total budget in dollars for the trip",
        "nullable": True,
    },
    "interests": {
        "type": "list",
        "description": "What delights the travelers - activities, themes, places",
        "required": True,
    },
}


class Oracle:
    """The voice that speaks through the compass."""

    _cached_provider: Optional[Provider] = None

    def __init__(self, provider: Provider = None):
        """
        Initialize the Oracle.

        Args:
            provider: Explicit Provider (for tests or overrides).
                      If None, lazily resolves from COMPASS_MODEL env var.
        """
        self._explicit_provider = provider

        # Pending images for vision (cleared after each ask call)
        self._pending_images = []

    @classmethod
    def invalidate(cls):
        """Invalidate cached provider (call after model change)."""
        cls._cached_provider = None

    @property
    def provider(self) -> Provider:
        """The configured provider. Lazy init from COMPASS_MODEL if not explicit."""
        if self._explicit_provider:
            return self._explicit_provider
        if Oracle._cached_provider is None:
            from compass.llm.ladder_policy import get_model_spec
            from compass.llm.providers import get_provider_by_id
            Oracle._cached_provider = get_provider_by_id(get_model_spec())
        return Oracle._cached_provider

    def _get_provider(self, task: Optional[str] = None, for_vision: bool = False) -> Provider:
        """Get the provider. Uses VISION_MODEL when images are present."""
        if for_vision and not self._explicit_provider:
            from compass.llm.ladder_policy import get_vision_model_spec
            vision_spec = get_vision_model_spec()
            # Only construct a separate provider if vision model differs
            default_spec = self.provider.name
            if vision_spec not in default_spec:
                from compass.llm.providers import get_provider_by_id
                try:
                    return get_provider_by_id(vision_spec)
                except Exception:
                    pass  # Fall back to default
        return self.provider

    def set_images(self, images: list) -> None:
        """Set images to include in the next ask() call."""
        self._pending_images = images

    def speak(self, prompt: str, max_tokens: Optional[int] = None, task: Optional[str] = None, provider: "Provider" = None, max_retries: int = 1) -> str:
        """Have the oracle speak, with conversational retry on truncation.

        If the model burns all tokens on thinking and produces empty content,
        retries with feedback asking her to be concise -- same retry loop as ask().
        """
        from compass.core.retry import retry_with_messages, AskResult
        from compass.core.telemetry import with_provider_timing
        from compass.llm.ask import append_retry_feedback
        from compass.llm.ladder_policy import get_max_tokens

        effective_max = max_tokens or get_max_tokens()
        active_provider = provider if provider else self._get_provider(task)
        messages = [{"role": "user", "content": prompt}]

        def ask_once(msgs):
            response = with_provider_timing(
                provider_name=active_provider.name,
                call=lambda: active_provider.complete(msgs, effective_max),
                task=task or "speak",
            )
            return AskResult(
                text=response.text.strip(),
                thinking=response.thinking or "",
                truncated=(response.done_reason == "length"),
            )

        result = retry_with_messages(
            ask_once=ask_once,
            initial_messages=messages,
            parse=lambda text: text,
            validate=lambda text: "Response was empty. Be concise and respond directly." if not text else None,
            append_feedback=lambda msgs, resp, err, thinking: append_retry_feedback(
                msgs, resp, err, thinking, feedback_suffix="Please respond concisely.",
            ),
            max_retries=max_retries,
        )

        return result.value if result.success else result.last_response

    def speak_stream(self, prompt: str, max_tokens: Optional[int] = None, task: Optional[str] = None, max_retries: int = 1) -> Generator[str, None, None]:
        """Have the oracle speak, streaming token by token.

        If the first stream yields nothing (all tokens on thinking),
        retries with a conversational nudge to be concise.
        """
        from compass.core.telemetry import with_provider_timing_stream
        from compass.llm.ladder_policy import get_max_tokens

        effective_max = max_tokens or get_max_tokens()
        active_provider = self._get_provider(task)
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(max_retries + 1):
            chunks = []
            for chunk in with_provider_timing_stream(
                provider_name=active_provider.name,
                stream_call=lambda: active_provider.stream(messages, effective_max),
                task=task or "speak_stream",
            ):
                chunks.append(chunk)
                yield chunk

            # If we got content, done
            if chunks:
                return

            # Empty stream -- check if truncated (all thinking, no content)
            done_reason = getattr(active_provider, '_last_done_reason', '')
            thinking = getattr(active_provider, '_last_thinking', '')
            if done_reason != "length" or attempt >= max_retries:
                return

            # Conversational retry: append what she was thinking + nudge
            messages = list(messages)
            if thinking:
                messages.append({"role": "assistant", "content": thinking})
            messages.append({"role": "user", "content":
                "Your response was cut off -- all tokens went to thinking. "
                "Please respond concisely and directly."
            })

    def converse(self, messages: List[Dict[str, str]], max_tokens: int = 300, task: Optional[str] = None, model: Optional[str] = None, seed: Optional[int] = None, temperature: Optional[float] = None, think_level: Optional["ThinkLevel"] = None) -> str:
        """Multi-turn conversation with the oracle."""
        from compass.core.telemetry import with_provider_timing

        provider = self._get_provider(task)
        return with_provider_timing(
            provider_name=provider.name,
            call=lambda: provider.complete(
                messages,
                max_tokens,
                model=model,
                seed=seed,
                temperature=temperature,
                think_level=think_level,
            ).text,
            think_level=think_level,
            temperature=temperature,
            seed=seed,
            task=task or "converse",
        )

    def converse_raw(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4000,
        task: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        think_level: Optional["ThinkLevel"] = None,
        provider: "Optional[Provider]" = None,
    ) -> "RawResponse":
        """
        Single LLM call with messages, returns text + thinking.

        Composable primitive for building retry loops. Unlike converse()
        which returns only text, this preserves thinking for proper
        message composition in retry scenarios.

        Example - compose with retry_with_messages:
            from compass.core.retry import retry_with_messages, AskResult

            def ask_once(msgs):
                r = oracle.converse_raw(msgs, max_tokens=4000, task="my-task")
                return AskResult(text=r.text, thinking=r.thinking)

            result = retry_with_messages(
                ask_once, messages, parse, validate, append_feedback
            )
        """
        import time
        from compass.core.telemetry import record_oracle_call, with_provider_timing
        start_time = time.time()

        provider = provider or self._get_provider(task)
        response = with_provider_timing(
            provider_name=provider.name,
            call=lambda: provider.complete(
                messages, max_tokens, seed=seed, temperature=temperature, think_level=think_level
            ),
            think_level=think_level,
            temperature=temperature,
            seed=seed,
            task=task or "converse",
        )

        # Record to telemetry with duration
        record_oracle_call(task or "converse", time.time() - start_time)

        return RawResponse(text=response.text, thinking=response.thinking or "", done_reason=response.done_reason or "")

    def converse_stream(self, messages: List[Dict[str, str]], max_tokens: int = 300, task: Optional[str] = None, model: Optional[str] = None) -> Generator[str, None, None]:
        """Multi-turn conversation, streaming.

        Args:
            model: Optional model override (e.g., vision model when images present)
        """
        from compass.core.telemetry import with_provider_timing_stream

        provider = self._get_provider(task)
        yield from with_provider_timing_stream(
            provider_name=provider.name,
            stream_call=(
                lambda: provider.stream(messages, max_tokens, model=model)
                if model and hasattr(provider, 'stream')
                else provider.stream(messages, max_tokens)
            ),
            task=task or "converse_stream",
        )

    @property
    def supports_streaming(self) -> bool:
        """Check if current provider supports streaming."""
        return self.provider.supports_streaming

    def _ask_json_legacy(
        self,
        prompt: str,
        response_schema: Dict[str, str],
        max_tokens: int,
        max_retries: int,
        task: str,
        provider: "Provider",
        iteration: int,
        think_level: Optional[ThinkLevel],
        on_thinking: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """JSON-based ask (legacy) - for backwards compatibility with dict schemas."""
        from compass.llm.ask import build_schema_prompt

        full_prompt = build_schema_prompt(prompt, response_schema)

        def validate_json(parsed: Dict) -> Optional[str]:
            try:
                validate_response(parsed, response_schema)
                return None
            except OracleSchemaError as e:
                return str(e)

        return self._ask_structured(
            full_prompt=full_prompt,
            parse=parse_json_response,
            validate=validate_json,
            feedback_suffix="Please correct and respond with valid JSON.",
            max_tokens=max_tokens,
            max_retries=max_retries,
            task=task,
            provider=provider,
            iteration=iteration,
            think_level=think_level,
            handle_truncation=True,
            on_thinking=on_thinking,
        )

    def _ask_structured(
        self,
        full_prompt: str,
        parse: Callable[[str], T],
        validate: Callable[[T], Optional[str]],
        feedback_suffix: str,
        max_tokens: int,
        max_retries: int,
        task: str,
        provider: "Provider" = None,
        iteration: int = 0,
        think_level: Optional["ThinkLevel"] = None,
        handle_truncation: bool = False,
        on_thinking: Optional[Callable[[str], None]] = None,
    ) -> T:
        """
        Structured ask with retry - composes ask_once with with_structured_retry.

        Args:
            full_prompt: Complete prompt with format instructions
            parse: Parse response text to value (raises on failure)
            validate: Validate value, returns error string or None
            feedback_suffix: What to ask for on retry
            max_tokens: Max tokens for response
            max_retries: How many correction attempts
            task: Task type for provider routing
            provider: Explicit provider override
            iteration: Problem-solving iteration
            think_level: Thinking level for extended reasoning
            handle_truncation: Whether to handle response truncation
            on_thinking: Optional callback for thinking chunks (str -> None)

        Returns:
            Parsed and validated result

        Raises:
            OracleSchemaError: If valid response not obtained after retries
        """
        from compass.llm.ask import (
            compute_retry_params, build_messages, append_retry_feedback,
        )
        from compass.core.retry import retry_with_messages, AskResult
        from compass.core.telemetry import (
            record_task_attempt,
            record_task_attempt_failure,
            record_task_provider_outcome,
        )

        # Handle images for vision
        has_images = hasattr(self, '_pending_images') and self._pending_images
        images = tuple(self._pending_images) if has_images else ()

        # Determine provider
        active_provider = provider if provider else self._get_provider(task)
        if has_images:
            self._pending_images = []  # Clear after use

        # Build initial messages
        messages = build_messages(full_prompt, active_provider.name, images)

        # Track attempt for retry params
        attempt_count = 0
        active_provider_time = 0.0

        # ask_once: single LLM call, returns AskResult
        def ask_once(msgs: List[Dict]) -> AskResult:
            nonlocal attempt_count, active_provider_time
            import time

            attempt_index = attempt_count
            params = compute_retry_params(iteration, attempt_index, think_level, _get_think_floor())
            attempt_count += 1
            record_task_attempt(task, attempt_index)

            start_time = time.monotonic()
            response = self._call_provider(
                active_provider, msgs, max_tokens,
                params.seed, params.temperature, params.think_level,
                iteration, attempt_index, think_level,
                task,
                on_thinking=on_thinking,
            )
            elapsed = time.monotonic() - start_time
            active_provider_time += elapsed

            _debug(f"Oracle {task} (attempt {attempt_count}) [{active_provider.name}]: {response.text[:500]}...")

            return AskResult(
                text=response.text.strip(),
                thinking=response.thinking or "",
                truncated=(response.done_reason == "length"),
            )

        # Compose: ask_once + retry
        def append_feedback(msgs, response, error, thinking):
            return append_retry_feedback(msgs, response, error, thinking, feedback_suffix)

        def on_retry_failure(attempt_index: int, failure_type: str, error_message: str) -> None:
            record_task_attempt_failure(task, attempt_index, failure_type, error_message)

        result = retry_with_messages(
            ask_once=ask_once,
            initial_messages=messages,
            parse=parse,
            validate=validate,
            append_feedback=append_feedback,
            max_retries=max_retries,
            on_retry_failure=on_retry_failure,
        )

        if result.success:
            record_task_provider_outcome(task, active_provider.name, "success")
            _debug(f"Success: {result.value}")
            return result.value

        record_task_provider_outcome(task, active_provider.name, "failed")
        raise OracleSchemaError(
            f"Failed after {max_retries + 1} attempts: {result.error}",
            raw_response=result.last_response
        )

    def _call_provider(
        self,
        provider: "Provider",
        messages: List[Dict],
        max_tokens: int,
        seed: Optional[int],
        temperature: Optional[float],
        think_level: "ThinkLevel",
        iteration: int,
        attempt: int,
        explicit_think: Optional["ThinkLevel"],
        task: str,
        on_thinking: Optional[Callable[[str], None]] = None,
    ) -> ProviderResponse:
        """
        Call provider with appropriate method based on capabilities.

        Args:
            on_thinking: Optional callback for thinking chunks. If provided,
                         bypasses default stdout streaming. Signature: (str) -> None

        Returns ProviderResponse with text, thinking, done_reason.
        No duck punching - all data comes from the response.
        """
        # Show thinking if: DEBUG, COMPASS_THINKING, escalating, or explicit think_level
        show_thinking = (
            os.getenv("DEBUG") or os.getenv("COMPASS_THINKING") or
            (iteration != 0 or attempt > 0) or explicit_think
        )

        # Determine thinking callback:
        # - If on_thinking provided, use it (injectable callback)
        # - Else if show_thinking, use default stdout stream
        # - Else None (no thinking output)
        thinking_cb = (
            on_thinking if on_thinking else
            show_thinking_stream if show_thinking else
            None
        )

        def run_call() -> ProviderResponse:
            from compass.llm.ladder_policy import get_max_tokens
            effective_max = max_tokens or get_max_tokens()
            if hasattr(provider, 'complete_with_thinking') and thinking_cb:
                # Only wrap with start/end if using default stdout stream
                if thinking_cb is show_thinking_stream:
                    start_thinking_stream()
                response = provider.complete_with_thinking(
                    messages, on_thinking=thinking_cb,
                    seed=seed, temperature=temperature, think_level=think_level,
                    max_tokens=effective_max,
                )
                if thinking_cb is show_thinking_stream:
                    end_thinking_stream()
                return response
            if hasattr(provider, 'complete_with_thinking'):
                return provider.complete_with_thinking(
                    messages, on_thinking=None,
                    seed=seed, temperature=temperature, think_level=think_level,
                    max_tokens=effective_max,
                )
            return provider.complete(
                messages, effective_max,
                seed=seed, temperature=temperature
            )

        from compass.core.telemetry import with_provider_timing
        return with_provider_timing(
            provider_name=provider.name,
            call=run_call,
            think_level=think_level,
            temperature=temperature,
            seed=seed,
            task=task,
        )

    def ask(
        self,
        prompt: str,
        response_type: "Optional[Union[Type[T], Dict[str, Any]]]" = None,
        max_tokens: Optional[int] = None,
        max_retries: int = 2,
        task: str = "ask",
        provider: "Optional[Provider]" = None,
        iteration: int = 0,
        think_level: Optional[ThinkLevel] = None,
        response_schema: "Optional[Dict[str, Any]]" = None,  # Legacy alias
        validate: "Optional[Callable[[T], Optional[str]]]" = None,  # External validation
        feedback_suffix: Optional[str] = None,  # Custom feedback on retry
        on_thinking: Optional[Callable[[str], None]] = None,  # Injectable thinking callback
        on_prompt: Optional[Callable[[str], None]] = None,  # Injectable prompt visitor
    ) -> "Union[T, Dict[str, Any], RawResponse]":
        """
        Ask the model for a response.

        Three modes based on response_type:
        - None: Raw output - model writes freely, returns RawResponse(text, thinking)
        - Type: Python path - model writes constructor, we eval and validate
        - Dict: JSON path (legacy) - model writes JSON, we parse and validate

        Args:
            prompt: The prompt to send (used to build full_prompt with type instructions)
            response_type: None (raw), Type (Python), or Dict schema (JSON)
            max_tokens: Max tokens for response
            max_retries: How many correction attempts before giving up
            task: Task type for provider routing
            provider: Explicit provider override
            iteration: Problem-solving iteration (higher = more creative)
            think_level: Thinking level for extended reasoning
            response_schema: Legacy alias for response_type
            validate: External validation function (result -> error_string or None)
            feedback_suffix: Custom feedback message on retry
            on_thinking: Optional callback for thinking chunks. Receives thinking
                         text as it streams. If None, uses default stdout streaming.

        Returns:
            RawResponse (if response_type=None), instance of response_type (if Type)

        Raises:
            OracleSchemaError: If valid response not obtained after retries

        Example (typed):
            assessment = oracle.ask("Are we making progress?", ProgressAssessment)

        Example (typed with external validation):
            result = oracle.ask(
                prompt, FileEditorResponse,
                validate=lambda r: _validate_file_edit(r, file_content),
                feedback_suffix="Make sure target is copied EXACTLY from the file.",
            )

        Example (raw - for code generation):
            response = oracle.ask(prompt, max_tokens=4000)  # Returns RawResponse

        Example (with thinking callback):
            chunks = []
            response = oracle.ask(prompt, on_thinking=chunks.append)
        """
        import time
        start_time = time.time()

        # Helper to record telemetry with duration before returning
        def record_and_return(result):
            from compass.core.telemetry import record_oracle_call
            record_oracle_call(task, time.time() - start_time)
            return result

        # Handle legacy response_schema parameter
        if response_schema is not None:
            response_type = response_schema

        # Dispatch based on response_type
        # None -> raw text output
        if response_type is None:
            return record_and_return(self._ask_raw_with_thinking(
                prompt, max_tokens, max_retries, task, provider, iteration, think_level,
                on_thinking=on_thinking,
            ))

        # Dict -> JSON path (legacy - for backwards compat)
        if isinstance(response_type, dict):
            return record_and_return(self._ask_json_legacy(
                prompt, response_type, max_tokens, max_retries,
                task, provider, iteration, think_level,
                on_thinking=on_thinking,
            ))

        from compass.core.python_schema import (
            get_type_source,
            collect_dependencies,
            parse_typed_response,
            validate_instance,
        )

        # Gather type definitions - what the model sees (alphabetical for predictability)
        deps = collect_dependencies(response_type)
        sorted_deps = sorted(deps, key=lambda t: t.__name__)
        type_defs = "\n\n".join(get_type_source(dep) for dep in sorted_deps)

        # Content blocks: if any type docstring shows the syntax, don't say "Nothing else"
        has_content_blocks = "# ===" in type_defs and "# === end ===" in type_defs

        closing = (
            f"RESPOND IN PYTHON. Write the {response_type.__name__}(...) expression first, then content blocks AFTER it (not inside strings). No XML tags, no markdown fences."
            if has_content_blocks else
            f"RESPOND IN PYTHON. Write ONLY the {response_type.__name__}(...) expression. Nothing else. No XML tags, no markdown fences."
        )

        # Build the prompt
        full_prompt = f"""{prompt}

Respond with a Python expression constructing {response_type.__name__}.
Use the exact enum values (e.g., EnumType.VALUE), not strings.

{type_defs}

{closing}"""

        # Visit prompt if callback provided
        if on_prompt:
            on_prompt(full_prompt)

        # Parse: use pure function from python_schema
        parse_type = lambda text: parse_typed_response(text, response_type)

        # Compose validate for Type mode: type check + field validation + external
        def validate_type(result: T) -> Optional[str]:
            """Validate result is correct type with correct field types."""
            if not isinstance(result, response_type):
                return f"Expected {response_type.__name__}, got {type(result).__name__}"
            field_error = validate_instance(result, response_type)
            if field_error:
                return f"Type error: {field_error}"
            return None

        # Compose internal + external validation
        def combined_validate(result: T) -> Optional[str]:
            type_error = validate_type(result)
            if type_error:
                return type_error
            if validate:
                return validate(result)
            return None

        # Default feedback suffix -- reinforce type names so model doesn't forget on retry
        type_names = ", ".join(t.__name__ for t in sorted_deps)
        if feedback_suffix:
            suffix = feedback_suffix
        elif has_content_blocks:
            suffix = (
                f"Write a Python constructor: {response_type.__name__}(field=value, ...). NOT XML, NOT JSON. "
                f"Available types: {type_names}. "
                f"Content blocks go AFTER the expression, not inside strings. "
                f"Leave field=None, then: # === name ===\\n...\\n# === end ==="
            )
        else:
            suffix = (
                f"Write a Python constructor: {response_type.__name__}(field=value, ...). NOT XML, NOT JSON. "
                f"Available types: {type_names}."
            )

        return record_and_return(self._ask_structured(
            full_prompt=full_prompt,
            parse=parse_type,
            validate=combined_validate,
            feedback_suffix=suffix,
            max_tokens=max_tokens,
            max_retries=max_retries,
            task=task,
            provider=provider,
            iteration=iteration,
            think_level=think_level,
            handle_truncation=False,
            on_thinking=on_thinking,
        ))

    def _ask_raw_with_thinking(
        self,
        prompt: str,
        max_tokens: int,
        max_retries: int,
        task: str,
        provider: "Provider",
        iteration: int,
        think_level: Optional[ThinkLevel],
        on_thinking: Optional[Callable[[str], None]] = None,
    ) -> "RawResponse":
        """
        Raw output - model writes freely, returns text + thinking.

        No schema instructions appended. Caller handles validation/retry.
        Returns RawResponse so caller can use thinking for retry context.

        Args:
            prompt: The prompt (used as-is, no modifications)
            max_tokens: Max tokens for response
            max_retries: Unused (caller handles retry)
            task: Task type for provider routing
            provider: Explicit provider override
            iteration: Problem-solving iteration
            think_level: Thinking level for extended reasoning
            on_thinking: Optional callback for thinking chunks (str -> None)

        Returns:
            RawResponse with text and thinking
        """
        from compass.llm.ask import compute_retry_params, build_messages

        # Handle images for vision
        has_images = hasattr(self, '_pending_images') and self._pending_images
        images = tuple(self._pending_images) if has_images else ()

        # Determine provider (vision provider if images present)
        if has_images:
            active_provider = provider if provider else self._get_provider(task, for_vision=True)
            self._pending_images = []  # Clear after use
        else:
            active_provider = provider if provider else self._get_provider(task)

        # Build messages with images if present
        messages = build_messages(prompt, active_provider.name, images)
        params = compute_retry_params(iteration, 0, think_level, _get_think_floor())

        response = self._call_provider(
            active_provider, messages, max_tokens,
            params.seed, params.temperature, params.think_level,
            iteration, 0, think_level, task,
            on_thinking=on_thinking,
        )

        _debug(f"Oracle {task} (raw) [{active_provider.name}]: {response.text[:500]}...")

        return RawResponse(text=response.text, thinking=response.thinking or "", done_reason=response.done_reason or "")
