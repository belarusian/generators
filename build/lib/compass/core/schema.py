"""
Schema derivation from Python types.

The machinery that makes typed prompts work:
- dataclass_to_schema: Python dataclass -> JSON Schema
- dataclass_from_dict: raw dict -> typed dataclass instance
- TypedPrompt[T]: prompt that knows its return type

Single source of truth: the dataclass IS the schema.
"""

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Type, TypeVar,
    Union, get_type_hints, get_origin, get_args
)

T = TypeVar("T")


@dataclass(frozen=True)
class TypedPrompt(Generic[T]):
    """
    A prompt coupled to its response type.

    The type IS the schema - no drift possible.
    """
    text: str
    response_type: Type[T]


# --- Schema Derivation ---

def dataclass_to_schema(cls: Type) -> Dict[str, Any]:
    """
    Derive JSON Schema from a dataclass.

    Handles:
    - Basic types: str, int, float, bool
    - Optional[T] -> nullable field
    - List[T] -> array with items schema
    - Enum -> string with enum values
    - Nested dataclasses -> nested object schema

    Example:
        @dataclass
        class Learning:
            type: LearningType  # enum
            summary: str
            facts: Optional[List[str]] = None

        schema = dataclass_to_schema(Learning)
        # -> {"type": "object", "properties": {...}, "required": [...]}
    """
    from dataclasses import MISSING

    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    properties = {}
    required = []

    hints = get_type_hints(cls)

    for field in fields(cls):
        field_type = hints.get(field.name, field.type)
        field_schema, is_optional = _type_to_schema(field_type)
        properties[field.name] = field_schema

        # Required if no default AND not Optional
        has_default = field.default is not MISSING
        has_default_factory = field.default_factory is not MISSING
        if not has_default and not has_default_factory and not is_optional:
            required.append(field.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


def _type_to_schema(python_type: Type) -> tuple[Dict[str, Any], bool]:
    """
    Convert a Python type to JSON Schema.

    Returns (schema_dict, is_optional).
    """
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Optional[T] -> T with nullable (Union[T, None])
    if origin is Union:
        # Filter out NoneType
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner_schema, _ = _type_to_schema(non_none[0])
            return inner_schema, True
        # Multiple types - just use first non-None for now
        if non_none:
            inner_schema, _ = _type_to_schema(non_none[0])
            return inner_schema, True
        return {"type": "string"}, True

    # List[T] -> array
    if origin is list:
        if args:
            items_schema, _ = _type_to_schema(args[0])
            return {"type": "array", "items": items_schema}, False
        return {"type": "array"}, False

    # Dict[K, V] -> object
    if origin is dict:
        return {"type": "object"}, False

    # Enum -> string with enum values
    if isinstance(python_type, type) and issubclass(python_type, Enum):
        return {
            "type": "string",
            "enum": [e.value for e in python_type]
        }, False

    # Nested dataclass -> nested object
    if is_dataclass(python_type):
        return dataclass_to_schema(python_type), False

    # Basic types
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }

    if python_type in type_map:
        return type_map[python_type], False

    # Fallback
    return {"type": "string"}, False


# --- Parsing ---

def dataclass_from_dict(raw: Dict[str, Any], cls: Type[T]) -> T:
    """
    Parse a raw dict into a typed dataclass instance.

    Handles:
    - Basic types (pass through)
    - Enum (string -> enum member)
    - Optional (None handling)
    - List (recursive for complex items)
    - Nested dataclasses (recursive)

    Example:
        raw = {"type": "file_read", "summary": "Found config"}
        learning = dataclass_from_dict(raw, Learning)
        # -> Learning(type=LearningType.FILE_READ, summary="Found config")
    """
    from dataclasses import MISSING

    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    hints = get_type_hints(cls)
    kwargs = {}

    for field in fields(cls):
        field_type = hints.get(field.name, field.type)

        if field.name in raw:
            value = raw[field.name]
            kwargs[field.name] = _parse_value(value, field_type)
        elif field.default is not MISSING:
            kwargs[field.name] = field.default
        elif field.default_factory is not MISSING:
            kwargs[field.name] = field.default_factory()
        # else: missing required field - let dataclass raise

    return cls(**kwargs)


def _parse_value(value: Any, target_type: Type) -> Any:
    """Parse a single value to target type."""
    if value is None:
        return None

    origin = get_origin(target_type)
    args = get_args(target_type)

    # Optional[T] - unwrap and parse inner
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _parse_value(value, non_none[0])
        return value

    # List[T] - parse each item
    if origin is list and args:
        item_type = args[0]
        return [_parse_value(item, item_type) for item in value]

    # Enum - string to member
    if isinstance(target_type, type) and issubclass(target_type, Enum):
        try:
            return target_type(value)
        except ValueError:
            # Return first member as fallback
            return next(iter(target_type))

    # Nested dataclass
    if is_dataclass(target_type) and isinstance(value, dict):
        return dataclass_from_dict(value, target_type)

    # Basic types - pass through
    return value


# --- Reflect ---

def create_typed_reflect(oracle: "Oracle") -> Callable[[TypedPrompt[T]], T]:
    """
    Create a reflect function with oracle baked in.

    Usage:
        reflect = create_typed_reflect(oracle)
        learning = reflect(TypedPrompt("What did we learn?", Learning))
    """
    def reflect(prompt: TypedPrompt[T]) -> T:
        schema = dataclass_to_schema(prompt.response_type)
        raw = oracle.ask(prompt.text, response_schema=schema, max_tokens=200, task="reflect")
        return dataclass_from_dict(raw, prompt.response_type)

    return reflect


# --- Convenience: Prompt Builders ---

def prompt(text: str, response_type: Type[T]) -> TypedPrompt[T]:
    """Shorthand for TypedPrompt creation."""
    return TypedPrompt(text=text, response_type=response_type)
