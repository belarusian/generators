"""
Python as Schema - machinery for letting models speak Python directly.

No JSON. No escaping. The type IS the schema.

Used by Oracle.ask_python():
    oracle.ask_python("Assess progress", ProgressAssessment)

Model writes:
    ProgressAssessment(
        signal=ProgressSignal.STALLED,
        confidence=0.8,
        reasoning="Too many reads"
    )

We eval it. Done.
"""

import ast
import inspect
import re
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set, Type, TypeVar, get_type_hints, get_origin, get_args

from compass.core.reasoning import debug

T = TypeVar('T')



def get_type_source(cls: Type) -> str:
    """
    Get the source definition of a type.

    For dataclasses: the full class definition
    For enums: the enum with all values
    """
    try:
        return inspect.getsource(cls)
    except (OSError, TypeError):
        # Can't get source, build a representation
        if is_dataclass(cls):
            return _build_dataclass_repr(cls)
        elif issubclass(cls, Enum):
            return _build_enum_repr(cls)
        return f"class {cls.__name__}: ..."


def _build_dataclass_repr(cls: Type) -> str:
    """Build a string representation of a dataclass."""
    lines = [f"@dataclass", f"class {cls.__name__}:"]
    hints = get_type_hints(cls)
    for field in fields(cls):
        type_name = _type_name(hints.get(field.name, field.type))
        lines.append(f"    {field.name}: {type_name}")
    return "\n".join(lines)


def _build_enum_repr(cls: Type) -> str:
    """Build a string representation of an enum."""
    lines = [f"class {cls.__name__}(Enum):"]
    for member in cls:
        lines.append(f"    {member.name} = {repr(member.value)}")
    return "\n".join(lines)


def _type_name(t: Type) -> str:
    """Get a readable name for a type."""
    origin = get_origin(t)
    args = get_args(t)

    if origin is list:
        return f"List[{_type_name(args[0])}]" if args else "List"
    if origin is dict:
        return "Dict[str, Any]"
    if hasattr(t, "__name__"):
        return t.__name__
    return str(t)


def collect_dependencies(cls: Type) -> Set[Type]:
    """
    Collect all types needed to construct this type.

    Walks the type tree: enums, nested dataclasses, etc.
    """
    deps = set()

    if is_dataclass(cls):
        deps.add(cls)
        hints = get_type_hints(cls)
        for field in fields(cls):
            field_type = hints.get(field.name, field.type)
            deps.update(_collect_from_type(field_type))
    elif isinstance(cls, type) and issubclass(cls, Enum):
        deps.add(cls)

    return deps


def _collect_from_type(t: Type) -> Set[Type]:
    """Collect dependencies from a type annotation."""
    deps = set()
    origin = get_origin(t)
    args = get_args(t)

    if origin is not None:
        for arg in args:
            if arg is not type(None):
                deps.update(_collect_from_type(arg))
    elif isinstance(t, type):
        if issubclass(t, Enum):
            deps.add(t)
        elif is_dataclass(t):
            deps.update(collect_dependencies(t))

    return deps


def build_eval_context(response_type: Type, *extra_types: Type) -> Dict[str, Any]:
    """
    Build the eval() context with all needed types.

    Includes the response type and all its dependencies. Optional *extra_types
    are unioned in (and their collect_dependencies), so callers can inject
    sibling types the model may reference (e.g. SpecPatch inside a Spec block)
    that are not reachable from the primary type's field annotations.
    """
    deps: Set[Type] = set(collect_dependencies(response_type))
    for et in extra_types:
        deps.update(collect_dependencies(et))

    context: Dict[str, Any] = {}
    for dep in deps:
        context[dep.__name__] = dep

    # Always include common types
    context["None"] = None
    context["True"] = True
    context["False"] = False

    return context


def extract_constructor(response: str, type_name: str) -> Optional[str]:
    """
    Extract a Python constructor expression from model response.

    Looks for: TypeName(...) pattern
    Handles multiline constructors with proper paren matching.
    """
    # Find the start of the constructor
    pattern = rf'\b{re.escape(type_name)}\s*\('
    match = re.search(pattern, response)

    if not match:
        return None

    start = match.start()

    # Find matching closing paren
    depth = 0
    i = match.end() - 1  # Start at the opening paren

    while i < len(response):
        char = response[i]
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                return response[start:i+1]
        elif char == '"' or char == "'":
            # Skip string literals
            quote = char
            i += 1
            while i < len(response) and response[i] != quote:
                if response[i] == '\\':
                    i += 1  # Skip escaped char
                i += 1
        i += 1

    return None


def validate_python_block(code: str) -> tuple[bool, Optional[str], str]:
    """
    Validate a Python block is safe to exec.

    Returns (is_safe, error_message, code).
    The third element is the (possibly repaired) code -- models miscount
    parens in nested constructors, so we try appending closing parens
    before giving up.

    The real security boundary is __builtins__: {} on exec. This check
    rejects only import statements (which bypass the namespace sandbox).
    Everything else -- definitions, control flow, expressions -- is safe
    when builtins are removed.
    """
    try:
        tree = ast.parse(code, mode='exec')
    except SyntaxError as e:
        # Models miscount parens in deeply nested constructors.
        # Try appending closing parens before giving up.
        if "was never closed" in str(e) or "unexpected EOF" in str(e):
            patched = code
            for _ in range(4):
                patched += ")"
                try:
                    tree = ast.parse(patched, mode='exec')
                    code = patched
                    break
                except SyntaxError:
                    continue
            else:
                return False, f"Syntax error: {e}", code
        else:
            return False, f"Syntax error: {e}", code

    # Only reject imports -- they can bypass the __builtins__: {} sandbox
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, f"Import statements not allowed: {ast.dump(node)}", code

    return True, None, code


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences if model wrapped its response in them.

    Handles: ```python ... ```, ``` ... ```, ```py ... ```
    Preserves content blocks that may follow the constructor.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text

    lines = stripped.split("\n")

    # Remove opening fence line (```python, ```py, ```, etc.)
    lines = lines[1:]

    # Find and remove closing fence -- last line that is just ```
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "```":
            lines = lines[:i] + lines[i+1:]
            break

    result = "\n".join(lines)
    debug("[parse] stripped markdown fences from response")
    from compass.core.telemetry import record_parse_recovery
    record_parse_recovery("markdown_fences")
    return result


def parse_typed_response(
    text: str,
    response_type: Type[T],
    *,
    extra_types: tuple[Type, ...] = (),
) -> T:
    """
    Parse text as a typed Python block.

    Pure function: text + type -> instance (or raises ValueError).

    The model writes Python: variable assignments (triple-quoted strings
    for multi-line content) followed by a constructor expression. We exec
    the block and scan the namespace for an instance of the response type.

    The type IS the schema. No JSON, no content blocks, no markers.

    extra_types: additional types (and their nested dataclass deps) merged into
    the exec namespace when the model references names not on the primary type
    (e.g. SpecPatch or ShellStep alongside Spec).
    """
    # Strip markdown fences if present
    text = _strip_markdown_fences(text)

    code = text.strip()
    if not code:
        raise ValueError("Empty response")

    # Validate it's safe Python (may auto-close trailing parens)
    is_safe, safety_error, code = validate_python_block(code)
    if not is_safe:
        raise ValueError(f"Invalid Python: {safety_error}")

    # Exec in a restricted namespace with all needed types
    eval_context = build_eval_context(response_type, *extra_types)
    namespace = dict(eval_context)

    try:
        exec(compile(code, "<model-response>", "exec"), {"__builtins__": {}}, namespace)
    except (SyntaxError, TypeError, NameError) as e:
        raise ValueError(f"Constructor error: {e}") from e

    # Scan namespace for an instance of the response type
    # Check newest bindings first (last assigned wins)
    for key in reversed(list(namespace)):
        if key in eval_context:
            continue  # Skip the type definitions we injected
        if isinstance(namespace[key], response_type):
            return namespace[key]

    # No named variable -- maybe the last statement is a bare expression
    # (e.g. just "Spec(...)" with no assignment). Try eval on the last line.
    try:
        tree = ast.parse(code, mode='exec')
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.get_source_segment(code, tree.body[-1])
            if last_expr:
                result = eval(last_expr, {"__builtins__": {}}, namespace)
                if isinstance(result, response_type):
                    return result
    except Exception:
        pass

    raise ValueError(
        f"No {response_type.__name__} instance found in response. "
        f"Assign it to a variable or end with a bare {response_type.__name__}(...) expression."
    )



_BANNER_RE = re.compile(r'^###\s+(.+?)\s+###\s*$', re.MULTILINE)


def split_file_sections(text: str) -> list[tuple[str, str]]:
    """Split text on ### filename ### banners.

    Returns list of (path, content) pairs.
    """
    matches = list(_BANNER_RE.finditer(text))
    if not matches:
        return []

    sections = []
    for i, m in enumerate(matches):
        path = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip('\n')
        sections.append((path, content))
    return sections


def parse_response_with_files(
    text: str,
    response_type: Type[T],
    *,
    extra_types: tuple[Type, ...] = (),
) -> tuple[T, list[tuple[str, str]]]:
    """Parse a response that has a constructor header + file sections.

    The model writes:
        TypeName(field=value, ...)

        ### path/to/file.py ###
        (raw file content)

        ### another/file.py ###
        (raw file content)

    Returns (metadata_instance, [(path, content), ...]).

    extra_types: forwarded to parse_typed_response for the constructor line.
    """
    text = _strip_markdown_fences(text)

    # Find the first banner -- everything before it is the constructor
    first_banner = _BANNER_RE.search(text)
    if not first_banner:
        raise ValueError(
            "No file sections found. Use ### filename ### banners to delimit files."
        )

    constructor_text = text[:first_banner.start()]
    files_text = text[first_banner.start():]

    # Parse the constructor for metadata
    meta = parse_typed_response(constructor_text, response_type, extra_types=extra_types)

    # Split file sections
    files = split_file_sections(files_text)
    if not files:
        raise ValueError("No files found after banners.")

    return meta, files


def validate_instance(obj, expected_type: Type) -> Optional[str]:
    """
    Validate that an instance has correct field types.

    Returns error message if any field has wrong type, None if valid.
    Checks enum fields especially - model often writes strings instead.
    """
    if not is_dataclass(obj):
        return None  # Only validate dataclasses

    try:
        hints = get_type_hints(type(obj))
    except Exception:
        return None  # Can't get hints, skip validation

    errors = []
    for field in fields(obj):
        value = getattr(obj, field.name)
        expected = hints.get(field.name)

        if expected is None:
            continue

        # Handle Optional[X] - extract X
        origin = get_origin(expected)
        if origin is type(None):
            continue
        if origin is not None:  # Optional, Union, etc.
            args = get_args(expected)
            # For Optional[X], check against X if value is not None
            if value is None:
                continue
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                expected = non_none_args[0]

        # Check enum fields - most common error
        if isinstance(expected, type) and issubclass(expected, Enum):
            if not isinstance(value, expected):
                errors.append(
                    f"{field.name}: expected {expected.__name__}.VALUE (e.g., {expected.__name__}.{list(expected)[0].name}), "
                    f"got {type(value).__name__} '{value}'"
                )

    return "; ".join(errors) if errors else None
