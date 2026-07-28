# FileEditor Architecture Guide

## Overview

The FileEditor system is a specialized agent that uses LLMs to intelligently edit files. It's used by the `edit_file` action handler to perform code modifications with validation and retry logic.

## Core Components

### 1. Oracle Pattern (Oracle.ask())

The `oracle.ask()` method is the core abstraction for structured LLM responses:

```python
response = oracle.ask(
    prompt,
    ResponseType,  # Python class to construct
    max_retries=3,
    validate=validation_func,  # Optional: check result
    feedback_suffix="Helpful hint on retry"
)
```

**How it works:**
1. Builds a prompt with `ResponseType` definition
2. Asks LLM to construct a Python expression: `ResponseType(...)`
3. Parses and validates the response
4. Retries with feedback on failure

### 2. FileEditor Types

#### FileEditorResponse
The LLM returns an instance of this class:

```python
@dataclass
class FileEditorResponse:
    operation: FileEditOperation  # REPLACE | INSERT | DELETE
    target: str                   # Text to replace (from file)
    content: str                  # Replacement content
    reasoning: str                # Why this edit is needed
```

#### FileEditResult
The handler returns this after parsing:

```python
@dataclass
class FileEditResult:
    success: bool
    operation: FileEditOperation
    target: str
    content: str
    reasoning: str
    error: Optional[str] = None
```

### 3. Validation Flow

`_validate_file_edit()` checks:
1. Operation is valid (REPLACE/INSERT/DELETE)
2. Target is provided
3. Target exists in file content
4. Target is unique (not duplicated)

```python
def _validate_file_edit(response, file_content) -> Optional[str]:
    if operation not in valid_ops:
        return "Invalid operation"
    if not target:
        return "No target specified"
    if target not in file_content:
        return "Target not found"
    if file_content.count(target) > 1:
        return "Target not unique"
```

### 4. Content Blocks System

Content blocks mark special content sections:

```python
# === content: path="file.py" ===
def add(a, b):
    return a + b
# === end ===
```

**Purpose:** Used to separate metadata from actual content, allowing zero-escaping and flexible structure.

### 5. Execution Flow

```
edit_file.py execute handler
    ↓
call_file_editor(oracle, file_path, file_content, instruction)
    ↓
oracle.ask(prompt, FileEditorResponse, validate=_validate_file_edit)
    ↓
LLM responds: FileEditorResponse(operation=..., target=..., content=..., reasoning=...)
    ↓
_parse_file_edit(response) → FileEditResult
    ↓
Apply edit with retry logic
```

## Key Patterns

### 1. External Validation Pattern

The `call_file_editor` uses external validation via `oracle.ask()`:

```python
response = oracle.ask(
    prompt,
    FileEditorResponse,
    validate=lambda r: _validate_file_edit(r, file_content),
    feedback_suffix="Fix the issue. Make sure target is copied EXACTLY from the file."
)
```

This pattern:
- Preserves full context in message history
- Allows custom validation logic
- Provides retry feedback to LLM

### 2. Retry with Feedback

The `edit_file` handler retries with error feedback:

```python
for attempt in range(max_retries):
    effective_instruction = (
        f"{instruction}\n\nPREVIOUS ATTEMPT FAILED:\n{last_error}"
        if last_error else instruction
    )
    result = call_file_editor(...)
    # ... apply and validate ...
```

### 3. Deterministic String Replacement

The actual edit uses whitespace-tolerant fallback:

```python
def _apply_edit_with_fallback(content, target, replacement, operation):
    # Try exact match
    if target in content:
        # ... apply edit
    # Try tab-expanded match
    normalized_content = content.expandtabs()
    normalized_target = target.expandtabs()
    # ... apply edit
```

## Usage in Actions

The `edit_file` action uses this pattern:

```python
action: EditFileAction = {
    "action_type": "edit_file",
    "path": "file.py",
    "instruction": "Add multiply function after add"
}

# execute handler calls:
result = call_file_editor(oracle, "file.py", file_content, instruction)
```

## File Locations

- **Oracle**: `compass/llm/oracle.py` - ask() method and type handling
- **FileEditor**: `compass/agents/neo/file_editor.py` - call_file_editor(), validation
- **Edit File Handler**: `compass/agents/neo/actions/edit_file.py` - execute handler
- **Content Blocks**: `compass/core/content_blocks.py` - parsing logic

## Design Principles

1. **Separation of Concerns**: FileEditor handles HOW (finding/editing), Handler handles WHAT (description)
2. **Zero-Escaping**: Content blocks allow raw content without escape issues
3. **Validation-First**: Check validity before writing to prevent corruption
4. **Retry Logic**: LLM self-corrects with feedback on failures