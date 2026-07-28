"""
Code artifact generator.

Produces standalone Python code artifacts (modules, scripts, CLI tools)
from natural language prompts. Uses the shared generation loop with
five wired functions:

    invoke   : G   -- Context -> Result[raw]
    parse    : V1  -- raw -> Result[CodeSpec]
    validate : V2  -- CodeSpec -> Result[ExecutedCode]
    fix      : G'  -- (CodeSpec, error, ctx) -> CodeSpec | None
    emit     : IO  -- (CodeSpec, ExecutedCode, ...) -> Result[Path]

Validation composes cheapest-first:
    structural (JSON shape) -> semantic (AST) -> executive (exec)
"""

from compass.generators.code._types import CodeSpec, CodeFile, CodeTestCase, validate_spec
