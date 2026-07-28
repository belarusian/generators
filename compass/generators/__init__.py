"""
compass.generators -- functional composition generators.

G -> V -> G', parameterized over Spec and Artifact.

Shared core:
    _types      : Result, DomainSection, FileSpec, GenerationContext, AskFn
    _loop       : generation_loop, refine_loop, repl_loop
    _invoke     : resolve_ask_fn, build_system_prompt
    _validation : validate_python_sources, summarize_variable_flow

Generator modules:
    notebook/   : .ipynb notebook generator
    meta/       : meta-generator (generates generators)
"""

from ._types import (
    AskFn,
    DomainSection,
    Err,
    FileSpec,
    GenerationContext,
    GenerationReport,
    Ok,
    Result,
)
from ._loop import generation_loop, repl_loop

__all__ = [
    "AskFn",
    "DomainSection",
    "Err",
    "FileSpec",
    "GenerationContext",
    "GenerationReport",
    "Ok",
    "Result",
    "generation_loop",
    "repl_loop",
]
