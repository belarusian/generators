"""Python file generator module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, Union

from compass.generators._types import Err, Ok, Result

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Spec:
    """Specification for a Python source file."""
    path: str
    content: str
    description: str


def validate_spec(raw: dict) -> Result[Spec, str]:
    """Validate raw dict into Spec. [STRUCTURAL]"""
    if not isinstance(raw, dict):
        return Err(f"expected dict, got {type(raw).__name__}")

    path = raw.get("path")
    if not isinstance(path, str) or not path.strip():
        return Err("path must be a non-empty string")
    path = path.strip()

    content = raw.get("content")
    if not isinstance(content, str) or not content.strip():
        return Err("content must be a non-empty string")
    content = content.strip()

    description = raw.get("description", "")
    if not isinstance(description, str):
        return Err("description must be a string")
    description = description.strip()

    return Ok(Spec(path=path, content=content, description=description))