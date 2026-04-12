"""CLI tool generator types."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Optional

from compass.generators._types import Err, Ok, Result


@dataclass(frozen=True)
class Argument:
    """A CLI argument for a subcommand."""

    name: str           # e.g. "--path", "--pattern", "filename"
    help_text: str
    required: bool = True
    default: Optional[str] = None
    arg_type: str = "str"  # str, int, float, bool


@dataclass(frozen=True)
class Subcommand:
    """A subcommand in the CLI tool."""

    name: str           # e.g. "count", "search"
    help_text: str
    arguments: tuple[Argument, ...]
    handler_body: str   # Python source for the handler function body


@dataclass(frozen=True)
class CliToolSpec:
    """Complete CLI tool specification.

    The model writes a Python constructor expression::

        CliToolSpec(
            name="fileutils",
            version="1.0.0",
            description="File utility CLI",
            subcommands=( ... ),
            source=None,
        )

        # === source ===
        #!/usr/bin/env python3
        ...
        # === end ===

    ``parse_typed_response`` fills *source* from the content block.

    The source field contains the full Python script. The structured
    fields (name, version, subcommands) are metadata used for
    validation -- they must be consistent with the source.
    """

    name: str
    version: str
    description: str
    subcommands: tuple[Subcommand, ...]
    source: str | None = None  # filled from content block


_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _validate_argument(raw: dict, prefix: str) -> Result[Argument, str]:
    """Validate a single argument dict."""
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return Err(f"{prefix}.name: must be a non-empty string")

    help_text = raw.get("help_text", "")
    if not isinstance(help_text, str):
        return Err(f"{prefix}.help_text: must be a string")

    required = raw.get("required", True)
    if not isinstance(required, bool):
        return Err(f"{prefix}.required: must be a boolean")

    default = raw.get("default")
    if default is not None and not isinstance(default, str):
        default = str(default)

    arg_type = raw.get("arg_type", "str")
    if arg_type not in ("str", "int", "float", "bool"):
        return Err(f"{prefix}.arg_type: must be str/int/float/bool, got '{arg_type}'")

    return Ok(Argument(
        name=name.strip(),
        help_text=help_text.strip(),
        required=required,
        default=default,
        arg_type=arg_type,
    ))


def _validate_subcommand(raw: dict, prefix: str) -> Result[Subcommand, str]:
    """Validate a single subcommand dict."""
    errors: list[str] = []

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return Err(f"{prefix}.name: must be a non-empty string")
    name = name.strip()

    help_text = raw.get("help_text", "")
    if not isinstance(help_text, str):
        return Err(f"{prefix}.help_text: must be a string")

    handler_body = raw.get("handler_body", "pass")
    if not isinstance(handler_body, str):
        return Err(f"{prefix}.handler_body: must be a string")

    raw_args = raw.get("arguments", [])
    if not isinstance(raw_args, list):
        return Err(f"{prefix}.arguments: must be a list")

    arguments: list[Argument] = []
    for i, ra in enumerate(raw_args):
        if not isinstance(ra, dict):
            errors.append(f"{prefix}.arguments[{i}]: expected dict")
            continue
        match _validate_argument(ra, f"{prefix}.arguments[{i}]"):
            case Ok(arg):
                arguments.append(arg)
            case Err(e):
                errors.append(e)

    if errors:
        return Err("; ".join(errors))

    return Ok(Subcommand(
        name=name,
        help_text=help_text.strip(),
        arguments=tuple(arguments),
        handler_body=handler_body,
    ))


def validate_spec(raw: dict) -> Result[CliToolSpec, str]:
    """Validate a raw JSON dict into a CliToolSpec. [STRUCTURAL]

    Cheapest validator -- pure dict inspection, no I/O.
    Checks field types, required fields, semver format.
    """
    if not isinstance(raw, dict):
        return Err(f"expected dict, got {type(raw).__name__}")

    errors: list[str] = []

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return Err("name must be a non-empty string")
    name = name.strip()
    if not name.isidentifier():
        return Err(f"name must be a valid Python identifier, got '{name}'")

    version = raw.get("version")
    if not isinstance(version, str) or not version.strip():
        return Err("version must be a non-empty string")
    version = version.strip()
    if not _SEMVER_RE.match(version):
        return Err(f"version must be semver (X.Y.Z), got '{version}'")

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        return Err("description must be a non-empty string")

    source = raw.get("source")
    if not isinstance(source, str) or not source.strip():
        return Err("source must be a non-empty string")

    raw_subcmds = raw.get("subcommands")
    if not isinstance(raw_subcmds, list) or not raw_subcmds:
        return Err("subcommands must be a non-empty list")

    subcommands: list[Subcommand] = []
    seen_names: set[str] = set()
    for i, rs in enumerate(raw_subcmds):
        if not isinstance(rs, dict):
            errors.append(f"subcommands[{i}]: expected dict")
            continue
        match _validate_subcommand(rs, f"subcommands[{i}]"):
            case Ok(sc):
                if sc.name in seen_names:
                    errors.append(f"subcommands[{i}]: duplicate name '{sc.name}'")
                else:
                    seen_names.add(sc.name)
                    subcommands.append(sc)
            case Err(e):
                errors.append(e)

    if errors:
        return Err("; ".join(errors))

    return Ok(CliToolSpec(
        name=name,
        version=version,
        description=description.strip(),
        subcommands=tuple(subcommands),
        source=source,
    ))


def validate_spec_instance(spec: CliToolSpec) -> Result[CliToolSpec, str]:
    """Validate a CliToolSpec instance (from parse_typed_response). [STRUCTURAL]

    Mirrors validate_spec but operates on the already-constructed dataclass
    rather than a raw dict.
    """
    errors: list[str] = []
    if not spec.name or not spec.name.isidentifier():
        errors.append(f"name must be a valid Python identifier, got '{spec.name}'")
    if not spec.version:
        errors.append("version must be non-empty")
    elif not _SEMVER_RE.match(spec.version):
        errors.append(f"version must be semver (X.Y.Z), got '{spec.version}'")
    if not spec.description:
        errors.append("description must be non-empty")
    if not spec.source:
        errors.append("source must be non-empty")
    if not spec.subcommands:
        errors.append("subcommands must be non-empty")
    if errors:
        return Err("; ".join(errors))
    return Ok(spec)
