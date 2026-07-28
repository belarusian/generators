"""IO boundaries for the meta-generator.

V_meta: materialize module -> import -> run inner generator.
G'_meta: ouroboros -- narrow patch loop, never rewrites the whole module.

The fix loop is deliberately narrow:
  1. Model receives its prior module source + the specific error
  2. Model returns a ModulePatch (only changed files, surgical edits)
  3. Patch is applied to produce corrected spec
  4. Re-validate from the cheapest validator that could catch the error

This keeps the inner loop fast and focused. The outer loop (wholesale
generation) handles total rewrites when ouroboros cannot converge.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from compass.generators._types import (
    AskFn,
    DomainSection,
    Err,
    GenerationContext,
    Ok,
    Result,
)
from compass.generators._invoke import (
    build_system_prompt,
    resolve_ask_fn,
)

from compass.generators.meta._types import (
    GeneratorModuleMeta,
    GeneratorModuleSpec,
    ModulePatch,
    SourceFile,
    _REQUIRED_FILES,
    apply_patch,
)

logger = logging.getLogger(__name__)

_META_TYPE_SOURCE = inspect.getsource(GeneratorModuleMeta)
_TYPES_SOURCE = (Path(__file__).parent / "_types.py").read_text()


# ---------------------------------------------------------------------------
# Model invocation -- G_meta
# ---------------------------------------------------------------------------


def invoke_model(
    ctx: GenerationContext,
    model_id: str = "",
    ask_fn: AskFn | None = None,
) -> Result:
    """Call the model and return a GeneratorModuleSpec. G_meta.

    Python-as-schema: model writes GeneratorModuleMeta constructor
    followed by ### banner ### file sections. No JSON, no escaping.
    """
    fn = resolve_ask_fn(model_id, ask_fn)

    system = build_system_prompt(
        ctx,
        _META_TYPE_SOURCE,
        role=(
            "You are an expert at building code generation pipelines. "
            "You generate Python generator modules that follow a specific architecture."
        ),
        contract_preamble=(
            "Respond as shown in the GeneratorModuleMeta docstring."
        ),
    )

    user = _build_meta_user_message(ctx)

    logger.debug("--- G_meta SYSTEM ---\n%s\n--- END SYSTEM ---", system)
    logger.debug("--- G_meta USER ---\n%s\n--- END USER ---", user)

    match fn(system, user):
        case Err() as e:
            return e
        case Ok(raw_text):
            pass

    logger.debug("--- G_meta RESPONSE ---\n%s\n--- END RESPONSE ---", raw_text)

    from compass.core.python_schema import parse_response_with_files
    try:
        meta, files = parse_response_with_files(raw_text, GeneratorModuleMeta)
    except ValueError as e:
        return Err(str(e))

    # Assemble full spec from metadata + file sections
    source_files = tuple(
        SourceFile(path=path, content=content)
        for path, content in files
    )
    spec = GeneratorModuleSpec(
        name=meta.name,
        purpose=meta.purpose,
        domain=meta.domain,
        files=source_files,
        test_prompt=meta.test_prompt,
        spec_type_name=meta.spec_type_name,
    )
    return Ok(spec)


def _build_meta_user_message(ctx: GenerationContext) -> str:
    """Build user message for meta-generation.

    The model has the type (GeneratorModuleSpec) and an exemplar in domain
    context. The user message is just the prompt + format hint + feedback.
    """
    primary = (
        ctx.user_prompt if ctx.user_prompt is not None else
        "Generate a generator module."
    )
    parts = [primary]

    parts.extend([
        "",
        "Write a GeneratorModuleMeta(...) constructor followed by ### banner ### file sections.",
        "Follow the shape of the exemplar in your system prompt.",
        "No markdown fencing.",
    ])

    if ctx.available_packages:
        parts.extend(["", f"Available packages: {ctx.available_packages}"])

    if ctx.feedback:
        parts.extend(["", "Your previous attempt had errors:", ""])
        for fb in ctx.feedback:
            parts.append(f"  {fb}")
        parts.extend(["", "Please fix these issues in your next attempt."])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Validation pipeline -- V_meta (cheapest first)
# ---------------------------------------------------------------------------


def validate_module_sources(spec: GeneratorModuleSpec) -> Result[None, str]:
    """V_meta layer 1 -- Semantic: ast.parse() + cross-module imports.

    Pure, cheapest. Two phases:
    1. Parse every .py file (catches syntax errors)
    2. Check intra-module imports resolve (catches missing functions)
    """
    from compass.generators._validation import collect_definitions

    errors: list[str] = []
    trees: dict[str, ast.Module] = {}

    # Phase 1: parse
    for sf in spec.files:
        if not sf.path.endswith(".py"):
            continue
        try:
            trees[sf.path] = ast.parse(sf.content)
        except SyntaxError as exc:
            loc = f" at line {exc.lineno}" if exc.lineno else ""
            errors.append(f"{sf.path}: SyntaxError{loc}: {exc.msg}")

    if errors:
        return Err("; ".join(errors))

    # Phase 2: cross-module import check
    defs_by_stem: dict[str, set[str]] = {}
    for path, tree in trees.items():
        stem = path.removesuffix(".py")
        defs_by_stem[stem] = set(collect_definitions(tree).keys())

    abs_prefix = f"compass.generators.{spec.name}."

    module_stems = set(defs_by_stem.keys())

    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue

            # Bare package import: from . import X
            # __init__.py is auto-generated (just a docstring), so only
            # submodule imports work (from . import _types), not names
            # (from . import Spec).
            if node.level == 1 and node.module is None:
                for alias in node.names:
                    if alias.name not in module_stems:
                        errors.append(
                            f"{path}: 'from . import {alias.name}' -- "
                            f"__init__.py does not export '{alias.name}'. "
                            f"Use 'from ._types import {alias.name}' instead"
                        )
                continue

            if not node.module:
                continue

            # Resolve target stem for both relative and absolute imports
            target = None
            if node.level == 1:
                # from .module import name
                target = node.module
            elif node.level == 0 and node.module.startswith(abs_prefix):
                # from compass.generators.<name>.module import name
                target = node.module[len(abs_prefix):]

            if target is None or target not in defs_by_stem:
                continue

            defined = defs_by_stem[target]
            for alias in node.names:
                if alias.name not in defined:
                    errors.append(
                        f"{path}: imports '{alias.name}' from {target}.py, "
                        f"but {target}.py does not define it"
                    )

    if errors:
        return Err("; ".join(errors))
    return Ok(None)


def materialize_module(spec: GeneratorModuleSpec) -> tuple[str, str]:
    """Write module files to a temp directory. Returns (tmpdir, module_name)."""
    tmpdir = tempfile.mkdtemp(prefix=f"gen_{spec.name}_")
    pkg_dir = Path(tmpdir) / spec.name
    pkg_dir.mkdir()

    init_path = pkg_dir / "__init__.py"
    init_content = next(
        (sf.content for sf in spec.files if sf.path == "__init__.py"),
        f'"""Generated generator: {spec.name}."""\n',
    )
    init_path.write_text(init_content)

    for sf in spec.files:
        fpath = pkg_dir / sf.path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(sf.content)

    return tmpdir, spec.name


def import_module(tmpdir: str, name: str) -> Result:
    """V_meta layer 2 -- Executive (lightweight): import the materialized module.

    Crosses representation boundary. Catches import errors that
    ast.parse misses (missing dependencies, circular imports, etc.).
    """
    sys.path.insert(0, tmpdir)
    try:
        mod = importlib.import_module(name)
        return Ok(mod)
    except Exception as exc:
        return Err(f"Import error: {type(exc).__name__}: {exc}")
    finally:
        if tmpdir in sys.path:
            sys.path.remove(tmpdir)


def _install_temporarily(spec: GeneratorModuleSpec) -> tuple[Path, Path | None]:
    """Install module to compass/generators/<name>/ for executive testing.

    Returns (pkg_dir, backup_dir). Caller must call _uninstall after.
    The module needs to be in its real home so it can import siblings
    via compass.generators.<name>.<module>.
    """
    import compass.generators
    generators_root = Path(compass.generators.__path__[0])
    pkg_dir = generators_root / spec.name

    backup = None
    if pkg_dir.exists():
        backup = pkg_dir.with_name(f".{spec.name}_backup")
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(pkg_dir, backup)
        shutil.rmtree(pkg_dir)

    pkg_dir.mkdir(parents=True, exist_ok=True)

    init_content = next(
        (sf.content for sf in spec.files if sf.path == "__init__.py"),
        f'"""Generated generator: {spec.name}."""\n',
    )
    (pkg_dir / "__init__.py").write_text(init_content)

    for sf in spec.files:
        fpath = pkg_dir / sf.path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(sf.content)

    return pkg_dir, backup


def _uninstall(pkg_dir: Path, backup: Path | None):
    """Restore previous state after executive testing."""
    shutil.rmtree(pkg_dir, ignore_errors=True)
    if backup and backup.exists():
        shutil.move(str(backup), str(pkg_dir))


def _clear_module_cache(prefix: str):
    """Remove all sys.modules entries matching prefix."""
    mods_to_remove = [k for k in sys.modules if k.startswith(prefix)]
    for k in mods_to_remove:
        del sys.modules[k]


def run_inner_generator(
    spec: GeneratorModuleSpec,
    model_id: str = "",
) -> Result:
    """V_meta layer 3 -- Executive (full): run the inner generator.

    The inner generator's generation_loop IS the test. If it produces
    an artifact, the module works. If it fails, the error bubbles up
    to the ouroboros fix loop.

    Installs to compass/generators/<name>/ so the module can find
    its siblings, then calls run() directly -- no sys.argv hacking.
    """
    cache_prefix = f"compass.generators.{spec.name}"
    pkg_dir, backup = _install_temporarily(spec)
    module_name = f"compass.generators.{spec.name}.generate"
    try:
        _clear_module_cache(cache_prefix)

        gen_mod = importlib.import_module(module_name)

        if not hasattr(gen_mod, "run"):
            return Err(f"{module_name} has no run() function")

        result = gen_mod.run(
            prompt=spec.test_prompt,
            model_id=model_id or "",
            max_rounds=1,
            max_fixes=1,
        )
        match result:
            case Ok(_):
                return Ok(True)
            case Err(e):
                return Err(f"Inner generator failed: {e}")

    except Exception as exc:
        return Err(f"Inner generator error: {type(exc).__name__}: {exc}")
    finally:
        _uninstall(pkg_dir, backup)
        _clear_module_cache(cache_prefix)


def validate_generated_module(
    spec: GeneratorModuleSpec,
    model_id: str = "",
    ask_fn: AskFn | None = None,
) -> Result:
    """Full V_meta pipeline. Cheapest first:

    1. Semantic: ast.parse() every .py file (pure, cheapest)
    2. Executive (light): materialize + import in tmpdir
    3. Executive (full): install to real location, run inner generator

    Each layer catches a different class of error:
    - Layer 1: syntax errors (typos, bad indentation)
    - Layer 2: import errors (missing deps, circular imports)
    - Layer 3: runtime errors (logic bugs, wrong API usage)
    """
    # 1. Semantic -- pure, cheapest
    match validate_module_sources(spec):
        case Err(e):
            return Err(e)

    logger.info("V_meta layer 1: all sources parse cleanly")

    # 2. Executive (light) -- materialize + import in tmpdir
    tmpdir, name = materialize_module(spec)
    try:
        match import_module(tmpdir, name):
            case Err(e):
                return Err(e)
            case Ok(mod):
                pass
        logger.info("V_meta layer 2: module imports successfully")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _clear_module_cache(name)

    # 3. Executive (full) -- run inner generator in real environment
    match run_inner_generator(spec, model_id):
        case Err(e):
            logger.warning("V_meta layer 3 failed:\n%s", str(e)[:3000])
            return Err(f"Inner generator failed: {e}")
        case Ok(_):
            pass

    logger.info("V_meta layer 3: inner generator ran successfully")
    return Ok(True)


# ---------------------------------------------------------------------------
# Ouroboros -- G'_meta (narrow patch loop)
# ---------------------------------------------------------------------------


def ouroboros_meta(
    spec: GeneratorModuleSpec,
    error: str,
    ctx: GenerationContext,
    model_id: str = "",
    ask_fn: AskFn | None = None,
) -> GeneratorModuleSpec | None:
    """G'_meta: model sees its prior module source + error, returns a PATCH.

    The ouroboros fix is deliberately narrow:
    - Model receives the full module listing (for context)
    - Model receives the specific error message
    - Model returns a ModulePatch with only the changed files
    - Each change is a surgical search-and-replace edit
    - The model NEVER rewrites the whole module here

    ctx is used to enrich the system prompt with domain knowledge:
    - Domain context sections (shared framework, exemplar, principles)
      help the fix model understand what APIs are available and what
      patterns to follow.
    - Available packages inform dependency choices.
    - Feedback history from prior rounds provides learning context.

    This keeps the inner fix loop fast while giving the model enough
    context to make correct fixes. If ouroboros cannot converge
    after max_fixes attempts, the outer loop scraps the whole spec
    and starts fresh with the error as feedback.
    """
    fn = resolve_ask_fn(model_id, ask_fn)

    # Build a compact listing of all module files
    file_listing: list[str] = []
    for sf in spec.files:
        file_listing.append(f"### {sf.path}")
        file_listing.append(sf.content)

    # Build domain context summary from ctx for the fix model
    domain_hints: list[str] = []
    for ds in ctx.domain_context:
        if ds.content and not ds.content.startswith("No "):
            # Include a condensed version -- heading + first 500 chars
            # The fix model needs awareness, not the full framework source
            preview = ds.content[:500]
            if len(ds.content) > 500:
                preview += "\n... (truncated)"
            domain_hints.append(f"## {ds.heading}\n{preview}")

    system_parts = [
        "Write a ModulePatch(...) expression to fix the error below.",
        "See the ModulePatch docstring in the types for the format.",
        f"Module: {spec.name} -- {spec.purpose}",
        "",
        _TYPES_SOURCE,
    ]

    user_parts = [
        f"# Your module: {spec.name} ({len(spec.files)} files)",
        "",
        "\n".join(file_listing),
        "",
        "# Error to fix",
        "",
        error[:4000],  # Truncate very long errors
        "",
        "Write a ModulePatch(...) expression with targeted edits to fix this error.",
        "Be surgical -- only change what is broken.",
        "No markdown fencing.",
    ]

    # Include feedback from prior rounds if available
    if ctx.feedback:
        user_parts.extend([
            "",
            "# Prior round feedback (for context)",
            "",
        ])
        for fb in ctx.feedback[-3:]:  # Last 3 feedback items to stay focused
            user_parts.append(f"  {fb}")

    system_text = "\n".join(system_parts)
    user_text = "\n".join(user_parts)

    logger.debug("--- G'_meta SYSTEM ---\n%s\n--- END SYSTEM ---", system_text)
    logger.debug("--- G'_meta USER ---\n%s\n--- END USER ---", user_text)

    match fn(system_text, user_text):
        case Err(e):
            logger.warning("Meta-ouroboros model error: %s", e)
            return None
        case Ok(raw_text):
            pass

    logger.debug("--- G'_meta RESPONSE ---\n%s\n--- END RESPONSE ---", raw_text)

    from compass.core.python_schema import parse_typed_response
    try:
        patch = parse_typed_response(raw_text, ModulePatch)
    except ValueError as e:
        logger.warning(
            "Meta-ouroboros parse error: %s\n--- raw (first 1000) ---\n%s",
            e, raw_text[:1000],
        )
        return None

    total_edits = sum(len(fp.edits) for fp in patch.files)
    full_replacements = sum(1 for fp in patch.files if fp.content is not None)
    logger.info(
        "Meta-ouroboros patch: %d file(s), %d edit(s), %d full replacement(s): %s",
        len(patch.files), total_edits, full_replacements,
        [fp.path for fp in patch.files],
    )

    match apply_patch(spec, patch):
        case Err(e):
            logger.warning("Meta-ouroboros patch apply failed: %s", e)
            return None
        case Ok(corrected):
            return corrected


# ---------------------------------------------------------------------------
# Emit -- write generated module to disk
# ---------------------------------------------------------------------------


def emit_module(spec: GeneratorModuleSpec, output_dir: Path) -> Path:
    """Write a GeneratorModuleSpec to disk as a Python package.

    Creates compass/generators/<name>/ with all source files.
    """
    pkg_dir = output_dir / spec.name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    init_path = pkg_dir / "__init__.py"
    init_content = next(
        (sf.content for sf in spec.files if sf.path == "__init__.py"),
        f'"""Generated generator: {spec.name} -- {spec.purpose}."""\n',
    )
    init_path.write_text(init_content)

    for sf in spec.files:
        fpath = pkg_dir / sf.path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(sf.content)

    logger.info("Generated module written to %s/", pkg_dir)
    return pkg_dir


# ---------------------------------------------------------------------------
# Load -- read generator directory back into GeneratorModuleSpec
# ---------------------------------------------------------------------------


def _extract_docstring(source: str) -> str:
    """Pull the module-level docstring from Python source, or return ''."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        return tree.body[0].value.value.strip()
    return ""


def load_module(path: Path) -> Result:
    """Load a generator directory into a GeneratorModuleSpec.

    Inverse of emit_module. Reads all .py files from the directory
    and reconstructs the spec. Purpose is extracted from the
    __init__.py docstring if present.
    """
    if not path.is_dir():
        return Err(f"Not a directory: {path}")

    name = path.name
    if not name.isidentifier():
        return Err(f"Directory name is not a valid identifier: {name}")

    files: list[SourceFile] = []
    for py_file in sorted(path.glob("*.py")):
        files.append(SourceFile(
            path=py_file.name,
            content=py_file.read_text(),
            description="",
        ))

    if not files:
        return Err(f"No .py files in {path}")

    missing = _REQUIRED_FILES - {sf.path for sf in files}
    if missing:
        return Err(f"Missing required files: {sorted(missing)}")

    init = next((sf for sf in files if sf.path == "__init__.py"), None)
    purpose = _extract_docstring(init.content) if init else name

    return Ok(GeneratorModuleSpec(
        name=name,
        purpose=purpose,
        domain=name,
        files=tuple(files),
        test_prompt=f"Test the {name} generator",
    ))


def serialize_spec(spec: GeneratorModuleSpec) -> str:
    """Serialize a GeneratorModuleSpec to a human-readable string.

    Used by refine_context to inject the existing artifact into
    the generation context.
    """
    parts = [
        f"# Module: {spec.name}",
        f"# Purpose: {spec.purpose}",
        f"# Domain: {spec.domain}",
        f"# Test prompt: {spec.test_prompt}",
        "",
    ]
    for sf in spec.files:
        parts.append(f"## {sf.path}")
        parts.append("")
        parts.append(sf.content)
        parts.append("")
    return "\n".join(parts)
