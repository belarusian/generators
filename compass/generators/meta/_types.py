"""Meta-generator types."""

from __future__ import annotations

from dataclasses import dataclass

from compass.generators._types import Err, Ok, Result


@dataclass(frozen=True)
class SourceFile:
    """A Python source file in the generated module."""

    path: str        # e.g. "_types.py", "_runtime.py"
    content: str = ""
    description: str = ""


@dataclass(frozen=True)
class GeneratorModuleMeta:
    """Metadata header for a generator module.

    A generator module is a Python package with four files.
    Write a GeneratorModuleMeta constructor, then raw file contents
    after ### banners:

    GeneratorModuleMeta(
        name="regex_tool",
        purpose="A regex pattern validation tool",
        domain="text processing",
        test_prompt="Validate phone numbers and emails",
    )

    ### _types.py ###
    from dataclasses import dataclass
    from compass.generators._types import Err, Ok, Result

    @dataclass(frozen=True)
    class Spec:
        name: str

    def validate_spec(raw: dict) -> Result[Spec, str]:
        ...

    ### _runtime.py ###
    from ._types import Spec
    from compass.generators._types import Err, Ok, Result

    def invoke_model(ctx, model_id, ask_fn=None) -> Result:
        ...

    ### _context.py ###
    from compass.generators._types import DomainSection, GenerationContext

    def build_context(prompt=None, root=None) -> GenerationContext:
        ...

    ### generate.py ###
    from ._types import Spec, validate_spec
    from ._runtime import invoke_model
    from compass.generators._types import Err, Ok, Result
    from compass.generators._loop import generation_loop, result_to_exit

    def run(prompt=None, *, model_id="", **kw) -> Result:
        ...
    """

    name: str
    purpose: str
    domain: str
    test_prompt: str
    spec_type_name: str = "Spec"


@dataclass(frozen=True)
class GeneratorModuleSpec:
    """Complete generator module specification (internal).

    Assembled from GeneratorModuleMeta + parsed file sections.
    Not written by the model directly.
    """

    name: str
    purpose: str
    domain: str
    files: tuple[SourceFile, ...]
    test_prompt: str
    spec_type_name: str = "Spec"


@dataclass(frozen=True)
class FileEdit:
    """A targeted edit within a file: find old text, replace with new."""

    old: str
    new: str


@dataclass(frozen=True)
class FilePatch:
    """Patch for a single file. Either targeted edits or full replacement.

    Edits (preferred -- surgical):
    FilePatch(path="_runtime.py", edits=(
        FileEdit(old="exact text to find", new="replacement text"),
    ))

    Full replacement:
    FilePatch(path="_context.py", content="full new content here")
    """

    path: str
    content: str | None = None
    edits: tuple[FileEdit, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.edits, FileEdit):
            object.__setattr__(self, "edits", (self.edits,))


@dataclass(frozen=True)
class ModulePatch:
    """A partial update to a GeneratorModuleSpec.

    Ouroboros returns this instead of the full spec. Each FilePatch
    targets a single file with search-and-replace edits. The model
    only returns the changes, not the full file content.

    ModulePatch(
        files=(
            FilePatch(path="_runtime.py", edits=(
                FileEdit(old="text to find", new="replacement text"),
            )),
        ),
    )
    """

    files: tuple[FilePatch, ...]

    def __post_init__(self) -> None:
        # Models often omit the trailing comma: files=(FilePatch(...)) is a bare
        # FilePatch, not a 1-tuple.
        if isinstance(self.files, FilePatch):
            object.__setattr__(self, "files", (self.files,))


_REQUIRED_FILES = frozenset({"_types.py", "_runtime.py", "_context.py", "generate.py"})


def validate_spec_instance(spec: GeneratorModuleSpec) -> Result[GeneratorModuleSpec, str]:
    """Validate a GeneratorModuleSpec instance.

    Checks what the constructor can't enforce: non-empty content,
    valid identifier, required files present.
    """
    from dataclasses import replace

    errors: list[str] = []

    if not spec.name or not spec.name.isidentifier():
        errors.append(f"name must be a valid Python identifier, got '{spec.name}'")

    if not spec.purpose:
        errors.append("purpose must be non-empty")

    if not spec.files:
        errors.append("files must be non-empty")

    seen: set[str] = set()
    for i, sf in enumerate(spec.files):
        if not sf.content:
            errors.append(f"files[{i}] ({sf.path}): content is empty")
        if sf.path in seen:
            errors.append(f"files[{i}]: duplicate path '{sf.path}'")
        seen.add(sf.path)

    filtered = tuple(sf for sf in spec.files if sf.path in _REQUIRED_FILES)
    missing = _REQUIRED_FILES - {sf.path for sf in filtered}
    if missing:
        errors.append(f"missing required files: {sorted(missing)}")

    if errors:
        return Err("; ".join(errors))

    return Ok(replace(spec, files=filtered))


def apply_patch(
    spec: GeneratorModuleSpec,
    patch: ModulePatch,
) -> Result[GeneratorModuleSpec, str]:
    """Apply a ModulePatch to a GeneratorModuleSpec.

    Each FilePatch targets a file by path. Each FileEdit within it
    is a search-and-replace on the file content. If an old string
    is not found, the patch fails with a precise error.
    """
    file_map = {sf.path: sf for sf in spec.files}
    errors: list[str] = []

    for fp in patch.files:
        if fp.path not in file_map:
            errors.append(f"{fp.path}: not in module")
            continue

        if fp.content is not None:
            file_map[fp.path] = SourceFile(
                path=fp.path,
                content=fp.content,
                description=file_map[fp.path].description,
            )
        else:
            content = file_map[fp.path].content
            for edit in fp.edits:
                if edit.old not in content:
                    errors.append(
                        f"{fp.path}: old text not found: {edit.old[:80]}..."
                    )
                    continue
                content = content.replace(edit.old, edit.new, 1)

            file_map[fp.path] = SourceFile(
                path=fp.path,
                content=content,
                description=file_map[fp.path].description,
            )

    if errors:
        return Err("; ".join(errors))

    merged = tuple(file_map[sf.path] for sf in spec.files)
    return Ok(GeneratorModuleSpec(
        name=spec.name,
        purpose=spec.purpose,
        domain=spec.domain,
        files=merged,
        test_prompt=spec.test_prompt,
        spec_type_name=spec.spec_type_name,
    ))
