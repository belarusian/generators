# Generation as Composition

Findings from generalizing a notebook generator into a parameterized
framework, testing it across five models and two transports, and building
a meta-generator that generates generators. This document picks up where
`tools/ai-code-generation-lessons.md` left off.

The prior paper established the core loop: `G -> V -> G'`, the two-tier
structure (wholesale rounds x ouroboros fixes), the cheapest-first
validator ordering, and the observation that the generator is not
domain-specific. This paper asks: if the loop is invariant, what exactly
varies? And if we can parameterize the variance, can we generate the
generator itself?


## 1. The loop is invariant; only five functions change

We migrated the generator from Bedrock to Compass Providers (`tools/notebook_gen/`
to `tools/compass_gen/`). Different transport, different model families,
same loop. Then we ran the same loop against five models:

| Model | Transport | First-pass fixes | Notes |
|-------|-----------|-------------------|-------|
| Claude Sonnet 4.6 | Anthropic API | 1 | Notebook-focused output |
| Claude Opus 4.6 | Anthropic API | 0 | Frontier |
| qwen3.5:122b | Ollama @big | 0 | Local MoE, competitive |
| qwen3-coder-next:q8_0 | Ollama @big | 0 | Compact, clean |
| glm-4.7-flash:bf16 | Ollama @big | N/A | Could not produce JSON |

The loop did not change between models. The prompt did not change. The
validators did not change. The only thing that changed was the `AskFn` --
the function that sends (system, user) and returns text. Everything else
was shared.

This meant the generation loop was already parameterized over one function
(invoke). But the loop also hardcoded the Spec type (NotebookSpec), the
validation pipeline (validate_python_cells + execute_cells), the ouroboros
strategy (cell-targeted), and the emit format (.ipynb).

The question became: what if those are parameters too?

We extracted five functions:

```
invoke   : G   -- Context -> Result[raw]
parse    : V1  -- raw -> Result[Spec]
validate : V2  -- Spec -> Result[Artifact]
fix      : G'  -- (Spec, error, ctx) -> Spec | None
emit     : IO  -- (Spec, Artifact, ...) -> Result[Path]
```

The loop takes these as arguments. It does not know what Spec it is
validating. It does not know what Artifact it is emitting. It only knows
how to orchestrate the five functions in the two-tier structure.


## 2. Type is identity

We ran a control experiment: Morpheus (action-based agent) writing a
notebook vs the generator writing the same notebook. Morpheus produced
invalid JSON -- an unterminated string in cell source. The .ipynb would
not open in Jupyter.

The generator cannot produce this failure. Not because it tries harder,
but because the type constraint makes it impossible. The model's output
must parse as `NotebookSpec`. If it does not parse, it does not exist.
The structural validator rejects it before it touches disk.

Morpheus wrote bytes to a file. The generator produced a typed value
that was validated before serialization. The difference is not in the
model's capability -- it is in what the system permits to exist.

This is the "type is identity" principle: the Spec type defines what
can exist. Validation is derived from the type. If you control the type,
you control the space of possible outputs. The model operates within
that space, not outside it.

The corollary for Neo (the action-based agent): Neo should not write
artifacts directly. Neo should choose a generator and delegate. The
generator enforces the type constraint. Neo chooses which constraint
to enforce.


## 3. A model that cannot parse is not reasoning

glm-4.7-flash:bf16 could not produce JSON. Not malformed JSON -- no JSON
at all. The response was prose. The structural validator rejected it at
character 0.

The instinct was to make the parser more robust -- extract JSON from
surrounding prose, strip markdown fencing more aggressively. We rejected
this. If the model cannot follow the instruction "return raw JSON only",
it is not reasoning about the task. Stripping its output to find
something that happens to parse is not convergence on truth -- it is
the filter of good intentions.

The model was discarded. The system moved on. This is the correct
response to a model that cannot meet the base contract. The ouroboros
cannot help because the model never produced a valid starting point for
the ouroboros to consume. You cannot fix what does not exist.


## 4. Error messages are the teaching mechanism -- and they must not lie

The prior paper established that error quality determines self-correction
quality. We observed the same pattern across all five models. But we
also observed a subtlety: error messages that are "helpful" but imprecise
are worse than error messages that are blunt but accurate.

When qwen3.5:122b produced a notebook with a variable flow error, the
error message said exactly where and what:

```
cells[7]: 'results' may be undefined -- defined conditionally in cells[5]
```

The model fixed it in one ouroboros pass. The structural view
(`summarize_variable_flow`) gave it the topology. The model reasoned
about the fix.

The temptation is to add "hints" -- "try moving the definition outside
the if block" or "consider using a default value." We did not. The
topology is the truth. The hint is an opinion. The model is better at
deriving fixes from topology than at following opinions for novel
situations.


## 5. The generator is a generic problem

Section 10 of the prior paper stated this as an observation. We now have
evidence.

The same loop produced validated notebooks from:
- 5 different models (Sonnet, Opus, qwen3.5, qwen3-coder-next, glm)
- 2 transports (Anthropic API, Ollama)
- 2 domains (Compass with ETHOS/design docs, generic with no context)
- Multiple artifact shapes (notebooks with/without FileSpec extraction)

The only things that changed between generators are:
1. The Spec type (what shape is the artifact?)
2. The validators (what does "valid" mean for that shape?)
3. The executive proof (how to exec it?)
4. The domain context (what does the model need to know?)

The loop itself -- invoke, parse, validate, fix, emit -- is invariant.


## 6. If the loop is invariant, it can be parameterized; if parameterized, it can be generated

This is the meta-generator insight. A generator module is:

```
_types.py     -- Spec type + structural validators
_runtime.py   -- execution, ouroboros, emit/load
_context.py   -- domain context builders
generate.py   -- wires five functions into generation_loop
```

This is a set of Python source files. A set of Python source files is an
artifact. An artifact can be generated by `G -> V -> G'`.

The meta-generator's Spec is `GeneratorModuleSpec` -- a name, purpose,
domain, and a tuple of `SourceFile` values. Its V is:

1. `ast.parse()` every .py file (semantic, cheapest)
2. Materialize to tmpdir, `importlib.import_module()` (executive)
3. Run the inner generator's `main()` with a test prompt (executive)

Step 3 is where it gets recursive. The inner generator runs its own
`G -> V -> G'` loop. If the inner loop produces a valid artifact, V_meta
passes. If the inner loop fails, the error bubbles up to the
meta-generator's ouroboros, which sees the module source + the inner
error and produces a corrected module.

Nested ouroboros: inner fixes the artifact, outer fixes the generator.

```
G_meta  : Query -> GeneratorModuleSpec
V_meta  : materialize + import + run inner G -> V -> G'
G_meta' : (GeneratorModuleSpec, Error) -> GeneratorModuleSpec
```

The meta-generator's context includes:
- The shared framework source (so the model knows what to import)
- The notebook generator source (so the model sees a working exemplar)
- FP architecture principles (so the model follows the shape)

The model studies the exemplar and produces a new generator that follows
the same shape. The V pipeline proves it works. If it does not work, the
error says why, and ouroboros corrects it.


## 7. Self-similarity as verification

The ultimate test of the meta-generator:

```
python -m compass generator --type meta \
  --prompt "Build a notebook generator"
```

The generated generator should produce valid .ipynb notebooks. This is
self-similarity -- the system generates a copy of itself and verifies
the copy works. If it does, the architecture is sound. If it does not,
the failure tells you what the architecture depends on that was not
captured in the types and the exemplar.

This is not circular. The meta-generator and the notebook generator are
different programs. The meta-generator produces source code. The notebook
generator produces notebooks. They share the loop, but the loop is
parameterized -- each instance fills in different functions. The
self-similarity is in the shape, not the identity.


## 8. From tools/ to compass/generators/

The generators moved from `tools/` to `compass/generators/`. This is not
a refactor -- it is a reclassification. A tool is something you use from
outside the system. A generator is part of the system's architecture.

The generation loop is the same mechanism that could power:
- Neo choosing a generator (via GenerateNotebookAction or similar)
- Refine mode for any artifact type
- CI validation of generated artifacts
- The meta-generator producing new generators

These are not tool invocations. They are architectural compositions.
The generator is a first-class component, not an external dependency.


## 9. Exemplar mimicry -- the model copies what it sees

The meta-generator shows the model a type docstring with an exemplar
response. The model copies it literally: import paths, function signatures,
patterns. When the exemplar's `generate.py` section didn't import `Result`
but used `-> Result:` in the function signature, every generated module had
the same `NameError`. When it didn't import `Ok` and `Err`, every generated
module failed at runtime.

The exemplar is not documentation. It is a program the model will reproduce
with variations. Every import, every type annotation, every pattern must be
a working example. If the exemplar is wrong, the output is wrong --
consistently, across models, across rounds.

This extends to all type docstrings used as response contracts. SHOW don't
TELL: the docstring example shows the format. The user message steers but
does not re-specify. One place defines the format. The model follows it.


## 10. The response format evolved: JSON -> Python -> banners

The original system used JSON. The model wrote `{"cells": [...]}` and the
parser validated it as a dict. This worked but had two problems: JSON
requires escaping (backslashes in code, quotes in strings) and JSON has no
type safety (the parser must validate every field manually).

Python-as-schema (Finding 8 of the prior paper) replaced JSON with typed
constructor expressions. The model writes `Spec(name="...", files=(...))`.
The type IS the schema. `parse_typed_response` execs the constructor in a
restricted namespace and scans for an instance of the response type. Type
safety is free -- the constructor enforces it.

But code-inside-strings still required escaping. A regex pattern
`r'[\w.+-]+@[\w-]+\.[\w.]+'` inside a Python string needs careful quoting.
Multi-line code needs `\n` or triple-quotes. Models struggled with this
consistently: broken escaping, truncated responses, `None` to avoid the
problem entirely.

The banner format is the third evolution: constructor for metadata, raw code
after `### name ###` delimiters. No string boundaries to cross.

```
Evolution:

JSON              -> escaping, no type safety
Python-as-schema  -> types as contract, code still in strings
Banners           -> constructor for metadata, raw code after ### markers
```

The parser (`parse_response_with_files`) splits at the first `### name ###`
banner, parses the constructor via `parse_typed_response`, and returns
`(instance, [(name, content), ...])`. A runtime helper (`_attach_banner_code`)
maps banner sections back to the typed fields (file content, test source).

The try-banner/fallback pattern handles model variation: try banner parse
first, fall back to `parse_typed_response` if no banners found. The lean
prompt (`inspect.getsource(Type)`) shows only the types the model writes,
with the docstring example demonstrating the banner format.


## Summary

| Finding | Implication |
|---------|-------------|
| The loop is invariant | Parameterize over five functions, not Spec types |
| Type is identity | The Spec constrains what can exist |
| Cannot-parse is not reasoning | Discard, don't accommodate |
| Error messages must not lie | Topology, not hints |
| Generation is generic | Same loop, different functions |
| Parameterized can be generated | Meta-generator generates generators |
| Self-similarity verifies architecture | Generate a copy, verify it works |
| Generators are architecture, not tools | compass/generators/, not tools/ |
| Exemplar mimicry | The model copies docstring examples literally; they must be correct |
| JSON -> Python -> banners | Constructor for metadata, raw code after ### markers, no escaping |


## Lineage

```
tools/notebook_gen/           -- The original. Bedrock + Neptune.
                                 Proved G -> V -> G' works.

tools/ai-code-generation-lessons.md
                              -- The paper. Documented findings.

tools/compass_gen/            -- Absorbed version. Compass Providers.
                                 Same architecture, different transport.
                                 Tested across 5 models, 2 transports.

compass/generators/           -- Generalized. Parameterized loop.
                                 Shared core, banner response format.
  _types.py _loop.py          -- The invariant.
  _invoke.py _validation.py   -- The shared machinery.
  core/python_schema.py       -- parse_typed_response, parse_response_with_files
  notebook/                   -- Notebooks (.ipynb).
  code/                       -- Code artifacts (files + tests).
  module/                     -- Python modules (package + test).
  trinity/                    -- Artifact orchestrator (plan + exec).
  neo/                        -- Plan generator (delegates to others).
  meta/                       -- The generator of generators.
```
