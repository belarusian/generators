# Generators

Functional composition generators. The generation loop is parameterized
over five functions -- the loop itself is invariant. The five functions
define the generator:

```
invoke   : G   -- Context -> Result[raw]
parse    : V1  -- raw -> Result[Spec]
validate : V2  -- Spec -> Result[Artifact]
fix      : G'  -- (Spec, error, ctx) -> Spec | None
emit     : IO  -- (Spec, Artifact, ...) -> Result[Path]
```

```
G -> V -> G', repeat until V(artifact) = Ok or budget exhausted
```

```
python -m compass generator --type notebook --prompt "write a fibonacci program"
python -m compass generator --type meta --prompt "build a CLI tool generator"
```


## Architecture

```
compass/generators/
  _types.py          # Shared foundation: Ok, Err, Result, DomainSection,
                     #   FileSpec, GenerationContext, GenerationReport, AskFn
  _loop.py           # Parameterized generation_loop, refine_loop, repl_loop
  _invoke.py         # Model invocation: resolve_ask_fn,
                     #   build_system_prompt, build_user_message
  _validation.py     # Python source validation: DefCollector,
                     #   validate_python_sources, summarize_variable_flow

  notebook/          # Notebook generator (.ipynb)
    _types.py        # NotebookSpec, NotebookCell, CellResult, validators
    _runtime.py      # execute_cells, ouroboros, emit_notebook, load_notebook
    _context.py      # build_compass_context, build_generic_context
    generate.py      # Wires five functions into generation_loop

  meta/              # Meta-generator (generates generators)
    _types.py        # GeneratorModuleSpec, SourceFile, validate_module_spec
    _runtime.py      # V_meta: ast.parse -> import -> run inner generator
    _context.py      # Model sees shared framework + exemplar + principles
    generate.py      # Wires five functions into generation_loop
```


### Shared Core

The shared core imports only stdlib (except `_invoke.py`, which crosses to
Compass's Provider abstraction). The four modules are:

- **`_types.py`** -- the Result monad, domain sections, file specs, the
  immutable `GenerationContext` accumulator, and the `AskFn` contract.
  Everything is frozen. Everything returns Result.

- **`_loop.py`** -- the generation loop parameterized over five functions.
  No classes, no inheritance, no abstract methods. Five callables. The loop
  does not know what Spec it is validating or what Artifact it is emitting.
  It only knows: invoke, parse, validate, fix, emit.

- **`_invoke.py`** -- model invocation. `resolve_ask_fn()` resolves
  `config.ask_fn > get_provider_by_id(model_id) > get_model_spec()`.
  `build_system_prompt()` and `build_user_message()` assemble prompts
  from context.

- **`_validation.py`** -- Python source validation via AST. Operates on
  raw strings, not notebook types. `validate_python_sources()` checks
  syntax and cross-source variable flow. `summarize_variable_flow()`
  produces a structural view for model feedback. Used by any generator
  that produces Python source.

### Generator Modules

Each generator module follows the same shape:

```
<name>/
  _types.py     -- Spec type + structural validators
  _runtime.py   -- execution, ouroboros, emit/load
  _context.py   -- domain context builders
  generate.py   -- wires five functions into generation_loop, CLI
```

The five functions are closures that capture the generator's config and
delegate to the module's types and runtime. The loop never sees the config
directly -- it sees five functions.


## The Generation Loop

Two-tier loop: `max_rounds` wholesale rounds x `max_fixes` ouroboros fixes.

```
for round in range(max_rounds):        # outer: wholesale
    raw = invoke(ctx)
    spec = parse(raw)

    for fix_attempt in range(max_fixes):  # inner: ouroboros
        match validate(spec):
            Ok(artifact) -> emit(spec, artifact), done
            Err(error)   -> spec = fix(spec, error, ctx)

    # ouroboros couldn't fix it -- scrap the page, keep the learnings
    ctx = ctx.with_feedback(errors)
```

The outer loop is the painter scrapping the page. The inner loop is the
painter correcting a line. The errors accumulate across rounds -- the
learnings travel forward even when the page is discarded.


## Notebook Generator

```
python -m compass generator --type notebook \
  --prompt "write a fibonacci program" \
  --model-id anthropic:sonnet

python -m compass generator --type notebook --live

python -m compass generator --type notebook \
  --refine notebooks/compass_notebook_v1.ipynb \
  --claim "cells[3]: add memoization"
```

The notebook generator's five functions:

| Function | Implementation |
|----------|---------------|
| invoke | Call model via AskFn, return raw text |
| parse | `validate_spec()` -- raw dict -> NotebookSpec |
| validate | `validate_python_cells()` (AST) then `execute_cells()` (exec) |
| fix | `try_ouroboros()` -- cell-targeted correction |
| emit | `emit_notebook()` -- .ipynb + report sidecar + file extraction |

Validation is cheapest-first: structural (parse) -> semantic (AST +
variable flow) -> executive (exec). The last step is always executive.

### Domain Contexts

Domain knowledge is injected via `--domain`:

- `compass` -- project ETHOS + design docs
- (omitted) -- generic mode, no domain context

Each domain is a pure function: `(prompt, root?) -> GenerationContext`.


## Meta-Generator

```
python -m compass generator --type meta \
  --prompt "Build a generator for Python CLI tools" \
  --model-id anthropic:opus
```

The meta-generator generates generator modules. Its output is a
`GeneratorModuleSpec` -- a set of Python source files that follow
the generator shape.

V_meta is the same pattern as every other validator, composed
cheapest-first:

1. **Semantic**: `ast.parse()` every .py file (pure, cheapest)
2. **Executive**: materialize module to tmpdir, `importlib.import_module()`
3. **Executive**: run the inner generator's `main()` with `test_prompt`

The inner generator's `generation_loop` IS the executive test. If the
inner loop fails, the error bubbles up to the meta-generator's ouroboros.
Nested ouroboros -- inner fixes the artifact, outer fixes the generator.

```
G_meta  : Query -> GeneratorModuleSpec
V_meta  : materialize + import + run inner G -> V -> G'
G_meta' : (GeneratorModuleSpec, Error) -> GeneratorModuleSpec
```


## CLI Reference

```
python -m compass generator [--type notebook|meta] \
  --prompt "..."              # What to generate
  --model-id <spec>           # Model (default: ladder policy worker)
  --output <path>             # Output path
  --max-rounds 3              # Max wholesale rounds
  --max-fixes 3               # Max ouroboros fixes per round
  --domain compass            # Domain context (notebook only)
  --refine <path>             # Refine existing notebook (notebook only)
  --claim "cells[N]: ..."     # Domain claim for refinement (notebook only)
  --live                      # Interactive REPL (notebook only)
  --dry-run                   # Print prompt, don't call model
  --verbose                   # Debug logging
```

Model specs follow Compass convention:
```
anthropic:sonnet              # Claude Sonnet 4.6
anthropic:opus                # Claude Opus 4.6
qwen3-coder:latest@local      # Ollama on local
qwen3.5:122b@big              # Ollama on big server
```
