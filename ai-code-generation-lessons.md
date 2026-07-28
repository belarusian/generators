# Lessons from AI-Driven Code Generation

Findings from building a system that uses Claude Sonnet (via Bedrock) to
generate Jupyter notebooks exploring a Neptune graph database. The system
uses a type-driven feedback loop: the model produces JSON conforming to a
Python type contract, validators catch errors, and error messages feed back
to the model for self-correction. Over several iterations, we added five
validation layers and observed sharp differences in the model's ability to
self-correct depending on the quality and structure of the feedback.

This document records what we learned. The findings generalise beyond
notebooks to any system where an LLM generates code that must be validated
against external constraints.


## 1. The quality of self-correction is a direct function of error message quality

This is the central finding. The model's ability to fix its own mistakes
depends almost entirely on how much the error message tells it about the
problem.

**Precise errors self-correct immediately.** SPARQL/Python binding mismatches
produce messages like:

```
cells[25]: binding key 'zoneId' not in SPARQL SELECT vars ['validFrom', 'zone'].
Did you mean 'zone'?
```

The fix is mechanical (rename the key). The model gets it right on the first
retry, every time.

**Vague errors do not self-correct.** SPARQL queries that fail at Neptune
produce:

```
SPARQL failed (500): 400 Client Error: Bad Request
```

The model cannot diagnose "Bad Request" without knowing what Neptune rejected
(missing closing brace? unsupported function? wrong property name?). We
observed the same SPARQL probes failing 6 consecutive rounds with identical
errors. The model has no signal to work with.

**Implication**: if you build a feedback loop, invest in error message quality.
The validator is not just a gate -- it is the primary teaching mechanism. An
opaque error is a wasted round.

Note that an opaque error is not necessarily an infrastructure error. A 500
from Neptune might mean the query is malformed in a way the error message
does not describe. If scrapping the page and regenerating gets past it, the
problem was in the notebook, not the infrastructure. If it persists, maybe
it IS infrastructure -- but the system cannot distinguish these cases. It can
only react to the signals it receives. The system's job is not to diagnose
root causes it cannot observe -- it is to respond to incomplete information
as well as it can.


## 2. Structural feedback beats prescriptive feedback

The model generates notebooks where variables defined inside `if` blocks in
one cell are referenced unconditionally in later cells. This produces
`NameError` at runtime when the condition is false.

We tried three approaches to help the model self-correct:

**A. Error messages only.** "cells[13]: 'df_zones' may be undefined -- defined
conditionally in cells[12]." The model understood the report but did not
converge on a fix across 5 rounds. It shuffled cells, renamed variables,
added more guards -- but the structural pattern persisted.

**B. Prescriptive fixes.** "Add `df_zones = None` before the if-block." This
was rejected as a design approach. It is a hard-coded heuristic that works
for one variable type in one context. The problem space is unbounded --
conditional variables are just one class of cross-cell issue.

**C. Structural view.** `summarize_cell_variable_flow()` appends a per-cell
topology of variable definitions and cross-cell references:

```
cells[12]:
  defines conditionally: df_zones, sample_site, ...
  references: df_sites (from cells[11], ok)
cells[13]:
  defines conditionally: df_equipment, ...
  references: df_zones (from cells[12], conditional)
```

The model sees the dependency cascade -- what flows where, what is conditional,
where the problem propagates. It reasons about the fix itself.

**Result**: with approach C, the model fixed all conditional variable issues
on the first retry. Without it, it never converged.

The lesson: don't tell the model what to do. Show it the shape of the problem.
The model is better at reasoning than at following instructions for novel
situations. Give it the topology and let it figure out the fix.


## 3. Validation layers should be ordered cheapest-first

Our pipeline runs five validation steps in sequence. Each step is a gate --
if it fails, the model retries without running later steps.

```
1. validate_spec()           -- structural, pure       ~0ms
2. validate_python_cells()   -- AST analysis, pure     ~1ms
3. probe_all_queries()       -- SPARQL via Resolver     ~2-5s
4. execute_cells()           -- exec() all code cells   ~5-15s
5. emit_notebook()           -- serialize to .ipynb     ~0ms
```

Ordering matters because each round costs ~70 seconds (Bedrock latency). A
structural error caught in step 1 saves the 20+ seconds that steps 3-4 would
have consumed. Over 5-9 rounds, this adds up.

More importantly, cheap validators produce better errors. `ast.parse()` gives
you a line number and a syntax message. `exec()` gives you an exception
traceback. The SPARQL probe gives you "400 Bad Request." The more expensive
the validation, the less actionable the error tends to be.


## 4. The model oscillates when satisfying multiple constraints simultaneously

In our 8-round wholesale-only run (commit `1c8403e`), we observed:

- Round 1: 15 conditional variable errors
- Round 2: 0 variable errors, 4 SPARQL errors
- Round 3: 0 SPARQL errors, 3 variable errors (regression)
- Round 4: 0 variable errors, 4 SPARQL errors (regression)
- ...

The model fixes one class of bug by restructuring the notebook, which
reintroduces another class. It is optimising against multiple constraints
but cannot hold all of them in working memory simultaneously. Each fix
is locally correct but globally destabilising.

This is the whack-a-mole problem in multi-objective optimisation. The model
has no mechanism to preserve prior fixes while addressing new errors. The
feedback contains the full error history, but the model regenerates the
entire notebook from scratch each round.


### Ouroboros as partial mitigation

The wholesale function `G : Context -> NotebookSpec` traverses the full
solution space each time. A fix to cells[18] may destabilise cells[12].

Ouroboros feeds the model its own output:

```
G' : (NotebookSpec, Error) -> NotebookSpec
```

Same type in, same type out. The model sees its prior output + the error.
It does not need to reconstruct the reasoning that led to cells[1..17] --
it wrote them. It only needs to reason about the local edit at cells[18] in
the context of its own prior decisions.

`G'` converges faster than `G` because it has fewer degrees of freedom. The
oscillation we observe with `G` (fix cells[18], break cells[12]) is less
likely with `G'` because cells[12] is not in the edit scope.

**Result**: 1 round + 1 ouroboros fix to converge, vs 8 wholesale rounds
for the same notebook. The control arm (wholesale-only, same code minus
ouroboros) failed after 3 rounds on the same error -- stuck on
`cells[26]: 'df_points' may be undefined`.

But ouroboros is not sufficient on its own. Sometimes the notebook has an
internal error that cannot be patched cell by cell -- the fix requires
restructuring across many cells, or the error is in the overall approach
rather than a single cell. Sometimes the signals from the world are opaque
(500s with no detail), and the model burns its fixes on something it cannot
diagnose from the information it has. In these cases, the only way forward
is to scrap the page and start fresh.


### The two-tier loop

The generation loop is two tiers:

```
for round in range(max_rounds):          # outer: wholesale (scrap + restart)
    spec = G(ctx)

    for fix in range(max_fixes):         # inner: ouroboros (patch in place)
        match V(spec):
            Ok -> emit, done
            Err(cell_i, error) ->
                spec = G'(spec, error, cell_i)

    # ouroboros couldn't fix it -- scrap the page, keep the learnings
    ctx = ctx.with_feedback(errors)
```

The outer loop is the painter scrapping the page. The inner loop is the
painter correcting a line. The errors accumulate across rounds -- the
learnings travel forward even when the page is discarded. A fresh round does
not start from zero knowledge; it starts with the context of everything that
failed before.

Total budget: `max_rounds` x `max_fixes` = 9 iterations (default 3 x 3).


## 5. Execution is the only real proof

We initially had three validation layers (structural, syntactic, SPARQL
probes) and considered the notebook "valid" if all three passed. The generated
notebooks contained:

- `import networkx` when networkx was not installed (ImportError)
- `execute_sparql(query)` when the function requires three arguments (TypeError)
- `os.getenv('AWS_REGION', 'us-east-1')` defaulting to the wrong region (runtime logic error)

None of these are catchable by AST analysis or SPARQL probing. They only
surface when you actually run the code. Adding `execute_cells()` -- which
runs every code cell via `exec()` in a shared namespace -- caught all three
immediately.

The corollary: **static analysis is necessary but never sufficient for
generated code.** The model produces code that is syntactically valid,
type-correct, and structurally sound, but fails at runtime. This is not a
limitation of the analysis -- it is inherent to the problem. The model can
produce code that satisfies any finite set of static checks while still being
wrong in ways that only execution reveals.


## 6. Tell the model what it has, not just what it must produce

The model repeatedly imported packages that were not installed, and repeatedly
tried to discover configuration that was already provided. Both are cases
where the model lacked context about its execution environment.

Two fixes:

**Available packages.** `importlib.metadata.distributions()` produces a list
of installed packages. This is passed to the model: "Available packages:
boto3, matplotlib, networkx, pandas, ..." The model stops guessing.

**Pre-configured variables.** The execution environment seeds `RESOLVER_URL`,
`RESOLVER_API_KEY`, `ENV_NAME`, `REGION` into the namespace. The prompt tells
the model these are available and not to rediscover them. Without this, the
model generated its own CloudFormation discovery code that conflicted with the
seeded values.

The lesson: the model's default behaviour is to be self-sufficient. It will
generate setup code, import fallbacks, and discovery logic because that is
what "good notebooks" look like in its training data. You must explicitly
tell it what is already provided, or it will re-derive it (often incorrectly).


## 7. One function, two callers

The same `execute_cells()` function serves both the generator (validation
step in the feedback loop) and the standalone runner (`run_notebook.py`,
which a human runs to re-validate a notebook at any time).

This is not just code reuse. It is a correctness guarantee: the notebook is
validated by the same code path that a human would use to run it. There is no
gap between "the generator says it works" and "it actually works when I open
it in Jupyter."

The same principle applies to `validate_python_cells()` and
`summarize_cell_variable_flow()`: they are used by the generator for feedback
and could be used by a linter or CI check. The validation functions are not
generator-specific -- they are notebook-specific.


## 8. The type system as communication protocol

The model sees the type definitions in its system prompt. The type IS the
schema. No separate JSON schema, no prose description of the format.

Originally this meant including the full `_types.py` source. This worked
but was wasteful -- the model saw patch types, validators, helpers it didn't
need. The prompt was large and unfocused.

The lean prompt approach: `inspect.getsource(Type)` extracts only the types
the model writes. The docstring example in the type IS the contract. The
model sees exactly what it needs to produce and nothing else.

```python
_SPEC_TYPE_SOURCE = (
    inspect.getsource(ModuleFile) + "\n\n"
    + inspect.getsource(Spec)
)
```

The docstring is not documentation -- it is a program the model will
reproduce with variations. If the docstring example has a missing import,
the model copies the bug. Treat type docstrings like production code.

SHOW don't TELL: the docstring shows the exact response shape. The user
message steers ("generate an email validator") but does not re-specify
the format. One place defines the format. One place.


## 9. Correctness is relative -- and the validation stack never terminates

Every validation layer we added revealed bugs. And every bug we caught raised
the question: is this a bug in the generated code, or a bug in the thing the
generated code is talking to?

The SPARQL probe returns 200 with bindings. The cell executes without
exception. The notebook renders in Jupyter. Is it correct? That depends on
whether the data in Neptune is correct. Which depends on whether the ingest
pipeline is correct. Which depends on whether the source telemetry is correct.
Which depends on whether the physical sensor is correct. The generated
notebook is the last link in a chain, and every link has its own notion of
"correct."

We saw this concretely: the model produced `b['brick_class']['value']` and
got a `KeyError` at runtime. Was this a model bug? Yes -- the SPARQL query
didn't SELECT `?brick_class`. But the model wrote that code because it
*expected* equipment to have a Brick class annotation. Some equipment does,
some doesn't. The model's assumption was correct for part of the data and
wrong for the rest. The "bug" lives at the intersection of the generated
code, the SPARQL schema, the ontology, and the data loading pipeline.

This is not a solvable problem. It is a structural property of validation
itself. Each layer we add answers one question ("does it parse?", "does it
run?", "does the SPARQL execute?") but opens another ("is the data right?",
"is the schema complete?", "is the ontology consistent with what was
ingested?"). The DAG from problem to solution has no terminal node -- there
is always another dimension of correctness we haven't checked.


### Why this reduces to function composition

Strip away the implementation and our pipeline is a composition of partial
functions:

```
V = v5 . v4 . v3 . v2 . v1

where
  v1 : RawJSON   -> NotebookSpec | Err
  v2 : NotebookSpec -> NotebookSpec | Err
  v3 : NotebookSpec -> NotebookSpec | Err
  v4 : NotebookSpec -> ExecutedNotebook | Err
  v5 : ExecutedNotebook -> Path | Err
```

Each `vi` is a partial function -- it maps its input to a value or to an
error. The composition `V` succeeds only if every `vi` succeeds. An error
at any stage short-circuits the rest and feeds back to the generator `G`,
which is itself a function `G : Context -> RawJSON`.

The generation loop is then the fixed-point iteration:

```
ctx_0 = initial context
ctx_{n+1} = ctx_n + feedback(V(G(ctx_n)))

terminate when V(G(ctx_n)) is not Err
```

This is a standard iterative refinement toward a fixed point: find an input
`ctx*` such that `V(G(ctx*))` produces `Ok`. The model `G` is a computable
function (it maps strings to strings). Each `vi` is a decidable predicate.
The composition `V` is a decidable predicate. The iteration is a computable
process searching for a fixed point.

The practical question is not "can it converge?" but "does it converge in a
reasonable number of steps?" That is where error message quality, structural
feedback, and the two-tier loop come in -- they are not theoretical
necessities but engineering optimisations that shrink the search space.

The non-termination of validation is also precise in this framing: we can
always define a new `v_{n+1}` that checks something `v_1 ... v_n` did not.
The composition grows monotonically. There is no final `V` that captures all
possible notions of correctness, because "correctness" is not a single
decidable property -- it is an open-ended family of predicates, each
corresponding to a different observer's requirements. The pipeline terminates
by policy (we choose to stop at `v5`), not by proof.

A domain expert constitutes yet another validation function:

```
v_expert : NotebookOutput -> Ok | Err("why are there only 3 zones?")
```

And there is not one `v_expert` but countably many. Each expert is a distinct
decidable predicate over the output. No finite composition can subsume all of
them. The system is sound (if `V` says `Err`, there is a real problem) but
incomplete (if `V` says `Ok`, there may still be problems that `V` does not
check). Soundness is achievable. Completeness is not. The engineering
question is always: which `v_{n+1}` do we add next?


### The proof-of-work chain

The version chain has the same structure as a blockchain. Each version is
a block:

```
block_n = (notebook_vN, validators, prev=block_{n-1}, work)
```

The notebook is the payload. The report (which validators ran, how many
rounds, what domain claims were applied) is the block header. The
`--refine` path is the `prev` pointer. The chain is append-only: you never
go back and mutate v1; you produce v2 that references v1.

The model invocation is the hash function. It takes the prior state
(notebook + error + context) and produces a candidate. The candidate is
validated against the predicate composition `V`. If `V(candidate) = Ok`,
the block is accepted. If not, the model burns more compute trying again.
In Bitcoin, `SHA256(block) < target`. In ours, `V(G(ctx)) = Ok`. Both are
computationally expensive to produce, cheap to verify.

The validator set is the difficulty. It only grows -- each domain expert
adds a `v_{n+1}`, raising the bar for future blocks. And like Bitcoin,
there is no finality. A new validator can always reject the chain. Version
N is "correct" only because nobody has produced `v_{N+1}` that breaks it.

The version number is not a quality ranking. It is the depth of the
validation chain -- how many observer sets have approved this artefact.
v0 (the original notebook) is correct as of its author. v1 is correct as
of `v1...v5`. v2 is correct as of `v1...v5` plus a domain expert's claim.
Each version is correct relative to its validator set -- and nothing more.


## 10. The generator is not domain-specific

Nothing in the generation loop is specific to Neptune, SPARQL, or Jupyter
notebooks. The loop is:

```
G : Prompt -> Artefact
V : Artefact -> Ok | Err
G' : (Artefact, Error) -> Artefact

repeat until V(artefact) = Ok or budget exhausted
```

In our implementation, `Prompt` is a text string assembled from graph model
docs, ontology, sample queries, and a user query. `Artefact` is a
`NotebookSpec` (a sequence of cells). `V` is a composition of type checking,
AST analysis, SPARQL probes, and cell execution.

But the prompt could be anything -- a feature request, a bug report, a
question about a codebase. The artefact could be a set of files rather than
a sequence of cells. The validators could be `pytest`, `mypy`, `cargo test`,
or a human looking at the output.

The notebook is a semantic tree of the solution. Each cell is a node. The
cells have dependencies (variable flow, imports). The tree can be validated
structurally (AST), semantically (probes), and empirically (execution). This
is not special to notebooks -- any structured program has these properties.
A notebook just makes the tree explicit: each cell is a visible node, the
execution order is the tree traversal, and the shared namespace is the
context that flows between nodes.

Extracting from notebooks to a complete program (a set of files that
encompass the solution) is a serialisation problem: flatten the semantic tree
into a file layout. The generation loop does not change -- only the artefact
type and the validators.


## 11. The escaping problem is structural, not a model limitation

Models write code. Code contains backslashes, quotes, newlines. When code
must live inside a string literal (JSON value, Python string), these
characters need escaping. The model must escape code it hasn't run yet --
predicting what escape sequences the parser needs. This is the wrong
abstraction.

We observed this across generators: regex patterns with `\d`, `\w`;
triple-quoted strings with nested quotes; multi-line code with `\n`. The
model would:

- Write `content=None` to avoid the problem (validation rejects empty content)
- Write code inline with broken escaping (unterminated string literals)
- Produce ouroboros responses truncated at parenthesis boundaries

The fix is not better escaping instructions. The fix is eliminating
escaping entirely.

**Banner-based response format**: the model writes a Python constructor
expression (metadata only), then raw code after `### filename ###`
delimiters. The code is never inside a string. No escaping needed.

```
Spec(
    name="email_validator",
    files=(
        ModuleFile(path="validator.py", description="Core logic"),
    ),
)

### validator.py ###
import re
pattern = r'^[\w.+-]+@[\w-]+\.[\w.]+$'

### test_source ###
from email_validator import validate
assert validate("a@b.com")
```

The parser (`parse_response_with_files`) splits at the first banner, parses
the constructor via `parse_typed_response`, and returns
`(instance, [(name, content), ...])`. Banner content is raw text -- never
parsed as Python by the response parser, never escaped.

The try-banner/fallback pattern handles model variation: try
`parse_response_with_files` first, fall back to `parse_typed_response` if
no banners are found. Both paths produce the same typed instance.


## 12. The sandbox is the namespace, not the AST

`parse_typed_response` execs the model's constructor expression. The
security boundary is `exec(code, {"__builtins__": {}}, namespace)` -- no
builtins means no `open()`, `__import__()`, `eval()`, `exec()`.

We initially had an AST whitelist that allowed only constructor-like nodes
(Call, Constant, Tuple, keyword, etc). This was too restrictive -- the model
sometimes writes class definitions or helper functions before the
constructor, and the fallback path may see code from banner sections. The
whitelist rejected `ClassDef`, which is a pure definition with no side
effects.

The fix: replace the whitelist with an import-only blacklist. Only
`ast.Import` and `ast.ImportFrom` are rejected (they can bypass the
namespace sandbox). Everything else is safe when builtins are removed.

Models also produce `TypeError` (missing constructor args) and `NameError`
(parroted type definitions with `@dataclass`, wrong type names like
`StepSpec` instead of `Step`). These must be caught and converted to
`ValueError` so the generation loop's retry machinery handles them
uniformly. An unhandled `NameError` crashes past the `except ValueError`
in `invoke_model` and kills the loop.


## 13. The model mimics; it does not understand implicit contracts

Banner-based field mapping in Trinity works like this: the model writes
a `Step(...)` constructor with `artifact_ref=None` for `inline_python`
steps, then writes the actual code after a `### step_id ###` banner.
`_attach_banner_code()` fills `artifact_ref` from the banner content.

The model does not know this. It follows the pattern because the Spec
docstring shows an example where `inline_python` steps omit
`artifact_ref` and code appears after banners. The model copies what it
sees. An integration test validates that the behavior holds -- not that
the model understands the mechanism.

This was proven empirically: when asked (as Trinity) why `artifact_ref`
gets populated, the model could not explain the mechanism. It
attributed the mapping to `extraction_expr`, which is a different
field entirely. It just followed the pattern.

The corollary: implicit contracts between the model's output and the
runtime are fragile. The model cannot extend them (banners for shell
steps produced Python code instead of shell commands) and cannot debug
them (it cannot reason about `_attach_banner_code`). Integration tests
are the only way to know the contract still holds.


## 14. When the model misuses a construct, fix the type -- not the runtime

Shell steps in Trinity posted literal `${COMMENT}` and
`$fact:formatted_comment` instead of resolved values. Four approaches
were tried:

**A. Regex interpolation.** `re.sub` to find `$fact:name` and
`${NAME}` patterns and replace them with resolved values. Brittle --
the naming conventions are unbounded, the model invents new ones each
run, and the regex can never cover all forms.

**B. Prompt instructions.** Tell the model "use $VAR syntax, not
$fact:VAR." The model ignored it. Instructions compete with the
pattern the model already learned from the type docstring.

**C. Banners for shell steps.** Let shell steps use `### step_id ###`
banners like `inline_python`. Backfired -- the model put Python code
in shell banners (`import os`, `subprocess.run`), producing
`/bin/sh: import: command not found`.

**D. Structural fix.** Three changes:
1. All accumulated facts injected as env vars in `_execute_shell()` --
   shell's own `$VAR` expansion works regardless of what the model
   calls the variable.
2. Semantic validation catches shell steps with `depends_on` but no
   `inputs` -- the model is told to add the mapping before execution.
3. A shell step example added to the Spec docstring showing proper
   `inputs: {"name": {"$fact": "name"}}` mapping.

Only D worked. The runtime (env vars) handles execution. The validator
catches missing mappings. The type example teaches the model the
pattern. Each component does one thing.

The lesson: when a model consistently misuses a construct, the problem
is in what the model sees (the type), not in what happens after (the
runtime). Fix the input to the model, not the output.


## Summary of findings

| Finding | Practical implication |
|---|---|
| Error quality determines self-correction | Invest in error messages, not retry count |
| Structural feedback > prescriptive feedback | Show the topology, not the fix |
| Order validation cheapest-first | Cheap steps produce better errors and save time |
| Models oscillate on multi-constraint problems | Two-tier loop: ouroboros for local fixes, wholesale for global reset |
| Execution is the only real proof | Always exec() generated code before shipping |
| Tell the model what it has | List available packages, pre-configured variables |
| One function, two callers | Generator validation = human validation |
| Type system as protocol | inspect.getsource for lean prompts; docstring examples are production code |
| Correctness is relative | Validation never terminates; decide when it's enough |
| The generator is not domain-specific | Prompt + artefact + validators = general code generation |
| Escaping is structural | Banner format: constructor + raw code after ### markers. No escaping. |
| Sandbox is the namespace | __builtins__: {} is the boundary; AST whitelist was too restrictive |
| Model mimics, doesn't understand | Implicit contracts need integration tests; the model cannot extend or debug them |
| Fix the type, not the runtime | When the model misuses a construct, change the docstring example -- not the runtime |


## Appendix: Pipeline in practice

**Wholesale-only** (commit `1c8403e`, 8 rounds):

```
Rounds 1-2: conditional variable errors -> fixed
Rounds 3-8: same 3 SPARQL probes fail every round
            error: "400 Bad Request" (no detail)
            model cannot diagnose -> stuck loop detected at round 5
            eventually converged at round 8
```

**With ouroboros** (two-tier loop, 1 round + 1 fix):

```
Round 1: spec produced (23 cells)
  Fix 0: cells[11] conditional variable error
         ouroboros targets cells[11], returns corrected notebook
  Fix 1: all validation passes, all cells execute
         notebook emitted
```

**Ouroboros failure mode** (3 wholesale x 3 ouroboros, Neptune 500s):

```
Round 1: spec produced
  Fix 0-2: ouroboros patches cells, but SPARQL probes return 500
  Fix 3: ouroboros exhausted
Round 1 scrapped, errors preserved in context
```

The system could not distinguish "bad query" from "Neptune is down" --
both produce `SPARQL failed (500)`. A fresh wholesale round might produce
different queries that avoid the failing probes. Or it might hit the same
500s. The system reacts to the signals; it does not diagnose causes it
cannot observe.
