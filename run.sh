#!/bin/bash
# Unified compass runner -- named examples for generation, refinement, and agent modes.
#
# Usage:
#   ./run.sh <example> [model-id]
#
# Examples:
#   ./run.sh generate:fibonacci qwen3-coder-next:q8_0@big
#   ./run.sh generate:sorting  glm-4.7-flash:bf16@big
#   ./run.sh refine:v2         qwen3-coder-next:q8_0@big
#   ./run.sh refine:v6         qwen3.5:122b@big
#   ./run.sh refine:v7         qwen3.5:122b@big
#   ./run.sh meta:notebook    anthropic:opus
#   ./run.sh morpheus:fibonacci
#   ./run.sh neo:fibonacci
#   ./run.sh neo:prompt anthropic:sonnet "Build a fibonacci notebook with plots"
#   ./run.sh list              # show available examples
#
# Direct invocation (no run.sh):
#   python -m compass neo --prompt "..." --model-id anthropic:sonnet --verbose
#   python -m compass generator --type meta --prompt "..." --model-id anthropic:opus --verbose
#   python -m compass generator --prompt "..." --model-id qwen3.5:122b@big --verbose

set -e
set -a; [ -f .env ] && source .env; set +a

EXAMPLE="${1:?Usage: $0 <example> [model-id]}"
MODEL_ID="${2:-}"

# -- Prompts ----------------------------------------------------------------

FIBONACCI_PROMPT="Write a fibonacci program with iterative and recursive \
implementations, a memoized version using functools.lru_cache, performance \
comparison across input sizes, and matplotlib plots: (1) linear time \
comparison, (2) log-scale time comparison, (3) recursive call count growth, \
(4) bar chart at n=10,20,30 for all three implementations."

SORTING_PROMPT="Write a sorting algorithm comparison: implement bubble sort, \
merge sort, quicksort, and Python's built-in timsort. Extract implementations \
into src/sorting.py. Benchmark all four on random arrays of sizes 100, 500, \
1000, 5000, 10000. Plot a 2x2 grid: (1) linear time comparison, (2) log-scale \
time comparison, (3) bar chart at n=1000 for all four, (4) time ratio relative \
to timsort. Include a cell that verifies all implementations produce correctly \
sorted output."

AGENT_FIBONACCI_PROMPT="Use your programmer to write a Jupyter notebook \
(.ipynb file) at notebooks/morpheus_fibonacci.ipynb with: iterative and \
recursive fibonacci implementations, a memoized version using \
functools.lru_cache, performance comparison across input sizes, and matplotlib \
plots: (1) linear time comparison, (2) log-scale time comparison, (3) \
recursive call count growth, (4) bar chart at n=10,20,30 for all three \
implementations. Extract the implementations into src/fibonacci.py and import \
them in the notebook. The notebook must have markdown cells explaining each \
section."

REFINE_V2_CLAIM="cells[7]: add a memoized fibonacci using functools.lru_cache \
as a third implementation. In the plotting cell, create a 2x2 grid of \
subplots: (1) linear time comparison, (2) log-scale time comparison, (3) \
recursive call count growth, (4) bar chart at n=10,20,30 for all three \
implementations. All variables must be defined at module level, not inside \
if/try blocks."

REFINE_V6_CLAIM="cells[1]: extract fibonacci implementations into \
src/fibonacci.py and import them. cells[12]: replace the 4 separate plots with \
a single 2x2 subplot grid using plt.subplots(2,2). Add a new cell after the \
bar chart that shows a heatmap of execution times (implementations x input \
sizes) using plt.imshow. All variables must be defined at module level, not \
inside if/try blocks."

REFINE_V7_CLAIM="cells[2]: extract the three fibonacci implementations into a \
file named src/fibonacci.py using the files field in the spec. Then import them \
in cells[2] with 'from src.fibonacci import recursive_fib, iterative_fib, \
memoized_fib'."

META_NOTEBOOK_PROMPT="Build a notebook generator. The spec type should be \
NotebookSpec with cells (markdown and code), title, purpose, and an optional \
files array for extracted deliverables. The executive validator must exec() \
every code cell in a shared namespace (same as Jupyter). Ouroboros targets the \
failing cell by index. The emitter serializes to .ipynb format with execution \
outputs. The test prompt should generate a simple fibonacci notebook."

REFINE_NEO_CLAIM="In _runtime.py: (1) _find_generators_root must use \
'import compass.generators; return Path(compass.generators.__path__[0])' \
instead of Path(__file__).resolve().parent.parent. \
(2) _run_generator must invoke generators in-process via importlib + main(), \
not via subprocess.run(). Remove the subprocess import. The pattern is: \
gen_mod = importlib.import_module(module_name); gen_mod.main() with sys.argv \
set to the appropriate arguments. No timeout. \
(3) _meta_generate must invoke the meta-generator in-process the same way: \
importlib.import_module('compass.generators.meta.generate').main() with \
sys.argv, not subprocess. No timeout. \
(4) _resolve_generator must not hardcode alias tables mapping artifact types \
to generators. Resolution is pure discovery: check if the directory exists, \
return None if not. The meta-generator handles unknown types. \
In _context.py: _discover_available_generators must use \
'import compass.generators; root = Path(compass.generators.__path__[0])' \
instead of Path(__file__).resolve().parent.parent."

NEO_PLAN_PROMPT="Build a Python module that computes fibonacci numbers \
with iterative and recursive implementations, and a test file that \
verifies both produce the same results for n=0..20."

NEO_COMPILER_PROMPT="Build a compiler for a minimal Python-like language called \
MiniPy. It should support: variable assignment, integer and string literals, \
arithmetic (+,-,*,/,%), comparison operators (==,!=,<,>,<=,>=), if/elif/else, \
while loops, function definitions with parameters and return, and print(). \
The compiler has 4 stages: a lexer that tokenizes source into tokens (with \
type and value), a parser that builds an AST from tokens (recursive descent), \
an interpreter that walks the AST and evaluates it (with a scope stack for \
function calls), and a test suite that compiles and runs programs like \
factorial(5), fibonacci(10), fizzbuzz(15), and a string reversal function."

# -- Test prompts (one per generator) ----------------------------------------

HELLO_WORLD_PROMPT="Write a hello world program that greets the user by name \
using sys.argv, with a default of 'World' if no argument is given."

CODE_PROMPT="Write a Python script that implements Conway's Game of Life on a \
30x30 grid. Initialize with a random seed, run 50 generations, print each \
generation as ASCII art (# for alive, . for dead) with a generation counter, \
and at the end print statistics: total cells born, total cells died, peak \
population, and final population."

TEST_SUITE_PROMPT="Generate a pytest test suite for a calculator module with \
add, subtract, multiply, and divide functions. Include tests for edge cases: \
division by zero, large numbers, negative numbers, and floating point precision."

TUTORIAL_PROMPT="Create a tutorial on Python list comprehensions covering \
basic syntax, filtering with conditions, nested comprehensions, and \
dictionary comprehensions. Each section should have exercises with solutions."

NOTEBOOK_PROMPT="Write a notebook that demonstrates three sorting algorithms \
(bubble sort, merge sort, quicksort) with implementations, correctness tests \
on random arrays, and a matplotlib bar chart comparing their performance on \
arrays of size 100, 500, and 1000."

META_TEST_PROMPT="Build a quiz generator. The spec type should be QuizSpec \
with a title, a list of Question dataclasses (each with question text, choices \
as a tuple of strings, and correct_index as int), and a difficulty level. \
The structural validator checks non-empty questions and valid correct_index. \
The executive validator verifies all questions have 2-4 choices and no \
duplicate choice text. The emitter writes a JSON file. The test prompt \
should generate a 5-question Python basics quiz."

META_REGEX_PROMPT="Build a regex engine generator. The spec type should be \
RegexSpec with a pattern string, a list of TestCase dataclasses (each with \
input_string str and should_match bool), and an optional flags field. \
The structural validator checks: pattern is non-empty, test cases are \
non-empty, no duplicate input strings. The semantic validator compiles \
the pattern with re.compile to catch syntax errors cheaply. The executive \
validator runs every test case against the compiled pattern and verifies \
match/no-match agrees with should_match. The emitter writes a Python module \
with a match(text) function using the compiled pattern. The test prompt \
should generate a regex that matches valid email addresses and include \
5 positive and 5 negative test cases."

META_FSM_PROMPT="Build a state machine generator. The spec type should be \
FSMSpec with a name, a list of State dataclasses (each with name, \
on_enter action as optional str, and transitions as a tuple of Transition \
dataclasses with event str, target state name, and optional guard str), \
and an initial_state str. The structural validator checks: states non-empty, \
initial_state exists in states, all transition targets reference valid states, \
no duplicate event names within a single state. The executive validator \
instantiates the FSM, walks a valid path from initial_state through at least \
3 transitions, and verifies the current state after each step. The emitter \
writes a Python module with a FSM class that has process_event(event) and \
current_state properties. The test prompt should generate a turnstile FSM \
with locked/unlocked states and coin/push events."

META_NEO_PROMPT="Build a plan generator called Neo. Neo is a plan generator -- \
his Spec is PlanSpec, a sequence of Steps, like NotebookSpec is a sequence of \
Cells. Each Step describes an artifact to produce, and the executive validator \
resolves each step to a generator and runs it.

The Spec type -- PlanSpec:
- goal: str -- what the plan achieves
- reasoning: str -- why this sequence of steps
- steps: tuple of Step dataclasses, each with:
  - description: str -- what this step produces (natural language)
  - artifact_type: str -- what kind of thing (e.g. 'notebook', 'cli_tool', 'module', 'test_suite')
  - prompt: str -- the generation prompt to pass to the resolved generator
  - depends_on: tuple of int -- indices of prior steps this one depends on

Structural validator (cheapest):
- Parse JSON into PlanSpec
- Steps must be non-empty, all descriptions non-empty
- Dependency indices must be valid (in range, no self-references)
- Dependency graph must be acyclic (topological sort must succeed)

Semantic validator:
- Each step's artifact_type should be a recognized generator name OR a \
description that could be meta-generated. Warn but don't fail on unknown types.

Executive validator (most expensive):
- Process steps in topological order (respecting depends_on)
- For each step:
  1. Try to resolve artifact_type to an existing generator: scan \
compass/generators/ for a matching subdirectory with generate.py
  2. If not found: invoke the meta-generator to create one on the fly -- call \
compass.generators.meta.generate.main() with artifact_type as the prompt. \
This is Neo's innate self-extension: he can generate any capability he needs.
  3. Run the resolved generator with step.prompt
  4. Accumulate the result (path to generated artifact) into shared context \
so later steps can reference earlier outputs
  5. Stop on first step failure -- return Err with step index and error
- The meta-generation itself uses the generation loop -- nested ouroboros. \
If the meta-generator fails to produce a working generator, that failure \
becomes the step failure.

Ouroboros (fix function):
- Model sees: the full plan, the error at steps[N], and successful results \
from steps[0..N-1]
- Returns a corrected PlanSpec -- may fix step N's prompt, change its \
artifact_type, restructure later steps, or add/remove steps
- Target the failing step by index, like notebook ouroboros targets cells[N]

Emitter:
- Write a plan report JSON with: goal, reasoning, per-step results (which \
generator was used or meta-generated, output path, success/failure)
- Individual artifacts are already emitted by their respective generators
- The report is the plan's artifact -- it documents what was produced

Test prompt for executive validation:
'Build a Python module that computes fibonacci numbers with iterative and \
recursive implementations, and a test file that verifies both implementations \
produce the same results for n=0..20.' This requires at minimum two steps \
(source module + test file) and may meta-generate a Python module generator \
if one does not exist."

# -- Dispatch ---------------------------------------------------------------

case "$EXAMPLE" in
  generate:fibonacci)
    : "${MODEL_ID:?generate requires a model-id argument}"
    python -m compass generator \
      --model-id "$MODEL_ID" \
      --prompt "$FIBONACCI_PROMPT" \
      --verbose
    ;;

  generate:sorting)
    : "${MODEL_ID:?generate requires a model-id argument}"
    python -m compass generator \
      --model-id "$MODEL_ID" \
      --prompt "$SORTING_PROMPT" \
      --verbose
    ;;

  refine:v2)
    : "${MODEL_ID:?refine requires a model-id argument}"
    python -m compass generator \
      --model-id "$MODEL_ID" \
      --refine notebooks/compass_notebook_v2.ipynb \
      --claim "$REFINE_V2_CLAIM" \
      --verbose
    ;;

  refine:v6)
    : "${MODEL_ID:?refine requires a model-id argument}"
    python -m compass generator \
      --model-id "$MODEL_ID" \
      --refine notebooks/compass_notebook_v6.ipynb \
      --claim "$REFINE_V6_CLAIM" \
      --verbose
    ;;

  refine:v7)
    : "${MODEL_ID:?refine requires a model-id argument}"
    python -m compass generator \
      --model-id "$MODEL_ID" \
      --refine notebooks/compass_notebook_v7.ipynb \
      --claim "$REFINE_V7_CLAIM" \
      --verbose
    ;;

  meta:notebook)
    : "${MODEL_ID:?meta requires a model-id argument}"
    python -m compass generator \
      --type meta \
      --model-id "$MODEL_ID" \
      --prompt "$META_NOTEBOOK_PROMPT" \
      --output-dir generated/ \
      --verbose
    ;;

  meta:meta)
    : "${MODEL_ID:?meta requires a model-id argument}"
    python -m compass generator \
      --type meta \
      --model-id "$MODEL_ID" \
      --prompt "$(cat prompts/meta_meta.txt)" \
      --output-dir generated/ \
      --verbose
    ;;

  meta:neo)
    : "${MODEL_ID:?meta requires a model-id argument}"
    python -m compass generator \
      --type meta \
      --model-id "$MODEL_ID" \
      --prompt "$META_NEO_PROMPT" \
      --output-dir generated/ \
      --verbose
    ;;

  refine:neo)
    : "${MODEL_ID:?refine requires a model-id argument}"
    python -m compass generator \
      --type meta \
      --model-id "$MODEL_ID" \
      --refine generated/neo \
      --output-dir generated/ \
      --claim "$REFINE_NEO_CLAIM" \
      --max-fixes 6 \
      --verbose
    ;;

  refine:trinity)
    : "${MODEL_ID:?refine requires a model-id argument}"
    python -m compass generator \
      --type meta \
      --model-id "$MODEL_ID" \
      --refine compass/generators/trinity \
      --output-dir generated/ \
      --claim "$(cat prompts/refine_trinity.txt)" \
      --max-fixes 6 \
      --verbose
    ;;

  neo:plan)
    : "${MODEL_ID:?neo requires a model-id argument}"
    python -m compass neo \
      --model-id "$MODEL_ID" \
      --prompt "$NEO_PLAN_PROMPT" \
      --verbose
    ;;

  neo:hello)
    : "${MODEL_ID:?neo requires a model-id argument}"
    python -m compass neo \
      --model-id "$MODEL_ID" \
      --prompt "Welcome Home, Neo" \
      --verbose
    ;;

  # -- Pass-through (custom prompt as 3rd arg) --------------------------------

  neo:prompt)
    : "${MODEL_ID:?neo:prompt requires a model-id argument}"
    PROMPT="${3:?neo:prompt requires a prompt as 3rd argument}"
    python -m compass neo \
      --model-id "$MODEL_ID" \
      --prompt "$PROMPT" \
      --verbose
    ;;

  meta:prompt)
    : "${MODEL_ID:?meta:prompt requires a model-id argument}"
    PROMPT="${3:?meta:prompt requires a prompt as 3rd argument}"
    python -m compass generator \
      --type meta \
      --model-id "$MODEL_ID" \
      --prompt "$PROMPT" \
      --output-dir generated/ \
      --verbose
    ;;

  # -- Individual generator tests ---------------------------------------------

  test:code)
    : "${MODEL_ID:?test requires a model-id argument}"
    python -m compass.generators.code.generate \
      --model-id "$MODEL_ID" \
      --prompt "$CODE_PROMPT" \
      --verbose
    ;;

  test:neo)
    : "${MODEL_ID:?test requires a model-id argument}"
    python -m generated.neo.generate \
      --model-id "$MODEL_ID" \
      --prompt "$NEO_PLAN_PROMPT" \
      --verbose
    ;;

  test:neo:compiler)
    : "${MODEL_ID:?test requires a model-id argument}"
    python -m generated.neo.generate \
      --model-id "$MODEL_ID" \
      --prompt "$NEO_COMPILER_PROMPT" \
      --verbose
    ;;

  test:meta)
    : "${MODEL_ID:?test requires a model-id argument}"
    python -m compass.generators.meta.generate \
      --model-id "$MODEL_ID" \
      --prompt "$META_TEST_PROMPT" \
      --output-dir generated/ \
      --verbose
    ;;

  test:meta:fsm)
    : "${MODEL_ID:?test requires a model-id argument}"
    python -m compass.generators.meta.generate \
      --model-id "$MODEL_ID" \
      --prompt "$META_FSM_PROMPT" \
      --output-dir generated/ \
      --verbose
    ;;

  test:meta:regex)
    : "${MODEL_ID:?test requires a model-id argument}"
    python -m compass.generators.meta.generate \
      --model-id "$MODEL_ID" \
      --prompt "$META_REGEX_PROMPT" \
      --output-dir generated/ \
      --verbose
    ;;

  # -- Batch: compile + dry-run all -------------------------------------------

  test:compile)
    echo "Compile-checking all generators..."
    python -m py_compile compass/generators/_loop.py && \
    python -m py_compile compass/generators/code/generate.py && \
    python -m py_compile compass/generators/meta/generate.py && \
    python -m py_compile compass/generators/meta/_runtime.py && \
    python -m py_compile generated/neo/generate.py && \
    python -m py_compile generated/neo/_runtime.py && \
    python -m py_compile compass/__main__.py && \
    echo "All generators compile clean."
    ;;

  test:dry)
    echo "Dry-run all generators..."
    for gen in code meta; do
      echo "--- $gen ---"
      python -m "compass.generators.$gen.generate" --prompt "test" --dry-run 2>&1 | tail -3
      echo ""
    done
    echo "--- neo ---"
    python -m generated.neo.generate --prompt "test" --dry-run 2>&1 | tail -3
    echo ""
    echo "All dry-runs complete."
    ;;

  morpheus:fibonacci)
    python -m compass \
      --morpheus "$AGENT_FIBONACCI_PROMPT"
    ;;

  neo:fibonacci)
    COMPASS_FAMILY=neo python -m compass \
      --red-pill "$AGENT_FIBONACCI_PROMPT"
    ;;

  list)
    echo "Available examples:"
    echo ""
    echo "  Generate:"
    echo "    generate:fibonacci  -- fibonacci with plots (requires model-id)"
    echo "    generate:sorting    -- sorting comparison (requires model-id)"
    echo ""
    echo "  Refine:"
    echo "    refine:v2           -- add memoized fib to v2 (requires model-id)"
    echo "    refine:v6           -- extract + subplot grid for v6 (requires model-id)"
    echo "    refine:v7           -- file extraction for v7 (requires model-id)"
    echo "    refine:neo          -- refine Neo with claim (requires model-id)"
    echo ""
    echo "  Meta:"
    echo "    meta:notebook       -- meta-generate notebook generator (requires model-id)"
    echo "    meta:neo            -- meta-generate Neo (requires model-id)"
    echo ""
    echo "  Neo:"
    echo "    neo:plan            -- Neo plan: fibonacci+tests (requires model-id)"
    echo "    neo:hello           -- Neo plan: welcome home (requires model-id)"
    echo ""
    echo "  Pass-through (custom prompt as 3rd arg):"
    echo "    neo:prompt          -- ./run.sh neo:prompt <model> \"your prompt\""
    echo "    meta:prompt         -- ./run.sh meta:prompt <model> \"your prompt\""
    echo ""
    echo "  Test individual generators (requires model-id):"
    echo "    test:code           -- code generator"
    echo "    test:neo            -- Neo plan (fibonacci+tests)"
    echo "    test:neo:compiler   -- Neo plan (expression compiler)"
    echo "    test:meta           -- meta-generate a quiz generator"
    echo "    test:meta:fsm       -- meta-generate a state machine generator"
    echo "    test:meta:regex     -- meta-generate a regex engine generator"
    echo ""
    echo "  Test batch (no model needed):"
    echo "    test:compile        -- py_compile all generators"
    echo "    test:dry            -- dry-run all generators"
    echo ""
    echo "  Agents:"
    echo "    morpheus:fibonacci  -- morpheus agent, fibonacci prompt"
    echo "    neo:fibonacci       -- neo agent, fibonacci prompt"
    ;;

  *)
    echo "Unknown example: $EXAMPLE" >&2
    echo "Run '$0 list' to see available examples." >&2
    exit 1
    ;;
esac
