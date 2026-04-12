"""
Type definitions for code mode.

Pure data types with no side effects. These define the contracts
between modules and make reasoning about the codebase simpler.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from compass.llm.oracle import Oracle
    from compass.agents.neo.memory import CodeMemory
    from compass.agents.neo.trace import ActionTrace, ExecutionTrace
    from compass.core.ui_adapter import UIAdapter
    from compass.core.stream_router import StreamRouter


# --- Enums ---

class QueryType(Enum):
    """Classification of user queries."""
    CODE_QUESTION = "code_question"
    CODE_MODIFICATION = "code_modification"
    SHELL_OPERATION = "shell_operation"
    GIT_OPERATION = "git_operation"
    FILE_OPERATION = "file_operation"
    CAPABILITY_EXTENSION = "capability_extension"
    PERSIST_CAPABILITY = "persist_capability"
    GENERAL = "general"


class LearningType(Enum):
    """Types of runtime-verified learnings."""
    FILE_READ = "file_read"           # verified file content/structure
    FILE_STRUCTURE = "file_structure" # symbols, line numbers in file
    SHELL_ENV = "shell_env"           # shell type, tools, versions
    CODEBASE_LAYOUT = "codebase_layout"  # where tests/src/configs live
    IMPORT_MAP = "import_map"         # module -> import path mappings
    CORRECTION = "correction"         # expected vs actual (calibration)


# --- Generic Response Types ---
# Simple typed responses for oracle.ask() - replaces legacy dict schemas
# For simple string responses, use response_type=None -> RawResponse

@dataclass
class AnyValue:
    """Generic value response - can hold int, string, etc."""
    value: Any


@dataclass
class IntLine:
    """Line number response."""
    line: int


@dataclass
class LearningResponse:
    """
    LLM response for learning extraction.

    Model writes:
        LearningResponse(
            learning_type=LearningType.CORRECTION,
            summary="The API returns JSON, not plain text",
            key_facts=["API is REST", "Returns application/json"]
        )
    """
    learning_type: LearningType
    summary: str
    key_facts: Optional[List[str]] = None


class ProgressSignal(Enum):
    """Progress trajectory evaluation."""
    PROGRESSING = "progressing"   # Making progress, continue normally
    STALLED = "stalled"           # Not progressing, try creativity bump
    OSCILLATING = "oscillating"   # Going in circles, need different approach
    STUCK = "stuck"               # Completely stuck, escalate to stronger model


@dataclass(frozen=True)
class ProgressAssessment:
    """
    Evaluation of whether we're making progress toward the goal.

    Unlike action success/failure which is per-action,
    this evaluates the *trajectory* of actions.
    """
    signal: ProgressSignal
    confidence: float  # 0.0-1.0
    reasoning: str
    suggestion: Optional[str] = None  # What to try differently


# --- Action Dataclasses (for Python-as-Schema) ---
# Each action the Actor can execute. Model writes Python constructors, we eval.

@dataclass
class ReadFileAction:
    """Read file contents with optional line range.

    Use for:
    - Reading files to understand their content
    - Checking current state before editing
    - Paginating through large files

    Fields:
    - path: File path (relative to project root, or absolute)
    - offset: Start at this line number (0-based). Default: 0
    - limit: Read this many lines. Default: all lines

    Read before you edit - understand existing code first.
    """
    path: str
    offset: Optional[int] = None  # Start at this line (0-based)
    limit: Optional[int] = None   # Read this many lines (None = all)
    reasoning: Optional[str] = None


@dataclass
class WriteFileAction:
    """Create or completely replace a file.

    Use for:
    - Creating NEW files that don't exist
    - Completely rewriting an existing file
    - When you have the full desired content

    One-liner (goes inline):
        WriteFileAction(path="config.txt", content="debug=true")

    Multi-line content -- use a banner (no escaping needed):
        WriteFileAction(path="file.py", content=None)

        ### file.py ###
        def greet():
            '''Any quotes work!'''
    """
    path: str
    content: Optional[str] = None  # Leave None; use content block for multi-line
    reasoning: Optional[str] = None


@dataclass
class EditFileAction:
    """Make targeted edits to specific code in an existing file.

    Use for:
    - Changing a function, class, or code block
    - Adding/removing imports, methods, lines
    - Renaming variables, fixing bugs in specific locations

    NOT for:
    - Creating new files (use WriteFileAction)
    - Replacing entire file content (use WriteFileAction)
    - Files that don't exist yet (use WriteFileAction)

    The instruction describes WHAT to change. FileEditor finds the exact
    target code and applies the edit. Be specific about the location.
    """
    path: str
    instruction: str  # Natural language: "Change X to Y" or "Add Z after W"
    reasoning: Optional[str] = None


@dataclass
class DeleteFileAction:
    """Delete a file. Does not delete directories."""
    path: str
    reasoning: Optional[str] = None


@dataclass
class CreateDirAction:
    """Create directories with parent directories (like mkdir -p)."""
    path: str
    reasoning: Optional[str] = None


@dataclass
class RunCommandAction:
    """Run a shell command directly.

    Use for simple commands where you know the exact syntax:
    - ls, git status, npm install, pytest
    - Commands without complex quoting or variables

    For commands with $variables, nested quotes, or special characters,
    use ShellCommandAction instead - ShellBuilder handles escaping.

    One-liner (goes inline):
        RunCommandAction(command="pytest tests/ -v")

    Multi-line or special chars -- use a banner (no escaping needed):
        RunCommandAction(command=None)

        ### command ###
        echo "Hello $USER"
    """
    command: Optional[str] = None  # Leave None; use content block for multi-line
    timeout: Optional[int] = None  # Seconds. Default 3600 (1 hour). Increase for builds/tests.
    reasoning: Optional[str] = None


@dataclass
class ShellCommandAction:
    """Run a complex shell command via ShellBuilder.

    Use when your command involves:
    - $variables or ${substitutions}
    - Quotes within quotes
    - Special characters: | & > < ; etc.
    - Complex pipelines

    You describe the INTENT, ShellBuilder generates the properly escaped command.

    Example:
        ShellCommandAction(
            intent="Write 'Price: $100' to output.txt",
            context="Dollar sign must be literal, not a variable"
        )

    For simple commands (ls, git status), use RunCommandAction directly.
    """
    intent: str  # Natural language: what the command should accomplish
    context: Optional[str] = None  # Files involved, special handling needed
    timeout: Optional[int] = None  # Seconds. Default 3600 (1 hour).
    reasoning: Optional[str] = None


@dataclass
class ExecAction:
    """Execute Python code in memory. Variables persist across calls.

    Use for:
    - Quick calculations and type checks
    - API calls and data processing
    - Validation and verification
    - Exploratory code that doesn't need to be saved

    Available: os, sys, json, Path, pathlib, cwd (project path)
    Variables you define persist to subsequent exec actions.

    For code worth keeping (logic to review, debug, or rerun),
    use WriteFileAction + RunCommandAction instead.

    One-liner (goes inline):
        ExecAction(code="print(2 + 2)", reasoning="Quick check")

    Multi-line code -- use a banner (no escaping needed):
        ExecAction(code=None, reasoning="Test it")

        ### code ###
        result = compute_something()
        print(result)
    """
    code: Optional[str] = None  # Leave None; use content block for multi-line
    timeout: Optional[int] = None  # Seconds. Default 30.
    reasoning: Optional[str] = None


@dataclass
class SearchAction:
    """Find code by name, concept, or meaning. Combines AST index + embeddings.

    Use for:
    - Finding functions or classes by name
    - Finding code related to a concept ("authentication", "error handling")
    - Discovering files that implement a feature

    search_type options:
    - "content" (default): Search code content semantically
    - "function": Find function definitions by name
    - "class": Find class definitions by name
    - "file": Find files by name pattern

    For exact text/regex matching, use GrepAction instead.
    """
    query: str  # What to search for (name, concept, description)
    search_type: Optional[str] = None  # "content", "function", "class", "file"
    reasoning: Optional[str] = None


@dataclass
class IndexAction:
    """Rebuild/refresh the semantic search index. Use when search seems stale."""
    force: Optional[bool] = None  # True = full rebuild, False/None = incremental
    reasoning: Optional[str] = None


@dataclass
class GrepAction:
    """Find exact text or regex patterns in files. Uses ripgrep.

    Use for:
    - Finding exact strings: "def get_session_context"
    - Pattern matching: "import.*json", "class.*Error"
    - When SearchAction didn't find what you need

    path: Directory or file to search. Default: project root.
    fixed: If True, treat pattern as literal text (no regex). Use this
           when pattern contains parentheses, dots, or special chars.

    For semantic/conceptual search, use SearchAction instead.
    Grep finds exact matches; Search finds related code by meaning.

    Simple pattern (goes inline):
        GrepAction(pattern="def get_session_context", path="src/")

    Literal text with special chars (no regex escaping needed):
        GrepAction(pattern="result.status in (SUCCESS, DONE)", fixed=True)

    Complex regex -- use a banner (no escaping needed):
        GrepAction(pattern=None, path="src/")

        ### pattern ###
        def\\s+\\w+.*->.*Optional
    """
    pattern: Optional[str] = None  # Leave None; use content block for complex patterns
    path: Optional[str] = None  # Directory or file. Default: project root.
    fixed: Optional[bool] = None  # True = literal text match (rg -F), no regex
    timeout: Optional[int] = None  # Seconds. Default 30.
    reasoning: Optional[str] = None


@dataclass
class AskOracleAction:
    """Ask the Oracle for help.

    THIS IS HOW YOU ASK THE ORACLE. When the user says "ask the oracle",
    use this action with your question.

    Use for:
    - Getting help with any problem you're stuck on
    - Reviewing code, designs, or approaches
    - Architectural guidance and design decisions
    - Debugging when you've tried and failed
    - Creative exploration and alternative perspectives
    - Any question where you need insight

    Short context (goes inline):
        AskOracleAction(question="How should I handle auth?", context="Using Flask")

    Long context -- use a banner (no escaping needed):
        AskOracleAction(question="How should I...", context=None)

        ### context ###
        ...code snippets, error traces, etc...

    The Oracle sees your question and context. Be specific about what you need.
    """
    question: str  # What to ask the Oracle
    context: Optional[str] = None  # Leave None; auto-filled from content block
    reasoning: Optional[str] = None


@dataclass
class ProgramAction:
    """Invoke Programmer NFA for complex multi-file code generation.

    Use for:
    - Multi-file features with interdependent components
    - Problems requiring design phase before implementation
    - Complex code generation needing Programmer + Scribe validation

    Short problem (goes inline):
        ProgramAction(problem="Add JWT auth to all API endpoints")

    Detailed problem -- use a banner (no escaping needed):
        ProgramAction(problem=None, reasoning="Multi-file feature")

        ### problem ###
        ...detailed problem statement...

    For simpler changes, use EditFileAction or WriteFileAction directly.
    Describe the abstract PROBLEM, not file-specific instructions.
    """
    problem: Optional[str] = None  # Leave None; auto-filled from content block
    constraints: Optional[List[str]] = None  # Explicit constraints on solution
    reasoning: Optional[str] = None


# --- Computer Use Actions ---
# Screenshot, click, type, scroll, keypress. The model sees the screen
# and acts on it. Text-based targeting via OCR -- no hallucinated coordinates.

@dataclass
class ScreenshotAction:
    """Capture the current screen state so you can see and reason about it.

    Only take a screenshot when you need to SEE the screen -- e.g. to
    answer a question about what's visible, or to understand layout
    before a complex interaction.  Don't take one routinely before or
    after every action; ClickAction and LocateAction already capture
    their own screenshots internally for detection.

    Example: ScreenshotAction(region="full")
    """
    region: str = "full"  # "full" or "active_window"
    reasoning: Optional[str] = None


@dataclass
class ClickAction:
    """Click a GUI element by text label or visual description.

    Prefer text targets over coordinates. Text is resilient to
    window position and resolution changes. Coordinates break
    when the window moves.

    Example: ClickAction(target="Save", button="left")
    Visual:  ClickAction(target="yellow button", button="left")
    Bad:     ClickAction(coords=(342, 187))  # fragile

    Takes its own screenshot internally, routes to OCR (text) or
    DINO (visual elements) based on the target, and clicks the
    detected coordinates. No separate ScreenshotAction needed.
    """
    target: str = ""  # visible text label (preferred)
    coords: Optional[Tuple[int, int]] = None  # fallback only
    button: str = "left"  # "left", "right", "double"
    reasoning: Optional[str] = None


@dataclass
class TypeAction:
    """Type text into the currently focused input field.

    If unsure which field has focus, use ClickAction to focus the
    right field first.

    Example: TypeAction(text="hello@example.com")
    """
    text: str = ""
    press_enter: bool = False
    reasoning: Optional[str] = None


@dataclass
class ScrollAction:
    """Scroll the active window or element.

    Example: ScrollAction(direction="down", amount=3)
    """
    direction: str = "down"  # "up", "down"
    amount: int = 3  # scroll clicks
    reasoning: Optional[str] = None


@dataclass
class KeyPressAction:
    """Press a key or key combination.

    Example: KeyPressAction(keys="cmd+s")
    """
    keys: str = ""  # "enter", "tab", "cmd+s", "ctrl+c"
    reasoning: Optional[str] = None


@dataclass
class LocateAction:
    """Find a UI element on screen without clicking it.

    Returns the element's screen coordinates, confidence, and
    detection method (OCR or DINO). Use this to probe the screen
    before acting, verify element positions, or plan multi-step
    interactions.

    Example: LocateAction(target="yellow button")
    Result:  "Found 'yellow button' at (505, 48) (conf=0.49, via dino)"
    """
    target: str = ""  # text label or visual description
    reasoning: Optional[str] = None


@dataclass
class SkillAction:
    """Run a learned skill from neo-lab.

    Skills are taught interactively (neo-lab learn.py) and saved as
    YAML in neo-lab/skills/. Each skill is a validated sequence:
    G . V . G . V -- actions alternating with OCR/DINO validation.

    Skills compose: a skill can invoke sub-skills. Each step validates
    its output via screen state. The skill either converges to the
    expected state or fails with a reason.

    Example: SkillAction(skill="chase_login")
    Compose: SkillAction(skill="yahoo_finance_nasdaq", expect="Nasdaq")
    """
    skill: str = ""  # skill name (matches skills/<name>.yaml in neo-lab)
    expect: str = ""  # optional OCR text to validate after skill completes
    reasoning: Optional[str] = None


@dataclass
class StateCheckAction:
    """Check what screen state Neo is currently looking at.

    Uses the neo-lab state graph -- registered states with OCR text
    markers and DINO visual markers. Neo screenshots, runs OCR+DINO,
    and matches against known states.

    If state is provided, checks "am I at this state?" (True/False).
    If state is empty, reports whatever state is detected.

    Example: StateCheckAction(state="amex_dashboard")
    Query:   StateCheckAction()  -- what state am I in?
    """
    state: str = ""  # specific state to check for (empty = identify any)
    reasoning: Optional[str] = None


# Forward reference for Learning (defined in memory.py)
if TYPE_CHECKING:
    from compass.agents.neo.memory import Learning

# Reflector: takes prompt -> structured Learning
# Schema baked in at creation - extractors just provide prompt
Reflector = Callable[[str], "Learning"]

# Type for learning extractor functions
# (action, success, result, reflect) -> List[Learning]
# The reflector is injected - no oracle dependency in the extractor
LearningExtractor = Callable[["Action", bool, str, Reflector], List["Learning"]]

# Type for action validator functions
# (action, project_path, files_read) -> (is_valid, error)
ActionValidator = Callable[["Action", str, Dict], Tuple[bool, Optional[str]]]


# --- Execution Context ---

@dataclass
class ExecutionContext:
    """
    Context passed to all action executors.

    Actions pull what they need from this context - no more ad-hoc parameter passing.
    Makes the execute_action dispatcher clean and uniform.
    """
    exec_globals: Optional[Dict[str, Any]] = None  # For exec action (variable persistence)
    oracle: Optional["Oracle"] = None              # For LLM-assisted actions (edit_file, shell_command, program)
    memory: Optional["CodeMemory"] = None          # For program action (sub-NFA)
    files_read: Optional[Dict] = None              # For validation (tracks known files)
    ui: Optional["UIAdapter"] = None               # For UI operations (show_action, show_result, etc.)
    on_thinking: Optional[Callable[[str], None]] = None  # Callback for thinking chunks
    stream_router: Optional["StreamRouter"] = None  # For NFA visualization (action events)
    actor_context: Optional[str] = None            # For ask_claude (full Actor context)
    execution_trace: Optional["ExecutionTrace"] = None  # For tracing execution flow
    codebase_index: Optional[Any] = None               # For search action (AST + RAG index)

    @classmethod
    def for_parallel(cls, base: "ExecutionContext") -> "ExecutionContext":
        """Create a context for parallel execution with collecting UI adapter."""
        from compass.core.ui_adapter import CollectingUIAdapter
        collector = CollectingUIAdapter()
        return cls(
            exec_globals=base.exec_globals.copy() if base.exec_globals else None,
            oracle=base.oracle,
            memory=base.memory,
            files_read=base.files_read.copy() if base.files_read else None,
            ui=collector,
            on_thinking=collector.show_thinking_chunk,
            stream_router=base.stream_router,  # Share router for action events
        )


# Type for action executor: (action, project_path, ctx) -> (success, message)
ActionExecutor = Callable[["Action", str, ExecutionContext], Tuple[bool, str]]


@dataclass
class ActionSpec:
    """
    Specification for an action type.

    All action modules in compass/agents/neo/rules/actions/ share this shape:
    - name: action type identifier (e.g. "read_file", "run_command")
    - validate: validates action, returns (is_valid, error_or_none)
    - execute: executes action, returns (success, message)
    - extract_learnings: extracts learnings from execution result

    This makes the contract explicit and enforced by types.
    """
    name: str
    validate: ActionValidator
    execute: Optional[ActionExecutor] = None  # None = use legacy executor
    extract_learnings: LearningExtractor = None


class CriticAction(Enum):
    """Neo Critic's decision after reviewing results."""
    DONE = "done"
    REPLAN = "replan"
    ASK_ORACLE = "ask_oracle"
    ASK_USER = "ask_user"


class ApprovalDecision(Enum):
    """Claude's approval decision for plans/actions."""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


@dataclass
class ModifiedStep:
    """A modified step in a plan."""
    step: str


@dataclass
class PlanApprovalResponse:
    """
    Claude's decision on a plan.

    Model writes:
        PlanApprovalResponse(
            decision=ApprovalDecision.APPROVED,
            reason="Plan looks good"
        )
    """
    decision: ApprovalDecision
    reason: Optional[str] = None
    modified_steps: Optional[List[ModifiedStep]] = None


@dataclass
class ActionApprovalResponse:
    """
    Claude's decision on an action.

    Model writes:
        ActionApprovalResponse(
            decision=ApprovalDecision.APPROVED,
            reason="Action is safe"
        )
    """
    decision: ApprovalDecision
    reason: Optional[str] = None
    modified_action: Optional[Dict[str, Any]] = None


@dataclass
class ClaudeCriticDecision:
    """
    Claude's decision when Critic escalates via ask_claude.

    Model writes:
        ClaudeCriticDecision(
            action="replan",
            explanation="Need different approach",
            feedback="Try using the API instead"
        )
    """
    action: str  # "replan" or "done"
    explanation: str
    feedback: Optional[str] = None


class ActorStatus(Enum):
    """Actor's execution status per iteration."""
    CONTINUE = "continue"  # More work to do, you get another turn
    # ^^^ ---- you get to see results of your actions
    COMPLETE = "complete"  # Request fulfilled
    DONE = "done" # Request fulfilled


# Union of all action types - for typing ActorResponsePython.actions
from typing import Union
Action = Union[
    ReadFileAction, WriteFileAction, EditFileAction, DeleteFileAction,
    CreateDirAction, RunCommandAction, ShellCommandAction, ExecAction,
    SearchAction, IndexAction, GrepAction, AskOracleAction, ProgramAction,
    ScreenshotAction, ClickAction, TypeAction, ScrollAction, KeyPressAction, LocateAction,
    SkillAction, StateCheckAction,
]

@dataclass
class ActorResponsePython:
    """Actor's response.

    One-liner actions (go inline):
        ActorResponsePython(
            status=ActorStatus.CONTINUE,
            actions=[RunCommandAction(command="pytest tests/ -v")]
        )

    Multi-line code/content -- use a ### banner ### (no escaping needed).
    Leave the field as None, then write raw content after a banner:
        ActorResponsePython(
            status=ActorStatus.CONTINUE,
            actions=[ExecAction(code=None, reasoning="Process data")]
        )

        ### code ###
        import json
        with open('data.json') as f:
            data = json.load(f)
        print(data['key'])

    Multiple actions with content -- banner name matches path or field:
        ActorResponsePython(
            status=ActorStatus.CONTINUE,
            actions=[
                WriteFileAction(path="math.py", content=None),
                ExecAction(code=None, reasoning="Test it")
            ]
        )

        ### math.py ###
        def add(a, b):
            return a + b

        ### code ###
        from math import add
        print(add(2, 3))

    Request fulfilled -- no more actions needed:
        ActorResponsePython(
            status=ActorStatus.COMPLETE,
            actions=[],
            reasoning="All tests pass and the file has been written."
        )

    Done after a final action:
        ActorResponsePython(
            status=ActorStatus.DONE,
            actions=[RunCommandAction(command="pytest tests/test_math.py -v")],
            reasoning="Running final verification before finishing."
        )
    """
    status: ActorStatus # <--- PAY ATTENTION TO THE STATUS
    ### ^^^ --- this allows you to continue to response or finish it ;
    actions: List[Action]
    reasoning: str = ""


@dataclass
class AnswerReference:
    """A file:line reference in an answer."""
    file: str
    line: int
    note: Optional[str] = None


@dataclass
class AnswerResponse:
    """
    Final answer to user after Actor completes.

    Short answer (goes inline):
        AnswerResponse(answer="The bug is on line 42: missing return.")

    Long answer -- use a banner (no escaping needed):
        AnswerResponse(answer=None, references=[...])

        ### answer ###
        ...detailed answer...
    """
    answer: str
    references: Optional[List[AnswerReference]] = None
    next_steps: Optional[List[str]] = None


class FileEditOperation(Enum):
    """Type of file edit operation."""
    REPLACE = "replace"
    INSERT = "insert"
    DELETE = "delete"


@dataclass
class FileEditorResponse:
    """FileEditor's output.

    Short target/content (go inline):
        FileEditorResponse(
            operation=FileEditOperation.REPLACE,
            reasoning="Fix typo",
            target="retrun x",
            content="return x",
        )

    Multi-line target/content -- use banners (no escaping needed).
    Leave target/content as None, then write raw content after banners:
        FileEditorResponse(
            operation=FileEditOperation.INSERT,
            reasoning="Add method"
        )

        ### target ###
        return "hello"

        ### content ###
        def farewell():
            return '''Goodbye!'''
    """
    operation: FileEditOperation
    reasoning: str
    target: Optional[str] = None    # Leave None; auto-filled from content block
    content: Optional[str] = None   # Leave None; auto-filled from content block


@dataclass
class ShellBuilderResponse:
    """ShellBuilder's output.

    One-liner (goes inline):
        ShellBuilderResponse(
            explanation="Single quotes preserve literal $",
            command="echo 'Price: $100'",
        )

    Multi-line command -- use a banner (no escaping needed):
        ShellBuilderResponse(explanation="...", command=None)

        ### command ###
        echo "Hello $USER" | tee output.txt
    """
    explanation: str
    command: Optional[str] = None  # Leave None; auto-filled from content block
    warnings: Optional[List[str]] = None


class ExecutionStatus(Enum):
    """Result of execute_request - what Actor returns to NFA."""
    SUCCESS = "success"   # Actor completed the request → REVIEW
    REPLAN = "replan"     # Actor retry with feedback → ACT
    EVALUATE = "evaluate" # Failures need Critic evaluation → EVALUATE
    FAILED = "failed"     # Exhausted retries, no actions generated
    DONE = "done"         # Give up, show what we have (circuit breaker, max iterations)


# --- TypedDicts for LLM responses ---

@dataclass(frozen=True)
class Classification:
    """
    Result from query classifier.

    Immutable, typed, no magic strings.
    `needs_extension` is derived: query_type == CAPABILITY_EXTENSION
    """
    query_type: QueryType
    requires_codebase_context: bool
    reasoning: str
    specific_files: List[str] = field(default_factory=list)

    @property
    def needs_extension(self) -> bool:
        """Derived: capability_extension implies needs_extension."""
        return self.query_type == QueryType.CAPABILITY_EXTENSION

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Classification":
        """Parse from LLM response dict."""
        query_type_str = d.get("query_type", "general")
        try:
            query_type = QueryType(query_type_str)
        except ValueError:
            query_type = QueryType.GENERAL
        return cls(
            query_type=query_type,
            requires_codebase_context=d.get("requires_codebase_context", True),
            reasoning=d.get("reasoning", ""),
            specific_files=d.get("specific_files", []),
        )

    @classmethod
    def default(cls, reasoning: str = "Classification unavailable") -> "Classification":
        """Safe fallback when classification fails."""
        return cls(
            query_type=QueryType.GENERAL,
            requires_codebase_context=True,
            reasoning=reasoning,
        )


# --- Dataclasses for internal state ---

@dataclass(frozen=True)
class ActionResult:
    """Immutable result from executing a single action."""
    success: bool
    message: str
    action_type: str
    target: str = ""

    def __str__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return f"[{status}] {self.action_type}: {self.message[:100]}"


@dataclass(frozen=True)
class FileReadRange:
    """Track what portion of a file was read."""
    start_line: int
    end_line: int
    total_lines: int

    def contains(self, line: int) -> bool:
        """Check if a line number is within this range."""
        return self.start_line <= line <= self.end_line

    def __str__(self) -> str:
        return f"[{self.start_line}-{self.end_line} of {self.total_lines}]"


@dataclass
class FilesRead:
    """Track all read ranges for files."""
    ranges: Dict[str, List[FileReadRange]] = field(default_factory=dict)
    content: Dict[str, str] = field(default_factory=dict)

    def add_read(self, path: str, start: int, end: int, total: int, content: str = "") -> None:
        """Record a file read."""
        if path not in self.ranges:
            self.ranges[path] = []
        self.ranges[path].append(FileReadRange(start, end, total))
        if content:
            self.content[path] = content

    def was_read(self, path: str) -> bool:
        """Check if a file was read at all."""
        return path in self.ranges

    def line_was_read(self, path: str, line: int) -> bool:
        """Check if a specific line was read."""
        if path not in self.ranges:
            return False
        return any(r.contains(line) for r in self.ranges[path])


@dataclass
class ExecutionState:
    """Mutable state during plan execution."""
    step_index: int = 0
    action_results: List[ActionResult] = field(default_factory=list)
    files_read: FilesRead = field(default_factory=FilesRead)
    exec_globals: Dict[str, Any] = field(default_factory=dict)
    retries: int = 0
    max_retries: int = 3

    def add_result(self, result: ActionResult) -> None:
        """Add an action result."""
        self.action_results.append(result)

    def get_results_text(self) -> str:
        """Format action results as text."""
        if not self.action_results:
            return "(no action results)"
        return "\n\n".join(str(r) for r in self.action_results)


# --- Pure Function Types ---
# Types for refactored agent functions (build_*_prompt, parse_*_response)

@dataclass
class ActionTarget:
    """Extracted target/display info from an action (pure)."""
    target: str           # The main target (path, command, etc.)
    display: str          # Human-readable display string
    content: Optional[Any] = None  # Content payload if any


@dataclass
class ValidationResult:
    """Result of validating an action (pure)."""
    is_valid: bool
    error_message: Optional[str] = None


@dataclass
class ActorOutput:
    """Typed output from parse_actor_response (pure)."""
    status: ActorStatus
    actions: List[Action]
    reasoning: str


@dataclass
class CriticOutput:
    """
    Critic's evaluation - typed response from LLM.

    Model writes:
        CriticOutput(
            action=CriticAction.REPLAN,
            explanation="The approach isn't working",
            feedback="Try using a different algorithm"
        )
    """
    action: CriticAction
    explanation: str
    feedback: Optional[str] = None
    rag_query: Optional[str] = None
    question: Optional[str] = None      # For ask_user/ask_claude
    context: Optional[str] = None       # Context for Claude


@dataclass
class ActionBatchResult:
    """Result of executing a batch of actions."""
    success: bool  # Did all actions succeed?
    results: List[str]  # Formatted result strings for context
    last_error: Optional[str]  # Error message if failed
    last_action: Optional[Dict[str, Any]]  # Last executed action (for Critic)
    files_read: Dict[str, List[Tuple[int, int]]]  # Updated files_read
    files_read_content: Dict[str, List[Tuple[int, int, str]]]  # Updated content
    traces: List["ActionTrace"] = field(default_factory=list)  # Structured trace
    errors_content: List[Tuple[str, str]] = field(default_factory=list)  # [(action_target, full_error), ...]
    circuit_breaker_halted: bool = False  # True if circuit breaker triggered - abort, don't retry
    file_snapshots: Dict[str, str] = field(default_factory=dict)  # Pre-modification snapshots for revert


@dataclass
class ExecutionResult:
    """Result of execute_request() - what the Actor returns to the NFA."""
    status: ExecutionStatus
    action_results: List[str] = field(default_factory=list)
    files_read: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    files_read_content: Dict[str, List[Tuple[int, int, str]]] = field(default_factory=dict)
    exec_globals: Dict[str, Any] = field(default_factory=dict)
    # REPLAN-specific fields
    feedback: Optional[str] = None
    rag_query: Optional[str] = None
    # Structured execution trace (for testing/debugging)
    trace: Optional["ExecutionTrace"] = None
    # Full error content for context (not truncated)
    errors_content: List[Tuple[str, str]] = field(default_factory=list)
    # Pre-modification snapshots for REVERT action
    file_snapshots: Dict[str, str] = field(default_factory=dict)
    # EVALUATE-specific fields (for failure evaluation)
    last_action: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None


# --- Type aliases ---

# Path -> [(start_line, end_line), ...]
FileRangesDict = Dict[str, List[Tuple[int, int]]]

# Path -> content string
FileContentDict = Dict[str, str]

# Action dict from LLM
ActionDict = Dict[str, Any]


# --- Pure utility functions ---

def parse_file_range(header: str) -> Optional[FileReadRange]:
    """Parse '[Lines X-Y of Z]' header into FileReadRange.

    Args:
        header: String like '[Lines 11-15 of 100]'

    Returns:
        FileReadRange or None if parsing fails
    """
    import re
    match = re.search(r'\[Lines?\s+(\d+)-(\d+)\s+of\s+(\d+)\]', header)
    if match:
        return FileReadRange(
            start_line=int(match.group(1)),
            end_line=int(match.group(2)),
            total_lines=int(match.group(3)),
        )
    return None


class ActionBase:
    """Base class for all actions in the neo agent system."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    def execute(self, context: dict) -> dict:
        """Execute the action with the given context."""
        raise NotImplementedError(f"{self.__class__.__name__}.execute() must be implemented")

    def to_dict(self) -> dict:
        """Convert action to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionBase":
        """Create an action from dictionary representation."""
        raise NotImplementedError(f"{cls.__name__}.from_dict() must be implemented")
