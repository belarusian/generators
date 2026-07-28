"""
Programmer Tool - Entry point for invoking the Programmer NFA.

This is what Actor calls when it needs to generate a programming solution.
Like call_file_editor() or call_shell_builder(), this provides a clean
interface that hides the NFA complexity.
"""

from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from compass.core.nfa import NFARunner
from compass.agents.programmer.context import (
    ProgrammerState,
    ProgrammerContext,
    ProgrammerResult,
)
from compass.agents.programmer.states import create_transitions
from compass.agents.programmer.trace import ProgrammerTrace
from compass.core.reasoning import debug

if TYPE_CHECKING:
    from compass.core.stream_router import StreamRouter


def call_programmer(
    oracle,
    problem: str,
    constraints: Optional[List[str]] = None,
    fetch_pattern: Optional[Callable[[str], str]] = None,
    get_file_structure: Optional[Callable[[], Dict[str, str]]] = None,
    get_coding_standards: Optional[Callable[[], List[str]]] = None,
    apply_chunks: Optional[Callable[[List[Dict]], Tuple[bool, str]]] = None,
    parent_feedback: Optional[str] = None,
    max_scribe_iterations: int = 3,
    is_cancelled: Optional[Callable[[], bool]] = None,
    stream_router: Optional["StreamRouter"] = None,
    show_prompts: bool = True,
) -> ProgrammerResult:
    """
    Invoke the Programmer NFA as a tool.

    This is the main entry point - what Actor calls when it needs to
    generate a programming solution. The Programmer NFA will:

    1. UNDERSTAND the problem
    2. DESIGN a solution
    3. IMPLEMENT as chunks
    4. SCRIBE_REVIEW validates against system
    5. (Loop with feedback if needed)
    6. DELIVER final chunks

    Args:
        oracle: Oracle instance for LLM calls
        problem: The problem statement to solve
        constraints: Optional explicit constraints
        fetch_pattern: Callback to fetch code patterns (for Scribe)
        get_file_structure: Callback to get file structure (for Scribe)
        get_coding_standards: Callback to get coding standards (for Scribe)
        max_scribe_iterations: Max feedback loops before forced delivery
        is_cancelled: Optional predicate for cooperative cancellation.
            If provided, NFA checks this at transition boundaries.
        stream_router: Optional StreamRouter for visualization.
            If provided, wraps oracle for streaming events and emits
            state transitions for real-time visualization.

    Returns:
        ProgrammerResult with success status, solution, and chunks
    """
    # Wrap oracle with streaming if router provided
    effective_oracle = oracle
    if stream_router:
        from compass.core.stream_router import StreamingOracle
        effective_oracle = StreamingOracle(oracle, stream_router)

    # Create context with bounded information
    ctx = ProgrammerContext(
        oracle=effective_oracle,
        problem=problem,
        constraints=constraints or [],
        parent_feedback=parent_feedback,
        show_prompts=show_prompts,
        fetch_pattern=fetch_pattern,
        get_file_structure=get_file_structure,
        get_coding_standards=get_coding_standards,
        apply_chunks=apply_chunks,
        max_scribe_iterations=max_scribe_iterations,
        trace=ProgrammerTrace(),  # Initialize execution trace
    )

    # Create transitions (wrapped with cancellation if provided)
    transitions = create_transitions(is_cancelled)

    # Transition callback - debug + streaming + telemetry
    def on_transition(from_state, to_state, ctx, iteration, duration=0.0):
        debug(f"Programmer NFA: {from_state.name} -> {to_state.name} (iteration {iteration}, {duration:.2f}s)")

        # Record to telemetry with duration
        from compass.core.telemetry import record_transition
        import os
        error = ctx.last_error if to_state == ProgrammerState.CRITIC_EVALUATE or to_state == ProgrammerState.FAILED else None
        source = os.environ.get("PYTEST_CURRENT_TEST", "")
        if "::" in source:
            source = source.split("::")[-1].split(" ")[0]
        record_transition(from_state.name, to_state.name, error=error, source=source or None, duration=duration)

        # Emit stream events if router provided
        if stream_router:
            from compass.core.stream_types import StreamEvent, StreamEventType
            stream_router.set_state(to_state.name)
            stream_router.emit(StreamEvent(
                type=StreamEventType.TRANSITION,
                data={
                    "from": from_state.name,
                    "to": to_state.name,
                    "iteration": iteration,
                    "duration": duration,
                },
            ))

    # Create and run the NFA
    runner = NFARunner(
        transitions=transitions,
        initial_state=ProgrammerState.UNDERSTAND,
        terminal_states={ProgrammerState.DONE, ProgrammerState.FAILED},
        success_states={ProgrammerState.DONE},
        max_iterations=50,  # Safety limit
        on_transition=on_transition,
    )

    # Run with stream context if router provided
    if stream_router:
        with stream_router.nfa_context("programmer"):
            stream_router.set_state(ProgrammerState.UNDERSTAND.name)
            result = runner.run(ctx)
    else:
        result = runner.run(ctx)

    # Build result
    final_ctx = result.context

    if result.success:
        return ProgrammerResult(
            success=True,
            solution_doc=final_ctx.solution_doc or "",
            chunks=final_ctx.chunks,
            reasoning=final_ctx.design or "",
            scribe_issues=final_ctx.scribe_view.issues if final_ctx.scribe_view else [],
            iterations=result.iterations,
            trace=final_ctx.trace,
        )
    else:
        return ProgrammerResult(
            success=False,
            solution_doc=final_ctx.solution_doc or "",
            chunks=final_ctx.chunks,
            reasoning=final_ctx.design or "",
            scribe_issues=final_ctx.scribe_view.issues if final_ctx.scribe_view else [],
            error=result.error or "Programmer NFA failed",
            iterations=result.iterations,
            trace=final_ctx.trace,
        )


def create_pattern_fetcher(oracle, memory=None) -> Callable[[str], str]:
    """
    Create a pattern fetcher callback for the Programmer.

    This callback allows Scribe to request code patterns from the system.
    The fetcher can use RAG, file reading, or other methods to find
    relevant code.

    Args:
        oracle: Oracle instance for semantic search
        memory: Optional CodeMemory for file access

    Returns:
        Callable that takes a query and returns matching code pattern
    """
    def fetch_pattern(query: str) -> str:
        # Simple implementation - could be enhanced with RAG
        if memory and memory.project_path:
            # Try to find relevant files
            import os
            import glob

            # Simple keyword-based search
            keywords = query.lower().split()
            matches = []

            for root, _, files in os.walk(memory.project_path):
                # Skip hidden and common ignore patterns
                if any(p in root for p in ['.git', '__pycache__', 'node_modules', '.venv']):
                    continue

                for f in files:
                    if not f.endswith('.py'):
                        continue

                    filepath = os.path.join(root, f)
                    try:
                        with open(filepath, 'r') as fh:
                            content = fh.read()
                            if any(kw in content.lower() for kw in keywords):
                                # Return full file content for pattern matching
                                # Scribe needs complete code to evaluate patterns properly
                                rel_path = os.path.relpath(filepath, memory.project_path)
                                matches.append(f"# {rel_path}\n{content}")
                    except Exception:
                        continue

                    if len(matches) >= 2:
                        break
                if len(matches) >= 2:
                    break

            if matches:
                return "\n\n".join(matches)

        return f"(no pattern found for: {query})"

    return fetch_pattern


def create_file_structure_getter(memory=None) -> Callable[[], Dict[str, str]]:
    """
    Create a file structure getter callback.

    Returns a dict of file paths to brief descriptions.
    """
    def get_file_structure() -> Dict[str, str]:
        if memory and memory.project_path:
            import os

            structure = {}
            for root, _, files in os.walk(memory.project_path):
                # Skip hidden and common ignore patterns
                if any(p in root for p in ['.git', '__pycache__', 'node_modules', '.venv']):
                    continue

                for f in files:
                    if not f.endswith('.py'):
                        continue

                    filepath = os.path.join(root, f)
                    rel_path = os.path.relpath(filepath, memory.project_path)

                    # Get file size as simple description
                    try:
                        size = os.path.getsize(filepath)
                        structure[rel_path] = f"{size} bytes"
                    except Exception:
                        structure[rel_path] = "unknown size"

            return structure

        return {}

    return get_file_structure


def create_coding_standards_getter() -> Callable[[], List[str]]:
    """
    Create a coding standards getter callback.

    Returns a list of coding standards/conventions.
    """
    def get_coding_standards() -> List[str]:
        # Default Python coding standards
        return [
            "Code is the documentation.",
            "FP principals to the core.",
            "DDD principals align with FP.",
            "Avoid mutation.",
            "Avoid programming by chance",
            "Avoid relying on mutation"
        ]

    return get_coding_standards
