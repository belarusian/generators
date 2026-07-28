"""
State machine for code mode request processing.

The request processing is modeled as a Non-deterministic Finite Automaton (NFA):
  ACT -> REVIEW -> ANSWER -> DONE

Each state is a pure function: (Context) -> (State, Context)
Accepts when Critic approves and Answerer completes.

This module uses the generic NFARunner from compass.core for execution,
while defining the specific states and context for code mode.
"""

import time
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from compass.llm.oracle import Oracle
from compass.agents.neo.memory import CodeMemory
from compass.agents.neo.index import CodebaseIndex
from compass.core.config import ExecutionConfig
from compass.agents.neo.rag import get_relevant_context
from compass.agents.neo.types import ExecutionResult, ExecutionStatus, CriticAction
from compass.cli import ui
from compass.cli.input import get_input
from compass.core.reasoning import debug
from compass.cli.driver import get_driver, UserDriver, ApprovalDecision
from compass.core.nfa import NFARunner, NFAResult

if TYPE_CHECKING:
    from compass.core.stream_router import StreamRouter
    from compass.agents.neo.trace import ExecutionTrace


class RequestState(Enum):
    """States in the request processing NFA."""
    ACT = auto()       # Actor executes request autonomously
    EVALUATE = auto()  # Critic evaluates failures (decides retry vs give up)
    REVIEW = auto()    # Critic reviews success (decides done vs replan)
    ANSWER = auto()    # Generate final response
    DONE = auto()      # Accept state - success
    FAILED = auto()    # Reject state - failure


def _snapshot_files(plan: Optional[Dict], project_path: str) -> Dict[str, str]:
    """
    Capture file contents before modification for potential revert.

    Args:
        plan: Plan dict with optional 'files_affected' list
        project_path: Base path for resolving relative file paths

    Returns:
        Dict mapping absolute file paths to their contents
    """
    if not plan:
        return {}

    files_affected = plan.get("files_affected", [])
    if not files_affected:
        return {}

    snapshots = {}
    for file_path in files_affected:
        # Handle both absolute and relative paths
        if Path(file_path).is_absolute():
            full_path = Path(file_path)
        else:
            full_path = Path(project_path) / file_path

        if full_path.exists():
            try:
                snapshots[str(full_path)] = full_path.read_text()
            except Exception:
                pass  # Skip files we can't read

    return snapshots


@dataclass
class RequestContext:
    """
    Immutable context that flows through the state machine.

    Implements the NFAContext protocol from compass.core.
    """
    oracle: Oracle
    request: str
    memory: CodeMemory
    codebase_index: Optional[CodebaseIndex] = None
    rag_context: Optional[str] = None
    action_results: List[str] = field(default_factory=list)
    files_read: Optional[Dict] = None
    files_read_content: Optional[Dict] = None
    exec_globals: Optional[Dict] = None
    feedback: Optional[str] = None
    retries: int = 0
    start_time: float = field(default_factory=time.time)
    file_snapshots: Dict[str, str] = field(default_factory=dict)
    config: Optional[ExecutionConfig] = None
    critic_summary: Optional[str] = None  # Critic's analysis flows to Answerer
    stream_router: Optional["StreamRouter"] = None  # For NFA visualization
    execution_trace: Optional["ExecutionTrace"] = None  # For expansion (content registry)
    # EVALUATE state context (from failed ACT)
    last_action: Optional[Dict] = None
    last_error: Optional[str] = None

    def describe(self) -> str:
        """Human-readable description for debugging (NFAContext protocol)."""
        return f"Request: {self.request[:50]}... | Retries: {self.retries}"


# Type alias for state transition functions
TransitionFn = Callable[[RequestContext], Tuple[RequestState, RequestContext]]

# Type alias for execute_request function
# (oracle, request, memory, rag_context?) -> ExecutionResult
ExecuteRequestFn = Callable[..., ExecutionResult]


def oracle_print(text: str) -> None:
    """Print in oracle voice (wrapper for backward compat)."""
    print()
    for char in text:
        print(char, end="", flush=True)
        time.sleep(0.02)
    print()


def create_act_state(execute_request_fn: ExecuteRequestFn) -> TransitionFn:
    """Factory for ACT state transition function."""
    def _act_state(ctx: RequestContext) -> Tuple[RequestState, RequestContext]:
        """
        ACT state: Actor executes request autonomously.

        Transitions:
            -> ACT (if retry requested with feedback)
            -> REVIEW (if execution completed)
            -> FAILED (if execution failed)
        """
        result = execute_request_fn(
            ctx.oracle,
            ctx.request,
            ctx.memory,
            rag_context=ctx.rag_context,
            stream_router=ctx.stream_router,
            directive=ctx.feedback,
            prior_results=ctx.action_results,  # Seed inner loop with outer context
        )

        if result.status == ExecutionStatus.REPLAN:
            # Retry with feedback - loop back to ACT
            rag_context = ctx.rag_context
            if ctx.memory.project_path and result.rag_query:
                rag_result = get_relevant_context(ctx.memory.project_path, result.rag_query, top_k=5)
                if rag_result:
                    rag_context = rag_result.context
                    debug(f"RAG refreshed for retry: {len(rag_context)} chars")

            return RequestState.ACT, RequestContext(
                oracle=ctx.oracle,
                request=ctx.request,
                memory=ctx.memory,
                codebase_index=ctx.codebase_index,
                rag_context=rag_context,
                action_results=result.action_results,  # Inner loop already accumulated
                files_read=result.files_read,
                files_read_content=result.files_read_content,
                feedback=result.feedback,
                retries=ctx.retries + 1,
                start_time=ctx.start_time,
                file_snapshots=result.file_snapshots,
                stream_router=ctx.stream_router,
                execution_trace=result.trace,
            )

        elif result.status in (ExecutionStatus.DONE, ExecutionStatus.SUCCESS):
            # Execution completed - go to REVIEW
            return RequestState.REVIEW, RequestContext(
                oracle=ctx.oracle,
                request=ctx.request,
                memory=ctx.memory,
                codebase_index=ctx.codebase_index,
                rag_context=ctx.rag_context,
                action_results=result.action_results,  # Inner loop already accumulated
                files_read=result.files_read,
                files_read_content=result.files_read_content,
                exec_globals=result.exec_globals,
                feedback=None,
                retries=ctx.retries,
                start_time=ctx.start_time,
                file_snapshots=result.file_snapshots,
                stream_router=ctx.stream_router,
                execution_trace=result.trace,
            )

        elif result.status == ExecutionStatus.EVALUATE:
            # Failures exhausted retries - go to EVALUATE for Critic decision
            return RequestState.EVALUATE, RequestContext(
                oracle=ctx.oracle,
                request=ctx.request,
                memory=ctx.memory,
                codebase_index=ctx.codebase_index,
                rag_context=ctx.rag_context,
                action_results=result.action_results,  # Inner loop already accumulated
                files_read=result.files_read,
                files_read_content=result.files_read_content,
                feedback=result.feedback,
                retries=ctx.retries,
                start_time=ctx.start_time,
                file_snapshots=result.file_snapshots,
                stream_router=ctx.stream_router,
                execution_trace=result.trace,
                last_action=result.last_action,
                last_error=result.last_error,
            )

        else:
            return RequestState.FAILED, ctx

    return _act_state


def create_evaluate_state(critic_evaluate_fn: Callable) -> TransitionFn:
    """Factory for EVALUATE state transition function."""
    def _evaluate_state(ctx: RequestContext) -> Tuple[RequestState, RequestContext]:
        """
        EVALUATE state: Critic evaluates failures.

        Called when Actor exhausted retries. Critic decides:
        - REPLAN: try again with feedback → ACT
        - DONE: give up → FAILED

        Transitions:
            -> ACT (if Critic says retry)
            -> FAILED (if Critic gives up)
        """
        ui.start_spinner("Evaluating")

        # Build context for critic_evaluate
        if not ctx.last_action:
            # No action to evaluate - skip critic and fail
            return RequestState.FAILED, RequestContext(
                oracle=ctx.oracle,
                request=ctx.request,
                memory=ctx.memory,
                codebase_index=ctx.codebase_index,
                action_results=ctx.action_results + ["No action to evaluate"],
                config=ctx.config,
                start_time=ctx.start_time,
                stream_router=ctx.stream_router,
            )
        last_action = ctx.last_action
        error = ctx.last_error or "Unknown error"
        critic_context = f"Request: {ctx.request}\n\nResults:\n" + "\n\n".join(ctx.action_results[-10:])
        if ctx.feedback:
            critic_context += f"\n\nPrevious feedback:\n{ctx.feedback}"

        review = critic_evaluate_fn(
            ctx.oracle,
            ctx.request,
            last_action,
            error,
            critic_context,
            config=ctx.config,
            execution_trace=ctx.execution_trace,
        )

        ui.stop_spinner()

        if review and review.action == CriticAction.REPLAN:
            # Retry with Critic's feedback
            feedback = review.feedback or review.explanation
            new_request = ctx.request
            if ctx.critic_summary:
                new_request += "\n" + ctx.critic_summary

            return RequestState.ACT, RequestContext(
                oracle=ctx.oracle,
                request=new_request,
                memory=ctx.memory,
                codebase_index=ctx.codebase_index,
                rag_context=ctx.rag_context,
                action_results=ctx.action_results,
                files_read=ctx.files_read,
                files_read_content=ctx.files_read_content,
                feedback=feedback,
                retries=ctx.retries + 1,
                start_time=ctx.start_time,
                stream_router=ctx.stream_router,
            )
        else:
            # Critic says give up
            return RequestState.FAILED, ctx

    return _evaluate_state


def create_review_state(critic_review_fn: Callable) -> TransitionFn:
    """Factory for REVIEW state transition function."""
    def _review_state(ctx: RequestContext) -> Tuple[RequestState, RequestContext]:
        """
        REVIEW state: Critic evaluates results.

        Transitions:
            -> ANSWER (if Critic approves)
            -> ACT (if Critic wants retry)
        """
        ui.start_spinner("Reviewing")

        review = critic_review_fn(
            ctx.oracle,
            ctx.request,  # Full request - never truncate
            ctx.action_results,  # Accumulated results from THIS request
            ctx.memory.get_answer_context(),  # Lightweight - no codebase index
            ctx.memory,
            exec_globals=ctx.exec_globals,
            files_read=ctx.files_read,
            config=ctx.config,
            execution_trace=ctx.execution_trace,
        )

        ui.stop_spinner()

        # CriticOutput is a dataclass - access attributes directly
        # Note: ASK_CLAUDE is handled internally by critic_review
        action = review.action if review else None

        if action == CriticAction.DONE:
            # Pass Critic's analysis to Answerer (pipeline, not parallel)
            summary = review.explanation if review else None
            return RequestState.ANSWER, replace(ctx, critic_summary=summary)

        elif action == CriticAction.ASK_USER:
            # Critic needs user clarification - pause for input
            question = review.question or review.explanation
            print(f"\n  {ui.Colors.yellow('[Needs clarification]')}: {question}")
            print()

            user_response = get_input("Your answer: ")

            if not user_response or user_response.lower() in ('q', 'quit', 'exit'):
                return RequestState.FAILED, ctx

            # Persist to conversation so future context includes this exchange
            ctx.memory.add_user_turn(f"[Clarification] Q: {question}\nA: {user_response}")

            feedback = f"User clarification for: {question}\nUser's answer: {user_response}"

            return RequestState.ACT, RequestContext(
                oracle=ctx.oracle,
                request=ctx.request,
                memory=ctx.memory,
                codebase_index=ctx.codebase_index,
                rag_context=ctx.rag_context,
                action_results=ctx.action_results,
                files_read=ctx.files_read,
                files_read_content=ctx.files_read_content,
                feedback=feedback,
                retries=ctx.retries + 1,
                start_time=ctx.start_time,
                stream_router=ctx.stream_router,
            )

        elif action == CriticAction.REPLAN:
            # Replan now means retry with Actor
            feedback = review.feedback or review.explanation
            rag_query = review.rag_query

            rag_context = ctx.rag_context
            if ctx.memory.project_path and rag_query:
                rag_result = get_relevant_context(ctx.memory.project_path, rag_query, top_k=5)
                if rag_result:
                    rag_context = rag_result.context
                    debug(f"RAG refreshed for retry ({rag_query}): {len(rag_context)} chars")

            return RequestState.ACT, RequestContext(
                oracle=ctx.oracle,
                request=ctx.request + "\n" + feedback,
                memory=ctx.memory,
                codebase_index=ctx.codebase_index,
                rag_context=rag_context,
                action_results=ctx.action_results,
                files_read=ctx.files_read,
                files_read_content=ctx.files_read_content,
                feedback=feedback,
                retries=ctx.retries + 1,
                start_time=ctx.start_time,
                stream_router=ctx.stream_router,
            )

        else:
            # Fallback to ANSWER with Critic's analysis
            summary = review.explanation if review else None
            return RequestState.ANSWER, replace(ctx, critic_summary=summary)

    return _review_state


def create_answer_state(
    generate_answer_fn: Callable,
    set_last_answer_fn: Callable,
) -> TransitionFn:
    """Factory for ANSWER state transition function."""
    def _answer_state(ctx: RequestContext) -> Tuple[RequestState, RequestContext]:
        """
        ANSWER state: Generate final response.

        Transitions:
            -> DONE (always)
        """
        ui.start_spinner("Answering")
        answer_data = generate_answer_fn(
            ctx.oracle,
            ctx.request,  # Full request - never truncate
            ctx.action_results,  # Accumulated results from THIS request
            ctx.memory.get_answer_context(),  # Lightweight context (no AST/learnings)
            memory=ctx.memory,
            exec_globals=ctx.exec_globals,
            files_read=ctx.files_read,
            config=ctx.config,
            critic_summary=ctx.critic_summary,  # Critic's analysis (pipeline)
            execution_trace=ctx.execution_trace,
        )
        ui.stop_spinner()

        if answer_data:
            set_last_answer_fn(
                answer_data.answer,
                ctx.action_results,
            )

            ctx.memory.set_last_answer(
                answer_data.answer,
                answer_data.next_steps or []
            )

            elapsed = time.time() - ctx.start_time
            ui.show_duration(elapsed)
            ui.show_answer(answer_data)

        return RequestState.DONE, ctx

    return _answer_state


def process_request(
    oracle: Oracle,
    request: str,
    memory: CodeMemory,
    transitions: Dict[RequestState, TransitionFn],
    codebase_index: Optional[CodebaseIndex] = None,
    rag_context: Optional[str] = None,
    max_retries: int = 5,
    config: Optional[ExecutionConfig] = None,
    stream_router: Optional["StreamRouter"] = None,
) -> bool:
    """
    Process a single request through the NFA state machine.

    Uses the generic NFARunner from compass.core for execution.

    Args:
        oracle: Oracle instance for LLM calls
        request: User's request string
        memory: CodeMemory for session state
        transitions: State transition function table
        codebase_index: Optional codebase index
        rag_context: Optional RAG context
        max_retries: Maximum number of retry attempts
        config: Execution config with provider/think_level overrides
        stream_router: Optional StreamRouter for visualization.
            If provided, wraps oracle for streaming events and emits
            state transitions for real-time visualization.

    Returns:
        True if request completed successfully
    """
    # Wrap oracle with streaming if router provided
    effective_oracle = oracle
    if stream_router:
        from compass.core.stream_router import StreamingOracle
        effective_oracle = StreamingOracle(oracle, stream_router)

    # Seed with history from previous requests (gives context without polluting)
    # This becomes the baseline; new actions accumulate on top
    historical_results = memory.get_recent_action_results(10) if memory else []
    if historical_results:
        # Add marker so Critic can distinguish history from new work
        historical_results = historical_results + ["\n🚀🚀🚀 ACTIONS FOR THIS REQUEST BELOW 🚀🚀🚀\n"]

    ctx = RequestContext(
        oracle=effective_oracle,
        request=request,
        memory=memory,
        codebase_index=codebase_index,
        rag_context=rag_context,
        config=config,
        stream_router=stream_router,
        action_results=historical_results,  # Seed with history + marker
    )

    # Wrap transitions to enforce max_retries
    def wrap_with_retry_check(transition_fn: TransitionFn) -> TransitionFn:
        def wrapped(ctx: RequestContext) -> Tuple[RequestState, RequestContext]:
            if ctx.retries > max_retries:
                debug(f"Max retries ({max_retries}) exceeded")
                return RequestState.FAILED, ctx
            return transition_fn(ctx)
        return wrapped

    wrapped_transitions = {
        state: wrap_with_retry_check(fn)
        for state, fn in transitions.items()
    }

    # Wrap with outer compaction (gentle: 2/3 retention)
    from compass.core.compaction import wrap_transitions_with_compaction
    wrapped_transitions = wrap_transitions_with_compaction(
        wrapped_transitions,
        oracle=effective_oracle,
    )

    # Debug callback for transition logging + stream events + telemetry
    def on_transition(from_state, to_state, ctx, step, duration=0.0):
        debug(f"NFA: {from_state.name} -> {to_state.name} (step {step}, {duration:.2f}s)")
        # Record to telemetry (include error and duration for failure states)
        from compass.core.telemetry import record_transition
        import os
        error = ctx.last_error if to_state in (RequestState.EVALUATE, RequestState.FAILED) else None
        # In tests: use test name (PYTEST_CURRENT_TEST). In production: use truncated request.
        source = None
        if error and not os.environ.get("PYTEST_CURRENT_TEST"):
            source = ctx.request[:50] if ctx.request else None
        record_transition(from_state.name, to_state.name, error=error, source=source, duration=duration)
        if stream_router:
            from compass.core.stream_types import StreamEvent, StreamEventType
            stream_router.set_state(to_state.name)
            stream_router.emit(StreamEvent(
                type=StreamEventType.TRANSITION,
                data={
                    "from": from_state.name,
                    "to": to_state.name,
                    "step": step,
                    "duration": duration,
                },
            ))

    # Create and run the NFA
    runner = NFARunner(
        transitions=wrapped_transitions,
        initial_state=RequestState.ACT,
        terminal_states={RequestState.DONE, RequestState.FAILED},
        success_states={RequestState.DONE},
        max_iterations=100,
        on_transition=on_transition,
    )

    # Run with stream context if router provided
    if stream_router:
        with stream_router.nfa_context("neo"):
            stream_router.set_state(RequestState.ACT.name)
            result = runner.run(ctx)
    else:
        result = runner.run(ctx)

    if result.error:
        debug(f"NFA error: {result.error}")

    return result.success


def create_transitions(
    execute_request_fn: Callable,
    critic_evaluate_fn: Callable,
    critic_review_fn: Callable,
    generate_answer_fn: Callable,
    set_last_answer_fn: Callable,
) -> Dict[RequestState, TransitionFn]:
    """
    Create the state machine transitions dict.

    Centralizes all wiring in one place for discoverability.

    Args:
        execute_request_fn: Actor function (oracle, request, memory, ...) -> ExecutionResult
        critic_evaluate_fn: Critic evaluate function for failure recovery
        critic_review_fn: Critic review function for final gate
        generate_answer_fn: Answerer function to generate final response
        set_last_answer_fn: Function to store last answer for context

    Returns:
        Dict mapping RequestState to transition functions
    """
    return {
        RequestState.ACT: create_act_state(execute_request_fn),
        RequestState.EVALUATE: create_evaluate_state(critic_evaluate_fn),
        RequestState.REVIEW: create_review_state(critic_review_fn),
        RequestState.ANSWER: create_answer_state(generate_answer_fn, set_last_answer_fn),
    }
