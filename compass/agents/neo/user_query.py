"""
User query execution - orchestrates the Actor/Critic loop.

Main entry point: execute_request()

The loop:
1. Actor generates actions to fulfill the request
2. Actions are executed via execute_action()
3. Critic reviews results, decides retry vs done
4. Answerer generates final response
"""

CYCLE_BREAKING = True  # Trinity halts and re-plans after calling this

import json
import os
import re
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from compass.core.stream_router import StreamRouter

from compass.llm.oracle import Oracle
from compass.agents.neo.memory import CodeMemory, Plan, Action as ActionRecord
from compass.agents.neo.types import (
    Action, ActorStatus, ActionBatchResult, ExecutionResult, ExecutionStatus,
    ExecutionContext, WriteFileAction, ReadFileAction, EditFileAction,
    AskOracleAction, RunCommandAction,
)
from compass.agents.neo.trace import ActionTrace, ExecutionTrace, trace_from_action
from compass.core.content import truncate_lines
from compass.agents.neo.rules import extract_learnings
from compass.agents.neo.rules import execute_action
from compass.agents.neo.dispatch import get_registered_types, display_name, format_result
from compass.cli.commands import (
    get_show_thinking,
    _build_files_read_section,
    _build_errors_section,
)
from compass.core.config import ExecutionConfig
from compass.core.actor_loop import LoopState, to_result, run_loop, with_hesitation
from compass.core.ui_adapter import ImmediateUIAdapter
from compass.core.reasoning import debug
from compass.agents.neo.dispatch import validate as validate_dispatch
import subprocess

# Import actor functions from dedicated module
from compass.agents.neo.actor import (
    parse_actor_response,
    check_circuit_breaker,
    extract_action_target,
    call_actor,
)



def _git_branch(project_path: str) -> str:
    """
    Return the name of the current git branch for *project_path*.
    Returns "unknown" on any failure (not a repo, git missing, etc.).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _parse_read_result_lines(result: str, action: Action) -> Tuple[int, int, Optional[int]]:
    """Parse line range from read_file result.

    Pure function - extracts (start_line, end_line, total_lines).
    Returns from header "[Lines X-Y of Z]" or falls back to offset/limit.
    """
    offset = action.offset or 0
    limit = action.limit

    # Try to extract from result header
    line_match = re.search(r'\[Lines? (\d+)-(\d+) of (\d+)', result or "")
    if line_match:
        start_line = int(line_match.group(1))
        end_line = int(line_match.group(2))
        total_lines = int(line_match.group(3))
        return start_line, end_line, total_lines

    # Fall back to offset/limit
    if limit:
        start_line = offset + 1  # Convert 0-based to 1-based
        end_line = offset + limit
        return start_line, end_line, None

    # Full file read - count lines
    line_count = result.count('\n') + 1 if result else 0
    start_line = 1
    end_line = line_count if line_count > 0 else 10000
    return start_line, end_line, None


def _format_read_result(target: str, start_line: int, end_line: int, total_lines: Optional[int]) -> str:
    """Format read_file result line (simplified, without full content).

    Pure function - returns "PASS: read_file target: lines X-Y of Z".
    """
    if total_lines:
        return f"PASS: read_file {target}: lines {start_line}-{end_line} of {total_lines}"
    return f"PASS: read_file {target}: lines {start_line}-{end_line}"


def _update_files_read(
    files_read: Dict[str, List[Tuple[int, int]]],
    abs_path: str,
    start_line: int,
    end_line: int
) -> Dict[str, List[Tuple[int, int]]]:
    """Return updated files_read dict with new range.

    Pure function - returns new dict, doesn't mutate input.
    """
    result = {k: list(v) for k, v in files_read.items()}  # Shallow copy
    if abs_path not in result:
        result[abs_path] = []
    result[abs_path].append((start_line, end_line))
    return result


def _refresh_files_content(
    files_read: Dict[str, List[Tuple[int, int]]],
    project_path: str,
) -> Dict[str, List[Tuple[int, int, str]]]:
    """Re-read files from disk to get current content.

    This ensures FILES READ shows current state after edits, not stale cached content.
    files_read uses absolute paths, we return relative paths for display.
    """
    result = {}
    for abs_path, ranges in files_read.items():
        # Convert to relative path for display
        try:
            rel_path = os.path.relpath(abs_path, project_path)
        except ValueError:
            rel_path = abs_path  # Fallback if paths on different drives

        # Read current file content
        try:
            with open(abs_path, 'r') as f:
                lines = f.readlines()
        except Exception:
            continue  # Skip files that can't be read

        total_lines = len(lines)
        chunks = []
        for start_line, end_line in ranges:
            # Format content like read_file does
            chunk_lines = []
            actual_end = min(end_line, total_lines)
            for i in range(start_line - 1, actual_end):
                if 0 <= i < total_lines:
                    chunk_lines.append(f"line {i+1}: {lines[i].rstrip()}")
            content = "\n".join(chunk_lines)
            chunks.append((start_line, actual_end, content))

        if chunks:
            result[rel_path] = chunks

    return result


def _update_files_read_content(
    files_read_content: Dict[str, List[Tuple[int, int, str]]],
    rel_path: str,
    start_line: int,
    end_line: int,
    content: str
) -> Dict[str, List[Tuple[int, int, str]]]:
    """Return updated files_read_content with new chunk, merging overlaps.

    Pure function - returns new dict, doesn't mutate input.

    Key behaviors:
    - If new range is fully contained in existing, skip it (no duplicate)
    - If new range fully contains existing, replace it
    - If ranges overlap partially, keep both (different content)
    - Non-overlapping ranges are kept separately
    """
    result = {k: list(v) for k, v in files_read_content.items()}  # Shallow copy
    if rel_path not in result:
        result[rel_path] = [(start_line, end_line, content)]
        return result

    existing = result[rel_path]

    # First pass: check if new range is contained in any existing range
    for exist_start, exist_end, _ in existing:
        if exist_start <= start_line and exist_end >= end_line:
            # New range is contained in (or equal to) existing - skip entirely
            return result  # Return unchanged

    # Second pass: filter out existing ranges that new range supersedes
    new_chunks = []
    for exist_start, exist_end, exist_content in existing:
        if start_line <= exist_start and end_line >= exist_end:
            # New range fully contains existing - don't keep it
            continue
        new_chunks.append((exist_start, exist_end, exist_content))

    # Add the new chunk
    new_chunks.append((start_line, end_line, content))

    # Sort by start line for clean display
    new_chunks.sort(key=lambda x: x[0])

    result[rel_path] = new_chunks
    return result


def _execute_actions(
    actions: List[Action],
    memory: CodeMemory,
    exec_globals: Dict,
    files_read: Dict[str, List[Tuple[int, int]]],
    files_read_content: Dict[str, List[Tuple[int, int, str]]],
    step_num: int = 0,
    step: str = "",
    actor_context: str = "",
    action_history: Optional[List[ActionTrace]] = None,
    step_results: Optional[List[str]] = None,
    plan: Optional[Dict] = None,
    planner_context: str = "",
    execution_trace: Optional[ExecutionTrace] = None,
    oracle: Optional[Oracle] = None,
    ctx: Optional[ExecutionContext] = None,
) -> ActionBatchResult:
    """
    Execute a batch of actions, stopping on first failure.

    Uses existing pure functions for validation and formatting.
    Returns results needed for Actor retry loop.

    Args:
        actor_context: Full context the Actor was working with (for ask_claude)
        action_history: Running list of actions for loop detection (persists across calls)
        step_results: All action results so far in this step (for bug reports)
        plan: Current plan being executed (for bug reports)
        planner_context: Full planner context (for bug reports)
        ctx: Execution context with UI adapter (defaults to ImmediateUIAdapter)
    """
    from dataclasses import asdict

    # Initialize ctx with default UI if not provided
    if ctx is None:
        ctx = ExecutionContext(ui=ImmediateUIAdapter())
    elif ctx.ui is None:
        ctx = ExecutionContext(
            exec_globals=ctx.exec_globals,
            oracle=ctx.oracle,
            memory=ctx.memory,
            files_read=ctx.files_read,
            ui=ImmediateUIAdapter(),
            on_thinking=ctx.on_thinking,
        )

    results: List[str] = []
    traces: List[ActionTrace] = []
    errors_content: List[Tuple[str, str]] = []  # Full error content for context
    last_error: Optional[str] = None
    last_action: Optional[Dict] = None
    actions_succeeded = 0

    def _truncate_result(result: str, label: str = "result") -> str:
        """Truncate long results."""
        MAX_RESULT_LINES = 50
        return truncate_lines(result, max_lines=MAX_RESULT_LINES, label=label)

    project_path = memory.project_path or "."
    file_snapshots: Dict[str, str] = {}
    if action_history is None:
        action_history = []

    # Track reads completed in this batch -- writes after reads are rejected
    # because write content was generated BEFORE reads executed (hallucination risk)
    reads_in_batch: List[str] = []

    for action in actions:
        action_name = type(action).__name__

        # Validate action type - must be a registered type
        if type(action) not in get_registered_types():
            from dataclasses import asdict
            action_str = json.dumps(asdict(action) if hasattr(action, '__dataclass_fields__') else str(action))
            ctx.ui.message(f"  [!] MALFORMED ACTION: {action_str[:200]}")

        # Extract target/display using existing helper
        target_info = extract_action_target(action)
        target = target_info.target
        display = target_info.display
        content = target_info.content
        action_reasoning = action.reasoning or ""

        # --- CIRCUIT BREAKER: Model stuck on same action 3+ times ---
        should_escalate, reason = check_circuit_breaker(action, action_history)
        if should_escalate:
            # Model isn't learning - hand off to a better model
            ctx.ui.message(f"  [^] {reason[:60]}...")
            results.append(f"[^] ESCALATE: {action_name} {target} - model stuck on same action. Try a DIFFERENT action type (e.g. RunCommandAction with sed, ExecAction, or WriteFileAction).")
            return ActionBatchResult(
                success=False,
                results=results,
                last_error=reason,
                last_action=action,
                files_read=files_read,
                files_read_content=files_read_content,
                traces=traces,
                errors_content=errors_content,
                circuit_breaker_halted=True,  # Signal: escalate to better model
                file_snapshots=file_snapshots,
            )

        ctx.ui.show_action(display_name(action), display, action_reasoning)

        # Show content preview for writes
        if content and isinstance(action, WriteFileAction):
            preview = str(content)[:300] + "..." if len(str(content)) > 300 else str(content)
            for line in preview.split("\n")[:10]:
                ctx.ui.message(f"      {line}")
            if len(str(content)) > 300 or str(content).count("\n") > 10:
                ctx.ui.message("      ...")

        # Validate via singledispatch -- each action type defines its own
        is_valid, error_message = validate_dispatch(action, project_path, files_read)
        if not is_valid:
            ctx.ui.show_result(False, error_message)
            results.append(f"FAIL: {action_name} {target}: {error_message}")
            errors_content.append((f"{action_name} {target}", error_message))
            # Record failed action so circuit breaker sees the attempt (not a gap)
            action_history.append(trace_from_action(action_name, target, False, error_message, action))
            last_error = error_message
            last_action = action
            # Don't abort -- skip bad action, try the rest
            continue

        # --- READ-BEFORE-WRITE GATE ---
        # Write content is baked at generation time. If a Read preceded this Write
        # in the same batch, the content was hallucinated (generated before the read
        # executed). Force the model to split: read first, write in next response.
        if isinstance(action, WriteFileAction) and reads_in_batch:
            read_list = ", ".join(reads_in_batch)
            error_message = (
                f"Read + Write in same response: you read [{read_list}] and write "
                f"'{action.path}' in one batch. Your write content was generated "
                f"BEFORE the reads executed -- it cannot reference read results. "
                f"Read first (return status=CONTINUE), then write with the actual "
                f"values in your next response."
            )
            ctx.ui.show_result(False, error_message)
            results.append(f"FAIL: {action_name} {target}: {error_message}")
            errors_content.append((f"{action_name} {target}", error_message))
            action_history.append(trace_from_action(action_name, target, False, error_message, action))
            return ActionBatchResult(
                success=False,
                results=results,
                last_error=error_message,
                last_action=action,
                files_read=files_read,
                files_read_content=files_read_content,
                traces=traces,
                errors_content=errors_content,
                file_snapshots=file_snapshots,
            )

        # --- APPROVAL GATE ---
        # Ask the Driver (user or Claude) to approve before executing
        from compass.cli.driver import get_driver, ApprovalDecision
        driver = get_driver()
        if driver.should_approve_actions():
            context = f"Step #{step_num}\nPrevious results: {results[-3:]}"  # Last 3 results
            approval = driver.approve_action(action, context)

            if approval.decision == ApprovalDecision.REJECTED:
                rejection_msg = f"REJECTED - {approval.feedback}"
                ctx.ui.show_result(False, f"Action rejected: {approval.feedback}")
                results.append(f"FAIL: {action_name} {target}: {rejection_msg}")
                errors_content.append((f"{action_name} {target}", rejection_msg))
                action_history.append(trace_from_action(action_name, target, False, rejection_msg, action))
                return ActionBatchResult(
                    success=False,
                    results=results,
                    last_error=f"Driver rejected: {approval.feedback}",
                    last_action=action,
                    files_read=files_read,
                    files_read_content=files_read_content,
                    traces=traces,
                    errors_content=errors_content,
                    file_snapshots=file_snapshots,
                )

            if approval.decision == ApprovalDecision.MODIFIED and approval.modified_action:
                action = approval.modified_action  # Use modified version

        # Pass extra context through ExecutionContext for actions that need it
        if isinstance(action, AskOracleAction):
            ctx = ExecutionContext(
                exec_globals=ctx.exec_globals,
                oracle=ctx.oracle,
                memory=ctx.memory,
                files_read=ctx.files_read,
                ui=ctx.ui,
                on_thinking=ctx.on_thinking,
                stream_router=ctx.stream_router,
                actor_context=actor_context,
                execution_trace=execution_trace,
            )

        # Emit ACTION_START for stream visualization
        if ctx.stream_router:
            from compass.core.stream_types import StreamEvent, StreamEventType
            from compass.agents.neo.types import ProgramAction
            action_data = {"action_type": action_name, "target": target}
            # For program actions, tell UI to look for programmer_*.jsonl files
            if isinstance(action, ProgramAction):
                action_data["programmer_pattern"] = "programmer_*.jsonl"
            ctx.stream_router.emit(StreamEvent(
                type=StreamEventType.ACTION_START,
                data=action_data,
            ))

        # Dispatch to unified execute_action (handles all action types)
        exec_result = execute_action(
            action, project_path,
            files_read=files_read,
            exec_globals=exec_globals,
            oracle=oracle,
            memory=memory,
            ctx=ctx,  # Pass full context for expand, ask_claude, etc.
        )
        if exec_result is None:
            success, result = False, f"Unknown action type: {action_name}"
        else:
            success, result = exec_result

        # Emit ACTION_END for stream visualization
        if ctx.stream_router:
            from compass.core.stream_types import StreamEvent, StreamEventType
            action_end_data = {
                "action_type": action_name,
                "target": target,
                "success": success,
            }
            # Include result/error for introspection - action controls its own formatting
            if result:
                action_end_data["result"] = format_result(action, result)
            ctx.stream_router.emit(StreamEvent(
                type=StreamEventType.ACTION_END,
                data=action_end_data,
            ))

        last_action = action

        # Record in memory
        action_record = ActionRecord(
            action_type=action_name,
            target=target,
            content=content,
            reasoning=action_reasoning,
            result=result,
            success=success,
        )
        memory.add_action(action_record)

        # Extract and store learnings (reflector is injected side effect)
        from compass.agents.neo.types import LearningType, LearningResponse
        from compass.agents.neo.memory import Learning

        def create_reflector(oracle: Optional["Oracle"]) -> Callable[[str], Learning]:
            """
            Create a reflector with oracle baked in.

            Uses Type path: model writes Python constructor, we eval it.
            Provider constructed once (not per-call) since reflect() is high-volume.
            """
            # Resolve learning provider once at creation time
            learning_provider = None
            if oracle is not None:
                try:
                    from compass.llm.ladder_policy import get_learning_model_spec
                    from compass.llm.providers import get_provider_by_id
                    learning_provider = get_provider_by_id(get_learning_model_spec())
                except Exception:
                    pass  # falls back to oracle default

            def reflect(prompt: str) -> Learning:
                """LLM reflects on action and returns structured Learning."""
                if oracle is None:
                    return Learning(type=LearningType.CORRECTION, data={"summary": "(no oracle)"})
                try:
                    from compass.llm.providers import ThinkLevel
                    response = oracle.ask(prompt, LearningResponse, max_tokens=500, task="actor:learning", think_level=ThinkLevel.LOW, provider=learning_provider)
                    # Convert to internal Learning type
                    return Learning(
                        type=response.learning_type,
                        data={
                            "summary": response.summary,
                            "key_facts": response.key_facts or [],
                        }
                    )
                except Exception:
                    return Learning(type=LearningType.CORRECTION, data={"summary": "(reflection failed)"})

            return reflect

        reflect = create_reflector(oracle)

        for learning in extract_learnings(action, success, result, reflect):
            memory.add_learning(learning)

        # Track action for loop detection (circuit breaker)
        # Build typed ActionTrace instead of raw dict
        action_trace = ActionTrace(
            action_type=action_name,
            target=target,
            success=success,
            result=result or "",
            params=asdict(action) if hasattr(action, '__dataclass_fields__') else dict(action),  # Store original params for matching
            reasoning=action_reasoning,
        )
        action_history.append(action_trace)

        if success:
            actions_succeeded += 1
            ctx.ui.show_result(True, result or "(no output)")

            # Special handling for read_file - track what was read
            if isinstance(action, ReadFileAction):
                reads_in_batch.append(action.path or "")
                read_path = action.path or ""
                if read_path:
                    abs_path = os.path.abspath(os.path.join(project_path, read_path))
                    start_line, end_line, total_lines = _parse_read_result_lines(result, action)

                    # Update tracking dicts
                    files_read = _update_files_read(files_read, abs_path, start_line, end_line)
                    files_read_content = _update_files_read_content(
                        files_read_content, read_path, start_line, end_line, result
                    )

                    results.append(_format_read_result(target, start_line, end_line, total_lines))
                else:
                    results.append(f"PASS: read_file {target}: OK")

            # Track write_file - model knows content, can overwrite later
            elif isinstance(action, WriteFileAction):
                write_path = action.path or ""
                if write_path:
                    abs_path = os.path.abspath(os.path.join(project_path, write_path))
                    content = action.content or ""
                    line_count = content.count('\n') + 1 if content else 1
                    files_read = _update_files_read(files_read, abs_path, 1, line_count)
                # Write results are usually short confirmations, no truncation needed
                results.append(f"PASS: {action_name}: {target}\n    -> {result}")

            elif isinstance(action, (AskOracleAction, RunCommandAction)):
                # LLM responses and command output are authoritative - never truncate
                results.append(f"PASS: {action_name}: {target}\n    -> {result}")

            else:
                # Truncate long results (grep, search, exec, run_command) with registry
                truncated = _truncate_result(result, label=action_name)
                results.append(f"PASS: {action_name}: {target}\n    -> {truncated}")

            # Add structured trace
            traces.append(trace_from_action(action_name, target, True, result or "", action))
        else:
            ctx.ui.show_result(False, result)
            # Track full error content for ERRORS section (no truncation)
            action_target = f"{action_name} {target}"
            errors_content.append((action_target, result or ""))
            # Short marker in results - full error goes to ERRORS section
            results.append(f"FAIL: {action_name} {target}: (see ERRORS section)")
            # Full error in trace (for registry and debugging)
            traces.append(trace_from_action(action_name, target, False, result or "", action))
            last_error = f"{action_name} on '{target}': {result}"
            # Stop on first failure
            return ActionBatchResult(
                success=False,
                results=results,
                last_error=last_error,
                last_action=action,
                files_read=files_read,
                files_read_content=files_read_content,
                traces=traces,
                errors_content=errors_content,
                file_snapshots=file_snapshots,
            )

    # Some actions may have been skipped (validation failure).
    # Success if at least one action ran. Fail only if nothing worked.
    return ActionBatchResult(
        success=actions_succeeded > 0,
        results=results,
        last_error=last_error,
        last_action=last_action,
        files_read=files_read,
        files_read_content=files_read_content,
        traces=traces,
        errors_content=errors_content,
        file_snapshots=file_snapshots,
    )


def build_actor_context(
    memory: CodeMemory,
    state: LoopState,
    rag_context: Optional[str],
    execution_trace: Optional["ExecutionTrace"] = None,
) -> str:
    """
    Build context for Actor. Pure function.

    Composes context from memory + accumulated results.
    If execution_trace provided, truncated content can be expanded.
    """

    parts = []

    # Directive from outer NFA (Critic/REPLAN) - show FIRST so Actor sees it
    if state.directive:
        parts.append(
            f"--- DIRECTIVE ---\n"
            f"{state.directive}"
        )
    project_path = memory.project_path or '.'
    branch = _git_branch(project_path)
    parts.append(f"--- CURRENT GIT BRANCH ---\n{branch}")

    parts.append(memory.get_actor_context())

    # Progress feedback from assessor - show FIRST so Actor sees it
    if state.progress_signal and state.progress_feedback:
        from compass.agents.neo.types import ProgressSignal
        signal_framing = {
            ProgressSignal.STALLED.value: "You are gathering information but not making progress. Time to ACT:",
            ProgressSignal.OSCILLATING.value: "You are going in circles (A→B→A→B). Break the pattern:",
            ProgressSignal.STUCK.value: "The same approach keeps failing. Try something DIFFERENT:",
        }
        framing = signal_framing.get(state.progress_signal, "Progress issue detected:")
        parts.append(
            f"--- PROGRESS FEEDBACK [{state.progress_signal.upper()}] ---\n"
            f"{framing}\n"
            f"{state.progress_feedback}"
        )

    # Diagonal hesitation: surface the rewrite question so Actor sees it
    if state.hesitation:
        parts.append(
            f"--- TRANSITION REFLECTION ---\n"
            f"{state.hesitation}\n"
            f"Consider: is this the right adaptation, or should you try a fundamentally different approach?"
        )

    if state.action_results:
        # this is the request actios - in the loop
        results_text = "\n\n".join(state.action_results[-20:])
        parts.append(f"--- PREVIOUS ACTION RESULTS ---\n{results_text}")

    # Add errors section
    errors_section = _build_errors_section(list(state.errors_content))
    if errors_section:
        parts.append(errors_section)

    # Add FILES READ section - refresh content from disk to show current state
    if state.files_read and memory.project_path:
        fresh_content = _refresh_files_content(dict(state.files_read), memory.project_path)
        files_section = _build_files_read_section(fresh_content)
        if files_section:
            parts.append(files_section)

    # Add RAG context
    if rag_context:
        parts.append(f"--- RELEVANT CODE CONTEXT ---\n{rag_context}")

    return "\n\n".join(parts)


# --- Trinity artifact entry point ---

def run(step, resolved_inputs: dict, workspace) -> "Result":
    """Trinity step dispatch contract: run(step, resolved_inputs, workspace).

    Bridges Trinity's plan execution to Neo's Actor/Critic loop.
    Neo gets an Oracle, a CodeMemory, and the request -- then does its thing.
    """
    from compass.generators._types import Ok, Err
    from compass.generators.trinity._types import Fact
    from compass.llm.oracle import Oracle as OracleClass
    from compass.agents.neo.memory import CodeMemory as CodeMemoryClass

    request = resolved_inputs.get("request", "") or resolved_inputs.get("prompt", "") or ""
    if not request:
        request = getattr(step, "description", "") or getattr(step, "artifact_ref", "") or ""
    if not request:
        return Err("Neo execute_request requires a 'request' or 'prompt' input")

    oracle = OracleClass()
    memory = CodeMemoryClass(session_id="trinity", project_path=str(workspace))

    rag_context = resolved_inputs.get("rag_context", None)
    directive = resolved_inputs.get("directive", None)

    result = execute_request(
        oracle=oracle,
        request=request,
        memory=memory,
        rag_context=rag_context,
        directive=directive,
    )

    step_id = getattr(step, "step_id", "neo_exec")
    fact_name = getattr(step, "expected_fact", "neo_result")

    if result.status == ExecutionStatus.SUCCESS:
        summary = "\n".join(result.action_results[-5:]) if result.action_results else "completed"
        expected_type = getattr(step, "expected_type", "any")

        if expected_type == "str":
            return Ok(Fact(
                step_id=step_id,
                name=fact_name,
                value=summary,
                fact_type="text",
                raw_value=summary,
            ))
        else:
            full_result = {
                "summary": summary,
                "files_modified": list(result.file_snapshots.keys()),
                "files_read": list(result.files_read.keys()),
            }
            return Ok(Fact(
                step_id=step_id,
                name=fact_name,
                value=json.dumps(full_result),
                fact_type="json",
                raw_value=full_result,
            ))
    else:
        error = result.last_error or result.feedback or "execution failed"
        return Err(f"Neo: {error}")


# --- Execution Entry Point ---

def execute_request(
    oracle: Oracle,
    request: str,
    memory: CodeMemory,
    rag_context: str = None,
    stream_router: Optional["StreamRouter"] = None,
    directive: Optional[str] = None,
    prior_results: Optional[List[str]] = None,  # Seeded from outer context
    codebase_index: Optional[Any] = None,
) -> ExecutionResult:
    """
    Execute a user request using the Actor directly.

    FP implementation: immutable LoopState threaded through iterations.
    Each step returns either a new state (continue) or a result (done).

    Args:
        oracle: LLM interface
        request: User's request (clean prose)
        memory: Session memory
        rag_context: Optional RAG context
        stream_router: Optional router for NFA visualization events
        directive: Optional feedback from outer NFA (Critic/REPLAN) - ephemeral

    Returns:
        ExecutionResult with status "success", "replan", or "done"
    """
    MAX_ITERATIONS = 20
    MAX_CONSECUTIVE_FAILURES = 3

    # Create execution config from user overrides
    config = ExecutionConfig.from_overrides()
    if config.think_level:
        debug(f"ExecutionConfig: think={config.think_level}")

    # Create execution trace ONCE for the whole request (persists registry across iterations)
    from compass.agents.neo.trace import ExecutionTrace
    execution_trace = ExecutionTrace()

    # Create execution context with UI adapter
    ctx = ExecutionContext(
        oracle=oracle,
        memory=memory,
        ui=ImmediateUIAdapter(),
        stream_router=stream_router,
        execution_trace=execution_trace,
        codebase_index=codebase_index,
    )

    # Initial immutable state - seed with prior results from outer context
    state = LoopState(
        directive=directive,
        action_results=tuple(prior_results) if prior_results else (),
    )

    def is_syntactic_error(error: str) -> bool:
        """Detect format/syntax errors vs semantic errors.

        Syntactic = the model produced malformed output (JSON, Python syntax).
        Semantic = the model's output was well-formed but wrong (file not found, logic error).
        """
        error_lower = error.lower()
        return any(kw in error_lower for kw in [
            "json", "malformed", "invalid syntax", "parse error",
            "unexpected token", "unterminated string", "unexpected eof",
            "indentation error", "invalid escape", "expecting value",
        ])

    def handle_no_response(s: LoopState) -> Union[LoopState, ExecutionResult]:
        """Handle Actor returning no response."""
        error = "Could not generate actions"
        ctx.ui.message(f"  [!] {error}")

        new_state = s.fail(error)
        if new_state.consecutive_failures >= config.max_consecutive_failures:
            return to_result(new_state, ExecutionStatus.DONE, trace=execution_trace)
        return new_state

    def handle_circuit_breaker(s: LoopState) -> Union[LoopState, ExecutionResult]:
        """Handle circuit breaker halt."""
        return to_result(s, ExecutionStatus.DONE, "Circuit breaker halted execution due to repeated actions.", trace=execution_trace)

    def handle_failure(s: LoopState, error: str, actions: List[Dict]) -> Union[LoopState, ExecutionResult]:
        """Handle action execution failure."""
        is_syntactic = is_syntactic_error(error)
        new_state = s.fail_syntactic(error) if is_syntactic else s.fail(error)

        # Always reduce creativity on failure (learning signal)
        pre = new_state
        new_state = new_state.adjust_creativity(-1)
        new_state = with_hesitation(pre, new_state, "failure recovery")

        if new_state.consecutive_failures >= config.max_consecutive_failures:
            # Exit to EVALUATE state for Critic decision
            if not actions:
                return to_result(
                    new_state,
                    ExecutionStatus.FAILED,
                    feedback=error,
                    last_action=None,
                    last_error="No actions generated",
                    trace=execution_trace,
                )
            return to_result(
                new_state,
                ExecutionStatus.EVALUATE,
                feedback=error,
                last_action=actions[-1],
                last_error=error,
                trace=execution_trace,
            )

        return new_state

    def execute_iteration(s: LoopState) -> Union[LoopState, ExecutionResult]:
        """
        Execute one iteration. Pure function.

        Returns:
            LoopState for continue, ExecutionResult for done
        """
        # Build context
        context = build_actor_context(memory, s, rag_context, execution_trace)

        # Get Actor response
        ctx.ui.start_spinner("Thinking")
        response = call_actor(
            oracle, request, context, memory.project_path, memory,
            iteration=s.creativity_iteration,
            think_level=config.think_level,
        )
        ctx.ui.stop_spinner()

        # No response
        if not response:
            return handle_no_response(s)

        # Debug: show raw response
        debug(f"Actor raw response: {response}")

        # Parse response
        actor_output = parse_actor_response(response)
        status, actions, reasoning = actor_output.status, actor_output.actions, actor_output.reasoning

        if reasoning and (not actions or os.getenv("DEBUG") or get_show_thinking()):
            ctx.ui.message(f"  {reasoning}")

        # No actions = model is done or confused
        if not actions:
            if status == ActorStatus.COMPLETE or status == ActorStatus.DONE:
                return to_result(s, ExecutionStatus.SUCCESS, trace=execution_trace)
            # Confused (no actions but not COMPLETE) - let Critic evaluate
            return to_result(s, ExecutionStatus.EVALUATE, trace=execution_trace)

        # Execute actions
        batch_result = _execute_actions(
            actions, memory, s.exec_globals,
            dict(s.files_read), dict(s.files_read_content), step_num=1,
            step=request, actor_context=context,
            action_history=list(s.action_history),
            step_results=list(s.action_results),
            oracle=oracle,
            ctx=ctx,
            execution_trace=execution_trace,  # Persist registry across iterations
        )

        # Update state with results
        new_state = s.with_results(
            batch_result.results,
            batch_result.files_read,
            batch_result.files_read_content,
            batch_result.errors_content,
            batch_result.traces,
            batch_result.file_snapshots,
        )

        # Circuit breaker
        if batch_result.circuit_breaker_halted:
            return handle_circuit_breaker(new_state)

        # Failure
        if not batch_result.success:
            error = batch_result.results[-1] if batch_result.results else "Action failed"
            return handle_failure(new_state, error, actions)

        # Success - check completion
        is_complete = status == ActorStatus.COMPLETE or status == ActorStatus.DONE
        if is_complete:
            final_state = new_state.reset_failures()
            return to_result(final_state, ExecutionStatus.SUCCESS, trace=execution_trace)

        # --- Progress assessment: are we making trajectory progress? ---
        from compass.agents.neo.actor import create_progress_assessor
        from compass.agents.neo.types import ProgressSignal

        assess_progress = create_progress_assessor(oracle)
        assessment = assess_progress(
            request,
            list(new_state.action_history),
            new_state.iteration,
        )

        # Get signal and suggestion if progress is poor
        signal = assessment.signal.value if assessment.signal != ProgressSignal.PROGRESSING else None
        suggestion = assessment.suggestion if signal else None

        next_state = (
            # Stuck: bump creativity significantly + feedback (same model)
            with_hesitation(new_state, new_state.adjust_creativity(+2), "stuck recovery")
                .with_progress_feedback(signal, suggestion)
            if assessment.signal == ProgressSignal.STUCK else

            # Oscillating: bump creativity significantly + pass feedback
            with_hesitation(new_state, new_state.adjust_creativity(+2), "oscillation break")
                .with_progress_feedback(signal, suggestion)
            if assessment.signal == ProgressSignal.OSCILLATING else

            # Stalled: bump creativity slightly + pass feedback
            with_hesitation(new_state, new_state.adjust_creativity(+1), "stall nudge")
                .with_progress_feedback(signal, suggestion)
            if assessment.signal == ProgressSignal.STALLED else

            # Progressing normally - clear any previous feedback, directive, and hesitation
            new_state.with_progress_feedback(None, None).with_directive(None)
        )

        if assessment.signal != ProgressSignal.PROGRESSING:
            debug(f"Progress: {assessment.signal.value} - {assessment.reasoning}")
            if suggestion:
                debug(f"Suggestion: {suggestion}")

        return next_state.reset_failures().next_iteration()

    # --- Main loop: thread state through iterations ---
    # Wrap with compaction to manage context size
    from compass.core.compaction import with_compaction
    step_fn = with_compaction(execute_iteration, oracle=oracle)
    return run_loop(state, step_fn, MAX_ITERATIONS)
