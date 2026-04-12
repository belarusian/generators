"""
program action - singledispatch handlers for ProgramAction.

Calls the Programmer NFA directly for complex code generation.
For multi-file features requiring architectural thinking.

Includes transaction model with snapshot/revert for safe file modifications.
"""

import json
import os
import concurrent.futures
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING

from compass.core.content import preview_head_tail

from compass.agents.neo.types import (
    ProgramAction,
    ActionTarget,
    ExecutionContext,
    Reflector,
)
from compass.agents.programmer.types import ParentCriticAction, ParentCriticOutput
from compass.agents.neo.dispatch import display, validate, execute, extract_learnings, action_key, hint, display_name, content_field
from compass.agents.neo.memory import Learning, CodeMemory
from compass.core.reasoning import debug
from compass.core.race import deep_thinking_enabled, DeepThinkingConfig
from compass.cli import ui

if TYPE_CHECKING:
    from compass.llm.oracle import Oracle
    from compass.agents.programmer.context import ProgrammerResult
    from compass.core.stream_router import StreamRouter, InstanceIdAllocator


# --- Helper functions ---

def _get_chunk_field(chunk, field: str, default=""):
    """Get field from chunk - handles both Chunk dataclass and dict."""
    if hasattr(chunk, field):
        val = getattr(chunk, field)
        # Handle enum values
        return val.value if hasattr(val, 'value') else (val or default)
    return chunk.get(field, default) if isinstance(chunk, dict) else default


def _create_chunk_applier(project_path: str) -> Callable[[List], Tuple[bool, str]]:
    """
    Create a callback for applying chunks to the filesystem.

    This is passed to the Programmer NFA so DELIVER can apply chunks
    and CRITIC_EVALUATE can see any errors.
    """
    def apply_chunks(chunks: List) -> Tuple[bool, str]:
        applied = []
        failed = []
        # Track files we've written - allows replace to work on files we created
        files_written = {}

        for chunk in chunks:
            chunk_id = _get_chunk_field(chunk, "id", "unknown")
            target = _get_chunk_field(chunk, "target", "")
            operation = _get_chunk_field(chunk, "operation", "create")
            content = _get_chunk_field(chunk, "content", "")

            if not target or not content:
                failed.append(f"[{chunk_id}] missing target or content")
                continue

            # For replace/append/insert on existing files, mark as known
            # This allows write_file to overwrite without blocking
            abs_path = os.path.join(project_path, target)
            if os.path.exists(abs_path) and operation in ("replace", "append", "insert"):
                files_written[target] = True

            # Map chunk operation to action type
            from compass.agents.neo.types import WriteFileAction
            if operation in ("create", "replace"):
                file_action = WriteFileAction(path=target, content=content)
            elif operation == "append":
                # Append: read existing + add new content + write back
                existing_content = ""
                if os.path.exists(abs_path):
                    try:
                        with open(abs_path, 'r') as f:
                            existing_content = f.read()
                    except Exception as e:
                        failed.append(f"[{chunk_id}] read failed for append: {e}")
                        continue
                file_action = WriteFileAction(
                    path=target,
                    content=existing_content + "\n" + content if existing_content else content,
                )
            elif operation == "insert":
                # Insert after a code marker
                insert_after = _get_chunk_field(chunk, "insert_after", "")
                if not insert_after:
                    # No marker - fallback to append
                    existing_content = ""
                    if os.path.exists(abs_path):
                        try:
                            with open(abs_path, 'r') as f:
                                existing_content = f.read()
                        except Exception as e:
                            failed.append(f"[{chunk_id}] read failed for insert: {e}")
                            continue
                    file_action = WriteFileAction(
                        path=target,
                        content=existing_content + "\n" + content if existing_content else content,
                    )
                else:
                    try:
                        with open(abs_path, 'r') as f:
                            file_content = f.read()
                    except FileNotFoundError:
                        failed.append(f"[{chunk_id}] insert target file not found: {target}")
                        continue
                    except Exception as e:
                        failed.append(f"[{chunk_id}] read failed: {e}")
                        continue

                    if insert_after not in file_content:
                        failed.append(f"[{chunk_id}] insert_after marker not found in {target}")
                        continue
                    count = file_content.count(insert_after)
                    if count > 1:
                        failed.append(f"[{chunk_id}] insert_after marker appears {count} times, must be unique")
                        continue

                    idx = file_content.find(insert_after) + len(insert_after)
                    new_content = file_content[:idx] + "\n" + content + file_content[idx:]

                    # Check for duplication before writing
                    from compass.agents.neo.actions.write_file import _detect_duplication
                    dup_warning = _detect_duplication(new_content)
                    if dup_warning:
                        failed.append(f"[{chunk_id}] insert blocked - {dup_warning}")
                        continue

                    try:
                        with open(abs_path, 'w') as f:
                            f.write(new_content)
                        applied.append(f"[{chunk_id}] insert -> {target}")
                        continue
                    except Exception as e:
                        failed.append(f"[{chunk_id}] write failed: {e}")
                        continue
            else:
                failed.append(f"[{chunk_id}] unknown operation: {operation}")
                continue

            # Execute the file action - pass files_written so replace works
            # Import here to avoid circular import at module load time
            from compass.agents.neo.rules import execute_action
            exec_result = execute_action(file_action, project_path, files_read=files_written)
            if exec_result is None:
                failed.append(f"[{chunk_id}] unknown action type")
                continue
            success, msg = exec_result
            if success:
                applied.append(f"[{chunk_id}] {operation} -> {target}")
                # Track this file so subsequent replace operations work
                files_written[target] = True
            else:
                failed.append(f"[{chunk_id}] {operation} -> {target}: {msg}")

        # Build result message
        output_parts = []
        if applied:
            output_parts.append(f"Applied {len(applied)} chunks:")
            output_parts.extend(f"  {a}" for a in applied)
        if failed:
            output_parts.append(f"\nFailed {len(failed)} chunks:")
            output_parts.extend(f"  {f}" for f in failed)

        if not applied and failed:
            return False, "\n".join(output_parts)
        return True, "\n".join(output_parts)

    return apply_chunks


def _snapshot_program_files(chunks: List, project_path: str) -> Dict[str, Optional[str]]:
    """Snapshot files that will be modified by chunks.

    Returns dict mapping file path to content (or None if file didn't exist).
    None values indicate files to DELETE on revert (newly created files).
    """
    snapshots = {}
    for chunk in chunks:
        target = _get_chunk_field(chunk, "target", "")
        if not target:
            continue
        full_path = Path(project_path) / target
        if full_path.exists() and full_path.is_file():
            try:
                snapshots[str(full_path)] = full_path.read_text()
            except Exception:
                snapshots[str(full_path)] = None  # Can't read - mark for delete
        else:
            # File doesn't exist - mark for deletion on revert
            snapshots[str(full_path)] = None
    return snapshots


def _revert_program_files(snapshots: Dict[str, Optional[str]]) -> List[str]:
    """Revert files to their snapshot state.

    Files with None value are deleted (they were newly created).
    Files with content are restored to their original state.
    """
    reverted = []
    for file_path, content in snapshots.items():
        try:
            if content is None:
                # File was created during operation - delete it
                path = Path(file_path)
                if path.exists():
                    path.unlink()
                    reverted.append(f"{file_path} (deleted)")
            else:
                # File existed before - restore content
                Path(file_path).write_text(content)
                reverted.append(file_path)
        except Exception:
            pass
    return reverted


def _parent_critic_review_program(
    oracle: "Oracle",
    problem: str,
    chunks: List[Dict],
    apply_result: str,
    memory: CodeMemory,
) -> ParentCriticOutput:
    """
    Parent Critic reviews Programmer's applied work.

    Returns ParentCriticOutput with action: done | revert | replan
    """
    def format_chunk(c):
        content = _get_chunk_field(c, 'content', '(no content)')
        chunk_id = _get_chunk_field(c, 'id', 'unknown')
        operation = _get_chunk_field(c, 'operation', '?')
        target = _get_chunk_field(c, 'target', '?')
        # No truncation - Critic sees everything Programmer wrote.
        # Same model, same context. If Programmer generated it, Critic can review it.
        return f"""[{chunk_id}] {operation} -> {target}
```
{content}
```"""

    chunks_detail = "\n\n".join(format_chunk(c) for c in chunks)

    prompt = f"""You are the Parent Critic reviewing Programmer's work.

PROBLEM: {problem}

CHUNKS APPLIED:
{chunks_detail}

APPLICATION RESULT:
{apply_result}

Decide:
- "done": Work looks correct, approve and continue
- "revert": Files are corrupted (duplicate code, broken syntax, wrong changes).
  Revert and retry with feedback to Programmer.
- "replan": Fundamental issue - need different approach from Planner.
NOTE: "create" and "replace" operations are both valid. "replace" means overwrite
the entire file - this is intentional when we want to rewrite a file completely.
Only revert if the code itself is wrong, not because of the operation type.

If reverting, provide feedback explaining what went wrong so Programmer can fix it."""

    try:
        # Show prompt for debugging (with schema) via on_prompt callback
        from compass.core.debug import show_prompt
        from compass.cli import ui
        on_prompt = lambda p: show_prompt("critic", "PARENT CRITIC PROMPT", p, ui.Colors.magenta)

        # Critic uses a dedicated model -- supervisory role needs discipline
        from compass.core.compose import with_fallback, with_logging
        from compass.llm.ladder_policy import get_critic_model_spec
        from compass.llm.providers import get_provider_by_id
        try:
            critic_provider = get_provider_by_id(get_critic_model_spec())
        except Exception:
            critic_provider = None

        fallback = ParentCriticOutput(action=ParentCriticAction.DONE, explanation="Critic unavailable")
        ask = with_fallback(with_logging(oracle.ask, "parent-critic"), fallback)
        return ask(prompt, ParentCriticOutput, task="parent-critic", on_prompt=on_prompt, provider=critic_provider)
    except Exception as e:
        debug(f"Parent Critic failed: {e}")
        return ParentCriticOutput(action=ParentCriticAction.DONE, explanation=f"Critic error: {e}")


# --- Deep Thinking for Programmer NFA ---

def _get_deep_providers(oracle: "Oracle") -> List:
    """Get providers for deep thinking (parallel exploration).

    The deep ladder was removed when routing moved to ladder_policy.py.
    Returns empty -- deep thinking branches are effectively disabled until
    the branching refactor (Tier B) is absorbed.
    """
    return []


def generate_programmer_results(
    oracle: "Oracle",
    problem: str,
    constraints: List[str],
    fetch_pattern: Callable,
    get_file_structure: Callable,
    get_coding_standards: Callable,
    parent_feedback: Optional[str] = None,
    session_dir: Optional[Path] = None,
    instance_allocator: Optional["InstanceIdAllocator"] = None,
    instance_ids: Optional[Dict[str, int]] = None,
    instance_routers: Optional[Dict[int, "StreamRouter"]] = None,
) -> Generator["ProgrammerResult", None, None]:
    """
    Generate Programmer results with deep thinking as optional enhancement.

    Architecture:
    - Default path ALWAYS runs (using configured oracle/coding ladder)
    - Deep thinking runs IN PARALLEL when enabled (using deep ladder)
    - Results yielded as they complete - consumer picks best

    Design principle: Deep thinking is ADDITIVE, not a replacement.
    - When disabled: direct call, no threading overhead
    - When enabled: default + deep branches compete in parallel

    Deep thinking uses the DEEP ladder (strongest models):
        OLLAMA_LADDER_DEEP="gpt-oss:120b@big,qwen3-coder:30b@big,anthropic:sonnet,anthropic:opus"

    Configuration:
        COMPASS_DEEP_THINKING=1: Enable deep thinking
        COMPASS_DEEP_THINKING_BRANCHES=N: Number of deep branches (default: 2)
        COMPASS_STREAM_LOG=1: Enable JSONL stream logging per instance

    Args:
        oracle: Oracle for default path (uses coding ladder)
        problem: Problem statement
        constraints: Solution constraints
        fetch_pattern, get_file_structure, get_coding_standards: Callbacks
        parent_feedback: Optional feedback from previous retry
        session_dir: Optional session directory for stream logs
        instance_allocator: Session-scoped allocator for instance IDs

    Yields:
        ProgrammerResults as they complete (fastest first)
    """
    from compass.agents.programmer import call_programmer

    # Check if stream logging is enabled
    from compass.core.stream_subscribers import stream_logging_enabled

    # Create allocator if not provided (for backwards compat)
    if instance_allocator is None:
        from compass.core.stream_router import InstanceIdAllocator
        instance_allocator = InstanceIdAllocator()

    # Fast path: no deep thinking - direct call, no threading overhead
    if not deep_thinking_enabled():
        # Create stream router for single instance if logging enabled
        stream_router = None
        if stream_logging_enabled():
            from compass.core.stream_router import StreamRouter
            from compass.core.stream_subscribers import JSONLSubscriber, create_instance_jsonl_path
            instance_id = instance_allocator.allocate()
            stream_router = StreamRouter(instance_id=instance_id)
            jsonl_path = create_instance_jsonl_path(instance_id, session_dir)
            stream_router.subscribe(JSONLSubscriber(jsonl_path))
            debug(f"Stream logging to: {jsonl_path}")

        result = call_programmer(
            oracle=oracle,
            problem=problem,
            constraints=constraints,
            fetch_pattern=fetch_pattern,
            get_file_structure=get_file_structure,
            get_coding_standards=get_coding_standards,
            apply_chunks=None,
            parent_feedback=parent_feedback,
            stream_router=stream_router,
        )
        yield result
        return

    # Deep thinking enabled: default + deep branches in parallel
    import threading
    from compass.llm.oracle import Oracle as OracleClass

    config = DeepThinkingConfig.from_env()
    stop_event = threading.Event()

    # Use passed-in dicts or create new ones (for backwards compat)
    # These persist across retries when passed from execute()
    if instance_ids is None:
        instance_ids = {}
    if instance_routers is None:
        instance_routers = {}
    if stream_logging_enabled():
        from compass.core.stream_router import StreamRouter
        from compass.core.stream_subscribers import JSONLSubscriber, create_instance_jsonl_path

    def run_branch(branch_oracle: "Oracle", branch_name: str) -> Tuple["ProgrammerResult", str]:
        """Run a single branch, return (result, branch_name)."""
        if os.getenv("DEBUG"):
            print(f"  {ui.Colors.dim(f'[{branch_name}] Starting...')}")

        # Get or create stream router for this instance
        stream_router = None
        if stream_logging_enabled():
            # Allocate instance ID for this branch (thread-safe)
            if branch_name not in instance_ids:
                instance_id = instance_allocator.allocate()
                instance_ids[branch_name] = instance_id
                router = StreamRouter(instance_id=instance_id)
                jsonl_path = create_instance_jsonl_path(instance_id, session_dir)
                router.subscribe(JSONLSubscriber(jsonl_path))
                instance_routers[instance_id] = router
                debug(f"[{branch_name}] instance_id={instance_id}, logging to: {jsonl_path}")
            stream_router = instance_routers.get(instance_ids.get(branch_name))

        result = call_programmer(
            oracle=branch_oracle,
            problem=problem,
            constraints=constraints,
            fetch_pattern=fetch_pattern,
            get_file_structure=get_file_structure,
            get_coding_standards=get_coding_standards,
            apply_chunks=None,
            parent_feedback=parent_feedback,
            is_cancelled=stop_event.is_set,
            stream_router=stream_router,
            show_prompts=(branch_name == "default"),
        )

        if os.getenv("DEBUG"):
            status = "cancelled" if stop_event.is_set() else ("success" if result.success else "failed")
            chunks = len(result.chunks) if result.chunks else 0
            print(f"  {ui.Colors.dim(f'[{branch_name}] {status}, {chunks} chunks')}")

        return result, branch_name

    # Build branches: default always, deep branches from DEEP ladder
    branches = [(oracle, "default")]

    deep_providers = _get_deep_providers(oracle)
    for i, provider in enumerate(deep_providers[:config.num_branches]):
        branch_name = f"deep:{provider.name}"
        branches.append((OracleClass(provider=provider), branch_name))

    if os.getenv("DEBUG"):
        branch_names = [name for _, name in branches]
        print(f"  {ui.Colors.dim(f'[Deep Thinking] Branches: {branch_names}')}")

    # Single branch after filtering = still no parallelism
    if len(branches) == 1:
        orc, name = branches[0]
        result, _ = run_branch(orc, name)
        yield result
        return

    # Parallel execution - no timeout, branches run until one sticks or all complete
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(branches))
    futures = {
        executor.submit(run_branch, orc, name): name
        for orc, name in branches
    }

    try:
        for future in concurrent.futures.as_completed(futures):
            try:
                result, branch_name = future.result()
                if os.getenv("DEBUG"):
                    print(f"  {ui.Colors.dim(f'[{branch_name}] completed, yielding...')}")
                yield result
            except Exception as e:
                debug(f"Branch failed: {e}")
    finally:
        stop_event.set()
        for f in futures:
            f.cancel()
        executor.shutdown(wait=True)


# --- Singledispatch handlers ---

@content_field.register(ProgramAction)
def _(action): return "problem"


@display.register(ProgramAction)
def _(action: ProgramAction) -> ActionTarget:
    """Get display info for program action."""
    problem = action.problem or "(no problem)"
    # Truncate problem for display
    problem_short = problem[:80] + "..." if len(problem) > 80 else problem
    constraints = action.constraints or []
    return ActionTarget(
        target=problem_short,
        display=f"program: {problem_short}",
        content=f"Problem: {problem}\nConstraints: {constraints}" if constraints else problem,
    )


@validate.register(ProgramAction)
def _(
    action: ProgramAction,
    project_path: str = ".",
    files_read: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate program action.

    Required fields:
    - problem: Abstract problem description (not file-specific)

    Optional fields:
    - constraints: List of explicit constraints on the solution

    Use program for complex code generation that needs architectural thinking:
    - Multi-file features with interdependent components
    - Problems requiring design phase before implementation
    - When you need validated output through Programmer+Scribe NFA

    For simpler changes, use edit_file or write_file directly.
    """
    problem = action.problem

    if not problem:
        return False, "Missing required field: problem"

    return True, None


@execute.register(ProgramAction)
def _(
    action: ProgramAction,
    project_path: str,
    ctx: ExecutionContext = None,
) -> Tuple[bool, str]:
    """
    Execute program action via the Programmer NFA directly.

    Calls call_programmer() -- the NFA that understands, designs,
    implements, and validates code through the Scribe review loop.
    Neo stays in the driver seat; no nested Trinity plan-execute-fix.

    Args:
        action: ProgramAction with problem statement
        project_path: Base project path
        ctx: ExecutionContext with memory and oracle

    Returns:
        (success, result_message)
    """
    from compass.llm.oracle import Oracle
    from compass.agents.programmer.tool import (
        call_programmer,
        create_pattern_fetcher,
        create_file_structure_getter,
        create_coding_standards_getter,
    )

    problem = action.problem or ""
    if not problem:
        return False, "program requires 'problem'"

    constraints = action.constraints or []

    workspace = Path(project_path)
    oracle = Oracle()

    # Duck-type workspace ref for pattern fetcher
    class _WsRef:
        def __init__(self, path):
            self.project_path = str(path)

    ws_ref = _WsRef(workspace)
    if ctx and ctx.memory:
        ws_ref = ctx.memory  # CodeMemory has .project_path

    fetch_pattern = create_pattern_fetcher(oracle, ws_ref)
    get_file_structure = create_file_structure_getter(ws_ref)
    get_coding_standards = create_coding_standards_getter()

    def _apply_chunks(chunks):
        applied, failed = [], []
        for chunk in chunks:
            target = getattr(chunk, "target", None) or (chunk.get("target") if isinstance(chunk, dict) else "")
            content = getattr(chunk, "content", None) or (chunk.get("content") if isinstance(chunk, dict) else "")
            if not target or not content:
                failed.append("missing target or content")
                continue
            out = workspace / target
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content)
            applied.append(str(target))
        msg = f"Applied {len(applied)} chunks"
        if failed:
            msg += f", {len(failed)} failed: {failed}"
        return len(failed) == 0, msg

    try:
        result = call_programmer(
            oracle=oracle,
            problem=problem,
            constraints=constraints,
            fetch_pattern=fetch_pattern,
            get_file_structure=get_file_structure,
            get_coding_standards=get_coding_standards,
            apply_chunks=_apply_chunks,
        )
    except Exception as exc:
        return False, f"Programmer failed: {type(exc).__name__}: {exc}"

    if not result.success:
        return False, f"Programmer failed: {result.error or 'unknown'}"

    files = [str(c.target) for c in (result.chunks or []) if hasattr(c, "target")]
    summary = f"Programmer completed in {result.iterations} iterations"
    if files:
        summary += f", wrote: {', '.join(files)}"
    return True, summary


@extract_learnings.register(ProgramAction)
def _(
    action: ProgramAction,
    success: bool,
    result: str,
    reflect: Reflector,
) -> List[Learning]:
    """Extract learnings. LLM reflects and chooses learning type - no heuristics."""
    from dataclasses import asdict

    action_data = asdict(action) if hasattr(action, '__dataclass_fields__') else action

    prompt = f"""Action: program
Input: {json.dumps(action_data)}
Success: {success}
Result:
{preview_head_tail(result, max_lines=23)}

What did we learn from this?"""

    return [reflect(prompt)]


@action_key.register(ProgramAction)
def _(action: ProgramAction) -> tuple:
    """Hashable key for ProgramAction comparison."""
    problem = action.problem or ""
    # Use first 60 chars of problem as identifier
    problem_short = problem[:60] if problem else ""
    return ("program", problem_short)


@hint.register(ProgramAction)
def _(action: ProgramAction) -> str:
    """Hint for Critic when program fails."""
    return "Invoke Programmer NFA. For complex multi-file solutions. Clarify problem statement."


@display_name.register(ProgramAction)
def _(action: ProgramAction) -> str:
    """Human-friendly name for UI."""
    return "Program"
