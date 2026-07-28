"""
Programmer States - State transition functions for the Programmer NFA.

Each state is a factory function that returns a transition function.
The transition function takes context and returns (new_state, new_context).
"""

import ast
import os
from typing import Callable, Dict, List, Optional, Tuple

from compass.cli import ui
from compass.core.debug import show_prompt

from compass.agents.programmer.context import (
    ProgrammerState,
    ProgrammerContext,
    ScribeView,
)
from compass.agents.programmer.trace import TransitionReason
from compass.agents.programmer.types import (
    ImplementResponse,
    ScribeReviewResponse, ScribeAction,
    ProgrammerCriticReviewResponse, CriticReviewAction,
    ProgrammerCriticEvaluateResponse, CriticEvaluateAction, RetryFromState,
    Chunk, ChunkOperation,
)
from compass.core.code_chunks import parse_code_chunks, CodeChunk
from compass.agents.programmer.prompts import (
    UNDERSTAND_PROMPT,
    DESIGN_PROMPT,
    IMPLEMENT_PROMPT,
    IMPLEMENT_FEEDBACK,
    SCRIBE_REVIEW_PROMPT,
    SCRIBE_CONTINUE_PROMPT,
    PROGRAMMER_AMEND_PROMPT,
    CRITIC_REVIEW_PROMPT,
    CRITIC_EVALUATE_PROMPT,
)
from compass.core.retry import retry_with_messages, AskResult
from compass.core.content import truncate_lines
# ask_with_expand removed - expansion disabled
from compass.llm.ask import append_retry_feedback


def _show_programmer_prompt(state_name: str, prompt: str, color_fn=None, show: bool = True) -> None:
    """Show Programmer NFA prompt when enabled via DEBUG_PROMPTS.

    Args:
        show: If False, suppresses output (for non-default parallel branches).
    """
    if not show:
        return
    show_prompt("programmer", f"PROGRAMMER {state_name}", prompt, color_fn or ui.Colors.green)


def _record_transition(
    ctx: ProgrammerContext,
    from_state: ProgrammerState,
    to_state: ProgrammerState,
    reason: TransitionReason,
    feedback: Optional[str] = None,
    error: Optional[str] = None,
    chunks_affected: Optional[List[str]] = None,
) -> None:
    """Record a state transition in the local trace.

    Note: Telemetry is now recorded centrally via NFARunner's on_transition callback,
    which includes timing information.
    """
    if ctx.trace:
        ctx.trace.add(from_state, to_state, reason, feedback, error, chunks_affected)


def create_understand_state() -> Callable[[ProgrammerContext], Tuple[ProgrammerState, ProgrammerContext]]:
    """
    Factory for UNDERSTAND state.

    Programmer (Actor) analyzes the problem and builds understanding.
    """
    def _understand(ctx: ProgrammerContext) -> Tuple[ProgrammerState, ProgrammerContext]:
        # Include parent feedback if this is a retry
        feedback_section = ""
        if ctx.parent_feedback:
            feedback_section = f"\nPREVIOUS ATTEMPT FEEDBACK (from parent Critic):\n{ctx.parent_feedback}\n"

        prompt = UNDERSTAND_PROMPT.format(
            problem=ctx.problem,
            constraints="\n".join(f"- {c}" for c in ctx.constraints) or "(none specified)",
            parent_feedback=feedback_section,
        )

        try:
            _show_programmer_prompt("UNDERSTAND", prompt, ui.Colors.green, ctx.show_prompts)
            raw = ctx.oracle.speak(prompt, task="programmer-understand")

            ctx.understanding = raw

            _record_transition(ctx, ProgrammerState.UNDERSTAND, ProgrammerState.DESIGN, TransitionReason.SUCCESS)
            return ProgrammerState.DESIGN, ctx

        except Exception as e:
            ctx.last_error = f"UNDERSTAND failed: {e}"
            _record_transition(ctx, ProgrammerState.UNDERSTAND, ProgrammerState.CRITIC_EVALUATE,
                             TransitionReason.ORACLE_ERROR, error=str(e))
            return ProgrammerState.CRITIC_EVALUATE, ctx

    return _understand


def create_design_state() -> Callable[[ProgrammerContext], Tuple[ProgrammerState, ProgrammerContext]]:
    """
    Factory for DESIGN state.

    Programmer (Actor) creates solution architecture.
    """
    def _design(ctx: ProgrammerContext) -> Tuple[ProgrammerState, ProgrammerContext]:
        # Include critic feedback if we're revising
        feedback_section = ""
        if ctx.critic_feedback:
            feedback_section = f"\nPREVIOUS ATTEMPT FEEDBACK:\n{ctx.critic_feedback}\n"

        prompt = DESIGN_PROMPT.format(
            problem=ctx.problem,
            understanding=ctx.understanding or "(no understanding)",
            constraints="\n".join(f"- {c}" for c in ctx.constraints) or "(none)",
            feedback=feedback_section,
        )

        try:
            # Free-form design: model writes naturally, no struct parsing
            raw = ctx.oracle.speak(prompt, task="programmer-design")

            # Extract architecture from header if present, else use first line
            lines = raw.strip().splitlines()
            arch_idx = next(
                (i for i, l in enumerate(lines)
                 if l.strip().upper().startswith("ARCHITECTURE SUMMARY:")),
                None,
            )
            ctx.design = (
                lines[arch_idx + 1].strip()
                if arch_idx is not None and arch_idx + 1 < len(lines)
                else lines[0].strip() if lines else "(no design)"
            )
            ctx.solution_doc = raw

            _record_transition(ctx, ProgrammerState.DESIGN, ProgrammerState.IMPLEMENT,
                             TransitionReason.SUCCESS)
            return ProgrammerState.IMPLEMENT, ctx

        except Exception as e:
            ctx.last_error = f"DESIGN failed: {e}"
            _record_transition(ctx, ProgrammerState.DESIGN, ProgrammerState.CRITIC_EVALUATE,
                             TransitionReason.ORACLE_ERROR, error=str(e))
            return ProgrammerState.CRITIC_EVALUATE, ctx

    return _design


def _validate_implement_response(response: ImplementResponse) -> Optional[str]:
    """
    Validate IMPLEMENT response before proceeding.

    Returns error string if invalid, None if valid.
    """
    chunks = response.chunks or []

    if not chunks:
        return "No chunks produced. Break the solution into at least one chunk."

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.id if hasattr(chunk, 'id') else f"chunk_{i}"

        # Required fields - Chunk dataclass has these as required
        if not chunk.id:
            return f"Chunk {i} missing 'id' field"
        if not chunk.content:
            return f"Chunk [{chunk_id}] missing 'content' field"
        if not chunk.target:
            return f"Chunk [{chunk_id}] missing 'target' field"
        if not chunk.operation:
            return f"Chunk [{chunk_id}] missing 'operation' field"

        # Valid operation - ChunkOperation enum handles this
        op_value = chunk.operation.value if hasattr(chunk.operation, 'value') else chunk.operation
        if op_value not in ("create", "replace", "append", "insert"):
            return f"Chunk [{chunk_id}] has invalid operation '{op_value}'. Use: create, replace, append, insert"

        # Insert requires insert_after
        if op_value == "insert" and not chunk.insert_after:
            return f"Chunk [{chunk_id}] uses 'insert' but missing 'insert_after' marker"

    return None  # Valid


def _code_chunk_to_chunk(code_chunk: CodeChunk) -> Chunk:
    """Convert CodeChunk (from parser) to Chunk (from types)."""
    # Map operation
    op_map = {
        "create": ChunkOperation.CREATE,
        "replace": ChunkOperation.REPLACE,
        "append": ChunkOperation.APPEND,
        "insert": ChunkOperation.INSERT,
    }
    operation = op_map.get(code_chunk.operation.value, ChunkOperation.CREATE)

    return Chunk(
        id=code_chunk.id,
        content=code_chunk.content,
        target=code_chunk.target,
        operation=operation,
        insert_after=code_chunk.insert_after,
        reasoning=code_chunk.reasoning,
    )


def _validate_implement_chunks(response) -> Optional[str]:
    """Validate raw code output for chunk markers. Returns error or None."""
    # Handle RawResponse from oracle.ask(response_type=None)
    raw_code = response.text if hasattr(response, 'text') else str(response)

    code_chunks = parse_code_chunks(raw_code)
    if not code_chunks:
        return "No code chunks found. Use # === chunk: target=\"file.py\", operation=create === markers."

    chunks = [_code_chunk_to_chunk(c) for c in code_chunks]
    return _validate_implement_response(ImplementResponse(chunks=chunks))


def create_implement_state() -> Callable[[ProgrammerContext], Tuple[ProgrammerState, ProgrammerContext]]:
    """
    Factory for IMPLEMENT state.

    Programmer (Actor) breaks solution into deliverable chunks.
    Model writes raw code with chunk markers - no serialization tax.

    Uses FP composition: retry_with_messages + converse_raw + append_retry_feedback.
    Full context preserved in message history (no arbitrary truncation).
    """
    def _implement(ctx: ProgrammerContext) -> Tuple[ProgrammerState, ProgrammerContext]:
        from compass.core.telemetry import record_task_attempt, record_task_attempt_failure

        # Get components from design if available
        components = []
        if ctx.design:
            components = ["main component"]

        # Include feedback from previous failures
        feedback_section = ""
        if ctx.critic_feedback:
            feedback_section = f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{ctx.critic_feedback}"
        if ctx.last_error and "Unresolved issues" in ctx.last_error:
            feedback_section += f"\n\nPREVIOUS FAILURE DETAILS:\n{ctx.last_error}"

        base_prompt = IMPLEMENT_PROMPT.format(
            solution=ctx.solution_doc or "(no solution)",
            components="\n".join(f"- {c}" for c in components) or "(none)",
        )
        if feedback_section:
            base_prompt = base_prompt + feedback_section

        _show_programmer_prompt("IMPLEMENT", base_prompt, ui.Colors.cyan, ctx.show_prompts)

        # Compose: converse_raw (single call) + retry_with_messages (loop)
        attempt_counter = [0]

        def ask_once(msgs):
            attempt_index = attempt_counter[0]
            attempt_counter[0] += 1
            record_task_attempt("programmer-implement", attempt_index)
            r = ctx.oracle.converse_raw(msgs, max_tokens=4000, task="programmer-implement")
            return AskResult(text=r.text, thinking=r.thinking, truncated=(r.done_reason == "length"))

        def append_feedback(msgs, response, error, thinking):
            return append_retry_feedback(
                msgs, response, error, thinking,
                feedback_template=IMPLEMENT_FEEDBACK,
            )

        def on_retry_failure(attempt_index: int, failure_type: str, _error_message: str) -> None:
            record_task_attempt_failure("programmer-implement", attempt_index, failure_type)

        result = retry_with_messages(
            ask_once=ask_once,
            initial_messages=[{"role": "user", "content": base_prompt}],
            parse=lambda text: text,  # Raw mode - no parsing
            validate=_validate_implement_chunks,
            append_feedback=append_feedback,
            max_retries=3,
            on_retry_failure=on_retry_failure,
        )

        if result.success:
            # Parse chunks from successful response
            raw_code = result.value
            code_chunks = parse_code_chunks(raw_code)
            ctx.chunks = [_code_chunk_to_chunk(c) for c in code_chunks]
            _record_transition(ctx, ProgrammerState.IMPLEMENT, ProgrammerState.SCRIBE_REVIEW,
                             TransitionReason.SUCCESS)
            return ProgrammerState.SCRIBE_REVIEW, ctx
        else:
            ctx.last_error = result.error
            _record_transition(ctx, ProgrammerState.IMPLEMENT, ProgrammerState.CRITIC_EVALUATE,
                             TransitionReason.PARSE_ERROR, error=result.error)
            return ProgrammerState.CRITIC_EVALUATE, ctx

    return _implement


# Unicode characters that look like ASCII but cause syntax errors
UNICODE_LOOKALIKES = {
    '\u2011': ('NON-BREAKING HYPHEN', '-'),
    '\u2010': ('HYPHEN', '-'),
    '\u2212': ('MINUS SIGN', '-'),
    '\u2013': ('EN DASH', '-'),
    '\u2014': ('EM DASH', '-'),
    '\u2018': ('LEFT SINGLE QUOTE', "'"),
    '\u2019': ('RIGHT SINGLE QUOTE', "'"),
    '\u201c': ('LEFT DOUBLE QUOTE', '"'),
    '\u201d': ('RIGHT DOUBLE QUOTE', '"'),
    '\u00a0': ('NO-BREAK SPACE', ' '),
    '\u200b': ('ZERO WIDTH SPACE', ''),
    '\u2026': ('HORIZONTAL ELLIPSIS', '...'),
}


def _detect_unicode_lookalikes(content: str) -> Optional[str]:
    """
    Detect Unicode characters that look like ASCII but will cause syntax errors.

    Returns error message if found, None if clean.
    """
    found = []
    for char, (name, replacement) in UNICODE_LOOKALIKES.items():
        if char in content:
            # Find position for helpful error
            pos = content.find(char)
            context = content[max(0, pos-10):pos+10]
            found.append(f"  - {name} (U+{ord(char):04X}) near '...{context}...' -> use '{replacement}' instead")

    if found:
        return "Unicode lookalike characters found (will cause syntax errors):\n" + "\n".join(found)
    return None


def _validate_chunks_syntax(chunks: List[Dict]) -> List[Dict]:
    """
    Validate that Python code chunks are syntactically valid.

    Returns a list of issues for chunks that fail to parse.
    This is the Scribe's first line of defense - no LLM call needed
    to catch basic syntax errors.

    Only validates files with .py extension.
    Detects Unicode lookalikes and provides feedback to model.
    """
    from compass.agents.programmer.types import ScribeIssue, IssueSeverity

    issues = []
    for chunk in chunks:
        # Handle both Chunk dataclass and dict (for backward compat)
        content = chunk.content if hasattr(chunk, 'content') else chunk.get("content", "")
        chunk_id = chunk.id if hasattr(chunk, 'id') else chunk.get("id", "unknown")
        target = chunk.target if hasattr(chunk, 'target') else chunk.get("target", "")

        # Only validate Python files (skip non-.py files if target is specified)
        # If no target, assume Python and validate
        if target and not target.endswith(".py"):
            continue

        if not content or not content.strip():
            continue

        # Check for Unicode lookalikes BEFORE parsing - give specific feedback
        unicode_error = _detect_unicode_lookalikes(content)
        if unicode_error:
            issues.append({
                "chunk_id": chunk_id,
                "severity": "error",
                "description": unicode_error,
                "suggestion": "Replace Unicode characters with their ASCII equivalents",
            })
            continue  # Skip ast.parse - we know it will fail

        try:
            ast.parse(content)
        except SyntaxError as e:
            issues.append({
                "chunk_id": chunk_id,
                "severity": "error",
                "description": f"Syntax error: {e.msg} at line {e.lineno}",
                "suggestion": f"Fix the syntax error near: {e.text.strip() if e.text else '(unknown)'}",
            })

    return issues


def create_scribe_review_state() -> Callable[[ProgrammerContext], Tuple[ProgrammerState, ProgrammerContext]]:
    """
    Factory for SCRIBE_REVIEW state.

    Scribe (Critic) evaluates solution against system constraints.
    First validates syntax with ast.parse(), then does LLM review.
    """
    def _scribe_review(ctx: ProgrammerContext) -> Tuple[ProgrammerState, ProgrammerContext]:
        # Build Scribe's constrained view
        file_structure = {}
        if ctx.get_file_structure:
            file_structure = ctx.get_file_structure()

        coding_standards = []
        if ctx.get_coding_standards:
            coding_standards = ctx.get_coding_standards()

        # Create ScribeView with callback
        ctx.scribe_view = ScribeView(
            solution_chunks=ctx.chunks,
            request_pattern=ctx.fetch_pattern or (lambda q: "(no pattern available)"),
            file_structure=file_structure,
            coding_standards=coding_standards,
        )

        # FIRST: Validate syntax before wasting an LLM call
        syntax_issues = _validate_chunks_syntax(ctx.chunks)
        if syntax_issues:
            # Immediate feedback - code doesn't even parse
            ctx.scribe_view.issues = syntax_issues
            ctx.scribe_view.feedback_for_programmer = (
                "Code failed syntax validation. The following chunks have Python syntax errors "
                "and will not compile. Please fix the syntax errors before proceeding."
            )
            _record_transition(ctx, ProgrammerState.SCRIBE_REVIEW, ProgrammerState.SCRIBE_FEEDBACK,
                             TransitionReason.VALIDATION_ERROR, feedback="Syntax validation failed")
            return ProgrammerState.SCRIBE_FEEDBACK, ctx

        prompt = SCRIBE_REVIEW_PROMPT.format(
            chunks=_format_chunks(ctx.chunks),
            file_structure=_format_file_structure(file_structure),
            standards="\n".join(f"- {s}" for s in coding_standards) or "(none specified)",
        )

        _show_programmer_prompt("SCRIBE_REVIEW", prompt, ui.Colors.magenta, ctx.show_prompts)

        try:
            # Direct oracle.ask - expansion removed
            result = ctx.oracle.ask(prompt, ScribeReviewResponse, task="scribe-review")
            return _handle_scribe_action(result, ctx, ProgrammerState.SCRIBE_REVIEW)

        except Exception as e:
            ctx.last_error = f"SCRIBE_REVIEW failed: {e}"
            _record_transition(ctx, ProgrammerState.SCRIBE_REVIEW, ProgrammerState.CRITIC_EVALUATE,
                             TransitionReason.ORACLE_ERROR, error=str(e))
            return ProgrammerState.CRITIC_EVALUATE, ctx

    return _scribe_review


def create_scribe_fetch_state() -> Callable[[ProgrammerContext], Tuple[ProgrammerState, ProgrammerContext]]:
    """
    Factory for SCRIBE_FETCH state.

    Scribe (Critic) requests code patterns from the system.
    Includes loop detection to prevent infinite fetch cycles.
    """
    def _scribe_fetch(ctx: ProgrammerContext) -> Tuple[ProgrammerState, ProgrammerContext]:
        query = (ctx.scribe_view.pattern_query if ctx.scribe_view else None) or ""

        # Fetch pattern - use cached if available (includes expanded content)
        pattern = "(no pattern available)"
        if ctx.scribe_view:
            if query in ctx.scribe_view.fetched_patterns:
                # Already fetched or expanded - use full content, no truncation
                pattern = ctx.scribe_view.fetched_patterns[query]
            elif ctx.scribe_view.request_pattern:
                # New fetch - truncate and register for expansion
                full_pattern = ctx.scribe_view.request_pattern(query)
                ctx.scribe_view.fetched_patterns[query] = full_pattern
                pattern = truncate_lines(full_pattern, max_lines=100, label="pat")

        prompt = SCRIBE_CONTINUE_PROMPT.format(
            chunks=_format_chunks(ctx.chunks),
            original_query=query,
            pattern=pattern,
        )

        try:
            # Direct oracle.ask - expansion removed
            result = ctx.oracle.ask(prompt, ScribeReviewResponse, task="scribe-continue")
            # check_loop=True for loop detection on repeated pattern fetches
            return _handle_scribe_action(result, ctx, ProgrammerState.SCRIBE_FETCH, check_loop=True)

        except Exception as e:
            ctx.last_error = f"SCRIBE_FETCH failed: {e}"
            _record_transition(ctx, ProgrammerState.SCRIBE_FETCH, ProgrammerState.CRITIC_EVALUATE,
                             TransitionReason.ORACLE_ERROR, error=str(e))
            return ProgrammerState.CRITIC_EVALUATE, ctx

    return _scribe_fetch


def create_scribe_feedback_state() -> Callable[[ProgrammerContext], Tuple[ProgrammerState, ProgrammerContext]]:
    """
    Factory for SCRIBE_FEEDBACK state.

    Feed Scribe's critique back to Programmer for amendments.
    Model writes raw code with chunk markers - no serialization tax.
    """
    def _scribe_feedback(ctx: ProgrammerContext) -> Tuple[ProgrammerState, ProgrammerContext]:
        # Check iteration limit
        if ctx.scribe_iterations >= ctx.max_scribe_iterations:
            # Max iterations - go to critic to decide if we can proceed
            issues = ctx.scribe_view.issues if ctx.scribe_view else []
            issue_details = _format_issues(issues) if issues else "(no details)"
            ctx.last_error = (
                f"SCRIBE_FEEDBACK max iterations ({ctx.max_scribe_iterations}) reached. "
                f"Unresolved issues:\n{issue_details}"
            )
            _record_transition(ctx, ProgrammerState.SCRIBE_FEEDBACK, ProgrammerState.CRITIC_EVALUATE,
                             TransitionReason.SCRIBE_MAX_ITERATIONS)
            return ProgrammerState.CRITIC_EVALUATE, ctx

        ctx.scribe_iterations += 1

        feedback = ctx.scribe_view.feedback_for_programmer if ctx.scribe_view else ""
        issues = ctx.scribe_view.issues if ctx.scribe_view else []

        prompt = PROGRAMMER_AMEND_PROMPT.format(
            original_solution=ctx.solution_doc or "(no solution)",
            chunks=_format_chunks(ctx.chunks),
            scribe_feedback=feedback,
            issues=_format_issues(issues),
        )

        try:
            # Raw output - returns RawResponse(text, thinking)
            response = ctx.oracle.ask(prompt, max_tokens=4000, task="programmer-amend")

            # Parse code chunks from raw text
            code_chunks = parse_code_chunks(response.text)

            if code_chunks:
                # Convert to Chunk objects
                amended_chunks = [_code_chunk_to_chunk(c) for c in code_chunks]

                # Merge amended chunks with existing - don't lose chunks LLM didn't mention
                amended_by_id = {c.id: c for c in amended_chunks}
                merged = []
                seen_ids = set()

                # Update existing chunks with amendments
                for chunk in ctx.chunks:
                    chunk_id = chunk.id if hasattr(chunk, 'id') else chunk.get("id")
                    if chunk_id in amended_by_id:
                        merged.append(amended_by_id[chunk_id])
                    else:
                        merged.append(chunk)
                    seen_ids.add(chunk_id)

                # Add any new chunks from amendments
                for chunk in amended_chunks:
                    if chunk.id not in seen_ids:
                        merged.append(chunk)

                ctx.chunks = merged

                # Back to Scribe for re-review
                amended_ids = [c.id for c in amended_chunks]
                _record_transition(ctx, ProgrammerState.SCRIBE_FEEDBACK, ProgrammerState.SCRIBE_REVIEW,
                                 TransitionReason.SUCCESS, chunks_affected=amended_ids)
                return ProgrammerState.SCRIBE_REVIEW, ctx
            else:
                # No chunks parsed - go back anyway
                _record_transition(ctx, ProgrammerState.SCRIBE_FEEDBACK, ProgrammerState.SCRIBE_REVIEW,
                                 TransitionReason.SUCCESS)
                return ProgrammerState.SCRIBE_REVIEW, ctx

        except Exception as e:
            ctx.last_error = f"SCRIBE_FEEDBACK failed: {e}"
            _record_transition(ctx, ProgrammerState.SCRIBE_FEEDBACK, ProgrammerState.CRITIC_EVALUATE,
                             TransitionReason.ORACLE_ERROR, error=str(e))
            return ProgrammerState.CRITIC_EVALUATE, ctx

    return _scribe_feedback


def create_critic_review_state() -> Callable[[ProgrammerContext], Tuple[ProgrammerState, ProgrammerContext]]:
    """
    Factory for CRITIC_REVIEW state.

    Holistic requirements validation - does solution solve the problem?
    Scribe validated system constraints. Critic validates REQUIREMENTS.
    """
    def _critic_review(ctx: ProgrammerContext) -> Tuple[ProgrammerState, ProgrammerContext]:
        # Check iteration limit
        if ctx.critic_review_iterations >= ctx.max_critic_review_iterations:
            next_state = ProgrammerState.DELIVER if ctx.apply_chunks else ProgrammerState.DONE
            _record_transition(ctx, ProgrammerState.CRITIC_REVIEW, next_state,
                             TransitionReason.CRITIC_MAX_ITERATIONS)
            return next_state, ctx

        prompt = CRITIC_REVIEW_PROMPT.format(
            problem=ctx.problem,
            constraints="\n".join(f"- {c}" for c in ctx.constraints) or "(none)",
            chunks=_format_chunks_full(ctx.chunks),
        )

        try:
            result = ctx.oracle.ask(prompt, ProgrammerCriticReviewResponse, task="critic-review")

            # Get action value - handle enum or string
            action = result.action.value if hasattr(result.action, 'value') else result.action

            if action == "approve":
                # Requirements met -- deliver chunks if callback exists
                next_state = ProgrammerState.DELIVER if ctx.apply_chunks else ProgrammerState.DONE
                _record_transition(ctx, ProgrammerState.CRITIC_REVIEW, next_state,
                                 TransitionReason.CRITIC_APPROVED)
                return next_state, ctx

            elif action == "revise":
                # Needs revision - go back to design with feedback
                ctx.critic_review_iterations += 1
                feedback = result.feedback or ""
                missing = result.missing_requirements or []
                if missing:
                    feedback += f"\n\nMissing requirements:\n" + "\n".join(f"- {m}" for m in missing)
                ctx.critic_feedback = feedback

                # Reset for new design cycle
                ctx.scribe_view = None
                ctx.scribe_iterations = 0

                _record_transition(ctx, ProgrammerState.CRITIC_REVIEW, ProgrammerState.DESIGN,
                                 TransitionReason.CRITIC_REVISE, feedback=feedback)
                return ProgrammerState.DESIGN, ctx

            else:
                # Unknown action - approve by default
                next_state = ProgrammerState.DELIVER if ctx.apply_chunks else ProgrammerState.DONE
                _record_transition(ctx, ProgrammerState.CRITIC_REVIEW, next_state,
                                 TransitionReason.CRITIC_APPROVED)
                return next_state, ctx

        except Exception as e:
            # Critic failure - proceed to deliver/done
            next_state = ProgrammerState.DELIVER if ctx.apply_chunks else ProgrammerState.DONE
            _record_transition(ctx, ProgrammerState.CRITIC_REVIEW, next_state,
                             TransitionReason.ORACLE_ERROR, error=str(e))
            return next_state, ctx

    return _critic_review


def create_critic_evaluate_state() -> Callable[[ProgrammerContext], Tuple[ProgrammerState, ProgrammerContext]]:
    """
    Factory for CRITIC_EVALUATE state.

    Tactical failure recovery - retry from earlier state or give up.
    """
    def _critic_evaluate(ctx: ProgrammerContext) -> Tuple[ProgrammerState, ProgrammerContext]:
        # Check retry limit - don't loop forever
        ctx.critic_evaluate_retries += 1
        if ctx.critic_evaluate_retries >= ctx.max_critic_evaluate_retries:
            _record_transition(ctx, ProgrammerState.CRITIC_EVALUATE, ProgrammerState.FAILED,
                             TransitionReason.CRITIC_GIVE_UP, error="Max retries exceeded")
            return ProgrammerState.FAILED, ctx

        # Build trace summary for informed decision-making
        trace_summary = ""
        if ctx.trace:
            trace_summary = f"\n\n{ctx.trace.summary_for_critic()}"

        prompt = CRITIC_EVALUATE_PROMPT.format(
            problem=ctx.problem,
            error=ctx.last_error or "(unknown error)",
            understanding=ctx.understanding or "(none)",
            design=ctx.design or "(none)",
            chunk_count=len(ctx.chunks),
        )
        # Append trace summary to prompt
        if trace_summary:
            prompt = f"{prompt}{trace_summary}"

        _show_programmer_prompt("CRITIC_EVALUATE", prompt, ui.Colors.yellow, ctx.show_prompts)

        try:
            result = ctx.oracle.ask(prompt, ProgrammerCriticEvaluateResponse, task="critic-evaluate")

            # Get action value - handle enum or string
            action = result.action.value if hasattr(result.action, 'value') else result.action

            if action == "retry":
                # Get retry_from value - handle enum or string
                retry_from = result.retry_from
                if hasattr(retry_from, 'value'):
                    retry_from = retry_from.value
                retry_from = retry_from or "implement"
                feedback = result.feedback or ""

                # Store feedback for the retry
                ctx.critic_feedback = feedback

                # Go to the appropriate state
                if retry_from == "understand":
                    ctx.understanding = None
                    ctx.design = None
                    ctx.solution_doc = None
                    ctx.chunks = []
                    _record_transition(ctx, ProgrammerState.CRITIC_EVALUATE, ProgrammerState.UNDERSTAND,
                                     TransitionReason.CRITIC_RETRY, feedback=feedback)
                    return ProgrammerState.UNDERSTAND, ctx
                elif retry_from == "design":
                    ctx.design = None
                    ctx.solution_doc = None
                    ctx.chunks = []
                    _record_transition(ctx, ProgrammerState.CRITIC_EVALUATE, ProgrammerState.DESIGN,
                                     TransitionReason.CRITIC_RETRY, feedback=feedback)
                    return ProgrammerState.DESIGN, ctx
                else:  # implement
                    ctx.chunks = []
                    _record_transition(ctx, ProgrammerState.CRITIC_EVALUATE, ProgrammerState.IMPLEMENT,
                                     TransitionReason.CRITIC_RETRY, feedback=feedback)
                    return ProgrammerState.IMPLEMENT, ctx

            else:
                # fail - terminal failure
                _record_transition(ctx, ProgrammerState.CRITIC_EVALUATE, ProgrammerState.FAILED,
                                 TransitionReason.CRITIC_GIVE_UP)
                return ProgrammerState.FAILED, ctx

        except Exception as e:
            # Critic failure - give up
            _record_transition(ctx, ProgrammerState.CRITIC_EVALUATE, ProgrammerState.FAILED,
                             TransitionReason.ORACLE_ERROR, error=str(e))
            return ProgrammerState.FAILED, ctx

    return _critic_evaluate


def create_deliver_state() -> Callable[[ProgrammerContext], Tuple[ProgrammerState, ProgrammerContext]]:
    """
    Create DELIVER state handler.

    DELIVER applies chunks to the filesystem. If application fails,
    transitions to CRITIC_EVALUATE for error handling/retry decision.
    """
    def _deliver(ctx: ProgrammerContext) -> Tuple[ProgrammerState, ProgrammerContext]:
        # If no apply callback, just return success (chunks returned to caller)
        if not ctx.apply_chunks:
            _record_transition(ctx, ProgrammerState.DELIVER, ProgrammerState.DONE,
                             TransitionReason.DELIVER_SUCCESS)
            return ProgrammerState.DONE, ctx

        # Apply chunks to filesystem
        success, message = ctx.apply_chunks(ctx.chunks)

        if success:
            _record_transition(ctx, ProgrammerState.DELIVER, ProgrammerState.DONE,
                             TransitionReason.DELIVER_SUCCESS)
            return ProgrammerState.DONE, ctx
        else:
            # Application failed - let CRITIC_EVALUATE decide what to do
            ctx.last_error = f"Chunk application failed: {message}"
            _record_transition(ctx, ProgrammerState.DELIVER, ProgrammerState.CRITIC_EVALUATE,
                             TransitionReason.DELIVER_FAILED, error=message)
            return ProgrammerState.CRITIC_EVALUATE, ctx

    return _deliver


def create_transitions(
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[ProgrammerState, Callable]:
    """
    Create the transition dict for the Programmer NFA.

    Args:
        is_cancelled: Optional predicate for cooperative cancellation.
            If provided, all transitions are wrapped to check cancellation
            at transition boundaries. FP pattern: compose, don't mutate.

    Returns:
        Dict mapping ProgrammerState to transition functions
    """
    from compass.core.nfa import with_cancellation

    base = {
        ProgrammerState.UNDERSTAND: create_understand_state(),
        ProgrammerState.DESIGN: create_design_state(),
        ProgrammerState.IMPLEMENT: create_implement_state(),
        ProgrammerState.SCRIBE_REVIEW: create_scribe_review_state(),
        ProgrammerState.SCRIBE_FETCH: create_scribe_fetch_state(),
        ProgrammerState.SCRIBE_FEEDBACK: create_scribe_feedback_state(),
        ProgrammerState.CRITIC_REVIEW: create_critic_review_state(),
        ProgrammerState.CRITIC_EVALUATE: create_critic_evaluate_state(),
        ProgrammerState.DELIVER: create_deliver_state(),
    }

    return (
        {
            state: with_cancellation(fn, is_cancelled, ProgrammerState.FAILED)
            for state, fn in base.items()
        }
        if is_cancelled else
        base
    )


# Helper functions for formatting

def _handle_scribe_action(
    result: ScribeReviewResponse,
    ctx: ProgrammerContext,
    from_state: ProgrammerState,
    check_loop: bool = False,
) -> Tuple[ProgrammerState, ProgrammerContext]:
    """
    Common handler for Scribe action responses.

    Handles: approve, fetch_pattern, feedback actions.

    Args:
        result: ScribeReviewResponse from oracle
        ctx: Current context
        from_state: State we're transitioning FROM (for trace)
        check_loop: If True, detect pattern fetch loops

    Returns:
        (next_state, context) tuple
    """
    if result.action == ScribeAction.APPROVE:
        if ctx.scribe_view:
            ctx.scribe_view.approved_chunks = [
                c.id if hasattr(c, 'id') else c.get("id", "")
                for c in ctx.chunks
            ]
        # Reset critic review counter - fresh solution to evaluate
        ctx.critic_review_iterations = 0
        _record_transition(ctx, from_state, ProgrammerState.CRITIC_REVIEW,
                         TransitionReason.SCRIBE_APPROVED)
        return ProgrammerState.CRITIC_REVIEW, ctx

    elif result.action == ScribeAction.FETCH_PATTERN:
        new_query = result.query or ""

        # Loop detection: if we already fetched this pattern, force approval
        if check_loop and ctx.scribe_view and new_query in ctx.scribe_view.fetched_patterns:
            if ctx.scribe_view:
                ctx.scribe_view.approved_chunks = [
                    c.id if hasattr(c, 'id') else c.get("id", "")
                    for c in ctx.chunks
                ]
            ctx.critic_review_iterations = 0
            _record_transition(ctx, from_state, ProgrammerState.CRITIC_REVIEW,
                             TransitionReason.SCRIBE_APPROVED, feedback="Pattern loop detected, auto-approved")
            return ProgrammerState.CRITIC_REVIEW, ctx

        # New pattern request
        if ctx.scribe_view:
            ctx.scribe_view.pattern_query = new_query
        _record_transition(ctx, from_state, ProgrammerState.SCRIBE_FETCH,
                         TransitionReason.SCRIBE_NEEDS_PATTERN)
        return ProgrammerState.SCRIBE_FETCH, ctx

    elif result.action == ScribeAction.FEEDBACK:
        feedback = result.feedback or ""
        if ctx.scribe_view:
            ctx.scribe_view.feedback_for_programmer = feedback
            # Convert ScribeIssue to dict for backward compat with ScribeView
            ctx.scribe_view.issues = [
                {"chunk_id": i.chunk_id, "severity": i.severity.value if hasattr(i.severity, 'value') else i.severity,
                 "description": i.description, "suggestion": i.suggestion}
                for i in (result.issues or [])
            ] if result.issues else []
        _record_transition(ctx, from_state, ProgrammerState.SCRIBE_FEEDBACK,
                         TransitionReason.SCRIBE_REJECTED, feedback=feedback)
        return ProgrammerState.SCRIBE_FEEDBACK, ctx

    else:
        # Unknown action - let critic evaluate
        ctx.last_error = f"{from_state.name} unknown action: {result.action}"
        _record_transition(ctx, from_state, ProgrammerState.CRITIC_EVALUATE,
                         TransitionReason.PARSE_ERROR, error=ctx.last_error)
        return ProgrammerState.CRITIC_EVALUATE, ctx


def _format_chunks(chunks: list) -> str:
    """Format chunks for prompt inclusion.

    Shows full content so Scribe can properly evaluate code quality.
    Truncating caused Scribe to think code was incomplete.
    Handles both Chunk dataclass and dict for backward compat.
    """
    if not chunks:
        return "(no chunks)"

    lines = []
    for i, chunk in enumerate(chunks):
        # Handle both Chunk dataclass and dict
        chunk_id = chunk.id if hasattr(chunk, 'id') else chunk.get("id", f"chunk_{i}")
        target = chunk.target if hasattr(chunk, 'target') else chunk.get("target", "unknown")
        op = chunk.operation if hasattr(chunk, 'operation') else chunk.get("operation", "unknown")
        operation = op.value if hasattr(op, 'value') else op
        content = chunk.content if hasattr(chunk, 'content') else chunk.get("content", "")

        lines.append(f"[{chunk_id}] {operation} -> {target}")
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


def _format_chunks_full(chunks: list) -> str:
    """Format chunks with full content for Critic.

    Handles both Chunk dataclass and dict for backward compat.
    """
    if not chunks:
        return "(no chunks)"

    lines = []
    for i, chunk in enumerate(chunks):
        # Handle both Chunk dataclass and dict
        chunk_id = chunk.id if hasattr(chunk, 'id') else chunk.get("id", f"chunk_{i}")
        target = chunk.target if hasattr(chunk, 'target') else chunk.get("target", "unknown")
        op = chunk.operation if hasattr(chunk, 'operation') else chunk.get("operation", "unknown")
        operation = op.value if hasattr(op, 'value') else op
        content = chunk.content if hasattr(chunk, 'content') else chunk.get("content", "")

        lines.append(f"[{chunk_id}] {operation} -> {target}")
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


def _format_file_structure(structure: dict) -> str:
    """Format file structure for prompt inclusion."""
    if not structure:
        return "(no file structure available)"

    lines = []
    for path, info in structure.items():
        lines.append(f"  {path}: {info}")

    return "\n".join(lines) or "(empty)"


def _format_issues(issues: list) -> str:
    """Format issues for prompt inclusion."""
    if not issues:
        return "(no issues)"

    lines = []
    for issue in issues:
        chunk_id = issue.get("chunk_id", "unknown")
        severity = issue.get("severity", "unknown")
        description = issue.get("description", "")
        suggestion = issue.get("suggestion", "")
        lines.append(f"[{severity}] {chunk_id}: {description}")
        if suggestion:
            lines.append(f"  Suggestion: {suggestion}")

    return "\n".join(lines)
