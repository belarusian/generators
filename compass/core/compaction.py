"""
Context Compaction System.

Two-layer context management:
1. Truncation (per-item) - ContentRegistry handles individual results
2. Compaction (aggregate) - This module drops/summarizes old items at budget threshold

Design principles:
- Type IS the dispatch key - singledispatch for content-specific strategies
- Pure functions - compaction takes state, returns new state
- Composition - with_compaction wraps step functions
- Immutable data - LoopState remains frozen

Enable/disable via COMPASS_COMPACTION_ENABLED env var (default: enabled).
"""

import os


def compaction_enabled() -> bool:
    """Check if compaction is enabled via environment."""
    return os.getenv("COMPASS_COMPACTION_ENABLED", "1").lower() in ("1", "true", "yes")


def _debug_enabled() -> bool:
    """Check if debug output is enabled (DEBUG or COMPASS_DEBUG)."""
    return (os.getenv("COMPASS_DEBUG", "").lower() in ("1", "true", "yes") or
            os.getenv("DEBUG", "").lower() in ("1", "true", "yes"))
from dataclasses import dataclass, field, replace
from functools import singledispatch
from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from compass.core.actor_loop import LoopState
    from compass.agents.neo.trace import ActionTrace
    from compass.agents.neo.state import RequestContext, RequestState
    from compass.llm.oracle import Oracle


# =============================================================================
# Core Types
# =============================================================================

@dataclass(frozen=True)
class ContextBudget:
    """Budget for context size. Immutable config."""
    max_chars: int = 100_000           # ~25k tokens at 4 chars/token
    max_action_results: int = 30       # Keep last N results
    max_errors: int = 10               # Keep last N errors
    max_file_content_chars: int = 40_000  # Files read section budget
    max_history_traces: int = 50       # Action history for loop detection
    compaction_threshold: float = 0.8  # Trigger at 80% of budget

    @classmethod
    def from_env(cls) -> "ContextBudget":
        """Load budget from environment overrides."""
        return cls(
            max_chars=int(os.getenv("COMPASS_CTX_MAX_CHARS", 100_000)),
            max_action_results=int(os.getenv("COMPASS_CTX_MAX_RESULTS", 30)),
            max_errors=int(os.getenv("COMPASS_CTX_MAX_ERRORS", 10)),
            max_file_content_chars=int(os.getenv("COMPASS_CTX_MAX_FILES", 40_000)),
            max_history_traces=int(os.getenv("COMPASS_CTX_MAX_HISTORY", 50)),
            compaction_threshold=float(os.getenv("COMPASS_CTX_THRESHOLD", 0.8)),
        )


@dataclass(frozen=True)
class ContextMetrics:
    """Metrics about current context size. Immutable snapshot."""
    total_chars: int
    action_results_chars: int
    action_results_count: int
    files_read_chars: int
    files_read_count: int
    errors_chars: int
    errors_count: int
    history_count: int
    file_snapshots_chars: int

    @property
    def utilization(self) -> float:
        """Budget utilization as fraction (0.0-1.0+)."""
        budget = ContextBudget()
        return self.total_chars / budget.max_chars if budget.max_chars > 0 else 0.0

    def exceeds_threshold(self, budget: ContextBudget) -> bool:
        """Check if we should trigger compaction."""
        return self.total_chars > (budget.max_chars * budget.compaction_threshold)


@dataclass(frozen=True)
class CompactionResult:
    """Result of compaction. Pure data."""
    compacted: bool
    metrics_before: ContextMetrics
    metrics_after: ContextMetrics
    summary: str  # Human-readable summary


# =============================================================================
# Content Wrapper Types (for singledispatch)
# =============================================================================

@dataclass(frozen=True)
class ActionResultsContent:
    """Wrapper type for action_results compaction."""
    results: Tuple[str, ...]


@dataclass(frozen=True)
class FilesReadContent:
    """Wrapper type for files_read_content compaction."""
    content: Dict[str, List[Tuple[int, int, str]]]


@dataclass(frozen=True)
class ErrorsContent:
    """Wrapper type for errors_content compaction."""
    errors: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class ActionHistoryContent:
    """Wrapper type for action_history compaction."""
    history: Tuple["ActionTrace", ...]


# =============================================================================
# Measurement: Pure Functions
# =============================================================================

def measure_context(state: "LoopState") -> ContextMetrics:
    """
    Measure context size. Pure function.

    Calculates character counts for each accumulating field.
    """
    action_results_chars = sum(len(r) for r in state.action_results)

    files_read_chars = sum(
        sum(len(content) for _, _, content in chunks)
        for chunks in state.files_read_content.values()
    )

    errors_chars = sum(
        len(target) + len(error)
        for target, error in state.errors_content
    )

    snapshots_chars = sum(
        len(content)
        for content in state.file_snapshots.values()
    )

    total = action_results_chars + files_read_chars + errors_chars + snapshots_chars

    return ContextMetrics(
        total_chars=total,
        action_results_chars=action_results_chars,
        action_results_count=len(state.action_results),
        files_read_chars=files_read_chars,
        files_read_count=len(state.files_read_content),
        errors_chars=errors_chars,
        errors_count=len(state.errors_content),
        history_count=len(state.action_history),
        file_snapshots_chars=snapshots_chars,
    )


def should_compact(state: "LoopState", budget: ContextBudget = None) -> bool:
    """Check if state should be compacted. Pure function."""
    budget = budget or ContextBudget.from_env()
    metrics = measure_context(state)
    return metrics.exceeds_threshold(budget)


# =============================================================================
# LLM Summarization
# =============================================================================

def _get_compaction_provider():
    """
    Get provider for compaction summarization.

    Uses COMPACTION_MODEL env var, falls back to COMPASS_MODEL.
    Compaction summaries feed back into context - use a capable model.
    """
    from compass.llm.providers import get_provider_by_id

    model_id = os.getenv("COMPACTION_MODEL") or os.getenv("COMPASS_MODEL")
    try:
        return get_provider_by_id(model_id)
    except Exception:
        return None  # Fall back to oracle's default routing


def summarize_results(results: Tuple[str, ...], oracle: "Oracle" = None) -> str:
    """
    Summarize old action results into compact form.

    Truncates each result but keeps ALL visible to LLM.
    Uses LLM if available, otherwise returns simple marker.
    """
    if not results:
        return ""

    if not oracle:
        return f"[{len(results)} earlier actions compacted]"

    # Truncate each result to essence, but keep ALL results visible
    truncated = [r[:150] + "..." if len(r) > 150 else r for r in results]
    results_text = "\n".join(f"- {t}" for t in truncated)

    prompt = f"""Summarize these {len(results)} action results in 2-3 sentences.
What was explored, modified, and learned?

{results_text}
"""
    try:
        provider = _get_compaction_provider()
        response = oracle.ask(
            prompt,
            response_type=None,
            max_tokens=200,  # 2-3 sentences
            task="compaction",
            provider=provider,
        )
        return f"[Summary of {len(results)} actions]: {response.text.strip()}"
    except Exception:
        return f"[{len(results)} earlier actions compacted]"


def summarize_errors(errors: Tuple[Tuple[str, str], ...], oracle: "Oracle" = None) -> str:
    """
    Summarize old errors into key learnings.

    Truncates each error but keeps ALL visible to LLM.
    Uses LLM if available, otherwise returns simple marker.
    """
    if not errors:
        return ""

    if not oracle:
        return f"[{len(errors)} earlier errors compacted]"

    # Truncate each error to essence, but keep ALL errors visible
    truncated = [f"{target}: {error[:150]}..." if len(error) > 150 else f"{target}: {error}"
                 for target, error in errors]
    errors_text = "\n".join(f"- {t}" for t in truncated)

    prompt = f"""Summarize these {len(errors)} errors in 1-2 sentences.
What approaches failed? What should be avoided?

{errors_text}
"""
    try:
        provider = _get_compaction_provider()
        response = oracle.ask(
            prompt,
            response_type=None,
            max_tokens=150,  # 1-2 sentences
            task="compaction",
            provider=provider,
        )
        return f"[Learnings from {len(errors)} errors]: {response.text.strip()}"
    except Exception:
        return f"[{len(errors)} earlier errors compacted]"


# =============================================================================
# Compaction Strategies via Singledispatch
# =============================================================================

@singledispatch
def compact_content(content, budget: ContextBudget, metrics: ContextMetrics, oracle: "Oracle" = None):
    """
    Compact content based on its type. Singledispatch pattern.

    Returns compacted content (same type as input).
    Default: return unchanged.
    """
    return content


@compact_content.register(ActionResultsContent)
def _(content: ActionResultsContent, budget: ContextBudget, metrics: ContextMetrics, oracle: "Oracle" = None):
    """
    Compact action results: summarize old, keep recent.

    Strategy: LLM summarizes old results into compact summary.
    """
    if len(content.results) <= budget.max_action_results:
        return content

    # Split into old (to summarize) and recent (to keep)
    keep_count = budget.max_action_results
    old_results = content.results[:-keep_count]
    recent_results = content.results[-keep_count:]

    # Summarize old results
    summary = summarize_results(old_results, oracle)

    return ActionResultsContent(results=(summary,) + recent_results)


@compact_content.register(FilesReadContent)
def _(content: FilesReadContent, budget: ContextBudget, metrics: ContextMetrics, oracle: "Oracle" = None):
    """
    Compact files read: keep most recent chunk per file.

    Strategy: For each file, keep only the most recent chunk.
    No LLM needed - file content can always be re-read.
    """
    total_chars = sum(
        sum(len(c) for _, _, c in chunks)
        for chunks in content.content.values()
    )

    if total_chars <= budget.max_file_content_chars:
        return content

    # Keep most recent chunk per file, prioritize recently accessed files
    compacted = {}
    chars_used = 0

    # Process files in reverse order (most recent first)
    for file_path, chunks in reversed(list(content.content.items())):
        if not chunks:
            continue

        # Keep most recent chunk (highest end line)
        most_recent = max(chunks, key=lambda x: x[1])
        chunk_chars = len(most_recent[2])

        if chars_used + chunk_chars <= budget.max_file_content_chars:
            compacted[file_path] = [most_recent]
            chars_used += chunk_chars

    # Reverse back to original order
    return FilesReadContent(content=dict(reversed(list(compacted.items()))))


@compact_content.register(ErrorsContent)
def _(content: ErrorsContent, budget: ContextBudget, metrics: ContextMetrics, oracle: "Oracle" = None):
    """
    Compact errors: summarize old, keep recent.

    Strategy: LLM extracts key learnings from old errors.
    """
    if len(content.errors) <= budget.max_errors:
        return content

    # Split into old (to summarize) and recent (to keep)
    keep_count = budget.max_errors
    old_errors = content.errors[:-keep_count]
    recent_errors = content.errors[-keep_count:]

    # Summarize old errors
    summary = summarize_errors(old_errors, oracle)

    # Prepend summary as a pseudo-error
    return ErrorsContent(errors=(("compaction_summary", summary),) + recent_errors)


@compact_content.register(ActionHistoryContent)
def _(content: ActionHistoryContent, budget: ContextBudget, metrics: ContextMetrics, oracle: "Oracle" = None):
    """
    Compact action history: keep recent for loop detection.

    Strategy: Simple rolling window - no summarization needed.
    """
    if len(content.history) <= budget.max_history_traces:
        return content

    return ActionHistoryContent(history=content.history[-budget.max_history_traces:])


# =============================================================================
# Total-Budget Enforcement (shared by inner + outer)
# =============================================================================

def _truncate_result(result: str, max_chars: int) -> str:
    """Truncate a single result, preserving head and tail for context. Pure."""
    if len(result) <= max_chars:
        return result
    keep = max_chars - 60  # room for marker
    head = keep * 2 // 3
    tail = keep // 3
    omitted = len(result) - head - tail
    return f"{result[:head]}\n[...{omitted:,} chars truncated...]\n{result[-tail:]}"


def _enforce_total_budget(
    results: Tuple[str, ...],
    non_result_chars: int,
    target: int,
) -> Tuple[str, ...]:
    """
    Truncate oversized action results until total fits within target. Pure.

    Strategy: calculate a fair per-result budget from what's left after
    files/errors/snapshots, then truncate from oldest to newest (preserve
    recent context).
    """
    result_budget = max(0, target - non_result_chars)
    result_chars = sum(len(r) for r in results)

    if result_chars + non_result_chars <= target:
        return results

    per_result_max = max(1500, result_budget // max(len(results), 1))
    out = list(results)
    excess = result_chars + non_result_chars - target

    for i in range(len(out)):
        if excess <= 0:
            break
        if len(out[i]) > per_result_max:
            truncated = _truncate_result(out[i], per_result_max)
            excess -= (len(out[i]) - len(truncated))
            out[i] = truncated

    return tuple(out)


# =============================================================================
# Compaction Orchestration
# =============================================================================

def compact_state(
    state: "LoopState",
    budget: ContextBudget = None,
    oracle: "Oracle" = None,
) -> Tuple["LoopState", CompactionResult]:
    """
    Compact LoopState to fit within budget. Pure function.

    Two phases:
    1. Per-type compaction (count-based: summarize old, keep recent)
    2. Total-budget enforcement (size-based: truncate oversized results)

    Returns (new_state, result) tuple.
    The new state is a compacted copy; original is unchanged.
    """
    budget = budget or ContextBudget.from_env()
    metrics_before = measure_context(state)

    # Check if compaction needed
    if not metrics_before.exceeds_threshold(budget):
        return state, CompactionResult(
            compacted=False,
            metrics_before=metrics_before,
            metrics_after=metrics_before,
            summary="No compaction needed",
        )

    # Phase 1: per-type compaction (manages count)
    compacted_results = compact_content(
        ActionResultsContent(state.action_results),
        budget, metrics_before, oracle
    )

    compacted_files = compact_content(
        FilesReadContent(dict(state.files_read_content)),
        budget, metrics_before, oracle
    )

    compacted_errors = compact_content(
        ErrorsContent(state.errors_content),
        budget, metrics_before, oracle
    )

    compacted_history = compact_content(
        ActionHistoryContent(state.action_history),
        budget, metrics_before, oracle
    )

    # Phase 2: enforce total budget (manages size)
    # Aim for 85% of max so we don't re-trigger immediately next iteration
    target = int(budget.max_chars * 0.85)
    non_result_chars = (
        sum(sum(len(c) for _, _, c in chunks)
            for chunks in compacted_files.content.values())
        + sum(len(t) + len(e) for t, e in compacted_errors.errors)
        + sum(len(content) for content in state.file_snapshots.values())
    )
    enforced_results = _enforce_total_budget(
        compacted_results.results, non_result_chars, target,
    )

    # Build new state
    new_state = replace(
        state,
        action_results=enforced_results,
        files_read_content=compacted_files.content,
        errors_content=compacted_errors.errors,
        action_history=compacted_history.history,
    )

    metrics_after = measure_context(new_state)

    # Build summary
    summary_parts = []
    if len(enforced_results) < len(state.action_results):
        summary_parts.append(
            f"results: {len(state.action_results)} -> {len(enforced_results)}"
        )
    if len(compacted_files.content) < len(state.files_read_content):
        summary_parts.append(
            f"files: {len(state.files_read_content)} -> {len(compacted_files.content)}"
        )
    if len(compacted_errors.errors) < len(state.errors_content):
        summary_parts.append(
            f"errors: {len(state.errors_content)} -> {len(compacted_errors.errors)}"
        )
    if len(compacted_history.history) < len(state.action_history):
        summary_parts.append(
            f"history: {len(state.action_history)} -> {len(compacted_history.history)}"
        )

    chars_saved = metrics_before.total_chars - metrics_after.total_chars
    summary_parts.append(f"saved {chars_saved:,} chars")

    return new_state, CompactionResult(
        compacted=True,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        summary="; ".join(summary_parts),
    )


# =============================================================================
# Composition Pattern
# =============================================================================

def with_compaction(
    step_fn: Callable[["LoopState"], Union["LoopState", "ExecutionResult"]],
    budget: ContextBudget = None,
    oracle: "Oracle" = None,
) -> Callable[["LoopState"], Union["LoopState", "ExecutionResult"]]:
    """
    Wrap step function with compaction. Composition pattern.

    Checks and compacts state after each step if needed.
    Disabled when COMPASS_COMPACTION_ENABLED=0.
    """
    # Early return if compaction disabled
    if not compaction_enabled():
        return step_fn

    from compass.core.actor_loop import LoopState
    from compass.agents.neo.types import ExecutionResult

    budget = budget or ContextBudget.from_env()

    def wrapped(state: LoopState):
        result = step_fn(state)

        # Only compact if continuing (not terminal)
        if isinstance(result, LoopState) and should_compact(result, budget):
            compacted, compaction_result = compact_state(result, budget, oracle)
            if compaction_result.compacted:
                # Reset circuit breaker — re-reading after compaction is recovery, not looping
                compacted = replace(compacted, action_history=())
                if _debug_enabled():
                    print(f"[Compaction] {compaction_result.summary}")
                emit_compaction_metrics(compaction_result, layer="inner")
            return compacted

        return result

    return wrapped


# =============================================================================
# Metrics Formatting
# =============================================================================

def format_metrics(metrics: ContextMetrics, budget: ContextBudget = None) -> str:
    """Format metrics for debug output."""
    budget = budget or ContextBudget()
    utilization_pct = metrics.utilization * 100

    return (
        f"Context: {metrics.total_chars:,} chars ({utilization_pct:.0f}% of {budget.max_chars:,})\n"
        f"  results: {metrics.action_results_count} ({metrics.action_results_chars:,} chars)\n"
        f"  files: {metrics.files_read_count} ({metrics.files_read_chars:,} chars)\n"
        f"  errors: {metrics.errors_count} ({metrics.errors_chars:,} chars)\n"
        f"  history: {metrics.history_count} traces\n"
        f"  snapshots: {metrics.file_snapshots_chars:,} chars"
    )


def emit_compaction_metrics(result: CompactionResult, layer: str = "inner") -> None:
    """
    Emit compaction metrics for observability.

    Controlled by COMPASS_COMPACTION_METRICS env var.
    Outputs JSON for easy parsing by monitoring tools.
    """
    import json

    if not os.getenv("COMPASS_COMPACTION_METRICS"):
        return

    metrics = {
        "layer": layer,
        "compacted": result.compacted,
        "before": {
            "total_chars": result.metrics_before.total_chars,
            "results_count": result.metrics_before.action_results_count,
            "files_count": result.metrics_before.files_read_count,
            "errors_count": result.metrics_before.errors_count,
        },
        "after": {
            "total_chars": result.metrics_after.total_chars,
            "results_count": result.metrics_after.action_results_count,
            "files_count": result.metrics_after.files_read_count,
            "errors_count": result.metrics_after.errors_count,
        },
        "chars_saved": result.metrics_before.total_chars - result.metrics_after.total_chars,
        "summary": result.summary,
    }

    print(f"[COMPACTION_METRICS] {json.dumps(metrics)}")


# =============================================================================
# Outer NFA Compaction (RequestContext)
# =============================================================================

@dataclass(frozen=True)
class OuterBudget:
    """Budget for outer NFA context. Immutable config."""
    max_chars: int = 150_000           # ~37k tokens - larger than inner loop
    retention_ratio: float = 0.67      # Gentle: keep 2/3 on compaction
    compaction_threshold: float = 0.85 # Trigger at 85% of budget

    @classmethod
    def from_env(cls) -> "OuterBudget":
        """Load budget from environment overrides."""
        return cls(
            max_chars=int(os.getenv("COMPASS_OUTER_MAX_CHARS", 150_000)),
            retention_ratio=float(os.getenv("COMPASS_OUTER_RETENTION", 0.67)),
            compaction_threshold=float(os.getenv("COMPASS_OUTER_THRESHOLD", 0.85)),
        )


def measure_request_context(ctx: "RequestContext") -> int:
    """
    Measure RequestContext size in characters. Pure function.

    Counts accumulated content that flows between NFA states.
    """
    total = 0

    # action_results: List[str]
    if ctx.action_results:
        total += sum(len(r) for r in ctx.action_results)

    # files_read_content: Dict[str, List[Tuple[int, int, str]]]
    if ctx.files_read_content:
        for chunks in ctx.files_read_content.values():
            total += sum(len(content) for _, _, content in chunks)

    # feedback flows between states
    if ctx.feedback:
        total += len(ctx.feedback)

    # rag_context
    if ctx.rag_context:
        total += len(ctx.rag_context)

    # critic_summary
    if ctx.critic_summary:
        total += len(ctx.critic_summary)

    return total


def _summarize_action_results_outer(
    results: List[str],
    keep_count: int,
    oracle: "Oracle" = None,
) -> List[str]:
    """
    Summarize old action results, keeping recent ones.

    Returns list with summary prepended to recent results.
    """
    if len(results) <= keep_count:
        return results

    old_results = results[:-keep_count]
    recent_results = results[-keep_count:]

    # Use LLM if available
    summary = summarize_results(tuple(old_results), oracle)

    return [summary] + list(recent_results)


def _compact_files_read_outer(
    files_content: Dict[str, List[Tuple[int, int, str]]],
    target_chars: int,
) -> Dict[str, List[Tuple[int, int, str]]]:
    """
    Compact files_read_content to target size.

    Strategy: Keep most recent chunk per file, prioritize recent files.
    """
    if not files_content:
        return files_content

    current_chars = sum(
        sum(len(c) for _, _, c in chunks)
        for chunks in files_content.values()
    )

    if current_chars <= target_chars:
        return files_content

    # Keep most recent chunk per file, prioritize recent
    compacted = {}
    chars_used = 0

    for file_path, chunks in reversed(list(files_content.items())):
        if not chunks:
            continue

        most_recent = max(chunks, key=lambda x: x[1])
        chunk_chars = len(most_recent[2])

        if chars_used + chunk_chars <= target_chars:
            compacted[file_path] = [most_recent]
            chars_used += chunk_chars

    return dict(reversed(list(compacted.items())))


def compact_request_context(
    ctx: "RequestContext",
    budget: OuterBudget = None,
    oracle: "Oracle" = None,
) -> "RequestContext":
    """
    Compact RequestContext to fit within budget. Pure function.

    Uses gentle compaction: keeps retention_ratio (2/3) of content.
    Returns new context; original is unchanged.
    """
    from dataclasses import replace
    from compass.agents.neo.state import RequestContext

    budget = budget or OuterBudget.from_env()
    current_size = measure_request_context(ctx)

    if current_size <= budget.max_chars * budget.compaction_threshold:
        return ctx  # No compaction needed

    target_chars = int(budget.max_chars * budget.retention_ratio)

    # Capture before counts
    results_before = len(ctx.action_results) if ctx.action_results else 0
    files_before = len(ctx.files_read_content) if ctx.files_read_content else 0

    # Compact action_results (biggest contributor usually)
    compacted_results = ctx.action_results
    if ctx.action_results:
        # Keep 2/3 of results
        keep_count = max(5, int(len(ctx.action_results) * budget.retention_ratio))
        compacted_results = _summarize_action_results_outer(
            list(ctx.action_results), keep_count, oracle
        )

    # Compact files_read_content
    compacted_files = ctx.files_read_content
    if ctx.files_read_content:
        # Allocate 1/3 of target to files
        file_budget = target_chars // 3
        compacted_files = _compact_files_read_outer(
            dict(ctx.files_read_content), file_budget
        )

    # Phase 2: enforce total budget (size-based truncation)
    files_chars = (
        sum(sum(len(c) for _, _, c in chunks)
            for chunks in compacted_files.values())
        if compacted_files else 0
    )
    non_result_chars = (
        files_chars
        + (len(ctx.feedback) if ctx.feedback else 0)
        + (len(ctx.rag_context) if ctx.rag_context else 0)
        + (len(ctx.critic_summary) if ctx.critic_summary else 0)
    )
    enforced_results = _enforce_total_budget(
        tuple(compacted_results) if compacted_results else (),
        non_result_chars,
        target_chars,
    )

    new_ctx = replace(
        ctx,
        action_results=list(enforced_results),
        files_read_content=compacted_files,
    )

    new_size = measure_request_context(new_ctx)
    chars_saved = current_size - new_size

    if _debug_enabled():
        print(f"[Outer Compaction] {current_size:,} -> {new_size:,} chars")

    # Emit metrics
    if os.getenv("COMPASS_COMPACTION_METRICS"):
        import json
        metrics = {
            "layer": "outer",
            "compacted": True,
            "before": {
                "total_chars": current_size,
                "results_count": results_before,
                "files_count": files_before,
            },
            "after": {
                "total_chars": new_size,
                "results_count": len(enforced_results),
                "files_count": len(compacted_files) if compacted_files else 0,
            },
            "chars_saved": chars_saved,
            "actual_retention": round(new_size / current_size, 2) if current_size else 0,
        }
        print(f"[COMPACTION_METRICS] {json.dumps(metrics)}")

    return new_ctx


def with_outer_compaction(
    transition_fn: Callable[["RequestContext"], Tuple["RequestState", "RequestContext"]],
    budget: OuterBudget = None,
    oracle: "Oracle" = None,
) -> Callable[["RequestContext"], Tuple["RequestState", "RequestContext"]]:
    """
    Wrap transition function with compaction check. Composition pattern.

    Checks and compacts context after each state transition if needed.
    Gentle compaction preserves 2/3 of context to maintain coherence.
    Disabled when COMPASS_COMPACTION_ENABLED=0.
    """
    # Early return if compaction disabled
    if not compaction_enabled():
        return transition_fn

    budget = budget or OuterBudget.from_env()

    def wrapped(ctx: "RequestContext") -> Tuple["RequestState", "RequestContext"]:
        state, new_ctx = transition_fn(ctx)

        # Check if compaction needed
        ctx_size = measure_request_context(new_ctx)
        if ctx_size > budget.max_chars * budget.compaction_threshold:
            new_ctx = compact_request_context(new_ctx, budget, oracle)

        return state, new_ctx

    return wrapped


def wrap_transitions_with_compaction(
    transitions: Dict["RequestState", Callable],
    budget: OuterBudget = None,
    oracle: "Oracle" = None,
) -> Dict["RequestState", Callable]:
    """
    Wrap all transition functions with compaction. Bulk wrapper.

    Returns new dict with wrapped transitions; original unchanged.
    Disabled when COMPASS_COMPACTION_ENABLED=0.
    """
    # Early return if compaction disabled
    if not compaction_enabled():
        return transitions

    return {
        state: with_outer_compaction(fn, budget, oracle)
        for state, fn in transitions.items()
    }
