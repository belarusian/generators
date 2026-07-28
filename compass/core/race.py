"""
Environment-aware parallel racing for LLM providers.

Provides infrastructure for racing multiple providers in parallel,
with environment-aware configuration:
- @local: Fast but single-threaded, NO parallel racing
- @big: Server with resources, CAN race multiple providers

Pure FP patterns: composition, immutable results, typed strategies.
"""

import os
import time
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, List, Optional

from compass.core.branch import BranchResult
from compass.agents.neo.types import ExecutionContext, ExecutionResult, ExecutionStatus
from compass.core.ui_events import UIEventStream


# --- Configuration ---

@dataclass
class DeepThinkingConfig:
    """
    Configuration for deep thinking (parallel model exploration).

    Deep thinking runs stronger models in parallel alongside the default path.
    The default path uses the configured coding ladder (fast, local-first).
    Deep branches use the deep ladder (strongest models on @big).

    Branches run until one sticks or all complete - no artificial timeout.
    Cooperative cancellation stops remaining branches once a solution is approved.

    Environment variables:
        COMPASS_DEEP_THINKING=1: Enable deep thinking
        COMPASS_DEEP_THINKING_BRANCHES=N: Number of deep branches (default: 2)
    """
    enabled: bool = False
    num_branches: int = 2

    @classmethod
    def from_env(cls) -> "DeepThinkingConfig":
        """Create config from environment variables."""
        enabled = os.getenv("COMPASS_DEEP_THINKING") == "1"
        num_branches = int(os.getenv("COMPASS_DEEP_THINKING_BRANCHES", "2"))

        return cls(
            enabled=enabled,
            num_branches=num_branches,
        )


# Backwards compatibility alias
RaceConfig = DeepThinkingConfig


# --- Environment Detection ---

def deep_thinking_enabled() -> bool:
    """Check if deep thinking is enabled.

    Deep thinking runs stronger models in parallel alongside the default path.
    Requires explicit opt-in via COMPASS_DEEP_THINKING=1.
    """
    return os.getenv("COMPASS_DEEP_THINKING") == "1"


# Backwards compatibility alias
def can_race() -> bool:
    """Deprecated: Use deep_thinking_enabled() instead."""
    return deep_thinking_enabled()


# --- Selection Strategies ---

SelectionStrategy = Callable[[List[BranchResult]], Optional[BranchResult]]


def select_first_success(results: List[BranchResult]) -> Optional[BranchResult]:
    """Return first completed result, else best partial.

    Strategy: Speed over quality. Take the first successful completion.
    Fallback: If none complete, pick the one with most successful actions.
    """
    completed = [r for r in results if r.is_complete]
    if completed:
        return completed[0]
    # Fallback: most progress
    return max(results, key=lambda r: r.actions_succeeded) if results else None


def select_best_quality(results: List[BranchResult]) -> Optional[BranchResult]:
    """Of completed: min retries, max success, min time.

    Strategy: Quality over speed. Among successful completions,
    prefer the one with fewest retries, most successful actions,
    and shortest duration (in that priority order).
    """
    completed = [r for r in results if r.is_complete]
    if not completed:
        return select_first_success(results)
    return min(completed, key=lambda r: (r.retries_used, -r.actions_succeeded, r.duration_seconds))


# --- Branch Execution Helpers ---

def _run_branch(
    execute_fn: Callable[[Any, ExecutionContext], ExecutionResult],
    provider: Any,
    base_ctx: ExecutionContext,
) -> BranchResult:
    """Execute a single branch and capture result.

    Creates isolated context for parallel execution, runs the provider,
    and packages everything into a BranchResult.
    """
    ctx = ExecutionContext.for_parallel(base_ctx)
    started_at = datetime.now().isoformat()
    start_time = time.time()

    try:
        result = execute_fn(provider, ctx)
        completed_at = datetime.now().isoformat()
        duration = time.time() - start_time

        # Get UI events if available
        ui_events = (
            ctx.ui.get_events() if hasattr(ctx.ui, 'get_events')
            else UIEventStream()
        )

        return BranchResult(
            execution_result=result,
            provider_name=str(provider),
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            ui_events=ui_events,
        )
    except Exception as e:
        # Return failed result
        completed_at = datetime.now().isoformat()
        duration = time.time() - start_time

        failed_result = ExecutionResult(
            status=ExecutionStatus.DONE,
            action_results=[f"Branch execution failed: {e}"],
        )

        return BranchResult(
            execution_result=failed_result,
            provider_name=str(provider),
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            ui_events=UIEventStream(),
        )


def _run_single(
    execute_fn: Callable[[Any, ExecutionContext], ExecutionResult],
    provider: Any,
    base_ctx: ExecutionContext,
) -> BranchResult:
    """Run single provider without parallel infrastructure.

    Optimization: When only one provider or max_branches=1,
    skip ThreadPoolExecutor overhead and run directly.
    """
    return _run_branch(execute_fn, provider, base_ctx)


# --- Main Race Function ---

def race_branches(
    execute_fn: Callable[[Any, ExecutionContext], ExecutionResult],
    providers: List[Any],
    base_ctx: ExecutionContext,
    config: RaceConfig = None,
    select: SelectionStrategy = select_first_success,
) -> Optional[BranchResult]:
    """
    Race multiple providers in parallel (if environment supports it).

    If max_branches=1 or only one provider, runs sequentially.
    Otherwise uses ThreadPoolExecutor for true parallelism.

    Args:
        execute_fn: Function that takes (provider, ctx) and returns ExecutionResult
        providers: List of providers to race (e.g., different LLM backends)
        base_ctx: Base execution context (will be cloned for each branch)
        config: Race configuration (defaults to environment-aware config)
        select: Strategy for selecting winner from results

    Returns:
        BranchResult from winning provider, or None if all fail
    """
    config = config or RaceConfig.from_env()
    providers = providers[:config.max_branches]

    if not providers:
        return None

    if len(providers) == 1 or config.max_branches == 1:
        # Sequential - just run with first provider
        return _run_single(execute_fn, providers[0], base_ctx)

    # Parallel racing
    results: List[BranchResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = {
            executor.submit(_run_branch, execute_fn, p, base_ctx): p
            for p in providers
        }

        try:
            for future in concurrent.futures.as_completed(futures, timeout=config.timeout_seconds):
                try:
                    result = future.result()
                    results.append(result)

                    if result.is_complete:
                        # Cancel remaining futures - we have a winner
                        for f in futures:
                            f.cancel()
                        break
                except Exception:
                    pass  # Log and continue - one branch failing shouldn't stop others
        except concurrent.futures.TimeoutError:
            # Timeout hit - work with what we have
            for f in futures:
                f.cancel()

    winner = select(results)

    if winner and winner.ui_events and winner.ui_events.events:
        # Render winner's UI events
        from compass.core.ui_adapter import ImmediateUIAdapter, replay_events
        replay_events(winner.ui_events, ImmediateUIAdapter())

    return winner
