"""
NFA Telemetry - Track actions and transitions.

Provides insight into model behavior during execution:
- Actions: what tools Neo uses (ReadFile, WriteFile, etc.)
- Transitions: state flow in Neo and Programmer NFAs

Note: "Decisions" was the bug - using singledispatch for routing.
Transitions are enum-based, tracked as (from_state, to_state) pairs.

Usage:
    from compass.core.telemetry import collect_stats, record_action, record_transition

    with collect_stats() as stats:
        # ... run NFA ...
        print(stats.report())
"""

import json
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import local
from typing import Any, Callable, Generator, Optional, Set, Tuple, TypeVar, TYPE_CHECKING

T = TypeVar("T")
ProviderParamsKey = Tuple[str, str, str, str]
TaskProviderKey = Tuple[str, str]
TaskProviderOutcomeKey = Tuple[str, str, str]
TaskAttemptKey = Tuple[str, int]
TaskAttemptFailureKey = Tuple[str, int, str]
if TYPE_CHECKING:
    from compass.llm.providers import ThinkLevel

ThinkLevelParam = Optional["ThinkLevel"]


def _normalize_think_level(think_level: ThinkLevelParam) -> str:
    """Normalize think level to a stable telemetry label."""
    if think_level is None:
        return "off"
    return str(think_level.value)


def _normalize_temperature(temperature: Optional[float]) -> str:
    """Normalize temperature to a stable telemetry label."""
    if temperature is None:
        return "default"
    return f"{temperature:g}"


def _normalize_seed(seed: Optional[int]) -> str:
    """Normalize seed to a stable telemetry label."""
    if seed is None:
        return "default"
    return str(seed)


def _provider_params_key(
    provider_name: str,
    think_level: ThinkLevelParam = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
) -> ProviderParamsKey:
    """Build normalized key for provider parameter telemetry."""
    return (
        provider_name,
        _normalize_think_level(think_level),
        _normalize_temperature(temperature),
        _normalize_seed(seed),
    )


def get_registered_action_types() -> Set[str]:
    """Discover all registered action types from dispatch registry."""
    from compass.agents.neo.dispatch import execute
    return {
        t.__name__ for t in execute.registry.keys()
        if t is not object  # Skip base registration
    }


# Thread-local storage for stats collector
_local = local()

# Global stats for cross-thread aggregation (set by test fixtures)
# When threads run Programmer NFAs, they need to record to a shared collector
_stats: Optional["NFAStats"] = None


@dataclass
class FailureDetail:
    """Details about a failure transition."""
    from_state: str
    to_state: str
    error: str
    source: str  # test name or context


@dataclass
class NFAStats:
    """Collected stats during NFA execution."""
    actions: Counter = field(default_factory=Counter)      # action type -> count
    action_time: Counter = field(default_factory=Counter)  # action type -> total seconds
    transitions: Counter = field(default_factory=Counter)  # (from_state, to_state) -> count
    transition_time: Counter = field(default_factory=Counter)  # (from_state, to_state) -> total seconds
    oracle_calls: Counter = field(default_factory=Counter) # task -> count
    oracle_time: Counter = field(default_factory=Counter)  # task -> total seconds
    provider_calls: Counter = field(default_factory=Counter)  # provider name -> count
    provider_time: Counter = field(default_factory=Counter)  # provider name -> total seconds
    provider_params_calls: Counter = field(default_factory=Counter)  # (provider, think, temp, seed) -> count
    provider_params_time: Counter = field(default_factory=Counter)  # (provider, think, temp, seed) -> total seconds
    task_provider_calls: Counter = field(default_factory=Counter)  # (task, provider) -> count
    task_provider_time: Counter = field(default_factory=Counter)  # (task, provider) -> total seconds
    task_provider_outcomes: Counter = field(default_factory=Counter)  # (task, provider, outcome) -> count
    task_attempts: Counter = field(default_factory=Counter)  # (task, attempt_index) -> count
    task_attempt_failures: Counter = field(default_factory=Counter)  # (task, attempt_index, failure_type) -> count
    failure_details: list = field(default_factory=list)    # List[FailureDetail]
    parse_recoveries: Counter = field(default_factory=Counter)  # recovery_type -> count

    def record_action(self, action: Any, duration: float = 0.0) -> None:
        """Record an action execution with duration."""
        name = type(action).__name__
        self.actions[name] += 1
        self.action_time[name] += duration

    def record_transition(self, from_state: str, to_state: str, error: str = None, source: str = None, duration: float = 0.0) -> None:
        """Record an NFA state transition with duration, optionally with error details."""
        self.transitions[(from_state, to_state)] += 1
        self.transition_time[(from_state, to_state)] += duration

        # Capture failure details whenever an error is provided
        if error:
            import os
            # Use provided source, or fall back to PYTEST_CURRENT_TEST, or "unknown"
            if not source:
                # PYTEST_CURRENT_TEST format: "tests/test_foo.py::TestClass::test_method (call)"
                source = os.environ.get("PYTEST_CURRENT_TEST", "unknown")
                # Clean up: extract just test name
                if "::" in source:
                    source = source.split("::")[-1].split(" ")[0]
            self.failure_details.append(FailureDetail(
                from_state=from_state,
                to_state=to_state,
                error=error[:200],  # Truncate long errors
                source=source,
            ))

    def record_oracle_call(self, task: str, duration: float = 0.0) -> None:
        """Record an Oracle call by task type with duration."""
        self.oracle_calls[task] += 1
        self.oracle_time[task] += duration

    def record_provider_call(
        self,
        provider_name: str,
        duration: float = 0.0,
        think_level: ThinkLevelParam = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        task: Optional[str] = None,
    ) -> None:
        """Record provider usage (and generation params) with duration."""
        self.provider_calls[provider_name] += 1
        self.provider_time[provider_name] += duration

        params_key = _provider_params_key(
            provider_name=provider_name,
            think_level=think_level,
            temperature=temperature,
            seed=seed,
        )
        self.provider_params_calls[params_key] += 1
        self.provider_params_time[params_key] += duration

        if task:
            task_key: TaskProviderKey = (task, provider_name)
            self.task_provider_calls[task_key] += 1
            self.task_provider_time[task_key] += duration

    def record_task_provider_outcome(
        self,
        task: str,
        provider_name: str,
        outcome: str,
    ) -> None:
        """Record outcome for a task served by a provider."""
        key: TaskProviderOutcomeKey = (task, provider_name, outcome)
        self.task_provider_outcomes[key] += 1

    def record_task_attempt(self, task: str, attempt_index: int) -> None:
        """Record attempt index distribution per task."""
        key: TaskAttemptKey = (task, attempt_index)
        self.task_attempts[key] += 1

    def record_task_attempt_failure(self, task: str, attempt_index: int, failure_type: str, error_message: str = "") -> None:
        """Record retry failure category per task/attempt."""
        key: TaskAttemptFailureKey = (task, attempt_index, failure_type)
        self.task_attempt_failures[key] += 1
        if error_message:
            self.failure_details.append(FailureDetail(
                from_state=f"{task}[{attempt_index}]",
                to_state=failure_type,
                error=error_message[:200],
                source=task,
            ))

    def record_parse_recovery(self, recovery_type: str) -> None:
        """Record a parse recovery (e.g., markdown fence stripping)."""
        self.parse_recoveries[recovery_type] += 1

    def metrics(self) -> dict:
        """Compute derived metrics from raw telemetry.

        Returns dict with rates and counts. Missing data = None.
        """
        t = self.transitions

        # === Programmer NFA ===
        # Scribe: SCRIBE_REVIEW -> CRITIC_REVIEW (approve) vs SCRIBE_FEEDBACK (reject)
        scribe_approve = t.get(("SCRIBE_REVIEW", "CRITIC_REVIEW"), 0)
        scribe_reject = t.get(("SCRIBE_REVIEW", "SCRIBE_FEEDBACK"), 0)
        scribe_total = scribe_approve + scribe_reject

        # Programmer Critic: CRITIC_REVIEW -> DONE (approve) vs DESIGN (revise)
        prog_critic_approve = t.get(("CRITIC_REVIEW", "DONE"), 0)
        prog_critic_reject = t.get(("CRITIC_REVIEW", "DESIGN"), 0)
        prog_critic_total = prog_critic_approve + prog_critic_reject

        # Programmer runs: count UNDERSTAND->DESIGN as "started"
        programmer_started = t.get(("UNDERSTAND", "DESIGN"), 0)

        # === Neo NFA ===
        # Neo Critic: REVIEW -> ANSWER (approve) vs ACT (retry)
        neo_critic_approve = t.get(("REVIEW", "ANSWER"), 0)
        neo_critic_reject = t.get(("REVIEW", "ACT"), 0)
        neo_critic_total = neo_critic_approve + neo_critic_reject

        # Neo runs: count ACT->REVIEW as "iterations"
        neo_iterations = t.get(("ACT", "REVIEW"), 0)

        # === Programmer Stage Failures ===
        # UNDERSTAND: success (->DESIGN) vs failure (->CRITIC_EVALUATE)
        understand_ok = t.get(("UNDERSTAND", "DESIGN"), 0)
        understand_fail = t.get(("UNDERSTAND", "CRITIC_EVALUATE"), 0)
        understand_total = understand_ok + understand_fail

        # DELIVER: success (->DONE) vs failure (->CRITIC_EVALUATE)
        deliver_ok = t.get(("DELIVER", "DONE"), 0)
        deliver_fail = t.get(("DELIVER", "CRITIC_EVALUATE"), 0)
        deliver_total = deliver_ok + deliver_fail

        # IMPLEMENT: success (->SCRIBE_REVIEW) vs failure (->CRITIC_EVALUATE)
        implement_ok = t.get(("IMPLEMENT", "SCRIBE_REVIEW"), 0)
        implement_fail = t.get(("IMPLEMENT", "CRITIC_EVALUATE"), 0)
        implement_total = implement_ok + implement_fail

        return {
            # Programmer reviews
            "scribe_approval_rate": scribe_approve / scribe_total if scribe_total else None,
            "scribe_approved": scribe_approve,
            "scribe_rejected": scribe_reject,
            "prog_critic_approval_rate": prog_critic_approve / prog_critic_total if prog_critic_total else None,
            "prog_critic_approved": prog_critic_approve,
            "prog_critic_rejected": prog_critic_reject,
            "programmer_runs": programmer_started,
            # Programmer stage failures
            "understand_success_rate": understand_ok / understand_total if understand_total else None,
            "understand_ok": understand_ok,
            "understand_fail": understand_fail,
            "implement_success_rate": implement_ok / implement_total if implement_total else None,
            "implement_ok": implement_ok,
            "implement_fail": implement_fail,
            "deliver_success_rate": deliver_ok / deliver_total if deliver_total else None,
            "deliver_ok": deliver_ok,
            "deliver_fail": deliver_fail,
            # Neo
            "neo_critic_approval_rate": neo_critic_approve / neo_critic_total if neo_critic_total else None,
            "neo_critic_approved": neo_critic_approve,
            "neo_critic_rejected": neo_critic_reject,
            "neo_iterations": neo_iterations,
        }

    def metrics_report(self) -> str:
        """Human-readable metrics summary."""
        m = self.metrics()
        lines = []

        # Neo section
        neo_lines = []
        if m["neo_critic_approval_rate"] is not None:
            pct = m["neo_critic_approval_rate"] * 100
            neo_lines.append(f"Critic: {pct:.0f}% approval ({m['neo_critic_approved']}/{m['neo_critic_approved'] + m['neo_critic_rejected']})")
        if m["neo_iterations"]:
            neo_lines.append(f"Iterations: {m['neo_iterations']}")
        if neo_lines:
            lines.append("Neo: " + ", ".join(neo_lines))

        # Programmer section - reviews
        prog_lines = []
        if m["scribe_approval_rate"] is not None:
            pct = m["scribe_approval_rate"] * 100
            prog_lines.append(f"Scribe {pct:.0f}% ({m['scribe_approved']}/{m['scribe_approved'] + m['scribe_rejected']})")
        if m["prog_critic_approval_rate"] is not None:
            pct = m["prog_critic_approval_rate"] * 100
            prog_lines.append(f"Critic {pct:.0f}% ({m['prog_critic_approved']}/{m['prog_critic_approved'] + m['prog_critic_rejected']})")
        if m["programmer_runs"]:
            prog_lines.append(f"Runs: {m['programmer_runs']}")
        if prog_lines:
            lines.append("Programmer: " + ", ".join(prog_lines))

        # Programmer section - stage failures (only show if there are failures)
        failures = []
        if m["understand_fail"]:
            total = m["understand_ok"] + m["understand_fail"]
            failures.append(f"UNDERSTAND {m['understand_fail']}/{total} failed")
        if m["implement_fail"]:
            total = m["implement_ok"] + m["implement_fail"]
            failures.append(f"IMPLEMENT {m['implement_fail']}/{total} failed")
        if m["deliver_fail"]:
            total = m["deliver_ok"] + m["deliver_fail"]
            failures.append(f"DELIVER {m['deliver_fail']}/{total} failed")
        if failures:
            lines.append("Failures: " + ", ".join(failures))

        return "  " + "\n  ".join(lines) if lines else ""

    def to_dict(self) -> dict:
        """
        Serialize telemetry to structured JSON-friendly data.

        This is the canonical machine format. Rows are key-sorted for deterministic
        output; human presentation ordering is handled by report().
        """
        actions_rows = [
            {
                "action": name,
                "count": count,
                "duration_seconds": self.action_time.get(name, 0.0),
            }
            for name, count in sorted(self.actions.items(), key=lambda item: item[0])
        ]

        transition_rows = [
            {
                "from_state": from_state,
                "to_state": to_state,
                "count": count,
                "duration_seconds": self.transition_time.get((from_state, to_state), 0.0),
            }
            for (from_state, to_state), count in sorted(self.transitions.items(), key=lambda item: item[0])
        ]

        oracle_rows = [
            {
                "task": task,
                "count": count,
                "duration_seconds": self.oracle_time.get(task, 0.0),
            }
            for task, count in sorted(self.oracle_calls.items(), key=lambda item: item[0])
        ]

        provider_rows = [
            {
                "provider": provider_name,
                "count": count,
                "duration_seconds": self.provider_time.get(provider_name, 0.0),
            }
            for provider_name, count in sorted(self.provider_calls.items(), key=lambda item: item[0])
        ]

        provider_param_rows = [
            {
                "provider": provider_name,
                "think_level": think_level,
                "temperature": temperature,
                "seed": seed,
                "count": count,
                "duration_seconds": self.provider_params_time.get(params_key, 0.0),
            }
            for params_key, count in sorted(self.provider_params_calls.items(), key=lambda item: item[0])
            for provider_name, think_level, temperature, seed in [params_key]
        ]

        task_provider_rows = [
            {
                "task": task,
                "provider": provider_name,
                "count": count,
                "duration_seconds": self.task_provider_time.get(key, 0.0),
            }
            for key, count in sorted(self.task_provider_calls.items(), key=lambda item: item[0])
            for task, provider_name in [key]
        ]

        task_outcome_rows = [
            {
                "task": task,
                "provider": provider_name,
                "outcome": outcome,
                "count": count,
            }
            for key, count in sorted(self.task_provider_outcomes.items(), key=lambda item: item[0])
            for task, provider_name, outcome in [key]
        ]

        task_attempt_rows = [
            {
                "task": task,
                "attempt_index": attempt_index,
                "count": count,
            }
            for key, count in sorted(self.task_attempts.items(), key=lambda item: item[0])
            for task, attempt_index in [key]
        ]

        task_attempt_failure_rows = [
            {
                "task": task,
                "attempt_index": attempt_index,
                "failure_type": failure_type,
                "count": count,
            }
            for key, count in sorted(self.task_attempt_failures.items(), key=lambda item: item[0])
            for task, attempt_index, failure_type in [key]
        ]

        return {
            "schema_version": 1,
            "actions": {
                "rows": actions_rows,
                "total_count": sum(self.actions.values()),
                "total_duration_seconds": sum(self.action_time.values()),
            },
            "transitions": {
                "rows": transition_rows,
                "total_count": sum(self.transitions.values()),
                "total_duration_seconds": sum(self.transition_time.values()),
            },
            "oracle_calls": {
                "rows": oracle_rows,
                "total_count": sum(self.oracle_calls.values()),
                "total_duration_seconds": sum(self.oracle_time.values()),
            },
            "providers": {
                "rows": provider_rows,
                "total_count": sum(self.provider_calls.values()),
                "total_duration_seconds": sum(self.provider_time.values()),
            },
            "provider_params": {
                "rows": provider_param_rows,
                "total_count": sum(self.provider_params_calls.values()),
                "total_duration_seconds": sum(self.provider_params_time.values()),
            },
            "task_providers": {
                "rows": task_provider_rows,
                "total_count": sum(self.task_provider_calls.values()),
                "total_duration_seconds": sum(self.task_provider_time.values()),
            },
            "task_outcomes": {
                "rows": task_outcome_rows,
                "total_count": sum(self.task_provider_outcomes.values()),
            },
            "task_attempts": {
                "rows": task_attempt_rows,
                "total_count": sum(self.task_attempts.values()),
            },
            "task_attempt_failures": {
                "rows": task_attempt_failure_rows,
                "total_count": sum(self.task_attempt_failures.values()),
            },
            "failures": [
                {
                    "from_state": failure.from_state,
                    "to_state": failure.to_state,
                    "error": failure.error,
                    "source": failure.source,
                }
                for failure in self.failure_details
            ],
            "parse_recoveries": {
                "rows": [
                    {"type": recovery_type, "count": count}
                    for recovery_type, count in sorted(self.parse_recoveries.items())
                ],
                "total_count": sum(self.parse_recoveries.values()),
            },
            "metrics": self.metrics(),
        }

    def to_json(self, indent: Optional[int] = 2, sort_keys: bool = True) -> str:
        """Serialize telemetry to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)

    def report(self, show_zeros: bool = True) -> str:
        """Generate human-readable report.

        Args:
            show_zeros: If True, include registered action types with count=0
        """
        _ = show_zeros  # currently preserved for API compatibility
        data = self.to_dict()
        lines = []

        def add_section(title: str, row_lines: list, total_line: Optional[str] = None) -> None:
            if not row_lines:
                return
            if lines:
                lines.append("")
            lines.append(title)
            lines.extend(row_lines)
            if total_line:
                lines.append(total_line)

        def format_count_duration_row(label: str, count: int, duration_seconds: float) -> str:
            if duration_seconds > 0:
                return f"  {label}: {count} ({duration_seconds:.1f}s)"
            return f"  {label}: {count}"

        actions_rows = sorted(
            data["actions"]["rows"],
            key=lambda row: (
                -row["duration_seconds"],
                -row["count"],
                row["action"],
            ),
        )
        add_section(
            "ACTIONS:",
            [
                format_count_duration_row(
                    label=row["action"],
                    count=row["count"],
                    duration_seconds=row["duration_seconds"],
                )
                for row in actions_rows
            ],
            (
                f"  -- total: {data['actions']['total_count']} actions, "
                f"{data['actions']['total_duration_seconds']:.1f}s"
                if data["actions"]["total_duration_seconds"] > 0
                else None
            ),
        )

        transition_rows = sorted(
            data["transitions"]["rows"],
            key=lambda row: (
                -row["duration_seconds"],
                -row["count"],
                row["from_state"],
                row["to_state"],
            ),
        )
        add_section(
            "TRANSITIONS:",
            [
                format_count_duration_row(
                    label=f"{row['from_state']} -> {row['to_state']}",
                    count=row["count"],
                    duration_seconds=row["duration_seconds"],
                )
                for row in transition_rows
            ],
            (
                f"  -- total: {data['transitions']['total_count']} transitions, "
                f"{data['transitions']['total_duration_seconds']:.1f}s"
                if data["transitions"]["total_duration_seconds"] > 0
                else None
            ),
        )

        oracle_rows = sorted(
            data["oracle_calls"]["rows"],
            key=lambda row: (
                -row["duration_seconds"],
                -row["count"],
                row["task"],
            ),
        )
        add_section(
            "ORACLE CALLS:",
            [
                format_count_duration_row(
                    label=row["task"],
                    count=row["count"],
                    duration_seconds=row["duration_seconds"],
                )
                for row in oracle_rows
            ],
            (
                f"  -- total: {data['oracle_calls']['total_count']} calls, "
                f"{data['oracle_calls']['total_duration_seconds']:.1f}s"
                if data["oracle_calls"]["total_duration_seconds"] > 0
                else None
            ),
        )

        provider_rows = sorted(
            data["providers"]["rows"],
            key=lambda row: (
                -row["duration_seconds"],
                -row["count"],
                row["provider"],
            ),
        )
        add_section(
            "PROVIDERS:",
            [
                format_count_duration_row(
                    label=row["provider"],
                    count=row["count"],
                    duration_seconds=row["duration_seconds"],
                )
                for row in provider_rows
            ],
            (
                f"  -- total: {data['providers']['total_count']} calls, "
                f"{data['providers']['total_duration_seconds']:.1f}s"
                if data["providers"]["total_duration_seconds"] > 0
                else None
            ),
        )

        provider_params_rows = sorted(
            data["provider_params"]["rows"],
            key=lambda row: (
                -row["duration_seconds"],
                -row["count"],
                row["provider"],
                row["think_level"],
                row["temperature"],
                row["seed"],
            ),
        )
        add_section(
            "PROVIDER PARAMS:",
            [
                format_count_duration_row(
                    label=(
                        f"{row['provider']} "
                        f"(think={row['think_level']}, temp={row['temperature']}, seed={row['seed']})"
                    ),
                    count=row["count"],
                    duration_seconds=row["duration_seconds"],
                )
                for row in provider_params_rows
            ],
            (
                f"  -- total: {data['provider_params']['total_count']} calls, "
                f"{data['provider_params']['total_duration_seconds']:.1f}s"
                if data["provider_params"]["total_duration_seconds"] > 0
                else None
            ),
        )

        task_provider_rows = sorted(
            data["task_providers"]["rows"],
            key=lambda row: (
                -row["duration_seconds"],
                -row["count"],
                row["task"],
                row["provider"],
            ),
        )
        add_section(
            "TASK PROVIDERS:",
            [
                format_count_duration_row(
                    label=f"{row['task']} -> {row['provider']}",
                    count=row["count"],
                    duration_seconds=row["duration_seconds"],
                )
                for row in task_provider_rows
            ],
            (
                f"  -- total: {data['task_providers']['total_count']} calls, "
                f"{data['task_providers']['total_duration_seconds']:.1f}s"
                if data["task_providers"]["total_duration_seconds"] > 0
                else None
            ),
        )

        task_outcome_rows = sorted(
            data["task_outcomes"]["rows"],
            key=lambda row: (
                -row["count"],
                row["task"],
                row["provider"],
                row["outcome"],
            ),
        )
        add_section(
            "TASK OUTCOMES:",
            [
                f"  {row['task']} -> {row['provider']} [{row['outcome']}]: {row['count']}"
                for row in task_outcome_rows
            ],
        )

        task_attempt_rows = sorted(
            data["task_attempts"]["rows"],
            key=lambda row: (
                row["task"],
                row["attempt_index"],
            ),
        )
        add_section(
            "TASK ATTEMPTS:",
            [
                f"  {row['task']} attempt {row['attempt_index']}: {row['count']}"
                for row in task_attempt_rows
            ],
        )

        task_attempt_failure_rows = sorted(
            data["task_attempt_failures"]["rows"],
            key=lambda row: (
                row["task"],
                row["attempt_index"],
                row["failure_type"],
            ),
        )
        add_section(
            "TASK ATTEMPT FAILURES:",
            [
                (
                    f"  {row['task']} attempt {row['attempt_index']} "
                    f"[{row['failure_type']}]: {row['count']}"
                )
                for row in task_attempt_failure_rows
            ],
        )

        if data["parse_recoveries"]["total_count"]:
            add_section(
                "PARSE RECOVERIES:",
                [
                    f"  {row['type']}: {row['count']}"
                    for row in data["parse_recoveries"]["rows"]
                ],
                f"  -- total: {data['parse_recoveries']['total_count']} recoveries",
            )

        add_section(
            "FAILURES:",
            [
                f"  [{row['source']}] {row['from_state']} -> {row['to_state']}: {row['error']}"
                for row in data["failures"]
            ],
        )

        metrics_str = self.metrics_report()
        if metrics_str:
            if lines:
                lines.append("")
            lines.append(f"METRICS:\n{metrics_str}")

        return "\n".join(lines) if lines else "(no stats recorded)"

    def __str__(self) -> str:
        return self.report()


def _get_stats() -> Optional[NFAStats]:
    """Get current stats collector (if active).

    Checks global _stats first (for cross-thread aggregation in tests),
    then falls back to thread-local (for normal collect_stats() usage).
    """
    if _stats is not None:
        return _stats
    return getattr(_local, 'stats', None)


@contextmanager
def collect_stats():
    """Context manager to collect NFA stats.

    Uses global _stats for cross-thread visibility (Programmer runs in ThreadPoolExecutor).

    Usage:
        with collect_stats() as stats:
            result = process_request(...)
        print(stats.report())
    """
    global _stats
    _stats = NFAStats()
    try:
        yield _stats
    finally:
        _stats = None


def record_action(action: Any, duration: float = 0.0) -> None:
    """Record an action execution with duration (call from execute_action)."""
    stats = _get_stats()
    if stats:
        stats.record_action(action, duration)


def record_transition(from_state: str, to_state: str, error: str = None, source: str = None, duration: float = 0.0) -> None:
    """Record a state transition with duration (call from NFA runners).

    Args:
        from_state: State transitioning from
        to_state: State transitioning to
        error: Optional error message (for failure transitions)
        source: Optional source identifier (e.g., request text, test name)
        duration: Time spent in the from_state before transitioning
    """
    stats = _get_stats()
    if stats:
        stats.record_transition(from_state, to_state, error, source, duration)


def record_oracle_call(task: str, duration: float = 0.0) -> None:
    """Record an Oracle call with duration (call from oracle.ask)."""
    stats = _get_stats()
    if stats:
        stats.record_oracle_call(task, duration)


def record_provider_call(
    provider_name: str,
    duration: float = 0.0,
    think_level: ThinkLevelParam = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    task: Optional[str] = None,
) -> None:
    """Record provider call (and generation params) with duration."""
    stats = _get_stats()
    if stats:
        stats.record_provider_call(
            provider_name=provider_name,
            duration=duration,
            think_level=think_level,
            temperature=temperature,
            seed=seed,
            task=task,
        )


def with_provider_timing(
    provider_name: str,
    call: Callable[[], T],
    think_level: ThinkLevelParam = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    task: Optional[str] = None,
) -> T:
    """Execute call and record provider telemetry (including params)."""
    import time
    start = time.monotonic()
    try:
        return call()
    finally:
        record_provider_call(
            provider_name=provider_name,
            duration=time.monotonic() - start,
            think_level=think_level,
            temperature=temperature,
            seed=seed,
            task=task,
        )


def with_provider_timing_stream(
    provider_name: str,
    stream_call: Callable[[], Generator[T, None, None]],
    think_level: ThinkLevelParam = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    task: Optional[str] = None,
) -> Generator[T, None, None]:
    """Execute streaming call and record provider telemetry on close/exhaustion."""
    import time
    start = time.monotonic()
    try:
        yield from stream_call()
    finally:
        record_provider_call(
            provider_name=provider_name,
            duration=time.monotonic() - start,
            think_level=think_level,
            temperature=temperature,
            seed=seed,
            task=task,
        )


def record_task_provider_outcome(task: str, provider_name: str, outcome: str) -> None:
    """Record outcome for task/provider pair."""
    stats = _get_stats()
    if stats:
        stats.record_task_provider_outcome(task, provider_name, outcome)


def record_task_attempt(task: str, attempt_index: int) -> None:
    """Record attempt index distribution per task."""
    stats = _get_stats()
    if stats:
        stats.record_task_attempt(task, attempt_index)


def record_task_attempt_failure(task: str, attempt_index: int, failure_type: str, error_message: str = "") -> None:
    """Record retry failure category for task/attempt."""
    stats = _get_stats()
    if stats:
        stats.record_task_attempt_failure(task, attempt_index, failure_type, error_message)


def record_parse_recovery(recovery_type: str) -> None:
    """Record a parse recovery (e.g., stripping markdown fences)."""
    stats = _get_stats()
    if stats:
        stats.record_parse_recovery(recovery_type)
