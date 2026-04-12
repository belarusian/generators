"""
Actor module - autonomous action execution.

The Actor is the primary agent that receives requests and executes actions
until complete. It self-loops: request -> actions -> observe -> continue/complete.

Key components:
- call_actor: Main entry point, generates actions from LLM
- _execute_actions: Executes action batch with validation and circuit breaker
- Circuit breaker: Prevents infinite loops on repeated actions
"""

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from compass.core.ui_adapter import UIAdapter

from compass.llm.oracle import Oracle
from compass.llm.providers import ThinkLevel
from compass.agents.neo.memory import CodeMemory
from compass.agents.neo.types import (
    Action, ActionTarget, ActionBatchResult, ActorOutput, ActorStatus,
)
from compass.agents.neo.dispatch import display, action_key
from compass.agents.neo.trace import trace_from_action
from compass.agents.neo.rules import extract_learnings
from compass.agents.neo.rules import execute_action
from compass.core.compose import with_fallback, with_logging
from compass.cli import ui
from compass.core.reasoning import debug


# --- Response Parsing ---

def parse_actor_response(response: "ActorResponsePython") -> ActorOutput:
    """Parse actor response into typed ActorOutput.

    Pure function - no side effects, just parsing.

    Args:
        response: ActorResponsePython from LLM

    Returns:
        ActorOutput with typed status, actions (typed dataclasses), reasoning
    """
    return ActorOutput(
        status=response.status,
        actions=list(response.actions),
        reasoning=response.reasoning,
    )


# --- Circuit Breaker (Loop Detection) ---

def _detect_repeated_action(
    action: Action, action_history: List["ActionTrace"]
) -> Tuple[bool, int]:
    """Detect if an action is being repeated CONSECUTIVELY.

    A loop is when the model does the exact same thing multiple times in a row
    with no other actions in between. Any different action = progression.

    Examples:
        read x, read x, read x  -> LOOP (3 consecutive)
        read x, write y, read x -> OK (write shows progression)
        read x, read y, read x  -> OK (different target shows progression)

    Args:
        action: Current action being attempted (typed action)
        action_history: List of previous ActionTraces

    Returns:
        (is_repeated, streak) - whether in a loop and current streak length
    """
    if not action_history:
        return False, 0

    # Count consecutive matching actions from the END of history
    streak = 0
    for prev in reversed(action_history):
        if _actions_match(action, prev):
            streak += 1
        else:
            # Different action breaks the streak - this is progression
            break

    # Threshold: 3+ consecutive identical actions = loop
    # (allows one retry for verification, but not infinite)
    return streak >= 2, streak

def check_circuit_breaker(
    action: Action, action_history: List["ActionTrace"]
) -> Tuple[bool, str]:
    """Check if circuit breaker should trigger for consecutive repeated actions.

    Only triggers when the EXACT same action is done 3+ times in a row.
    Any different action in between = progression, not a loop.

    Args:
        action: Current action being attempted (typed action)
        action_history: List of previous ActionTraces

    Returns:
        (should_trigger, reason) - whether to halt and why
    """
    is_loop, streak = _detect_repeated_action(action, action_history)

    if is_loop:
        action_name = type(action).__name__
        # Extract target based on action type
        target = (
            getattr(action, "path", None) or
            getattr(action, "command", None) or
            getattr(action, "pattern", None) or
            getattr(action, "query", None) or
            getattr(action, "question", None) or  # ask_claude
            getattr(action, "code", None) or
            "?"
        )
        reason = (
            f"Loop detected: {action_name} on '{target}' done {streak}x consecutively. "
            f"No progression. Escalating to stronger model."
        )
        return True, reason

    return False, ""

# --- Progress Assessment (Trajectory Evaluation) ---

from compass.agents.neo.types import ProgressSignal, ProgressAssessment

# Type: (request, action_history, iteration) -> ProgressAssessment
ProgressAssessor = Callable[[str, List["ActionTrace"], int], ProgressAssessment]


def _format_action_history(actions: List["ActionTrace"], max_actions: int = 8) -> str:
    """Format recent actions for LLM review."""
    from compass.agents.neo.trace import ActionTrace
    recent = actions[-max_actions:]
    lines = []
    for a in recent:
        success = "+" if a.success else "x"
        result_preview = a.result[:100] if a.result else ""
        lines.append(f"  [{success}] {a.action_type} {a.target}: {result_preview}")
    return "\n".join(lines)


def create_progress_assessor(
    oracle: "Oracle",
    min_iterations: int = 3,
    model: Optional[str] = None,
) -> ProgressAssessor:
    """
    Create a progress assessor with oracle injected.

    Uses oracle.ask_python: the model writes Python, we eval it.
    No JSON. No escaping. The type IS the schema.

    Args:
        oracle: Oracle instance for LLM calls
        min_iterations: Minimum iterations before assessing
        model: Model ID for the assessor (default: LOOP_OBSERVER env var)

    Returns: (request, action_history, iteration) -> ProgressAssessment
    """
    from compass.llm.ladder_policy import get_loop_observer_spec
    from compass.llm.providers import get_provider_by_id

    model = model or get_loop_observer_spec()

    # Create dedicated provider for progress assessment
    try:
        assessor_provider = get_provider_by_id(model)
    except Exception:
        assessor_provider = None  # Fall back to oracle's default

    # Signal descriptions for the prompt
    _SIGNAL_DESCRIPTIONS = {
        ProgressSignal.PROGRESSING: "actions are moving toward the goal",
        ProgressSignal.STALLED: "gathering info but not advancing (many reads, no writes)",
        ProgressSignal.OSCILLATING: "going in circles (doing A then B then A then B)",
        ProgressSignal.STUCK: "same thing failing repeatedly, need different approach",
    }

    def assess(request: str, action_history: List["ActionTrace"], iteration: int, directive: Optional[str] = None, prior_results: Optional[List[str]] = None) -> ProgressAssessment:
        # Too early to assess
        if iteration < min_iterations or len(action_history) < min_iterations:
            return ProgressAssessment(
                signal=ProgressSignal.PROGRESSING,
                confidence=0.5,
                reasoning="Too early to assess progress",
            )

        # Prompt - clean, no schema cruft
        signal_options = "\n".join(
            f"- {sig.name}: {desc}" for sig, desc in _SIGNAL_DESCRIPTIONS.items()
        )

        # Build directive and prior_results context
        directive_context = f"\nDirective: {directive}" if directive else ""

        # Build detailed action results context with success/failure status and error messages
        action_results_context = ""

        # Process action_history for success/failure details
        if action_history:
            recent_actions = action_history[-8:]  # Show up to 8 most recent actions
            results_lines = []
            for trace in recent_actions:
                status_symbol = "✓" if trace.success else "✗"
                error_msg = f" (error: {trace.result})" if not trace.success and trace.result else ""
                result_preview = trace.result[:150] if trace.result else ""
                result_preview = result_preview.replace("\n", " ").replace("\r", "")
                if trace.success and trace.result:
                    result_line = f"  [{status_symbol}] {trace.action_type} on {trace.target}: {result_preview}{error_msg}"
                else:
                    result_line = f"  [{status_symbol}] {trace.action_type} on {trace.target}: FAILED{error_msg}"
                results_lines.append(result_line)
            action_results_context = f"\n\nAction Results (with success/failure details):\n" + "\n".join(results_lines)

        # Process prior_results to include success/failure context where applicable
        if prior_results:
            prior_lines = []
            for i, res in enumerate(prior_results[-3:]):  # Show up to 3 most recent results
                # Try to infer success/failure from content patterns
                is_failure = any(pattern in res.lower() for pattern in ["error", "failed", "exception", "traceback", "no such file"])
                status_symbol = "✗" if is_failure else "✓"
                res_preview = res[:150].replace("\n", " ").replace("\r", "")
                prior_lines.append(f"  [{status_symbol}] Prior result {i+1}: {res_preview}")
            if prior_lines:
                prior_results_section = "\n".join(prior_lines)
                if action_results_context:
                    action_results_context += f"\n\nPrior Results (processed with success/failure context):\n{prior_results_section}"
                else:
                    action_results_context = f"\n\nAction Results (with success/failure and prior context):\n{prior_results_section}"

        prior_results_context = ""
        if prior_results:
            prior_results_preview = "\n".join(
                f"  - {res}" for res in prior_results[-3:]  # Show up to 3 most recent results
            )
            prior_results_context = f"\n\nPrior Results:\n{prior_results_preview}"

        # Build oracle thinking instructions for the prompt
        oracle_thinking_instructions = """\n\nPlease think step by step about the following:
1. What has been attempted so far?
2. What is the current state based on results?
3. Are actions showing progression toward the goal?
4. What signal best describes the current trajectory?
\nFormat your thinking clearly before providing your final assessment."""

        # Capture oracle thinking for progress assessment - moved BEFORE prompt construction
        from compass.cli.ui import show_thinking_stream
        thinking_lines: List[str] = []

        def capture_thinking_stream(chunk: str):
            thinking_lines.append(chunk)
            show_thinking_stream(chunk)

        # Build oracle thinking stream placeholder - now uses defined thinking_lines
        oracle_thinking_placeholder = """\n\n--- Oracle Thinking Stream ---\n[Thinking will be captured and appended here]""" if not thinking_lines else ""

        prompt = f"""Evaluate if we're making progress toward the goal.

Goal: "{request}"

Recent actions:
{_format_action_history(action_history)}{directive_context}{action_results_context}{prior_results_context}{oracle_thinking_placeholder}

Signals:
{signal_options}{oracle_thinking_instructions}"""

        # Prompt visitor for debug output
        from compass.core.debug import show_prompt
        on_prompt = lambda p: show_prompt("judge", "LOOP JUDGE", p, ui.Colors.yellow)

        try:
            assessment = oracle.ask(
                prompt,
                ProgressAssessment,
                task="actor:progress-judge",
                provider=assessor_provider,
                on_prompt=on_prompt,
                on_thinking=capture_thinking_stream
            )
            # Inject oracle thinking into reasoning if available
            if thinking_lines:
                thinking_text = "".join(thinking_lines).strip()
                assessment.reasoning = f"[Oracle Thinking]\n{thinking_text}\n\n[Assessment]\n{assessment.reasoning}"
            return assessment
        except Exception:
            return ProgressAssessment(
                signal=ProgressSignal.PROGRESSING,
                confidence=0.5,
                reasoning="Assessment unavailable",
            )
        except Exception:
            return ProgressAssessment(
                signal=ProgressSignal.PROGRESSING,
                confidence=0.5,
                reasoning="Assessment unavailable",
            )

    return assess

def _actions_match(current, prev: "ActionTrace", _action_hash_cache: Dict[tuple, int] = {}) -> bool:
    """Check if current action matches a previous ActionTrace."""
    # Compare keys directly - prev.key was stored at trace creation time
    return action_key(current) == prev.key

# --- Action Target Extraction ---

def extract_action_target(action) -> ActionTarget:
    """Extract target/display info from an action.

    Uses singledispatch - each action type registers its own display handler.
    """
    return display(action)


# --- Actor Entry Point ---


def call_actor(
    oracle: Oracle,
    request: str,
    context: str,
    project_path: Optional[str] = None,
    memory: Optional[CodeMemory] = None,
    provider: "Provider" = None,
    iteration: int = 0,
    think_level: "ThinkLevel" = None,
    ui_adapter: "UIAdapter" = None,
) -> Optional["ActorResponsePython"]:
    """
    Generate actions for a user request.

    The Actor is the primary agent - receives the user request directly
    and executes it through a self-loop of actions until complete.

    Uses dynamically composed schema and rules from compass.agents.neo.rules.

    Args:
        oracle: LLM interface
        request: User's request (clean prose)
        context: Session context with previous results
        project_path: Base project path
        memory: Optional memory for images
        provider: Optional explicit provider (for escalation)
        iteration: Problem-solving iteration (0=first, higher=Critic retries).
                   Higher iterations use higher temperature for more creative solutions.
        think_level: Optional explicit ThinkLevel (OFF/LOW/MEDIUM/HIGH).

    Returns:
        ActorResponsePython with status, actions (typed), and reasoning.
    """
    from compass.agents.neo.types import ActorResponsePython

    # Discover neo-lab skills and states (RAG when available)
    from compass.generators.neo._context import _neo_lab_context
    neo_text = _neo_lab_context(query=request)
    neo_context = f"\nAVAILABLE SKILLS AND STATES (neo-lab):\n{neo_text}\n" if neo_text else ""

    prompt = f"""You are the Actor. Fulfill this request using actions.

Project: {project_path}

You are AUTONOMOUS. You decide what actions to take to fulfill the request.
Think step by step, execute actions, observe results, and continue until done.

STATUS VALUES:
- ActorStatus.CONTINUE = more work to do, I will be called again
- ActorStatus.COMPLETE = request fulfilled, stop execution

FILES READ:
When files appear in the FILES READ section, USE THAT CONTENT DIRECTLY.
Do NOT re-read files that are already shown - you have the content.

BANNERS (for multi-line content -- no escaping, any quotes):
  ### file.py ###
  def greet(name):
      '''Any quotes work!'''
{neo_context}
--- CONTEXT ---

REQUEST: "{request}"

{context}

--- END CONTEXT ---"""

    # Prompt visitor for debug output
    from compass.core.debug import show_prompt
    on_prompt = lambda p: show_prompt("actor", "ACTOR PROMPT", p, ui.Colors.cyan)

    # Pass any pending images to the Oracle for vision
    if memory and memory.images:
        oracle.set_images(memory.get_pending_images())

    ask = with_fallback(with_logging(oracle.ask, "actor"), None)
    if ui_adapter:
        ui_adapter.set_thinking_color("actor")
    result = ask(prompt, ActorResponsePython, task="actor", provider=provider, iteration=iteration, think_level=think_level, on_prompt=on_prompt)
    if ui_adapter:
        ui_adapter.set_thinking_color(None)
    return result
