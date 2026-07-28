"""
Critic module - reviews Actor's work and evaluates failures.

The Critic has two roles:
1. Review successful execution before generating final answer (critic_review)
2. Evaluate failures after Actor exhausted retries (critic_evaluate)

Key components:
- critic_review: Reviews successful execution before final answer
- critic_evaluate: Evaluates failures, decides replan vs done
- _ask_oracle_for_wisdom: Gets prose guidance from Oracle (Critic retries with it)
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from compass.core.config import ExecutionConfig
    from compass.core.ui_adapter import UIAdapter
    from compass.agents.neo.trace import ExecutionTrace

from compass.llm.oracle import Oracle
from compass.agents.neo.memory import CodeMemory
from compass.agents.neo.types import (
    CriticOutput, CriticAction,
)
from compass.agents.neo.dispatch import hint
from compass.core.compose import with_fallback, with_logging
from compass.cli import ui
from compass.core.reasoning import debug


# --- Oracle Wisdom (prose feedback for self-loop) ---

def _ask_oracle_for_wisdom(oracle: Oracle, question: str, context: str) -> str:
    """
    Ask Oracle for prose wisdom. Critic retries with this as feedback.

    Unlike the old ASK_CLAUDE which made the decision FOR the Critic,
    this returns prose guidance that enriches the Critic's next attempt.
    The Critic remains the judge.

    Args:
        oracle: Oracle instance
        question: The Critic's specific question
        context: Context the Critic provided

    Returns:
        Prose wisdom from Oracle (or error message)
    """
    prompt = f"""The Critic is uncertain and seeks your wisdom.

QUESTION:
{question}

CONTEXT:
{context}

Provide guidance - insight, considerations, what to weigh.
Do NOT make the decision. Help the Critic think through it."""

    try:
        response = oracle.ask(
            prompt=prompt,
            response_type=None,  # Prose, not structured
            max_tokens=2000,
            task="oracle_wisdom",
        )
        return response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        debug(f"Oracle wisdom failed: {e}")
        return f"(Oracle unavailable: {e})"


# --- Critic Review (success path) ---

def critic_review(
    oracle: Oracle,
    request: str,
    action_results: List[str],
    context: str,
    memory: Optional[CodeMemory] = None,
    exec_globals: Optional[Dict] = None,
    files_read: Optional[Dict[str, List]] = None,
    config: "ExecutionConfig" = None,
    ui_adapter: "UIAdapter" = None,
    execution_trace: "ExecutionTrace" = None,
    _oracle_wisdom: Optional[str] = None,  # Internal: Oracle's guidance from ASK_ORACLE
    _oracle_retries: int = 0,  # Internal: bounded self-loop counter
) -> Optional[CriticOutput]:
    """
    Critic role - reviews successful execution BEFORE final answer.

    DESIGN PRINCIPLE: If context can't expand/page, show everything.
    We use sliding windows for bounded context, but Critic sees full picture.
    If results are truncated with [ID], Critic can request expansion.

    Args:
        oracle: LLM interface
        request: Original user request (FULL - never truncate)
        action_results: Results from execution
        context: Session context
        memory: Optional memory for images
        exec_globals: Computed variables
        files_read: Files that were actually read/written during execution
        config: Execution config with provider/think_level overrides
        execution_trace: For expanding truncated content via [ID] markers

    Returns:
        {"action": "replan"|"done", "explanation": "...", "feedback?": "..."}
    """
    # Pass images to Critic so it can verify image descriptions
    if memory and memory.images:
        oracle.set_images(memory.get_pending_images())

    results_text = "\n\n".join(action_results) if action_results else "(no action results recorded)"
    # Files actually touched - derived from execution, not a stale plan
    files = list(files_read.keys()) if files_read else []
    files_text = ", ".join(files) if files else "None"

    # Format exec_globals so Critic sees actual computed values
    vars_text = ""
    if exec_globals:
        builtins = {"os", "sys", "json", "Path", "pathlib", "cwd", "__builtins__"}
        user_vars = []
        for name, val in exec_globals.items():
            if name in builtins:
                continue
            val_str = str(val)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            user_vars.append(f"  {name} = {val_str}")
        if user_vars:
            vars_text = "\n".join(user_vars)

    # Include Oracle's wisdom if this is a retry after ASK_ORACLE
    wisdom_section = ""
    if _oracle_wisdom:
        wisdom_section = f"""
--- ORACLE'S GUIDANCE ---
{_oracle_wisdom}
--- END GUIDANCE ---

Consider this wisdom as you make your decision.
"""

    prompt = f"""You are the Critic - quick check before the final answer.

{context}

User request: "{request}"
Files touched: {files_text}

Action results:
{results_text}{vars_text}
{wisdom_section}
===================================================
CRITICAL: Look for the 🚀🚀🚀 marker.
- Actions BEFORE 🚀🚀🚀 = history from previous requests (context only)
- Actions AFTER 🚀🚀🚀 = work done for THIS request
- If 🚀🚀🚀 exists but NOTHING follows it = Actor did NOTHING = replan!

DEFAULT TO "done" - approve unless there's a CRITICAL problem.

- "done": Actor did work for THIS request (has actions AFTER 🚀🚀🚀)
- "replan": Nothing after 🚀🚀🚀, OR Actor misunderstood the request
- "ask_user": ONLY if request is fundamentally impossible without more info
- "ask_oracle": ONLY if genuinely uncertain about a complex decision

Minor imperfections, edge cases, potential improvements = "done". Ship it.
"""

    from compass.core.debug import show_prompt
    on_prompt = lambda p: show_prompt("critic", "CRITIC PROMPT", p, ui.Colors.magenta)

    fallback = CriticOutput(
        action=CriticAction.DONE,
        explanation="Critic unavailable, proceeding to answer."
    )

    think_level = config.think_level if config else None

    if ui_adapter:
        ui_adapter.set_thinking_color("critic")

    # Direct oracle.ask - expansion removed, model uses ReadFileAction if needed
    ask = with_fallback(with_logging(oracle.ask, "critic_review"), fallback)
    result = ask(prompt, CriticOutput, task="critic", think_level=think_level, on_prompt=on_prompt)

    if ui_adapter:
        ui_adapter.set_thinking_color(None)

    # Handle ASK_ORACLE: get wisdom, then self-loop with enriched context
    if result and result.action == CriticAction.ASK_ORACLE:
        if _oracle_retries >= 2:
            # Bounded: after 2 oracle consultations, just proceed
            debug("Critic asked Oracle twice, proceeding with DONE")
            return CriticOutput(
                action=CriticAction.DONE,
                explanation="Proceeding after consulting Oracle.",
            )

        question = result.question or result.explanation
        context_str = result.context or ""
        print(f"  {ui.Colors.cyan('[Critic asks Oracle]')}: {question[:80]}...")

        wisdom = _ask_oracle_for_wisdom(oracle, question, context_str)
        print(f"  {ui.Colors.cyan('[Oracle speaks]')}: {wisdom[:100]}...")

        # Self-loop: retry with Oracle's wisdom in context
        return critic_review(
            oracle=oracle,
            request=request,
            action_results=action_results,
            context=context,
            memory=memory,
            exec_globals=exec_globals,
            files_read=files_read,
            config=config,
            ui_adapter=ui_adapter,
            execution_trace=execution_trace,
            _oracle_wisdom=wisdom,
            _oracle_retries=_oracle_retries + 1,
        )

    return result


# --- Critic Evaluate (failure path) ---

def critic_evaluate(oracle: Oracle, request: str, action, error: str, context: str, config: "ExecutionConfig" = None, ui_adapter: "UIAdapter" = None, execution_trace: "ExecutionTrace" = None, _oracle_wisdom: Optional[str] = None, _oracle_retries: int = 0) -> Optional[Dict]:
    """
    Critic role - evaluates failures AFTER Actor exhausted retries.

    Actor already tried 3 times. Critic decides:
    - "replan": Go back to Actor with feedback to try different approach
    - "done": Stop execution and return to user

    Args:
        oracle: LLM interface
        request: Request that failed
        action: Last action attempted (typed action dataclass)
        error: Error message
        context: Session context
        config: Execution config with provider/think_level overrides
        execution_trace: For expanding truncated content via [ID] markers

    Returns:
        CriticOutput with action, explanation, feedback
    """
    action_name = type(action).__name__
    # Extract target from typed action using display singledispatch
    from compass.agents.neo.dispatch import display
    target_info = display(action)
    target = target_info.target

    # Get hint via singledispatch
    action_hint = hint(action)
    hint_line = f"\nHint for {action_name}: {action_hint}" if action_hint else ""

    # Include Oracle's wisdom if this is a retry after ASK_ORACLE
    wisdom_section = ""
    if _oracle_wisdom:
        wisdom_section = f"""
--- ORACLE'S GUIDANCE ---
{_oracle_wisdom}
--- END GUIDANCE ---

Consider this wisdom as you make your decision.
"""

    prompt = f"""You are the Critic - Actor hit an error. Quick decision needed.

{context}

Request: "{request}"
Action: {action_name} on "{target}"
Error: {error}{hint_line}
{wisdom_section}
DEFAULT TO "done" - errors happen, return to user unless clearly fixable.

- "done": Error occurred, return to user with what we have (USE THIS BY DEFAULT)
- "replan": ONLY if error is clearly fixable with different approach AND you know exactly what to do
- "ask_oracle": ONLY if genuinely uncertain about a complex decision

If unsure whether replanning will help = "done". Don't loop endlessly.
One failed attempt is often enough - the user can clarify or retry.
"""

    from compass.core.debug import show_prompt
    on_prompt = lambda p: show_prompt("critic", "CRITIC FAILURE PROMPT", p, ui.Colors.yellow)

    fallback = CriticOutput(
        action=CriticAction.DONE,
        explanation="Critic unavailable, returning to user."
    )

    think_level = config.think_level if config else None

    if ui_adapter:
        ui_adapter.set_thinking_color("critic")

    # Direct oracle.ask - expansion removed, model uses ReadFileAction if needed
    ask = with_fallback(with_logging(oracle.ask, "critic_evaluate"), fallback)
    result = ask(prompt, CriticOutput, task="critic", think_level=think_level, on_prompt=on_prompt)
    if ui_adapter:
        ui_adapter.set_thinking_color(None)

    # Handle ASK_ORACLE: get wisdom, then self-loop with enriched context
    if result and result.action == CriticAction.ASK_ORACLE:
        if _oracle_retries >= 2:
            # Bounded: after 2 oracle consultations, just proceed
            debug("Critic asked Oracle twice, proceeding with DONE")
            return CriticOutput(
                action=CriticAction.DONE,
                explanation="Proceeding after consulting Oracle.",
            )

        question = result.question or result.explanation
        context_str = result.context or ""
        print(f"  {ui.Colors.cyan('[Critic asks Oracle]')}: {question[:80]}...")

        wisdom = _ask_oracle_for_wisdom(oracle, question, context_str)
        print(f"  {ui.Colors.cyan('[Oracle speaks]')}: {wisdom[:100]}...")

        # Self-loop: retry with Oracle's wisdom in context
        return critic_evaluate(
            oracle=oracle,
            request=request,
            action=action,
            error=error,
            context=context,
            config=config,
            ui_adapter=ui_adapter,
            execution_trace=execution_trace,
            _oracle_wisdom=wisdom,
            _oracle_retries=_oracle_retries + 1,
        )

    return result
