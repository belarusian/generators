"""
AskOracleAction handlers - singledispatch registration.

Type-based dispatch for ask_oracle action: display, validate, execute, extract_learnings.
Uses ORACLE_MODEL for deep wisdom and architectural insight.
"""

import json
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from compass.core.content import preview_head_tail

from compass.agents.neo.types import AskOracleAction
from compass.agents.neo.dispatch import display, validate, execute, extract_learnings, action_key, hint, display_name, content_field

if TYPE_CHECKING:
    from compass.agents.neo.types import ActionTarget, ExecutionContext
    from compass.agents.neo.memory import Learning


# --- Display ---

@content_field.register(AskOracleAction)
def _(action): return "context"


@display.register(AskOracleAction)
def _(action: AskOracleAction) -> "ActionTarget":
    """Get display info for AskOracleAction."""
    from compass.agents.neo.types import ActionTarget

    question = action.question
    # Truncate long questions for display
    display_question = question[:60] + "..." if len(question) > 60 else question

    return ActionTarget(target="oracle", display=display_question, content=None)


# --- Validation ---

@validate.register(AskOracleAction)
def _(
    action: AskOracleAction,
    project_path: str = ".",
    files_read: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate ask_oracle action.

    Returns (is_valid, error_message).

    Required fields:
    - question: What to ask the Oracle

    Optional fields:
    - context: Additional context (what you tried, errors, code snippets)

    Use ask_oracle when:
    - Seeking architectural wisdom or design guidance
    - Exploring possibilities before committing to an approach
    - Needing creative insight or alternative perspectives
    - Wanting prose reflection rather than structured decisions

    Reserve for genuinely hard problems. Use ask_claude for quick help.
    """
    question = action.question

    if not question:
        return False, "Missing required field: question"

    return True, None


# --- Execution ---

@execute.register(AskOracleAction)
def _(action: AskOracleAction, project_path: str, ctx: "ExecutionContext" = None) -> Tuple[bool, str]:
    """
    Execute ask_oracle action - consult the Oracle for deep wisdom.

    Formats the request as a "letter" from Neo to the Oracle,
    with Neo's message and full working context attached.

    Args:
        action: AskOracleAction
        project_path: Project root directory
        ctx: ExecutionContext with oracle and actor_context
    """
    question = action.question or ""
    context = action.context or ""
    actor_context = ctx.actor_context if ctx else ""

    # Build full prompt with context
    if actor_context:
        full_prompt = f"""FROM: Neo
TO: Oracle

QUESTION:
{question}

NEO'S CONTEXT (what they chose to include):
{context}

---
ATTACHED: Full context Neo was working with
---
{actor_context}
---
END ATTACHMENT
---"""
    else:
        full_prompt = f"{question}\n\nContext:\n{context}" if context else question

    # Use ORACLE_MODEL for wisdom tasks
    if ctx and ctx.oracle:
        try:
            from compass.llm.ladder_policy import get_oracle_model_spec
            from compass.llm.providers import get_provider_by_id

            provider = get_provider_by_id(get_oracle_model_spec())

            response = ctx.oracle.ask(
                prompt=full_prompt,
                response_type=None,  # Raw text response
                max_tokens=4000,
                task="oracle",
                provider=provider,
                on_thinking=ctx.on_thinking,
            )
            return True, f"Oracle speaks:\n{response.text}"
        except Exception as e:
            return False, f"Oracle is silent: {e}"
    else:
        # Fallback to bridge if no oracle in context
        from compass.llm.bridge import ask_claude as bridge_ask
        success, result = bridge_ask(question, context, model="opus")
        if success:
            return True, f"Oracle (Opus) speaks:\n{result}"
        else:
            return False, f"Oracle is silent: {result}"


# --- Learning Extraction ---

@extract_learnings.register(AskOracleAction)
def _(
    action: AskOracleAction,
    success: bool,
    result: str,
    reflect,
) -> List["Learning"]:
    """Extract learnings from ask_oracle action. LLM reflects and chooses learning type."""
    from dataclasses import asdict

    action_data = asdict(action) if hasattr(action, '__dataclass_fields__') else {"question": action.question, "context": action.context}

    prompt = f"""Action: ask_oracle
Input: {json.dumps(action_data)}
Success: {success}
Result:
{preview_head_tail(result, max_lines=202)}

What did we learn from this?"""

    return [reflect(prompt)]


# --- Action Key ---

@action_key.register(AskOracleAction)
def _(action: AskOracleAction) -> tuple:
    """Hashable key for AskOracleAction comparison."""
    return ("ask_oracle", action.question)


@hint.register(AskOracleAction)
def _(action: AskOracleAction) -> str:
    """Hint for Critic when ask_oracle fails."""
    return "Deep wisdom. Architecture, complex planning, creative insight."


@display_name.register(AskOracleAction)
def _(action: AskOracleAction) -> str:
    """Human-friendly name for UI."""
    return "Oracle"
