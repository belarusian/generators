"""
The Oracle - speaks wisdom in prose.

Unlike other agents that return structured JSON, the Oracle speaks freely.
She uses the big model for divergent, creative thinking.

Neo invokes her when:
- Stuck and needs a fresh perspective
- Problem is ambiguous and needs exploration
- Wants creative alternatives before committing
"""

from compass.llm.oracle import Oracle as OracleVoice


def ask_oracle(
    oracle: OracleVoice,
    question: str,
    context: str = "",
    max_tokens: int = 1500,
) -> str:
    """
    Ask the Oracle for wisdom.

    Unlike other agents, Oracle returns prose - no JSON schema.
    She thinks freely, explores, and speaks naturally.

    Args:
        oracle: The Oracle voice (LLM interface)
        question: What to ask
        context: Background context for the question
        max_tokens: How much she can speak

    Returns:
        Prose wisdom - unstructured, natural language
    """
    prompt = f"""You are the Oracle - a wise advisor who speaks in prose.

Think deeply, explore freely, and speak naturally.
Do not constrain yourself to JSON or structured formats. Just speak.

QUESTION:
{question}

CONTEXT:
{context or "(no additional context)"}

Speak your wisdom. Be thoughtful, be creative, explore alternatives.
If you see multiple paths, describe them. If you're uncertain, say so.
Your role is to dream, not to execute."""

    # Use REASONING_TASKS provider (big model) via task routing
    return oracle.speak(prompt, max_tokens=max_tokens, task="oracle-dream")
