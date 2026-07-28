"""
Bridge to Claude API for Oracle escalation.

When the local Oracle (Actor) gets stuck on complex reasoning,
it can escalate to Claude for help. This module handles that communication.
"""

import os
from typing import Optional


class ClaudeBridge:
    """Client for escalating questions to Claude."""

    # Model name mapping (shared with providers.py)
    MODELS = {
        "sonnet": "claude-sonnet-4-6",
        "opus": "claude-opus-4-6",
    }

    def __init__(self):
        """Initialize with Anthropic client."""
        try:
            from anthropic import Anthropic
            self.client = Anthropic()  # Uses ANTHROPIC_API_KEY env var
        except ImportError:
            raise ImportError(
                "anthropic package required for Claude escalation. "
                "Install with: pip install anthropic"
            )

        model = os.getenv("CLAUDE_MODEL", "sonnet")
        self.model = self.MODELS.get(model, model)  # Resolve friendly name

    def ask(self, question: str, context: str = "", max_tokens: int = 1024) -> str:
        """
        Ask Claude a question and return the answer.

        Args:
            question: The specific question from the Oracle
            context: Relevant code/error context to include
            max_tokens: Maximum response length

        Returns:
            Claude's response text
        """
        prompt = self._build_prompt(question, context)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def consult(self, message: str, context: str = "", max_tokens: int = 2048) -> str:
        """
        Consult Claude for /claude command - response becomes Oracle request.

        Unlike ask(), this tells Claude its response will be sent to Oracle
        as a task request, so it should produce actionable instructions.

        Args:
            message: User's message to Claude
            context: Session context (RAG, history, etc.)
            max_tokens: Maximum response length

        Returns:
            Claude's response (to be sent to Oracle as request)
        """
        prompt = self._build_consult_prompt(message, context)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for Oracle escalation (ask_claude action)."""
        parts = [
            "You are helping a local LLM (the 'Oracle') that got stuck on a task.",
            "",
            "The Oracle is executing a plan step-by-step and needs your help with",
            "complex reasoning, debugging, or architectural decisions.",
            "",
        ]

        if context:
            parts.extend([
                "Context from Oracle's execution:",
                "```",
                context,
                "```",
                "",
            ])

        parts.extend([
            "Oracle's question:",
            question,
            "",
            "Provide a specific, actionable answer. Be concise - the Oracle has",
            "limited context window. Focus on the key insight or solution.",
        ])

        return "\n".join(parts)

    def _build_consult_prompt(self, message: str, context: str) -> str:
        """Build prompt for /claude command - response becomes Oracle request."""
        parts = [
            "You are a senior developer helping a user work with a local LLM called 'Oracle'.",
            "",
            "The user is consulting you. If you approve, YOUR RESPONSE will be sent to the",
            "Oracle as a task request. The Oracle will create a plan and execute it.",
            "",
            "IMPORTANT:",
            "- Your response must be an actionable task/instruction for the Oracle",
            "- Do NOT greet the user or ask clarifying questions",
            "- Do NOT produce a plan yourself - just describe WHAT needs to be done",
            "- If the user's request is unclear, make reasonable assumptions and state them",
            "- If there's truly nothing to do, say 'No action needed' and explain why",
            "",
        ]

        if context:
            parts.extend([
                "Session context:",
                context,
                "",
            ])

        parts.extend([
            "User's message:",
            message,
            "",
            "Write a clear, actionable request for the Oracle:",
        ])

        return "\n".join(parts)


    def ask_json(
        self,
        prompt: str,
        schema: dict,
        max_tokens: int = 512,
        max_retries: int = 2,
    ) -> dict:
        """
        Ask Claude a question and get structured JSON response.

        Uses the shared ask_with_schema() retry loop from oracle.py.

        Args:
            prompt: The prompt to send
            schema: JSON schema for expected response
            max_tokens: Maximum response length
            max_retries: How many correction attempts before giving up

        Returns:
            Parsed and validated JSON response

        Raises:
            OracleSchemaError: If valid response not obtained after retries
        """
        from compass.llm.oracle import ask_with_schema

        def complete_fn(messages):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages
            )
            return response.content[0].text

        return ask_with_schema(complete_fn, prompt, schema, max_retries)


def ask_claude(question: str, context: str = "", model: str = None) -> tuple[bool, str]:
    """
    Convenience function to ask Claude a question.

    Args:
        question: The question to ask
        context: Additional context
        model: Model to use ("sonnet" or "opus"). Defaults to CLAUDE_MODEL env or "sonnet".

    Returns:
        (success, answer_or_error)
    """
    try:
        bridge = ClaudeBridge()
        if model:
            bridge.model = bridge.MODELS.get(model, model)
        answer = bridge.ask(question, context)
        return True, answer
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Failed to reach Claude: {e}"
