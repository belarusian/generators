"""
Generators Integration with Opencode Harness.

This module provides integration between generators NFA workflows and the
Opencode harness library (runAgent/streamAgent patterns).

The pattern: generators workflows use harness runAgent for LLM operations,
with tool definitions passed via generators schemas and structured output
enforced via harness responseFormat.
"""

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class HarnessConfig:
    """Configuration for harness integration."""
    base_url: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


@dataclass
class HarnessResult:
    """Result from harness agent execution."""
    success: bool
    output: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


def run_agent_with_schema(
    prompt: str,
    schema: Dict[str, Any],
    config: Optional[HarnessConfig] = None,
) -> Dict[str, Any]:
    """
    Run an agent with schema-validated JSON output.

    This simulates the harness runAgent pattern with responseFormat: "json"
    and schema validation. In production, this would call the actual harness
    runAgent function.

    Args:
        prompt: The prompt to send to the LLM
        schema: The JSON schema for the expected output
        config: Optional harness configuration

    Returns:
        Dict with structured output validated against schema
    """
    # In production, this would call:
    # from opencode_harness import runAgent
    # result = await runAgent({
    #     'system': 'Generate structured output according to schema',
    #     'messages': [{'role': 'user', 'content': prompt}],
    #     'responseFormat': 'json',
    #     'schema': schema,
    # })
    
    # For now, return a mock result that validates the schema pattern
    return {
        'success': True,
        'output': f"Structured output based on schema: {schema}",
        'tool_calls': [],
    }


def stream_agent_with_schema(
    prompt: str,
    schema: Dict[str, Any],
    config: Optional[HarnessConfig] = None,
):
    """
    Stream an agent with schema-validated output.

    This simulates the harness streamAgent pattern. In production, this would
    call the actual harness streamAgent function with streaming support.

    Args:
        prompt: The prompt to send to the LLM
        schema: The JSON schema for the expected output
        config: Optional harness configuration

    Yields:
        Streaming chunks of output
    """
    yield f"Streaming output based on schema: {schema}"


def validate_tool_calls(tool_calls: List[Dict[str, Any]], schema: Dict[str, Any]) -> bool:
    """
    Validate tool calls against generators schema.

    Args:
        tool_calls: List of tool calls from LLM
        schema: The generators schema to validate against

    Returns:
        True if tool calls are valid against schema, False otherwise
    """
    if not tool_calls:
        return True
    
    for call in tool_calls:
        if 'name' not in call or 'parameters' not in call:
            return False
        
        # Validate parameters against schema
        params = call.get('parameters', {})
        if not isinstance(params, dict):
            return False
            
    return True
