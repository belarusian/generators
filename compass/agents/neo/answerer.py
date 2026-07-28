"""
Answerer module - generates final responses to user requests.

The Answerer produces the final answer after Actor completes execution.
It formats computed values, action results, and file references into
a coherent response.

Key components:
- generate_answer: Generate final answer from execution results
- is_image_question: Detect if request is about an attached image
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from compass.core.config import ExecutionConfig
    from compass.agents.neo.trace import ExecutionTrace

from compass.llm.oracle import Oracle
from compass.agents.neo.memory import CodeMemory
from compass.agents.neo.types import AnswerResponse
from compass.core.compose import with_fallback, with_logging
# ask_with_expand removed - expansion disabled
from compass.core.reasoning import debug


def is_image_question(request: str) -> bool:
    """Detect if request is primarily about an attached image.

    Pure function: no side effects.

    Args:
        request: User's request string

    Returns:
        True if request appears to be about an image attachment
    """
    request_lower = request.lower()
    # Contains image reference and is a simple question about it
    has_image_ref = "image#" in request_lower or "[image" in request_lower
    image_keywords = ["what's in", "what is in", "describe", "show", "see", "look at", "analyze"]
    is_asking_about = any(kw in request_lower for kw in image_keywords)
    # Short request focused on the image
    is_short = len(request.split()) < 20
    return has_image_ref and (is_asking_about or is_short)


def generate_answer(
    oracle: Oracle,
    request: str,
    action_results: List[str],
    context: str,
    memory: Optional[CodeMemory] = None,
    exec_globals: Optional[Dict] = None,
    files_read: Optional[Dict[str, List]] = None,
    config: "ExecutionConfig" = None,
    critic_summary: Optional[str] = None,
    execution_trace: "ExecutionTrace" = None,
) -> Optional[AnswerResponse]:
    """Generate the final answer after Actor completes.

    DESIGN PRINCIPLE: Critic -> Answerer pipeline. Answerer receives Critic's analysis.
    If results are truncated with [ID], Answerer can request expansion.

    Args:
        oracle: LLM interface
        request: Original user request (FULL - never truncate)
        action_results: Results from execution
        context: Session context (same as Critic receives)
        memory: Optional memory for images
        exec_globals: Computed variables from exec actions
        files_read: Files that were actually read/written during execution
        critic_summary: Critic's analysis of what happened (pipeline input)
        execution_trace: For expanding truncated content via [ID] markers

    Returns:
        AnswerResponse with answer, references (optional), next_steps (optional)
    """
    results_text = "\n\n".join(action_results) if action_results else "(no action results recorded)"
    files = list(files_read.keys()) if files_read else []
    files_text = ", ".join(files) if files else "None"
    image_note = ""
    if memory and memory.images:
        labels = [f"Image#{img.id}: {img.filename}" for img in memory.get_pending_images()]
        image_note = (
            "\nImage attachments (visible to you): "
            + ", ".join(labels)
            + "\nDescribe what you see directly; do not claim you cannot view them."
        )

    # Format exec_globals so Answerer sees actual computed values
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
            vars_text = "\n\nComputed values:\n" + "\n".join(user_vars)

    # Critic's analysis is the primary source of truth (pipeline)
    critic_section = (
        f"\n--- CRITIC'S ANALYSIS (use this as your primary source) ---\n{critic_summary}\n"
        if critic_summary else ""
    )

    prompt = f"""You are the Answerer. Provide the final response to the user.

{context}

User request: "{request}"
Files touched: {files_text}
{image_note}
{critic_section}
Action results:
{results_text}{vars_text}

Guidelines:
- The Critic's analysis above summarizes what was accomplished - USE IT as your primary source
- Answer directly and concisely using the ACTUAL computed values above
- Only include "references" when action results contain file:line data (from read_file, search, grep)
- For exec-only results (API calls, computations), omit references - there are no grounded line numbers
- Keep the answer focused on what the user asked
- In next_steps, suggest 1-2 logical follow-up actions the user might want (even if answer is complete)
  Examples: "Check the 7-day forecast", "Compare with another city", "Run the tests", "Deploy to staging"
  Keep them short and actionable. Omit if no natural follow-up exists."""

    from compass.cli import ui
    from compass.core.debug import show_prompt
    on_prompt = lambda p: show_prompt("answerer", "ANSWERER PROMPT", p, ui.Colors.cyan)

    # Pass images to Answerer so it can reference them
    if memory and memory.images:
        oracle.set_images(memory.get_pending_images())

    fallback = AnswerResponse(answer="Unable to generate answer.")
    think_level = config.think_level if config else None

    # Direct oracle.ask - expansion removed, model uses ReadFileAction if needed
    ask = with_fallback(with_logging(oracle.ask, "answerer"), fallback)
    return ask(prompt, AnswerResponse, think_level=think_level, on_prompt=on_prompt)
