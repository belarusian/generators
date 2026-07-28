"""
Model invocation and prompt building for generators.

All side effects for model communication live here.
Uses Compass's Provider abstraction composed with with_retry / with_logging.
The model is a parameter, not a dependency.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ._types import AskFn, DomainSection, Err, GenerationContext, Ok, Result

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


def _provider_ask(provider: Any) -> AskFn:
    """Wrap a Compass Provider as an AskFn.

    The generator sends (system, user) and gets text back.
    The Provider handles the transport.

    For Ollama: num_predict=-1 means unlimited (fill context window).
    For Anthropic: provider clamps to model's max output tokens.
    """
    # -1 = unlimited / use model max. Provider clamps per-model.
    max_tokens = -1

    def ask(system: str, user: str) -> Result:
        sys_chars = len(system)
        usr_chars = len(user)
        # logger.debug(
        #     "--- SYSTEM (%d chars) ---\n%s\n--- END SYSTEM ---",
        #     sys_chars, system,
        # )
        # logger.debug(
        #     "--- USER (%d chars) ---\n%s\n--- END USER ---",
        #     usr_chars, user,
        # )
        t0 = time.monotonic()
        try:
            resp = provider.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            elapsed = time.monotonic() - t0
            text = resp.text.strip()
            logger.info(
                "[%s] %.1fs | prompt %d+%d chars | response %d chars",
                provider.name, elapsed, sys_chars, usr_chars, len(text),
            )
            logger.debug(
                "--- RESPONSE (%d chars) ---\n%s\n--- END RESPONSE ---",
                len(text), text,
            )
            return Ok(text) if text else Err("Empty response from provider")
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning(
                "[%s] %.1fs | prompt %d+%d chars | ERROR: %s",
                provider.name, elapsed, sys_chars, usr_chars, e,
            )
            return Err(f"Provider error: {e}")
    return ask


def resolve_ask_fn(model_id: str = "", ask_fn: AskFn | None = None) -> AskFn:
    """Build the model invocation function.

    Resolution: ask_fn > get_provider_by_id(model_id) > get_model_spec().

    The model_id is a spec string the rest of Compass already understands:
        "qwen3-coder:latest@local"   -> OllamaProvider on local server
        "qwen3-coder-next:q8_0@big"  -> OllamaProvider on big server
        "anthropic:sonnet"           -> AnthropicProvider (sonnet)
        "anthropic:opus"             -> AnthropicProvider (opus)

    When model_id is not explicitly set, falls back to the ladder policy's
    worker role (COMPASS_FAMILY -> Family.worker -> default).
    """
    if ask_fn is not None:
        return ask_fn

    from dotenv import load_dotenv
    load_dotenv()

    from compass.core.compose import with_retry, with_logging
    from compass.llm.providers import get_provider_by_id
    from compass.llm.ladder_policy import get_model_spec

    spec = model_id or get_model_spec()
    provider = get_provider_by_id(spec)
    logger.info("Generator using provider: %s", provider.name)

    return with_retry(
        with_logging(_provider_ask(provider), "generator"), 2,
    )


# ---------------------------------------------------------------------------
# Prompt building (pure)
# ---------------------------------------------------------------------------


def build_system_prompt(
    ctx: GenerationContext,
    types_source: str,
    *,
    role: str = "You are an expert at generating structured artifacts.",
    contract_preamble: str = (
        "Your output must conform to the Spec type below."
    ),
) -> str:
    """Assemble the system prompt from context.

    The types_source IS the contract. The model reads it verbatim.
    Domain context is injected via ctx.domain_context sections.
    """
    sections = [
        role,
        "",
        "## Output Contract (Python types)",
        "",
        contract_preamble,
        "",
        types_source,
    ]

    for ds in ctx.domain_context:
        if ds.content and not ds.content.startswith("No "):
            sections.extend(["", f"## {ds.heading}", "", ds.content])

    return "\n".join(sections)


def build_user_message(
    ctx: GenerationContext,
    *,
    focus: str | None = None,
    suffix_lines: tuple[str, ...] = (
        "Respond with the Spec type defined in the Output Contract.",
        "See its docstring for the response format.",
        "No markdown fencing, no explanation.",
    ),
) -> str:
    """Assemble the user message from context.

    Primary content comes from ctx.user_prompt, ctx.default_task,
    or a minimal fallback.
    """
    primary = (
        ctx.user_prompt if ctx.user_prompt is not None else
        ctx.default_task if ctx.default_task is not None else
        "Generate an artifact."
    )
    parts = [primary]

    parts.extend(["", *suffix_lines])

    if ctx.available_packages:
        parts.extend(["", f"Available packages: {ctx.available_packages}"])

    if focus:
        parts.extend(["", f"FOCUS AREA: Emphasise '{focus}' patterns."])

    if ctx.feedback:
        parts.extend(["", "Your previous attempt had errors:", ""])
        for fb in ctx.feedback:
            parts.append(f"  {fb}")
        parts.extend(["", "Please fix these issues in your next attempt."])

    return "\n".join(parts)
