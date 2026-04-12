"""Context builder for Trinity."""

from __future__ import annotations

import os
from pathlib import Path

from compass.generators._types import DomainSection, GenerationContext


def _neo_saved_skills_section(workspace: Path | None) -> DomainSection | None:
    """If workspace has a skills/ directory (e.g. neo-lab), list YAML skills for Trinity.

    The ``screen`` artifact replays those skills by name; models must not invent names.
    """
    if workspace is None:
        return None
    skills_dir = workspace / "skills"
    if not skills_dir.is_dir():
        return None
    names: list[str] = []
    for p in sorted(skills_dir.glob("*.yaml")):
        names.append(p.stem)
    body = [
        "When using the `screen` artifact with `inputs`: {\"skill\": \"...\"}, the name "
        "must be exactly one of the saved skills below (files in skills/*.yaml). "
        "Do not invent skill names (e.g. click_search_field). "
        "If the list is empty or no skill fits, do not use `screen` with a fictional "
        "skill — use inline_python (neo.screen/actions), user_query, action_invoker, "
        "or other discovered artifacts instead.",
        "",
    ]
    if names:
        body.append("Saved skill names in this workspace:")
        body.extend(f"  - {n}" for n in names)
    else:
        body.append("No saved skills yet (skills/*.yaml is empty). Do not use the screen "
                    "artifact with a skill input until the user has taught one.")
    return DomainSection(
        heading="Neo-lab saved skills (screen artifact)",
        content="\n".join(body),
    )


def _dreams_dir() -> Path:
    """~/.compass/dreams/ -- shared Trinity dream storage."""
    return Path(os.path.expanduser("~")) / ".compass" / "dreams"

def _discover_available_packages() -> str:
    """List installed Python packages."""
    try:
        from importlib.metadata import distributions
        pkgs = sorted(
            {d.metadata["Name"] for d in distributions() if d.metadata["Name"]},
            key=str.lower,
        )
        return ", ".join(pkgs)
    except Exception:
        return ""


_ROUTING_AND_META_QUESTIONS = """\
Routing (same request as any other — no extra model call):

Users may arrive from Neo learn.py. If they did not prefix ``trinity``, learn.py may
send UI phrasing to a small YAML action generator when a browser capture target is
set — that is separate from you. **You (Trinity) are still allowed to plan GUI and
screen work** using artifacts and inline_python (see "Screen and GUI via Trinity").

They may ask purely informational questions (e.g. whether you can see the screen,
how Neo vs Trinity differs, what commands to use).

- For those meta-only questions: do NOT plan git, shell, or repo-mutation steps
  unless the user also asked for concrete work. Answer with a minimal Spec —
  typically one inline_python step whose code assigns ``result`` to a short
  explanatory string, or the smallest artifact plan that answers honestly.
- For real tasks (code, tests, files, git, **or screen/GUI**): use the right
  artifacts and steps as usual.
"""

_SCREEN_AND_GUI_VIA_TRINITY = """\
**Screen and GUI via Trinity**

Trinity is not limited to code, git, or repo files. You may plan steps that capture
the screen, drive applications, or invoke neo-lab skills when the task requires it.

Mechanisms (prefer items that appear under **Discovered Artifacts**):
- **Dynamic artifacts** (e.g. ``screen``, ``user_query``, ``action_invoker``): set
  ``artifact_type`` to the module stem and supply ``inputs`` per that file's Plan guide.
  The ``screen`` artifact replays **saved** YAML skills by name — see "Neo-lab saved
  skills" when that section is present; do not invent skill names.
- **Screen capture inside Trinity** (no conflict with ``learn.py`` routing — these are
  steps inside a Spec):
  - **vision** step with ``artifact_ref="__TRINITY_SCREEN_CAPTURE__"`` — grabs the
    current display at step execution (neo-lab ``screen.capture()`` or pyautogui),
    then sends the image to ``VISION_MODEL``. Use ``inputs``: ``{"prompt": "..."}``.
  - **action_invoker** with ``inputs["action"]``: ``{"type": "ScreenshotAction", "region": "full"}``
    (or ``ClickAction``, ``TypeAction``, …) — uses the same Neo computer-use stack as
    ``compass.agents.neo.actions.computer``.
  - **inline_python**: when the workspace is neo-lab on ``PYTHONPATH``, ``from neo import screen, actions``
    and call ``screen.capture()``, OCR helpers, ``actions.click``, etc. — same primitives as ``learn.py``.

Do not refuse screen interaction for "Trinity only does code" — use the mechanisms
above. Prefer small, verifiable steps (observe → act → validate).

**Foreground vs background windows:** Screen capture (neo ``screen``, computer actions,
or vision with ``artifact_ref`` = ``__TRINITY_SCREEN_CAPTURE__``) sees the **active**
window or the neo capture target — not every stacked Safari window. Plans should
activate the correct tab/window first (navigate, click, or shell/osascript) before
assuming the viewport shows a specific site.
"""

_PLAN_PRINCIPLES = """\
Plan construction principles for Trinity:

- Each step should be atomic: one artifact, one fact.
- Steps form a DAG via depends_on. No cycles.
- Prefer inline_python for computation -- it's the most reliable.
- Use 'auto' for discovered artifacts -- Trinity will inspect their
  signatures and map your inputs to parameters automatically.
- Use 'module' only when you know the exact dotted module path.
- Use 'shell' only for system commands.
- extraction_expr is evaluated in the step's namespace after execution.
  For inline_python, it's a variable name from the executed code.
  For auto/module, 'result' holds the return value of the entry function.
- inputs can reference prior facts via {"$fact": "fact_name"}.
- The synthesis field should describe how facts combine to answer the question.
- Use expected_type to declare the Python type a step produces.
  "str" for text, "dict" for structure, "any" (default) for legacy.
  Downstream steps get that type via {"$fact": "name"}.
- Plans should be minimal: fewest steps that answer the question.
- Each step's artifact_ref (for inline_python) must be complete, runnable code.
  Do not assume imports or variables from other steps unless passed via inputs.
- When using discovered artifacts, match your input keys to the parameter
  names shown in the artifact listing. Trinity maps by name.
- **vision** steps: ``artifact_ref`` must be an existing image path, **or** the
  sentinel ``__TRINITY_SCREEN_CAPTURE__`` to grab the current screen at execution
  time (no PNG file required beforehand). Do not invent paths like
  ``artifacts/screenshot_foo.png`` unless a prior step writes that file.
"""


def build_trinity_context(
    prompt: str | None = None,
    workspace: Path | None = None,
) -> GenerationContext:
    """Build context for Trinity.

    The model needs to understand the plan contract and see what
    artifacts are available to construct an execution plan.
    Artifact discovery uses signature inspection, not hard-coded lists.
    """
    sections: list[DomainSection] = []

    sections.append(DomainSection(
        heading="Routing and informational questions",
        content=_ROUTING_AND_META_QUESTIONS,
    ))

    sections.append(DomainSection(
        heading="Screen and GUI via Trinity",
        content=_SCREEN_AND_GUI_VIA_TRINITY,
    ))

    # Discover artifacts with full signature information
    from compass.generators.trinity._runtime import discover_artifacts, format_artifacts_for_context

    artifacts = discover_artifacts(workspace)
    artifact_info = format_artifacts_for_context(artifacts)
    if artifact_info and not artifact_info.startswith("No "):
        sections.append(DomainSection(
            heading="Discovered Artifacts",
            content=artifact_info,
        ))

    neo_skills = _neo_saved_skills_section(workspace)
    if neo_skills is not None:
        sections.append(neo_skills)

    # Plan principles
    sections.append(DomainSection(
        heading="Plan Construction Principles",
        content=_PLAN_PRINCIPLES,
    ))

    # Load relevant dreams -- past experience as context
    if prompt:
        try:
            from compass.generators._transcript import DreamStore
            dreams = DreamStore(_dreams_dir())
            for section in dreams.search_and_inject(prompt, top_k=3, threshold=0.3):
                sections.append(section)
        except Exception:
            pass

    return GenerationContext(
        domain_context=tuple(sections),
        available_packages=_discover_available_packages(),
        user_prompt=prompt,
    )
