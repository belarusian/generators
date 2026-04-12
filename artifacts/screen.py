"""Screen skill artifact -- runs a neo-lab skill as a Trinity step.

Plan guide: inputs must include {"skill": "skill_name"}.
Optionally include {"expect": "text to validate"} for post-skill validation.
The skill YAML is loaded from neo-lab/skills/ and replayed.
Each step in the skill validates via OCR/DINO.

This is the bridge between neo-lab (teach-by-demonstration) and
Trinity (typed plan execution). A skill is G . V . G . V -- alternating
actions and validations. The screen is the artifact. OCR is exec().

Example Trinity step:
    Step(
        step_id="s1",
        description="log into Chase",
        artifact_type="screen",
        inputs={"skill": "chase_login"},
        expected_fact="authenticated",
    )
"""

import os
import sys
from pathlib import Path

CYCLE_BREAKING = True  # screen state is non-deterministic

# Companion neo-lab workspace for taught skills/state.
NEO_LAB = Path(
    os.environ.get(
        "COMPASS_NEO_LAB",
        str(Path.home() / ".compass" / "neo-lab"),
    )
)


def run(step, resolved_inputs, workspace):
    """Trinity artifact contract: run(step, resolved_inputs, workspace) -> Result."""
    from compass.generators._types import Ok, Err
    from compass.generators.trinity._types import Fact

    skill_name = resolved_inputs.get("skill", "") or getattr(step, "artifact_ref", "") or ""
    if not skill_name:
        return Err(f"step '{step.step_id}': screen artifact requires a 'skill' input")

    # Add neo-lab to path so we can import the skill module
    neo_lab_str = str(NEO_LAB)
    if neo_lab_str not in sys.path:
        sys.path.insert(0, neo_lab_str)

    try:
        from neo.skill import replay
    except ImportError:
        return Err(
            f"step '{step.step_id}': cannot import neo.skill. "
            f"Ensure neo-lab exists at {NEO_LAB} with .venv activated "
            f"or neo/ on PYTHONPATH"
        )

    try:
        success, facts = replay(skill_name, pause=0.2)
    except FileNotFoundError:
        return Err(f"step '{step.step_id}': skill '{skill_name}' not found in {NEO_LAB}/skills/")
    except Exception as e:
        return Err(f"step '{step.step_id}': skill '{skill_name}' failed: {e}")

    if not success:
        return Err(f"step '{step.step_id}': skill '{skill_name}' did not complete successfully")

    # Post-skill validation if requested
    expect = resolved_inputs.get("expect", "")
    if expect:
        from neo.skill import validate_screen
        valid, found = validate_screen(expect, timeout=10)
        if not valid:
            return Err(
                f"step '{step.step_id}': skill completed but "
                f"expected text '{expect}' not found on screen"
            )

    # Return facts as the step result
    fact_summary = ", ".join(f"{k}={v}" for k, v in facts.items()) if facts else "completed"
    return Ok(Fact(
        step_id=step.step_id,
        name=step.expected_fact or "screen_result",
        value=fact_summary,
        fact_type="text",
    ))
