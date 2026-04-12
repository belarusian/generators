"""State check artifact -- identifies current screen state via neo-lab.

Plan guide: optionally include {"state": "state_name"} to check for a
specific state. Without it, returns whatever state is detected.

Uses the neo-lab state graph -- registered states with OCR text markers
and DINO visual markers. Neo screenshots, runs detection, matches against
known state signatures.

This is the perception counterpart to screen.py (which executes skills).
screen.py = action (G), state.py = perception (V on the environment).

Example Trinity steps:
    Step(
        step_id="s1",
        description="verify we're on the Amex dashboard",
        artifact_type="state",
        inputs={"state": "amex_dashboard"},
        expected_fact="at_dashboard",
    )

    Step(
        step_id="s0",
        description="identify current screen",
        artifact_type="state",
        inputs={},
        expected_fact="current_state",
    )
"""

import os
import sys
from pathlib import Path

CYCLE_BREAKING = True  # screen state is non-deterministic

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

    neo_lab_str = str(NEO_LAB)
    if neo_lab_str not in sys.path:
        sys.path.insert(0, neo_lab_str)

    try:
        from neo.state_graph import where_am_i, identify
    except ImportError:
        return Err(
            f"step '{step.step_id}': cannot import neo.state_graph. "
            f"Ensure neo-lab exists at {NEO_LAB}"
        )

    target_state = resolved_inputs.get("state", "")

    if target_state:
        # Specific check: am I at this state?
        current, conf = where_am_i()
        if current == target_state:
            return Ok(Fact(
                step_id=step.step_id,
                name=step.expected_fact or "state_check",
                value=f"{target_state} ({conf:.0%})",
                fact_type="text",
            ))
        elif current:
            return Err(
                f"step '{step.step_id}': expected state '{target_state}' "
                f"but at '{current}' ({conf:.0%})"
            )
        else:
            return Err(
                f"step '{step.step_id}': expected state '{target_state}' "
                f"but cannot identify current screen"
            )
    else:
        # General query: what state am I in?
        matches = identify()
        if not matches:
            return Ok(Fact(
                step_id=step.step_id,
                name=step.expected_fact or "current_state",
                value="unknown",
                fact_type="text",
            ))

        best_name, best_ratio = matches[0][0], matches[0][1]
        summary = ", ".join(f"{m[0]}={m[1]:.0%}" for m in matches[:3])
        return Ok(Fact(
            step_id=step.step_id,
            name=step.expected_fact or "current_state",
            value=f"{best_name} ({best_ratio:.0%})" if best_ratio >= 0.5 else f"uncertain: {summary}",
            fact_type="text",
        ))
