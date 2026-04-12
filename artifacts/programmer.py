"""Programmer NFA -- structured code generation with solution docs and chunks.

Plan guide: inputs must include {"problem": "what to build"}.
Generates code, writes files to workspace. Returns structured JSON fact.
"""

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path

CYCLE_BREAKING = True


def run(step, resolved_inputs, workspace):
    """Trinity artifact contract: run(step, resolved_inputs, workspace).

    Bridges to call_programmer() -- the Programmer NFA that produces
    solution docs, code chunks, and applies them to the workspace.
    """
    from compass.generators._types import Ok, Err
    from compass.generators.trinity._types import Fact
    from compass.llm.oracle import Oracle
    from compass.agents.programmer.tool import (
        call_programmer,
        create_pattern_fetcher,
        create_file_structure_getter,
        create_coding_standards_getter,
    )

    workspace = Path(workspace)

    problem = resolved_inputs.get("problem", "") or getattr(step, "artifact_ref", "") or ""
    if not problem:
        # Fall back to step description -- the planner puts the task there
        problem = getattr(step, "description", "") or ""
    if not problem:
        return Err(f"step '{step.step_id}': programmer requires a 'problem' input")

    constraints = resolved_inputs.get("constraints", [])
    if isinstance(constraints, str):
        constraints = [c.strip() for c in constraints.split(",") if c.strip()]

    parent_feedback = resolved_inputs.get("feedback", None)

    oracle = Oracle()

    # Duck-type workspace ref for pattern fetcher (avoids importing CodeMemory)
    class _WsRef:
        def __init__(self, path):
            self.project_path = str(path)

    ws_ref = _WsRef(workspace)
    fetch_pattern = create_pattern_fetcher(oracle, ws_ref)
    get_file_structure = create_file_structure_getter(ws_ref)
    get_coding_standards = create_coding_standards_getter()

    def _apply_chunks(chunks):
        """Write chunks to workspace."""
        applied, failed = [], []
        for chunk in chunks:
            target = getattr(chunk, "target", None) or (chunk.get("target") if isinstance(chunk, dict) else "")
            content = getattr(chunk, "content", None) or (chunk.get("content") if isinstance(chunk, dict) else "")
            if not target or not content:
                failed.append("missing target or content")
                continue
            out = workspace / target
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content)
            applied.append(str(target))
        msg = f"Applied {len(applied)} chunks"
        if failed:
            msg += f", {len(failed)} failed: {failed}"
        return len(failed) == 0, msg

    try:
        result = call_programmer(
            oracle=oracle,
            problem=problem,
            constraints=constraints,
            fetch_pattern=fetch_pattern,
            get_file_structure=get_file_structure,
            get_coding_standards=get_coding_standards,
            apply_chunks=_apply_chunks,
            parent_feedback=parent_feedback,
        )
    except Exception as exc:
        return Err(
            f"step '{step.step_id}': programmer error: "
            f"{type(exc).__name__}: {exc}"
        )

    if not result.success:
        return Err(
            f"step '{step.step_id}': programmer failed: {result.error or 'unknown'}"
        )

    step_id = getattr(step, "step_id", "programmer")
    fact_name = getattr(step, "expected_fact", "code_result")

    def _default(o):
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    fact_value = json.dumps({
        "success": result.success,
        "solution_doc": result.solution_doc,
        "chunks": result.chunks or [],
        "reasoning": result.reasoning,
        "iterations": result.iterations,
    }, default=_default)

    return Ok(Fact(
        step_id=step_id,
        name=fact_name,
        value=fact_value,
        fact_type="json",
    ))
