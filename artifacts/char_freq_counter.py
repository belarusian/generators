from compass.generators._types import Ok
from compass.generators.trinity._types import Fact

def run(step, resolved_inputs, workspace):
    path = resolved_inputs.get('path', '')
    target_char = resolved_inputs.get('target_char', '')
    with open(path) as f:
        content = f.read()
    count = content.count(target_char)
    return Ok(Fact(
        step_id=step.step_id,
        name=step.expected_fact,
        value=str(count),
        fact_type='numeric',
    ))
