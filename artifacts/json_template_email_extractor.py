"""Reusable artifact: Define JSON template and extract emails using regex"""

from compass.generators._types import Ok
from compass.generators.trinity._types import Fact

_CODE = 'import re\nimport json\n\n# 1. Define a multi-line JSON template with nested double quotes and backslash-n literals\ntemplate = \'\'\'{\n    "greeting": "Hello\\\\nWorld",\n    "contact": {\n        "email": "user@example.com",\n        "note": "Send to \\\\\\"alice@example.com\\\\\\" or bob.smith@corp.co.uk"\n    }\n}\'\'\'\n\n# 2. Regex pattern to extract email addresses from text\npattern = r\'\\\\b[\\\\w.]+@[\\\\w.]+\\\\b\'\n\n# 3. Test text and extraction\ntext = \'Contact alice@example.com or bob.smith@corp.co.uk for info\'\nemails = re.findall(pattern, text)\n\n# 4. Build result dict\nresult = {\n    "template": template,\n    "emails": emails\n}'

_EXTRACTION = 'result'


def run(step, resolved_inputs, workspace):
    ns = dict(resolved_inputs)
    ns["inputs"] = dict(resolved_inputs)
    ns["workspace"] = workspace
    ns["__builtins__"] = __builtins__
    exec(_CODE, ns)
    value = eval(_EXTRACTION, ns)
    return Ok(Fact(
        step_id=step.step_id,
        name=step.expected_fact,
        value=str(value),
        fact_type="text",
    ))
