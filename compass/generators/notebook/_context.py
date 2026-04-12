"""Context builder for the notebook generator.

The model sees:
1. Available Python packages
2. Notebook best practices
3. Matplotlib non-interactive guidance
"""

from __future__ import annotations

from compass.generators._types import DomainSection, GenerationContext


_NOTEBOOK_PRINCIPLES = """\
Notebook authoring principles:

- Structure: Title (H1 markdown) -> imports code cell -> sections with markdown + code
- Imports: All imports in the first code cell. Use standard aliases:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
- Matplotlib: Always call matplotlib.use('Agg') BEFORE importing pyplot.
  Save figures with plt.savefig('output.png') instead of plt.show().
  Call plt.close() or plt.clf() after saving to free memory.
- Self-contained: Each code cell should work given prior cells' namespace.
- No user interaction: No input(), no blocking calls, no GUI windows.
- Explanatory: Markdown cells explain what the next code cell does.
- Concise: Prefer clear, direct code over verbose boilerplate.
- Error-free: All code must execute without exceptions.

Example matplotlib pattern:

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label='sin(x)')
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.savefig('sine_wave.png', dpi=100)
plt.close()
print('Figure saved to sine_wave.png')
"""


def _discover_available_packages() -> str:
    """List installed Python packages relevant to notebooks."""
    try:
        from importlib.metadata import distributions
        pkgs = sorted(
            {d.metadata["Name"] for d in distributions() if d.metadata["Name"]},
            key=str.lower,
        )
        return ", ".join(pkgs)
    except Exception:
        return "numpy, matplotlib, pandas"


def build_notebook_context(
    prompt: str | None = None,
) -> GenerationContext:
    """Build context for the notebook generator."""
    return GenerationContext(
        domain_context=(
            DomainSection(
                "Notebook Authoring Principles",
                _NOTEBOOK_PRINCIPLES,
            ),
        ),
        available_packages=_discover_available_packages(),
        user_prompt=prompt,
        default_task=(
            "Generate a well-structured Jupyter notebook that demonstrates "
            "a data science or Python programming concept."
        ),
    )
