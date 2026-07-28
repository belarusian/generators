"""
Capability persistence for code mode.

Saves successful solutions to .compass/capabilities/ for future RAG discovery.
"""

import hashlib
import re
from pathlib import Path
from typing import Tuple

from compass.agents.neo.memory import CodeMemory
from compass.core.reasoning import debug


def persist_capability(memory: CodeMemory, request: str) -> Tuple[bool, str]:
    """
    Persist the last successful plan's code as a reusable capability.

    Extracts actual code from exec/write_file actions and saves
    to .compass/capabilities/ so RAG can find it later.

    Args:
        memory: CodeMemory with plans and actions
        request: Original request (for naming)

    Returns:
        (success, message)
    """
    if not memory.project_path:
        return False, "No project path set"

    # Find the last successful plan
    successful_plans = [p for p in memory.plans if p.status == "executed"]
    if not successful_plans:
        return False, "No successfully executed plans to persist"

    last_plan = successful_plans[-1]
    original_request = last_plan.summary or "capability"

    # Extract actual code from recent actions
    exec_code = []
    written_files = {}  # path -> content

    for action in reversed(memory.actions):
        if not action.success:
            continue

        if action.action_type == "exec" and action.content:
            exec_code.insert(0, action.content)
        elif action.action_type == "write_file" and action.content:
            if action.target not in written_files:
                written_files[action.target] = action.content

        if len(exec_code) + len(written_files) >= 10:
            break

    if not exec_code and not written_files:
        return False, "No code found in recent actions"

    # Create capabilities directory
    capabilities_dir = Path(memory.project_path) / ".compass" / "capabilities"
    capabilities_dir.mkdir(parents=True, exist_ok=True)

    # Generate a name from the request
    safe_name = re.sub(r'[^a-z0-9]+', '_', original_request.lower())[:50].strip('_')
    if not safe_name:
        safe_name = hashlib.md5(original_request.encode()).hexdigest()[:8]

    capability_file = capabilities_dir / f"{safe_name}.py"

    # Build the capability content
    content_lines = [
        '"""',
        f'Capability: {original_request}',
        '',
        'Auto-persisted from successful plan execution.',
        'This file is indexed by RAG for future discovery.',
        '"""',
        '',
    ]

    if exec_code:
        content_lines.append('# === Executed Code ===')
        content_lines.append('')
        for code in exec_code:
            content_lines.extend(code.strip().split('\n'))
            content_lines.append('')

    if written_files:
        content_lines.append('# === Files Written ===')
        for path, content in written_files.items():
            content_lines.append(f'# File: {path}')
            file_lines = content.strip().split('\n')[:50]
            for line in file_lines:
                content_lines.append(line)
            if len(content.strip().split('\n')) > 50:
                content_lines.append('# ... (truncated)')
            content_lines.append('')

    capability_file.write_text('\n'.join(content_lines) + '\n')

    # Write the actual files that were written during the plan
    for path, content in written_files.items():
        file_path = Path(memory.project_path) / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    # Trigger RAG re-index after successful file writes
    try:
        from compass.agents.neo.rag import get_embedder
        embedder = get_embedder(str(memory.project_path))
        count = embedder.build_index(force=False)
        debug(f"RAG re-indexed, {count} chunks (including {capability_file.name})")
    except Exception as e:
        debug(f"RAG re-index failed (capability still saved): {e}")

    return True, f"Saved capability to {capability_file.relative_to(memory.project_path)}"
