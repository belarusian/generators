"""
Code Chunks - Parse real Python code with chunk markers.

The model writes actual code with lightweight markers:

    # === chunk: target="math.py", operation=create ===
    def add(a, b):
        return a + b
    # === end ===

No string serialization. No escaping. Pure code.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ChunkOperation(Enum):
    """Type of chunk operation."""
    CREATE = "create"
    REPLACE = "replace"
    APPEND = "append"
    INSERT = "insert"


@dataclass
class CodeChunk:
    """A parsed code chunk."""
    id: str
    target: str
    operation: ChunkOperation
    content: str
    insert_after: Optional[str] = None
    reasoning: Optional[str] = None


# Regex patterns for chunk markers
CHUNK_START = re.compile(
    r'^#\s*===\s*chunk:\s*(.+?)\s*===\s*$',
    re.MULTILINE
)
CHUNK_END = re.compile(
    r'^#\s*===\s*end\s*===\s*$',
    re.MULTILINE
)

# Pattern for parsing chunk metadata
METADATA_PATTERNS = {
    'id': re.compile(r'id\s*=\s*["\']([^"\']+)["\']'),
    'target': re.compile(r'target\s*=\s*["\']([^"\']+)["\']'),
    'operation': re.compile(r'operation\s*=\s*(\w+)'),
    'insert_after': re.compile(r'insert_after\s*=\s*["\']([^"\']+)["\']'),
    'after': re.compile(r'after\s*=\s*["\']([^"\']+)["\']'),  # alias
    'reasoning': re.compile(r'reasoning\s*=\s*["\']([^"\']+)["\']'),
}


def parse_chunk_metadata(metadata_str: str) -> dict:
    """Parse chunk metadata from marker string."""
    result = {}

    for key, pattern in METADATA_PATTERNS.items():
        match = pattern.search(metadata_str)
        if match:
            result[key] = match.group(1)

    # Handle 'after' as alias for 'insert_after'
    if 'after' in result and 'insert_after' not in result:
        result['insert_after'] = result.pop('after')

    return result


def parse_code_chunks(code: str) -> List[CodeChunk]:
    """
    Parse code with chunk markers into structured chunks.

    Args:
        code: Python code with chunk markers

    Returns:
        List of CodeChunk objects
    """
    chunks = []
    chunk_id = 0

    # Find all chunk start markers
    starts = list(CHUNK_START.finditer(code))

    for i, start_match in enumerate(starts):
        metadata_str = start_match.group(1)
        metadata = parse_chunk_metadata(metadata_str)

        # Find content between start and end
        start_pos = start_match.end()

        # Look for end marker
        end_match = CHUNK_END.search(code, start_pos)

        if end_match:
            content = code[start_pos:end_match.start()]
        else:
            # No end marker - take until next chunk or end of file
            if i + 1 < len(starts):
                content = code[start_pos:starts[i + 1].start()]
            else:
                content = code[start_pos:]

        # Clean up content - strip leading/trailing blank lines
        content = content.strip('\n')

        # Parse operation
        op_str = metadata.get('operation', 'create').lower()
        try:
            operation = ChunkOperation(op_str)
        except ValueError:
            operation = ChunkOperation.CREATE

        # Generate ID if not provided
        chunk_id += 1
        chunk_id_str = metadata.get('id', str(chunk_id))

        chunk = CodeChunk(
            id=chunk_id_str,
            target=metadata.get('target', 'unknown.py'),
            operation=operation,
            content=content,
            insert_after=metadata.get('insert_after'),
            reasoning=metadata.get('reasoning'),
        )
        chunks.append(chunk)

    return chunks


def format_chunks_for_prompt(chunks: List[CodeChunk]) -> str:
    """Format chunks for display in prompts."""
    if not chunks:
        return "(no chunks)"

    lines = []
    for chunk in chunks:
        lines.append(f"[{chunk.id}] {chunk.operation.value} -> {chunk.target}")
        lines.append(chunk.content)
        lines.append("")

    return "\n".join(lines)


def chunks_to_dicts(chunks: List[CodeChunk]) -> List[dict]:
    """Convert CodeChunks to dicts for backward compatibility."""
    return [
        {
            "id": c.id,
            "target": c.target,
            "operation": c.operation.value,
            "content": c.content,
            "insert_after": c.insert_after,
            "reasoning": c.reasoning,
        }
        for c in chunks
    ]
