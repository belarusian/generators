"""
search action - singledispatch handlers.

Registers handlers for SearchAction.

Unified structural + semantic code search.
Find code by name, concept, or description.
Combines AST-based index (exact matches) with RAG embeddings (semantic).
"""

import json
from typing import Dict, List, Optional, Set, Tuple

from compass.core.content import preview_head_tail
from compass.agents.neo.types import SearchAction, ActionTarget, ExecutionContext, Reflector
from compass.agents.neo.dispatch import display, validate, execute, extract_learnings, action_key, hint, display_name
from compass.agents.neo.memory import Learning


# =============================================================================
# Pure search functions
# =============================================================================

def _search_structural(
    project_path: str,
    query: str,
    search_type: str,
    max_results: int = 15,
    codebase_index=None,
) -> Tuple[List[Dict], Set[Tuple[str, int]]]:
    """
    Search codebase using AST-based index (exact/structural matches).

    Uses the session's cached index when available, falls back to
    rebuilding only if none was provided.

    Returns:
        Tuple of (list of match dicts, set of (file, line) tuples for dedup)
    """
    seen: Set[Tuple[str, int]] = set()
    matches = []

    try:
        if codebase_index is None:
            from compass.agents.neo.index import index_codebase
            codebase_index = index_codebase(project_path)

        results = codebase_index.search(query, search_type=search_type, max_results=max_results)

        for match in results.get("matches", []):
            file = match.get("file", "?")
            line = match.get("line", 0)
            seen.add((file, line))
            matches.append(match)
    except Exception:
        pass  # Structural search is optional

    return matches, seen


def _search_semantic(
    project_path: str,
    query: str,
    seen_locations: Set[Tuple[str, int]],
    top_k: int = 10,
) -> str:
    """
    Search codebase using RAG embeddings (semantic matches).

    Pure function: project_path, query, seen_locations -> formatted context

    Deduplicates against seen_locations from structural search.
    Returns full context with signatures and docstrings.

    Returns:
        Formatted string with code snippets (not just locations)
    """
    from compass.agents.neo.rag import get_embedder, CodeRetriever

    try:
        embedder = get_embedder(project_path)

        if embedder.embeddings is None:
            return ""

        retriever = CodeRetriever(embedder)
        results = retriever.query(query, top_k=top_k)

        if not results:
            return ""

        # Filter out already-seen locations and format with full context
        chunks = []
        for chunk, score in results:
            loc = (chunk.file, chunk.line)
            if loc not in seen_locations:
                seen_locations.add(loc)
                content_preview = _format_chunk_content(chunk, score)
                chunks.append(content_preview)

        return "\n\n".join(chunks)

    except Exception:
        return ""  # Semantic search is optional


def _format_chunk_content(chunk, score: float) -> str:
    """Format a code chunk with full content."""
    from compass.core.content import preview_head_tail

    header = f"# {chunk.file}:{chunk.line} ({score:.2f}) - {chunk.type}"
    content = preview_head_tail(chunk.content, max_lines=20, label="search")
    return header + "\n" + content


def _format_structural_match(match: Dict, search_type: str) -> str:
    """Format a single structural match for display."""
    file = match.get("file", "?")
    line = match.get("line", "?")

    if search_type == "file":
        return f"  {file} ({match.get('lines', '?')}L)"
    elif search_type in ("function", "class"):
        name = match.get("name", "?")
        result = f"  {name} - {file}:{line}"
        if match.get("methods"):
            result += f"\n    methods: {', '.join(match['methods'][:5])}"
        if match.get("args"):
            result += f"\n    args: {', '.join(match['args'][:5])}"
        return result
    else:  # content
        content = match.get("content", "")[:80]
        return f"  {file}:{line}: {content}"


def _unified_search(
    project_path: str,
    query: str,
    search_type: str = "content",
    codebase_index=None,
) -> str:
    """
    Unified search combining structural AND semantic search.

    Uses the session's cached index when available.

    Returns formatted string with sections for each search type,
    deduplicated to avoid showing the same location twice.
    """
    lines = []

    # --- Structural search (AST index) ---
    structural_matches, seen = _search_structural(
        project_path, query, search_type, codebase_index=codebase_index,
    )

    if structural_matches:
        lines.append(f"=== Structural matches ({len(structural_matches)}) ===")
        for match in structural_matches:
            lines.append(_format_structural_match(match, search_type))
        lines.append("")

    # --- Semantic search (RAG embeddings) ---
    semantic_context = _search_semantic(project_path, query, seen)

    if semantic_context:
        lines.append("=== Semantic matches ===")
        lines.append(semantic_context)
        lines.append("")

    if not lines:
        return f"No matches found for '{query}'"

    return "\n".join(lines)


# =============================================================================
# Singledispatch handlers
# =============================================================================

@display.register(SearchAction)
def _(action: SearchAction) -> ActionTarget:
    """Get display info for search."""
    search_type = action.search_type or "content"
    return ActionTarget(
        target=action.query,
        display=f"search({search_type}): {action.query}",
        content=None,
    )


@validate.register(SearchAction)
def _(
    action: SearchAction,
    project_path: str = ".",
    files_read: Optional[Dict] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate search action.

    Returns (is_valid, error_message).

    Required fields:
    - query: What to search for (name, concept, description)

    Optional fields:
    - search_type: "content" (default), "function", "class", "file"

    Search finds code semantically - by meaning, not just text match.
    Use search when you need to find:
    - Functions or classes by name
    - Code related to a concept
    - Files implementing a feature

    For exact text/pattern matching, use grep instead.
    If search returns no results, try different query terms or use grep.
    """
    query = action.query

    if not query:
        return False, "Missing required field: query"

    return True, None


@execute.register(SearchAction)
def _(action: SearchAction, project_path: str, ctx: ExecutionContext = None) -> Tuple[bool, str]:
    """
    Execute search action using unified structural + semantic search.

    Returns (success, message).

    Combines AST-based index (exact matches) with RAG embeddings (semantic).
    Results are deduplicated - same location won't appear twice.
    Long results are truncated with clean formatting.
    """
    query = action.query or ""
    search_type = action.search_type or "content"
    index = ctx.codebase_index if ctx else None

    try:
        result = _unified_search(project_path, query, search_type, codebase_index=index)
        return True, result
    except Exception as e:
        return False, f"Search failed: {e}"


@extract_learnings.register(SearchAction)
def _(
    action: SearchAction,
    success: bool,
    result: str,
    reflect: Reflector,
) -> List[Learning]:
    """Extract learnings from search action."""
    from dataclasses import asdict

    action_data = asdict(action) if hasattr(action, '__dataclass_fields__') else action

    prompt = f"""Action: search
Input: {json.dumps(action_data)}
Success: {success}
Result:
{preview_head_tail(result, max_lines=64)}

What did we learn from this?"""

    return [reflect(prompt)]


@action_key.register(SearchAction)
def _(action: SearchAction) -> tuple:
    """Hashable key for search comparison."""
    return ("search", action.query, action.search_type)


@hint.register(SearchAction)
def _(action: SearchAction) -> str:
    """Hint for Critic when search fails."""
    return "Semantic search. Try different query terms."


@display_name.register(SearchAction)
def _(action: SearchAction) -> str:
    """Human-friendly name for UI."""
    return "Search"
