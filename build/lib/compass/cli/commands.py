"""
Command handlers for code mode.

Slash commands like /help, /session, /think, etc.
Each handler receives (memory, project_path, **kwargs) and returns
None to continue or "break" to exit.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from compass.llm.oracle import Oracle
from compass.agents.neo.rag import get_embedder


@dataclass
class ValidationResponse:
    """Response type for /validate command."""
    validation: str

# Dynamic command registry for code mode
# Format: {"/command": {"handler": func, "help": "description"}}
# Handler signature: func(memory, project_path, **kwargs) -> str|None
CODE_COMMANDS: Dict[str, Dict[str, Any]] = {}

# State for /validate command
_last_answer: Optional[str] = None
_last_action_results: list = []

# State for /ask command (no longer needed - response is just input)

# Output visibility toggles
_show_critic: bool = True  # Show critic output

# Manual reasoning depth control
# None = auto, "off"/"low"/"medium"/"high" = override
_think_level_override: Optional[str] = None

# Original model spec before /model switched it (for /model off restore)
_original_model_spec: Optional[str] = None


def register_command(name: str, handler: Callable, help_text: str) -> None:
    """Register a slash command for code mode."""
    CODE_COMMANDS[name] = {"handler": handler, "help": help_text}


def unregister_command(name: str) -> None:
    """Remove a slash command."""
    CODE_COMMANDS.pop(name, None)


def get_show_thinking() -> bool:
    """True when a non-default think level is active."""
    return _think_level_override is not None


def get_show_critic() -> bool:
    """Get current critic display state."""
    return _show_critic


def get_think_level_override() -> Optional[str]:
    """Get manual think level override (off/low/medium/high) or None for auto.
    Falls back to COMPASS_THINK_LEVEL env var for test/CI use."""
    return _think_level_override or os.getenv("COMPASS_THINK_LEVEL") or None


def clear_reasoning_overrides() -> None:
    """Clear think level override after query completes."""
    global _think_level_override
    _think_level_override = None


def set_last_answer(answer: str, action_results: list) -> None:
    """Store last answer for validation."""
    global _last_answer, _last_action_results
    _last_answer = answer
    _last_action_results = action_results


def _cmd_help(memory, project_path, **kwargs) -> Optional[str]:
    """Show available commands."""
    print("\nAvailable commands:")
    print("  /help     - Show this help")
    print("  /session  - Show current session info")
    print("  /think    - Think floor: /think [off|low|med|high]")
    from compass.llm.ladder_policy import FAMILIES
    families = "|".join(FAMILIES)
    print(f"  /model    - Switch model: /model [off|{families}|<spec>]")
    print("  /critic   - Toggle critic display")
    print("  /debug    - Toggle debug mode (raw prompts/responses)")
    print("  /last     - Show last action's raw output")
    print("  /validate - Verify last answer against raw data")
    print("  /ask      - Ask question (Claude or Oracle via ASK_PROVIDER)")
    print("  /trace    - Show session trace metrics (actions, NFA states)")
    print("  /remember - Save last execution as reusable capability")
    print("  /new      - Start fresh session (re-indexes codebase)")
    print("  /quit     - Save and exit")
    # Show any dynamically registered commands
    builtin = {
        "/help",
        "/session",
        "/think",
        "/model",
        "/critic",
        "/debug",
        "/last",
        "/validate",
        "/ask",
        "/trace",
        "/new",
        "/quit",
    }
    for cmd, info in CODE_COMMANDS.items():
        if cmd not in builtin:
            print(f"  {cmd:10} - {info['help']}")
    print()
    return None


def _cmd_session(memory, project_path, **kwargs) -> Optional[str]:
    """Show current session info."""
    print(f"\nSession: {memory.session_id}")
    print(f"  Path:    {memory.save()}")
    print(f"  Path:    {memory.save()}")
    print(f"  Project: {memory.project_path}")
    if memory.index_summary:
        s = memory.index_summary
        print(
            f"  Index:   {s.get('file_count', 0)} files, {s.get('function_count', 0)} functions, {s.get('class_count', 0)} classes"
        )
    if memory.plans:
        print(f"  Plans:   {len(memory.plans)} executed")
    if memory.actions:
        print(f"  Actions: {len(memory.actions)} total")
    # Show runtime learnings
    learnings_ctx = memory.get_learnings_context()
    if learnings_ctx:
        print()
        print(learnings_ctx)
    print()
    return None


def _show_model_state() -> None:
    """Display current model. Pure output."""
    from compass.llm.ladder_policy import get_model_spec, FAMILIES, DEFAULT_FAMILY
    model = get_model_spec()
    family_name = os.getenv("COMPASS_FAMILY", DEFAULT_FAMILY).lower()
    family = FAMILIES.get(family_name)
    family_label = family_name if family and model == family.worker else None
    label = f"{family_label} ({model})" if family_label else model
    print(f"\n  model: {label}")
    print()


def _cmd_think(memory, project_path, **kwargs) -> Optional[str]:
    """
    Set floor think level. Retries can escalate above this, never below.
    Qwen can't think -- this is a no-op for qwen models.

    /think              - Show current think floor
    /think off          - Clear think floor (auto)
    /think low|med|high - Set think floor
    """
    global _think_level_override

    level_map = {"low": "low", "med": "medium", "medium": "medium", "high": "high"}
    args = kwargs.get("cmd_args", "").strip().lower()

    if not args:
        level = _think_level_override or "auto"
        print(f"\n  think floor: {level.upper()}")
        print()
        return None

    if args == "off":
        _think_level_override = None
    elif args in level_map:
        _think_level_override = level_map[args]
    else:
        print(f"\n  Unknown level: {args}. Use off|low|med|high.")
        print()
        return None

    level = _think_level_override or "auto"
    print(f"\n  think floor: {level.upper()}")
    print()
    return None


def _cmd_model(memory, project_path, **kwargs) -> Optional[str]:
    """
    Switch model at runtime.

    /model              - Show current model
    /model off          - Restore original model
    /model <family>     - Switch to a family (see FAMILIES in ladder_policy)
    /model <spec>       - Set arbitrary model spec (e.g. anthropic:sonnet)
    """
    global _original_model_spec
    from compass.llm.ladder_policy import get_model_spec, set_model_spec, set_family, FAMILIES

    args = kwargs.get("cmd_args", "").strip()
    parts = args.lower().split() if args else []

    if not parts:
        _show_model_state()
        return None

    if parts[0] == "off":
        if _original_model_spec:
            set_model_spec(_original_model_spec)
            _original_model_spec = None
        _show_model_state()
        return None

    # Save original model for /model off restore
    if _original_model_spec is None:
        _original_model_spec = get_model_spec()

    if parts[0] in FAMILIES:
        set_family(parts[0])
    else:
        set_model_spec(parts[0])

    _show_model_state()
    return None


def _cmd_critic(memory, project_path, **kwargs) -> Optional[str]:
    """Toggle display of critic evaluations."""
    global _show_critic
    _show_critic = not _show_critic
    state = "ON" if _show_critic else "OFF"
    print(f"\nCritic display: {state}")
    print()
    return None


def _cmd_debug(memory, project_path, **kwargs) -> Optional[str]:
    """Toggle debug mode via COMPASS_DEBUG env var."""
    current = os.getenv("COMPASS_DEBUG")
    if current:
        os.environ.pop("COMPASS_DEBUG", None)
        print("\nDebug mode: OFF")
    else:
        os.environ["COMPASS_DEBUG"] = "1"
        print("\nDebug mode: ON")
        print("  Raw oracle prompts/responses will be shown")
        print("  Use /debug again to turn off")
    print()
    return None


def _cmd_last(memory, project_path, **kwargs) -> Optional[str]:
    """Show last action's raw output."""
    if not memory.actions:
        print("\nNo actions executed yet.")
        print()
        return None

    last = memory.actions[-1]
    print(f"\nLast action: {last.action_type}")
    print(f"  Target:  {last.target}")
    print(f"  Success: {last.success}")
    print(f"  Time:    {last.executed_at}")
    if last.reasoning:
        print(f"  Reason:  {last.reasoning}")
    print()
    print("Raw result:")
    print("-" * 60)
    print(last.result or "(no output)")
    print("-" * 60)
    print()
    return None


def _cmd_validate(
    memory, project_path, oracle: Optional[Oracle] = None, **kwargs
) -> Optional[str]:
    """Ask model to verify its last answer against raw data."""
    global _last_answer, _last_action_results

    if not _last_answer:
        print("\nNo answer to validate yet.")
        print()
        return None

    if not oracle:
        print("\nNo oracle available for validation.")
        print()
        return None

    print("\n* Validating answer...")

    # Build validation prompt
    results_text = (
        "\n".join(_last_action_results) if _last_action_results else "(no raw data)"
    )
    prompt = f"""You are a fact-checker. Verify this answer against the raw data.

ANSWER GIVEN:
{_last_answer}

RAW DATA FROM EXECUTION:
{results_text}

Check for:
1. Factual accuracy - does the answer match the raw data?
2. Unit correctness - are units (m/s vs km/h, C vs F, etc.) correct?
3. Missing information - did the answer omit important details?
4. Misinterpretation - did the answer misread any values?

Respond with:
- VALID if the answer is correct
- INVALID with specific errors if problems found

Be precise about what's wrong."""

    # Simple validation - just get text response
    try:
        result = oracle.ask(prompt, ValidationResponse, task="cli:validate")
        validation = result.validation
        print()
        print("Validation result:")
        print("-" * 60)
        print(validation)
        print("-" * 60)
        print()
    except Exception as e:
        print(f"\nValidation failed: {e}")
        print()

    return None


def _cmd_ask(
    memory, project_path, oracle: Optional[Oracle] = None, cmd_args: str = "", **kwargs
) -> Optional[str]:
    """Ask a question - routes based on ASK_PROVIDER config.

    Usage: /ask <your question>

    Routes to:
    - neo (ASK_PROVIDER=neo): Oracle wisdom + Curator context -> Neo plans/acts
    - oracle (ASK_PROVIDER=oracle): Oracle responds directly with prose (terminal)
    - claude (ASK_PROVIDER=claude): Claude sees context, response drives Oracle

    Default: neo (Oracle -> Neo flow).
    """
    from compass.cli import ui
    from compass.agents.neo.rag import get_relevant_context

    if not cmd_args:
        print("\nUsage: /ask <your question>")
        print("Example: /ask What does the FileEditor do?")
        print()
        provider = os.getenv("ASK_PROVIDER", "").lower() or "neo"
        ask_format = os.getenv("ASK_FORMAT", "raw").lower()
        print(f"  ASK_PROVIDER: {provider} (neo|oracle|claude)")
        print(f"    neo:    Oracle wisdom + context -> Neo plans/acts")
        print(f"    oracle: Direct prose response (terminal)")
        print(f"    claude: Claude drives Oracle")
        print()
        return None

    # Get passed context
    codebase_index = kwargs.get("codebase_index")
    transitions = kwargs.get("transitions")

    if not oracle:
        print("\n  /ask requires oracle context")
        return None

    def build_rich_context():
        """Build rich context like Planner gets."""
        parts = []

        # Session context
        session_ctx = (
            memory.get_session_context()
            if hasattr(memory, "get_session_context")
            else ""
        )
        if session_ctx:
            parts.append(f"SESSION CONTEXT:\n{session_ctx}")

        # Codebase index summary
        if codebase_index:
            index_ctx = codebase_index.get_context(max_chars=2000)
            if index_ctx:
                parts.append(f"CODEBASE INDEX:\n{index_ctx}")

        # RAG context (semantic search for user's message)
        if memory.project_path and cmd_args:
            try:
                rag_result = get_relevant_context(
                    memory.project_path, cmd_args, top_k=5
                )
                if rag_result and rag_result.context:
                    parts.append(f"RELEVANT CODE (RAG):\n{rag_result.context[:3000]}")
            except:
                pass

        # Last answer from Oracle
        if _last_answer:
            parts.append(f"ORACLE'S LAST ANSWER:\n{_last_answer[:2000]}")

        # Action results
        if _last_action_results:
            results_text = "\n".join(_last_action_results[-15:])
            parts.append(f"ACTION RESULTS:\n{results_text[:3000]}")

        # Learnings from memory
        if hasattr(memory, "learnings") and memory.learnings:
            learnings_text = []
            for l in memory.learnings[-10:]:
                learnings_text.append(f"- [{l.type}] {l.data}")
            if learnings_text:
                parts.append(f"SESSION LEARNINGS:\n" + "\n".join(learnings_text))

        return "\n\n---\n\n".join(parts) if parts else "(new session, no prior context)"

    # Determine provider (default: neo)
    provider = os.getenv("ASK_PROVIDER", "").lower() or "neo"

    # Helper to create stream router if logging enabled
    def create_stream_router_if_enabled():
        from compass.core.stream_subscribers import stream_logging_enabled
        if not stream_logging_enabled():
            return None
        from compass.core.stream_router import StreamRouter
        from compass.core.stream_subscribers import JSONLSubscriber, create_session_jsonl_path
        from compass.agents.neo.memory import get_code_sessions_dir
        from compass.core.reasoning import debug

        # Get session directory if memory available
        session_dir = None
        if memory and hasattr(memory, 'session_id'):
            session_dir = get_code_sessions_dir() / memory.session_id

        router = StreamRouter()
        jsonl_path = create_session_jsonl_path(session_dir)
        router.subscribe(JSONLSubscriber(jsonl_path))
        debug(f"Stream logging to: {jsonl_path}")
        return router

    if provider == "neo":
        # Neo mode: Oracle wisdom -> Neo (Actor searches on-demand)
        from compass.agents.oracle import ask_oracle
        from compass.agents.neo.state import process_request

        if not transitions:
            print("\n  Neo mode requires transitions context")
            return None

        print()
        ui.start_spinner("Oracle dreaming")

        session_ctx = memory.get_session_context() if hasattr(memory, "get_session_context") else ""
        oracle_wisdom = ask_oracle(oracle, cmd_args, session_ctx)

        ui.stop_spinner()

        # Show Oracle's wisdom to user
        print()
        print("-" * 60)
        print("ORACLE WISDOM:")
        print("-" * 60)
        print(oracle_wisdom)
        print("-" * 60)
        print()

        # Store in memory
        memory.add_user_turn(cmd_args)
        memory.add_oracle_turn(oracle_wisdom)

        # Invoke Neo with Oracle context (Actor will search for code context on-demand)
        oracle_context = f"--- ORACLE WISDOM ---\n{oracle_wisdom}\n---"
        print("Neo acting with Oracle's wisdom...")
        print()
        process_request(
            oracle,
            cmd_args,
            memory,
            transitions,
            codebase_index=codebase_index,
            rag_context=oracle_context,
            stream_router=create_stream_router_if_enabled(),
        )
        return None

    # Build rich context for oracle/claude modes
    ui.start_spinner("Building context (RAG search)")
    context = build_rich_context()
    ui.stop_spinner()

    if provider == "oracle":
        # Oracle mode: direct prose response, streamed
        prompt = f"""You are an expert assistant helping with a software project.

{context}

The user asks: "{cmd_args}"

Respond thoughtfully and thoroughly. You have access to the session context above.
Be direct and helpful. If you see relevant code in the context, reference it specifically.
If you need more information to answer fully, say what you'd need."""

        print()
        print("-" * 60)

        response_chunks = []
        try:
            for chunk in oracle.speak_stream(prompt, max_tokens=2000, task="oracle"):
                print(chunk, end="", flush=True)
                response_chunks.append(chunk)
            print()
            print("-" * 60)
            print()
        except Exception as e:
            print(f"\nOracle error: {e}")
            print()
            return None

        full_response = "".join(response_chunks)

        # Optionally format through Answerer for structured output
        ask_format = os.getenv("ASK_FORMAT", "raw").lower()
        if ask_format == "formatted":
            from compass.agents.neo.types import AnswerResponse

            ui.start_spinner("Formatting")
            # Use Answerer to structure the Oracle's raw response
            answer_data = oracle.ask(
                f"""Format this response into structured output.

ORACLE'S RESPONSE:
{full_response}

Extract:
- answer: the main response (preserve content, improve clarity)
- references: any file:line mentions
- next_steps: suggested follow-ups if any""",
                AnswerResponse,
                max_tokens=2000,
                task="answerer",
            )
            ui.stop_spinner()
            ui.show_answer(answer_data)

        # Store in memory so subsequent requests can reference
        memory.add_user_turn(cmd_args)
        memory.add_oracle_turn(full_response)
        return None

    else:
        # Claude mode: interactive with approval flow
        from compass.agents.neo.state import process_request
        from compass.cli.driver import get_driver, set_driver, use_claude_driver
        from compass.cli.input import get_input

        if not transitions:
            print("\n  Claude mode requires transitions context")
            return None

        current_message = cmd_args

        try:
            from compass.llm.bridge import ClaudeBridge

            bridge = ClaudeBridge()

            while True:
                ui.start_spinner("Consulting Claude")
                response = bridge.consult(current_message, context)
                ui.stop_spinner()

                print()
                print("=" * 60)
                print("CLAUDE'S RESPONSE:")
                print("=" * 60)
                print(response)
                print("=" * 60)
                print()

                # Approval flow with modify option
                while True:
                    choice = (
                        input("[a]pprove (Claude drives Oracle) | [m]odify | [r]eject: ")
                        .strip()
                        .lower()
                    )

                    if choice in ["a", "approve", "y", "yes"]:
                        prev_driver = get_driver()
                        try:
                            use_claude_driver()
                            print("\n  Claude driving Oracle...")
                            print()

                            # Claude's response becomes the request - add to conversation
                            # so Actor sees it in RECENT CONVERSATION context
                            memory.add_user_turn(response)

                            process_request(
                                oracle,
                                response,
                                memory,
                                transitions,
                                codebase_index=codebase_index,
                                stream_router=create_stream_router_if_enabled(),
                            )
                        finally:
                            set_driver(prev_driver)
                            print(f"\n  Driver restored.")
                        return None

                    elif choice in ["m", "modify"]:
                        print("\nEnter updated message for Claude:")
                        new_message = get_input()
                        if new_message:
                            current_message = new_message
                            break  # Break inner loop, continue outer loop to re-ask Claude
                        else:
                            print("  (empty input, keeping current message)")

                    elif choice in ["r", "reject", "n", "no"]:
                        print("\n  Discarded.")
                        print()
                        return None

                    else:
                        print("Choose: [a]pprove, [m]odify, or [r]eject")

        except Exception as e:
            ui.stop_spinner()
            print(f"\nFailed to reach Claude: {e}")
            print()

        return None


def build_files_read_section(
    files_read_content: dict,
    max_lines_per_file: int = 52,
    max_total_lines: int = 10000,
) -> str:
    """Build consolidated FILES READ section from read file chunks.

    Coalesces multiple reads of the same file into a single view with
    gaps shown as "..." for unread sections.

    Shows last N lines per file - model can use read_file to get earlier parts.
    Also caps total lines across all files to prevent context overflow.

    Args:
        files_read_content: {rel_path: [(start_line, end_line, content), ...]}
        max_lines_per_file: Maximum lines to show per file (default 202, shows last N)
        max_total_lines: Maximum total lines across all files (default 10000)

    Returns:
        Formatted string with all files and their consolidated content.
    """
    if not files_read_content:
        return ""

    sections = []
    total_lines_used = 0
    for file_path, chunks in files_read_content.items():
        # Sort chunks by start line
        sorted_chunks = sorted(chunks, key=lambda x: x[0])

        # Track the max line we read AND file total (for tail indicator)
        max_line_read = max(end for _, end, _ in sorted_chunks) if sorted_chunks else 0
        file_total_lines = 0  # Will be extracted from header if present

        # Coalesce overlapping/adjacent chunks
        file_lines = {}  # {line_num: line_content}
        for start_line, end_line, content in sorted_chunks:
            # Parse the content - extract total from header if present
            lines = content.split("\n")
            if lines and lines[0].startswith("[Lines"):
                # Extract total from "[Lines X-Y of Z]" or "[Line X of Z]"
                import re
                total_match = re.search(r'of (\d+)', lines[0])
                if total_match:
                    file_total_lines = max(file_total_lines, int(total_match.group(1)))
                lines = lines[1:]  # Skip header

            for i, line in enumerate(lines):
                line_num = start_line + i
                if line_num <= end_line:
                    file_lines[line_num] = line

        if not file_lines:
            continue

        # Build output with gaps shown as "..."
        sorted_line_nums = sorted(file_lines.keys())

        # Rolling window: keep last N lines
        if len(sorted_line_nums) > max_lines_per_file:
            # Show note about earlier content
            first_shown = sorted_line_nums[-max_lines_per_file]
            sorted_line_nums = sorted_line_nums[-max_lines_per_file:]
            earlier_note = f"[{file_path} lines 1-{first_shown - 1} not shown - use read_file]\n"
        else:
            earlier_note = ""

        output_lines = []
        prev_line = 0

        for line_num in sorted_line_nums:
            # Show gap if there's a discontinuity
            if prev_line > 0 and line_num > prev_line + 1:
                gap_size = line_num - prev_line - 1
                output_lines.append(f"[... lines {prev_line+1}-{line_num-1} not read]")

            output_lines.append(file_lines[line_num])
            prev_line = line_num

        # Build header with line range
        first_line = sorted_line_nums[0] if sorted_line_nums else 0
        last_line = sorted_line_nums[-1] if sorted_line_nums else 0
        header = f"=== {file_path} [lines {first_line}-{last_line}] ==="

        # Tail note if we're not showing to the end
        later_note = ""
        file_end = file_total_lines if file_total_lines > 0 else max_line_read
        if last_line < file_end:
            later_note = f"\n[{file_path} lines {last_line + 1}-{file_end} not shown - use read_file]"

        # Format is self-documenting: "line N: content"
        section_text = f"{header}\n{earlier_note}" + "\n".join(output_lines) + later_note
        file_line_count = len(output_lines)

        # Check if adding this file would exceed total limit
        if total_lines_used + file_line_count > max_total_lines:
            remaining = max_total_lines - total_lines_used
            if remaining > 20:  # Only add if meaningful space left
                # Take last N lines that fit
                trimmed_lines = output_lines[-remaining:]
                skipped = len(output_lines) - remaining
                trim_note = f"[... {skipped} lines not shown - total limit reached]\n"
                section_text = f"=== {file_path} ===\n{trim_note}" + "\n".join(trimmed_lines)
                sections.append(section_text)
            # Stop processing more files
            break

        sections.append(section_text)
        total_lines_used += file_line_count

    if not sections:
        return ""

    return "--- FILES READ ---\n" + "\n\n".join(sections)


def _cmd_remember(memory, project_path, **kwargs) -> Optional[str]:
    """Save last successful execution as a reusable capability."""
    from compass.agents.neo.capabilities import persist_capability

    cmd_args = kwargs.get("cmd_args", "")
    success, message = persist_capability(memory, cmd_args or "capability")

    if success:
        print(f"\n  Remembered! {message}")
    else:
        print(f"\n  Couldn't persist: {message}")
    print()
    return None


def _cmd_trace(memory, project_path, **kwargs) -> Optional[str]:
    """Show session action history from memory."""
    from collections import Counter

    if not memory.actions:
        print("\nNo actions executed yet.")
        print("  Actions will be recorded as you work.")
        print()
        return None

    # Count action types
    action_counts = Counter(a.action_type for a in memory.actions)
    success_counts = Counter(a.action_type for a in memory.actions if a.success)

    print("\n=== ACTION HISTORY ===")
    print(f"\nTotal actions: {len(memory.actions)}")
    print("\nBy type:")
    for action_type, count in action_counts.most_common():
        success = success_counts.get(action_type, 0)
        rate = success / count if count > 0 else 0
        print(f"  {action_type}: {count} ({rate:.0%} success)")
    print()
    return None


def _cmd_stats(memory, project_path, **kwargs) -> Optional[str]:
    """Show NFA telemetry stats for current session."""
    from compass.core.telemetry import _get_stats

    stats = _get_stats()
    if not stats:
        print("\nNo telemetry stats being collected.")
        print("  Stats are collected during request processing.")
        print()
        return None

    print("\n" + "=" * 60)
    print("NFA TELEMETRY")
    print("=" * 60)
    print(stats.report(show_zeros=False))
    print("=" * 60)
    print()
    return None


def _cmd_rag(memory, project_path, **kwargs) -> Optional[str]:
    """RAG index management: /rag [rebuild|status|rebuild --force]."""
    import time
    from compass.agents.neo.rag import get_embedder, reindex

    cmd_args = (kwargs.get("cmd_args") or "").strip()
    parts = cmd_args.split()
    subcmd = parts[0] if parts else "status"
    force = "--force" in parts

    if subcmd == "rebuild":
        label = "full rebuild" if force else "incremental"
        print(f"\n  Indexing ({label})...")
        t0 = time.perf_counter()
        count = reindex(project_path, force=force)
        elapsed = time.perf_counter() - t0
        print(f"  Done: {count} chunks in {elapsed:.1f}s")

    elif subcmd == "status":
        embedder = get_embedder(project_path)
        n_chunks = len(embedder.chunks)
        n_embeds = len(embedder.embeddings) if embedder.embeddings is not None else 0
        dim = (
            embedder.embeddings.shape[1]
            if embedder.embeddings is not None and embedder.embeddings.ndim == 2
            else 0
        )
        files = {c.file for c in embedder.chunks.values()} if n_chunks else set()
        print(f"\n  Chunks:     {n_chunks}")
        print(f"  Embeddings: {n_embeds} x {dim}-dim")
        print(f"  Files:      {len(files)}")
        print(f"  On disk:    {'yes' if embedder.metadata_path.exists() else 'no'}")

    else:
        print("\n  Usage: /rag [rebuild [--force] | status]")

    print()
    return None


# Register built-in commands
register_command("/help", _cmd_help, "Show available commands")
register_command("/session", _cmd_session, "Show current session info")
register_command("/debug", _cmd_debug, "Toggle debug mode")
register_command("/last", _cmd_last, "Show last action's raw output")
register_command("/validate", _cmd_validate, "Verify last answer against raw data")
register_command("/think", _cmd_think, "Think floor: /think [off|low|med|high]")
register_command("/model", _cmd_model, "Switch model: /model [off|<family>|<spec>]")
register_command("/critic", _cmd_critic, "Toggle critic display")
register_command("/ask", _cmd_ask, "Ask question (Claude or Oracle via ASK_PROVIDER)")
register_command("/claude", _cmd_ask, "Ask Claude (alias for /ask with Claude mode)")
register_command("/trace", _cmd_trace, "Show session trace metrics")
register_command("/remember", _cmd_remember, "Save last execution as reusable capability")
register_command("/stats", _cmd_stats, "Show NFA telemetry (actions, transitions, oracle calls)")
register_command("/rag", _cmd_rag, "RAG index: /rag [rebuild [--force] | status]")


def build_errors_section(
    errors: list,
    max_total_lines: int = 100,
) -> str:
    """Build consolidated ERRORS section from accumulated errors.

    Pure function - takes errors, returns formatted string.

    Uses a rolling window that prioritizes the most recent error:
    1. Most recent error is ALWAYS shown in full (never truncated)
    2. Previous errors fill remaining line budget
    3. Oldest errors are dropped first when over budget

    Args:
        errors: List of (action_target, full_error) tuples in chronological order
        max_total_lines: Maximum total lines for all errors (default 100)

    Returns:
        Formatted string with error sections and delimiters.
    """
    if not errors:
        return ""

    # Work backwards from most recent
    sections = []
    total_lines = 0

    for i, (target, error) in enumerate(reversed(errors)):
        error_lines = error.split("\n")
        line_count = len(error_lines)
        is_most_recent = (i == 0)

        if is_most_recent:
            # Most recent error: always include in full
            sections.append((target, error, line_count))
            total_lines += line_count
        elif total_lines + line_count <= max_total_lines:
            # Fits in budget: include in full
            sections.append((target, error, line_count))
            total_lines += line_count
        elif total_lines < max_total_lines:
            # Partial fit: take last N lines that fit
            remaining = max_total_lines - total_lines
            if remaining >= 5:  # Only if meaningful
                truncated = "\n".join(error_lines[-remaining:])
                skipped = line_count - remaining
                note = f"[... {skipped} lines earlier in this error not shown]\n"
                sections.append((target, note + truncated, remaining + 1))
                total_lines = max_total_lines
            break
        else:
            # No room left
            break

    if not sections:
        return ""

    # Reverse back to chronological order
    sections.reverse()

    # Format output with delimiters
    output_parts = []
    for target, error, _ in sections:
        header = f"--- ERROR: {target} ---"
        output_parts.append(f"{header}\n{error}")

    return "--- ERRORS ---\n" + "\n\n".join(output_parts)


# Backward compatibility aliases
_build_files_read_section = build_files_read_section
_build_errors_section = build_errors_section
