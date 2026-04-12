"""Trinity interactive REPL.

Multi-turn conversation with session persistence, Ctrl+C pause
for mid-generation comments, and slash commands for control.

The generation loop IS the state machine. No NFA needed -- Trinity's
G -> V -> G' cycle is already a functional state machine. The REPL
wraps it with persistence and interrupt handling.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from compass.generators._types import (
    DomainSection,
    Err,
    GenerationContext,
    Ok,
    Result,
)
from compass.generators._loop import generation_loop
from compass.generators._ui import (
    Colors,
    Spinner,
    show_answer,
    show_duration,
    show_facts,
    show_result,
    show_separator,
    show_step,
    start_spinner,
    stop_spinner,
)
from compass.generators.trinity._session import TrinitySession
from compass.generators.trinity._types import ExecutionResult, Fact, FileFact
from compass.generators.trinity.fact_dispatch import display_fact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress callback -- oracle-style real-time action display
# ---------------------------------------------------------------------------


def make_progress_callback(workspace: Path | None = None):
    """Create a progress callback that shows real-time step execution.

    This is what makes Trinity's execution visible -- you see each step
    as it runs, each fact as it's collected, each fix as it's attempted.
    Ctrl+C during any of this interrupts and drops to comment prompt.

    When TRINITY_GUARD=1, the callback also:
      - Shows a preview box before each step and asks Y/n
      - Snapshots the workspace before execution
      - Shows colored diffs of any changed files after execution
    """
    from compass.generators.trinity._guard import (
        WorkspaceSnapshot,
        build_step_preview,
        format_diff,
        is_guard_enabled,
        prompt_approve,
    )

    _snapshot: list[WorkspaceSnapshot | None] = [None]  # mutable cell for nonlocal
    _ws = workspace.resolve() if workspace else None

    def on_progress(event: str, **kwargs):
        if event == "invoke_start":
            start_spinner("Thinking")

        elif event == "invoke_done":
            stop_spinner()
            spec = kwargs.get("spec")
            if spec and hasattr(spec, "steps"):
                n = len(spec.steps)
                print(f"  {Colors.cyan('*')} {Colors.bold('Plan')}({n} steps)")
                for s in spec.steps:
                    ref = ""
                    if s.artifact_type == "inline_python":
                        ref = "inline code"
                    elif s.artifact_ref:
                        ref = s.artifact_ref
                    print(f"    {Colors.dim('-')} {s.step_id}: {s.description} {Colors.dim(f'[{s.artifact_type}]')} {Colors.dim(ref)}")
                print()

        elif event == "invoke_error":
            stop_spinner()
            print(f"  {Colors.red('x')} Model error: {kwargs.get('error', '?')}")

        elif event == "parse_done":
            pass  # Already displayed in invoke_done

        elif event == "parse_error":
            print(f"  {Colors.red('x')} Parse error: {kwargs.get('error', '?')}")

        elif event == "validate_semantics":
            print(f"  {Colors.dim('*')} Validating plan semantics...")

        elif event == "validate_runtime_reality":
            print(f"  {Colors.dim('*')} Checking workspace paths and git...")

        elif event == "step_start":
            step = kwargs["step"]
            idx = kwargs["index"]
            total = kwargs["total"]
            target = ""
            if step.artifact_type == "inline_python":
                code = step.artifact_ref or ""
                first_line = code.strip().split("\n")[0][:80] if code.strip() else ""
                target = first_line
            elif step.artifact_ref:
                target = step.artifact_ref
            print(f"  {Colors.cyan('*')} Step {idx + 1}/{total} {Colors.cyan(step.step_id)}")
            print(f"    {Colors.dim('|')} {step.description}")
            if target:
                print(f"    {Colors.dim('|')} {Colors.dim(target)}")

        elif event == "step_approve":
            step = kwargs["step"]
            resolved_inputs = kwargs.get("resolved_inputs", {})
            if not is_guard_enabled():
                return True
            preview = build_step_preview(step, resolved_inputs)
            # Snapshot workspace before execution
            if _ws:
                _snapshot[0] = WorkspaceSnapshot(_ws)
            return prompt_approve(step, preview)

        elif event == "step_done":
            step = kwargs.get("step")
            fact = kwargs["fact"]
            displayed = display_fact(fact)

            # FileFact: show line-numbered content (dispatch produces it)
            if isinstance(fact, FileFact):
                lines = displayed.splitlines()
                if lines and lines[0].startswith("["):
                    print(f"    [{Colors.green('/')}] {Colors.dim(lines[0])}")
                    lines = lines[1:]
                else:
                    print(f"    [{Colors.green('/')}]")
                for line in lines:
                    print(f"    {line}")
            else:
                val = displayed.strip()
                if val:
                    if len(val) > 80:
                        val = val[:77] + "..."
                    print(f"    [{Colors.green('+')}] {fact.name}: {val}")
                else:
                    print(f"    [{Colors.green('+')}] {fact.name}")
            # Guard: show file diffs if we have a snapshot
            if _snapshot[0] is not None:
                diffs = _snapshot[0].diff()
                _snapshot[0] = None
                if diffs:
                    print(format_diff(diffs))

        elif event == "step_error":
            error = str(kwargs.get("error", "?"))
            # Show first line as header, then indented body (up to 1000 chars)
            lines = error.split("\n", 1)
            header = lines[0]
            if len(header) > 120:
                header = header[:117] + "..."
            print(f"    [{Colors.red('x')}] {header}")
            if len(lines) > 1 and lines[1].strip():
                body = lines[1][:1000]
                for line in body.split("\n"):
                    print(f"    {Colors.dim('|')} {line}")
            # Consume snapshot (partial writes may have occurred)
            if _snapshot[0] is not None:
                diffs = _snapshot[0].diff()
                _snapshot[0] = None
                if diffs:
                    print(format_diff(diffs))

        elif event == "step_skipped":
            step = kwargs["step"]
            print(f"    [{Colors.yellow('-')}] skipped by guard")

        elif event == "synthesize":
            start_spinner("Synthesizing answer")

        elif event == "fix_start":
            stop_spinner()
            error = kwargs.get("error", "")
            if len(error) > 100:
                error = error[:97] + "..."
            print(f"  {Colors.yellow('*')} {Colors.bold('Ouroboros')}: fixing...")
            print(f"    {Colors.dim('|')} {Colors.dim(error)}")

        elif event == "fix_done":
            fixed = kwargs.get("fixed", False)
            if fixed:
                print(f"    [{Colors.green('+')}] Fix applied, re-validating")
            else:
                print(f"    [{Colors.red('x')}] Could not fix")

    return on_progress


# ---------------------------------------------------------------------------
# Input -- bracketed paste, Ctrl+J newline, Ctrl+I image paste
# ---------------------------------------------------------------------------

# Attachments from the current input (images, large pastes)
_pending_attachments: list[dict] = []


def _grab_clipboard_image() -> str | None:
    """Check clipboard for image, save to temp file. Returns path or None."""
    try:
        from PIL import ImageGrab
        img = ImageGrab.grabclipboard()
        if img is None:
            return None
        import tempfile
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(tempfile.gettempdir()) / f"trinity_paste_{ts}.png"
        img.save(str(path), "PNG")
        return str(path)
    except ImportError:
        return None
    except Exception:
        return None


def _make_prompt_session():
    """Create a prompt_toolkit PromptSession with paste and image support.

    Key bindings:
      Enter       submit input
      Ctrl+J      insert newline (for multi-line typing)
      Ctrl+I      paste image from clipboard
      BracketedPaste  auto-detect pasted text (multi-line just works)
      Ctrl+R      search history
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys
    except ImportError:
        return None

    history_file = Path.home() / ".compass" / "trinity_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    bindings = KeyBindings()

    @bindings.add("c-j")
    def _newline(event):
        """Insert newline without submitting."""
        event.current_buffer.insert_text("\n")

    @bindings.add("c-i")
    def _paste_image(event):
        """Grab image from clipboard and attach."""
        path = _grab_clipboard_image()
        if path:
            _pending_attachments.append({"type": "image", "path": path})
            event.current_buffer.insert_text(f"[image: {path}]")
            print(f"\n  {Colors.green('*')} Clipboard image saved: {path}")
        else:
            print(f"\n  {Colors.dim('(no image in clipboard)')}")

    @bindings.add(Keys.BracketedPaste)
    def _handle_paste(event):
        """Handle pasted text -- multi-line just works."""
        normalized = event.data.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        # If large paste, save to temp file and attach
        if len(lines) > 50 or len(normalized) > 4000:
            import tempfile
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(tempfile.gettempdir()) / f"trinity_paste_{ts}.txt"
            path.write_text(normalized)
            _pending_attachments.append({
                "type": "text",
                "path": str(path),
                "lines": len(lines),
            })
            summary = f"[pasted {len(lines)} lines -> {path}]"
            event.current_buffer.insert_text(summary)
            print(f"\n  {Colors.green('*')} Large paste saved: {path} ({len(lines)} lines)")
        else:
            event.current_buffer.insert_text(normalized)

    session = PromptSession(
        history=FileHistory(str(history_file)),
        enable_history_search=True,
        key_bindings=bindings,
    )
    return session


def _read_input(ps, prompt: str = "> ") -> tuple[str | None, list[dict]]:
    """Read user input with paste/image support.

    Returns (text, attachments) where attachments is a list of
    {"type": "image"|"text", "path": str} dicts.
    Returns (None, []) on Ctrl+D or exit commands.
    """
    _pending_attachments.clear()

    try:
        if ps is not None:
            text = ps.prompt(prompt)
        else:
            text = input(prompt)
    except EOFError:
        return None, []
    except KeyboardInterrupt:
        print()
        return None, []

    attachments = list(_pending_attachments)
    _pending_attachments.clear()

    stripped = text.strip()
    if stripped.lower() in ("exit", "quit", ":q", "/quit"):
        return None, []
    return text, attachments


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def _cmd_help(**_kw) -> None:
    print()
    print(f"  {Colors.bold('Input:')}")
    print(f"    Enter           submit")
    print(f"    Ctrl+J          insert newline (multi-line typing)")
    print(f"    Ctrl+I          paste image from clipboard")
    print(f"    Ctrl+R          search command history")
    print(f"    Ctrl+C          pause generation, offer comment")
    print(f"    {Colors.dim('paste')}           auto-detected, large pastes saved to file")
    print()
    print(f"  {Colors.bold('Commands:')}")
    print(f"    {Colors.cyan('/help')}           show this help")
    print(f"    {Colors.cyan('/model')} [spec]   show or switch model")
    print(f"    {Colors.cyan('/facts')}          show accumulated facts")
    print(f"    {Colors.cyan('/history')}        show conversation history")
    print(f"    {Colors.cyan('/session')}        show session info")
    print(f"    {Colors.cyan('/sessions')}       list all saved sessions")
    print(f"    {Colors.cyan('/new')}            start a new session")
    print(f"    {Colors.cyan('/load')} [id]      load a previous session")
    print(f"    {Colors.cyan('/save')}           save session now")
    print(f"    {Colors.cyan('/guard')} [on|off]  toggle step approval and file diffs")
    print(f"    {Colors.cyan('/dream')} [name]   save session as dream / list / search / index")
    print(f"    {Colors.cyan('/clear')}          clear facts and history")
    print(f"    {Colors.dim('exit')}            quit")
    print()


def _cmd_facts(session: TrinitySession, **_kw) -> None:
    facts = session.get_facts()
    if not facts:
        print(f"  {Colors.dim('(no facts yet)')}")
        return
    print()
    show_facts(facts)
    print()


def _cmd_history(session: TrinitySession, **_kw) -> None:
    if not session.turns:
        print(f"  {Colors.dim('(no history)')}")
        return
    print()
    for t in session.turns:
        ts = t.timestamp[:19] if t.timestamp else ""
        content = t.content[:120]
        if len(t.content) > 120:
            content += "..."
        if t.role == "user":
            print(f"  {Colors.dim(ts)} {Colors.cyan('Q')}: {content}")
        else:
            print(f"  {Colors.dim(ts)} {Colors.green('A')}: {content}")
    print()


def _cmd_session(session: TrinitySession, **_kw) -> None:
    print()
    print(f"  {Colors.bold('session')}:  {session.session_id}")
    print(f"  {Colors.bold('created')}:  {session.created_at[:19]}")
    print(f"  {Colors.bold('updated')}:  {session.updated_at[:19]}")
    print(f"  {Colors.bold('turns')}:    {len(session.turns)}")
    print(f"  {Colors.bold('facts')}:    {len(session.get_facts())}")
    if session.project_path:
        print(f"  {Colors.bold('project')}:  {session.project_path}")
    print()


def _cmd_sessions(**_kw) -> None:
    sessions = TrinitySession.list_sessions()
    if not sessions:
        print(f"  {Colors.dim('(no saved sessions)')}")
        return
    print()
    for s in sessions:
        q = s["last_question"]
        sid = Colors.cyan(s["session_id"])
        turns = Colors.dim(f"[{s['turns']} turns]")
        if q:
            print(f"  {sid}  {turns}  {q}")
        else:
            print(f"  {sid}  {turns}")
    print()


def _cmd_new(**_kw) -> TrinitySession:
    session = TrinitySession.create()
    print(f"  {Colors.green('*')} new session: {Colors.cyan(session.session_id)}")
    return session


def _cmd_load(cmd_args: str, **_kw) -> TrinitySession | None:
    sid = cmd_args.strip()
    if sid:
        loaded = TrinitySession.load(sid)
    else:
        loaded = TrinitySession.get_latest()

    if loaded is None:
        print(f"  {Colors.red('x')} session not found")
        return None

    print(f"  {Colors.green('*')} loaded: {Colors.cyan(loaded.session_id)} ({len(loaded.turns)} turns)")
    return loaded


def _cmd_save(session: TrinitySession, **_kw) -> None:
    path = session.save()
    print(f"  {Colors.green('*')} saved: {path}")


def _cmd_clear(session: TrinitySession, **_kw) -> None:
    session.turns.clear()
    print(f"  {Colors.green('*')} cleared history and facts")


def _resolve_model(model_id: str) -> str:
    """Resolve model spec, showing what will actually be used."""
    if model_id:
        return model_id
    try:
        from compass.llm.ladder_policy import get_model_spec
        return get_model_spec()
    except Exception:
        return "(unknown)"


def _cmd_model(cmd_args: str, model_ref=None, **_kw) -> None:
    """Switch model. Usage: /model anthropic:sonnet"""
    if model_ref is None:
        print(f"  {Colors.red('x')} model switching not available")
        return

    new_model = cmd_args.strip()
    if not new_model:
        resolved = _resolve_model(model_ref.model_id)
        if model_ref.model_id:
            print(f"  {Colors.bold('model')}: {resolved}")
        else:
            print(f"  {Colors.bold('model')}: {resolved} {Colors.dim('(from config)')}")
        print(f"  {Colors.dim('Usage: /model <model-spec>')}")
        print(f"  {Colors.dim('Examples:')}")
        print(f"    {Colors.dim('/model anthropic:sonnet')}")
        print(f"    {Colors.dim('/model anthropic:opus')}")
        print(f"    {Colors.dim('/model qwen3-coder-next:latest@big')}")
        return

    old = _resolve_model(model_ref.model_id)
    model_ref.model_id = new_model
    print(f"  {Colors.green('*')} model: {old} -> {Colors.cyan(new_model)}")


def _cmd_guard(cmd_args: str, **_kw) -> None:
    """Toggle step guard. Usage: /guard [on|off]"""
    from compass.generators.trinity._guard import is_guard_enabled, set_guard

    arg = cmd_args.strip().lower()
    if arg in ("on", "1", "yes", "true"):
        set_guard(True)
        print(f"  {Colors.green('*')} guard: {Colors.green('on')}")
    elif arg in ("off", "0", "no", "false"):
        set_guard(False)
        print(f"  {Colors.green('*')} guard: {Colors.red('off')}")
    elif not arg:
        # Toggle
        current = is_guard_enabled()
        set_guard(not current)
        state = Colors.green("on") if not current else Colors.red("off")
        print(f"  {Colors.green('*')} guard: {state}")
    else:
        print(f"  {Colors.red('x')} usage: /guard [on|off]")


def _cmd_dream(cmd_args: str, session: TrinitySession, **_kw) -> None:
    """Save session as a dream, list dreams, search, or rebuild index.

    /dream                   -- list saved dreams
    /dream <name>            -- save current session as dream
    /dream index             -- rebuild dream embedding index
    /dream index --force     -- force full rebuild
    /dream search <query>    -- search dreams by query
    """
    from compass.generators._transcript import DreamStore
    from compass.generators.trinity._context import _dreams_dir

    store = DreamStore(_dreams_dir())
    args = cmd_args.strip()

    if not args:
        names = store.list_names()
        if not names:
            print(f"  {Colors.dim('(no saved dreams)')}")
        else:
            print(f"  {len(names)} dreams:")
            for n in names:
                print(f"    {n}")
        return

    parts = args.split(None, 1)
    subcmd = parts[0]

    if subcmd == "index":
        force = "--force" in args
        try:
            n = store.build_index(force=force)
            tag = " (forced)" if force else ""
            print(f"  {Colors.green('*')} indexed {n} dreams{tag}")
        except Exception as e:
            print(f"  {Colors.red('x')} index error: {e}")
        return

    if subcmd == "search":
        query = parts[1] if len(parts) > 1 else ""
        if not query:
            print(f"  {Colors.dim('usage: /dream search <query>')}")
            return
        matches = store.search(query)
        if not matches:
            print(f"  {Colors.dim('no matches')}")
        else:
            for name, score, _t in matches[:5]:
                print(f"  {name}: {score:.0%}")
        return

    # Save session as dream
    name = subcmd
    if not session.turns:
        print(f"  {Colors.red('x')} nothing to save -- session is empty")
        return

    entries = []
    for turn in session.turns:
        if turn.role == "user":
            entries.append({"said": turn.content})
        elif turn.role == "trinity" and turn.facts:
            did = []
            for f in turn.facts:
                did.append({
                    "step": f.get("name", ""),
                    "value": f.get("value", ""),
                    "fact_type": f.get("fact_type", ""),
                })
            entries.append({"did": did})

    store.save(name, entries, domain="trinity")
    print(f"  {Colors.green('*')} {len(entries)} entries saved as dream '{name}'")


_COMMANDS = {
    "/help": _cmd_help,
    "/facts": _cmd_facts,
    "/history": _cmd_history,
    "/session": _cmd_session,
    "/sessions": _cmd_sessions,
    "/save": _cmd_save,
    "/clear": _cmd_clear,
    "/dream": _cmd_dream,
    "/dreams": _cmd_dream,
}


def _handle_command(
    text: str,
    session: TrinitySession,
    model_ref=None,
) -> TrinitySession:
    """Dispatch a slash command. Returns (possibly new) session."""
    parts = text.strip().split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "/new":
        return _cmd_new()

    if cmd == "/load":
        loaded = _cmd_load(args)
        return loaded if loaded is not None else session

    if cmd == "/model":
        _cmd_model(args, model_ref=model_ref)
        return session

    if cmd == "/guard":
        _cmd_guard(args)
        return session

    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(f"  {Colors.red('x')} unknown command: {cmd} {Colors.dim('(try /help)')}")
        return session

    handler(cmd_args=args, session=session)
    return session


# ---------------------------------------------------------------------------
# Generation with interrupt handling
# ---------------------------------------------------------------------------


def _quiet_loggers():
    """Suppress logs from generator internals during REPL.

    The progress callback shows the same info in a nicer format,
    so the raw log lines (INFO and WARNING) are just noise.
    Skips suppression when the root logger is at DEBUG (--verbose).
    """
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        return
    for name in (
        "compass.generators._loop",
        "compass.generators._invoke",
        "compass.generators.trinity._runtime",
        "compass.generators.trinity._context",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def _restore_loggers():
    """Restore default log levels."""
    for name in (
        "compass.generators._loop",
        "compass.generators._invoke",
        "compass.generators.trinity._runtime",
        "compass.generators.trinity._context",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(name).setLevel(logging.NOTSET)


def _run_with_interrupt(
    ctx: GenerationContext,
    *,
    invoke,
    parse,
    validate,
    fix,
    emit,
    max_rounds: int,
    max_fixes: int,
    ps,
) -> tuple[Result, float]:
    """Run generation_loop with Ctrl+C interrupt handling.

    Returns (result, elapsed_seconds).
    """
    t0 = time.time()
    _quiet_loggers()

    while True:
        try:
            result = generation_loop(
                ctx,
                invoke=invoke,
                parse=parse,
                validate=validate,
                fix=fix,
                emit=emit,
                max_rounds=max_rounds,
                max_fixes=max_fixes,
            )
            stop_spinner()
            _restore_loggers()
            return result, time.time() - t0

        except KeyboardInterrupt:
            stop_spinner()
            _restore_loggers()
            print()
            show_separator("paused")
            print(f"  {Colors.italic('Type a comment to steer, or Enter to skip.')}")
            print()
            try:
                # Plain text prompt -- prompt_toolkit doesn't render ANSI in prompt strings
                if ps is not None:
                    comment = ps.prompt("comment> ").strip()
                else:
                    comment = input("comment> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return Err("interrupted by user"), time.time() - t0

            if not comment:
                return Err("interrupted by user"), time.time() - t0

            ctx = ctx.with_feedback(f"User comment: {comment}")
            print(f"  {Colors.dim('resuming...')}")
            print()
            _quiet_loggers()


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _display_result(path: Path) -> None:
    """Show the answer from a completed result."""
    result_file = path / "result.json" if path.is_dir() else path
    if not result_file.exists():
        return

    try:
        data = json.loads(result_file.read_text())
    except (json.JSONDecodeError, OSError):
        return

    answer = data.get("answer", "")
    facts = data.get("facts", [])

    if answer:
        print()
        print(f"  {Colors.cyan('*')} {Colors.bold('Answer')}")
        show_answer(answer)


def _load_execution_result(path: Path) -> ExecutionResult | None:
    """Load ExecutionResult from emitted files for session recording."""
    result_file = path / "result.json" if path.is_dir() else path
    if not result_file.exists():
        return None

    try:
        data = json.loads(result_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    facts = tuple(
        Fact(
            step_id=f.get("step_id", ""),
            name=f.get("name", ""),
            value=str(f.get("value", "")),
            fact_type=f.get("fact_type", "text"),
        )
        for f in data.get("facts", [])
    )

    return ExecutionResult(
        question=data.get("question", ""),
        facts=facts,
        answer=data.get("answer", ""),
        success=data.get("success", False),
    )


# ---------------------------------------------------------------------------
# REPL entry point
# ---------------------------------------------------------------------------


def _build_prior_facts(session: TrinitySession) -> dict | None:
    """Convert session facts to Fact objects for execution seeding."""
    raw = session.get_facts()
    if not raw:
        return None
    result = {}
    for f in raw:
        if f.get("fact_type") != "error":
            result[f["name"]] = Fact(
                step_id="prior_session",
                name=f["name"],
                value=f["value"],
                fact_type=f["fact_type"],
            )
    return result or None


def trinity_repl(
    ctx: GenerationContext,
    *,
    invoke,
    parse,
    validate,
    fix,
    emit,
    max_rounds: int = 3,
    max_fixes: int = 3,
    session: TrinitySession | None = None,
    model_ref=None,
    history: bool = False,
    facts_ref=None,
) -> Result:
    """Interactive Trinity REPL.

    Multi-turn conversation with:
    - Session persistence (facts accumulate across turns)
    - Ctrl+C pause (interrupt generation, add comments, resume)
    - Slash commands (/help, /facts, /history, /session, /model, etc.)
    """
    ps = _make_prompt_session()

    if session is None:
        session = TrinitySession.get_latest()
        if session is not None:
            print(f"  {Colors.green('*')} Resumed session {Colors.cyan(session.session_id)} ({len(session.turns)} turns)")
        else:
            session = TrinitySession.create()

    print()
    show_separator("Trinity")
    print()
    print(f"  Session: {Colors.cyan(session.session_id)}")
    model_display = _resolve_model(model_ref.model_id) if model_ref else "(unknown)"
    print(f"  Model:   {Colors.cyan(model_display)}")
    print(f"  Facts:   {len(session.get_facts())}")
    print()
    print(f"  {Colors.dim('Type a question, press Enter to submit.')}")
    print(f"  {Colors.dim('Ctrl+J for newline, Ctrl+I to paste image.')}")
    print(f"  {Colors.dim('Ctrl+C during generation to pause and comment.')}")
    print(f"  {Colors.dim('/help for commands, exit to quit.')}")
    print()

    while True:
        text, attachments = _read_input(ps)
        if text is None:
            session.save()
            print()
            print(f"  {Colors.dim('Session saved:')} {session.session_id}")
            return Ok(None)

        text = text.strip()
        if not text:
            continue

        # Slash commands
        if text.startswith("/"):
            session = _handle_command(text, session, model_ref=model_ref)
            continue

        # Handle attachments (images, large text pastes)
        turn_ctx = ctx
        for att in attachments:
            if att["type"] == "image":
                # Add image path as domain context for vision steps
                turn_ctx = turn_ctx.with_domain(DomainSection(
                    heading="Attached Image",
                    content=f"Image available at: {att['path']}\n"
                            f"Use a vision step with artifact_ref=\"{att['path']}\" to read it.",
                ))
                print(f"  {Colors.dim('*')} Image attached for vision")
            elif att["type"] == "text":
                # Load and add large pasted text as context
                try:
                    pasted = Path(att["path"]).read_text()
                    turn_ctx = turn_ctx.with_domain(DomainSection(
                        heading="Pasted Content",
                        content=pasted,
                    ))
                    print(f"  {Colors.dim('*')} Text attached ({att.get('lines', '?')} lines)")
                except OSError:
                    pass

        # Record user turn
        session.add_user_turn(text)

        # Seed prior session facts for execution
        if facts_ref is not None:
            facts_ref.facts = _build_prior_facts(session)

        # Enrich context with session history and facts
        turn_ctx = session.enrich_context(turn_ctx, text, history=history)

        # Run generation with interrupt handling and timing
        result, elapsed = _run_with_interrupt(
            turn_ctx,
            invoke=invoke,
            parse=parse,
            validate=validate,
            fix=fix,
            emit=emit,
            max_rounds=max_rounds,
            max_fixes=max_fixes,
            ps=ps,
        )

        match result:
            case Ok(path):
                if path is not None:
                    _display_result(path)
                    exec_result = _load_execution_result(path)
                    answer = exec_result.answer if exec_result else str(path)
                    session.add_trinity_turn(answer, exec_result)
                else:
                    session.add_trinity_turn("(no output)")

            case Err(e):
                error_msg = str(e)
                if "interrupted" not in error_msg:
                    show_result(False, error_msg)
                session.add_trinity_turn(f"failed: {error_msg}")

        show_duration(elapsed)
        print()
        session.save()
