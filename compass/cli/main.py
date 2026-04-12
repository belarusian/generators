#!/usr/bin/env python3
"""
CLI entry point for Compass.

This module provides the main() function that serves as the entry point
for the `compass` command when installed via pip.

The actual implementation lives in compass.py at the project root.
When installed as a package, we import from there dynamically.
"""

import sys
import argparse
from pathlib import Path


# --- Local source, not site-packages ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The sacred geometry
COMPASS_ART = r"""
         .  +  .
          \ | /
        --- * ---
          / | \
         '  +  '
"""
def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compass - self-rewriting code intelligence.",
        prog="compass",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=COMPASS_ART
    )
    parser.add_argument("--version", "-v", action="store_true", help="Show version and exit")
    parser.add_argument("--journey", action="store_true", help="Launch the Guide (journey mode)")
    parser.add_argument("--secret", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lodging", action="store_true", help="Find lodging for last journey")
    parser.add_argument("--online", action="store_true", help="Interactive follow-up on last journey")
    parser.add_argument("--code", action="store_true", help="Launch code mode (Planner/Actor)")
    parser.add_argument("--live", action="store_true", help="Continuous interaction mode")
    parser.add_argument("--resume", action="store_true", help="Resume last code session")
    parser.add_argument("--list", action="store_true", help="List all code sessions")
    parser.add_argument("--session", type=str, help="Resume specific session by ID")
    parser.add_argument("--model", type=str, help="Model spec (e.g. qwen3-coder:latest@local, anthropic:sonnet)")
    parser.add_argument("--red-pill", type=str, nargs="?", const="", metavar="TASK",
                        help="Red pill: straight to Neo (one-shot execution)")
    parser.add_argument("--blue-pill", type=str, nargs="?", const="", metavar="TASK",
                        help="Blue pill: through Trinity (Neo + reflection)")
    parser.add_argument("--morpheus", type=str, nargs="?", const="", metavar="TASK",
                        help="Morpheus: Neo with full autonomy (big server)")
    return parser.parse_args(args)


def _get_version():
    """Get version info from git commit."""
    # Try to read baked-in build info
    try:
        import compass._build_info as build_info
        commit = getattr(build_info, 'COMMIT', 'unknown')
        message = getattr(build_info, 'COMMIT_MSG', '')[:22]
        return f"compass {commit[:8]} {message}"
    except ImportError:
        return "compass (not installed - run make install)"


def main(args=None):
    """Main entry point for the compass CLI.

    This is called by:
    - `compass` command after pip install .
    - `python -m compass` via __main__.py
    """
    from dotenv import load_dotenv

    # Local .env first, global fills gaps. No override -- first writer wins.
    load_dotenv()                       # .env in cwd
    global_env = Path.home() / ".compass" / ".env"
    if global_env.exists():
        load_dotenv(global_env)         # ~/.compass/.env fills what's missing

    # Parse args if not provided
    if args is None:
        args = parse_args()
    elif isinstance(args, list):
        args = parse_args(args)

    # CLI --model overrides env var
    if getattr(args, 'model', None):
        import os as _os
        _os.environ["COMPASS_MODEL"] = args.model

    # Handle special modes that are self-contained
    if args.version:
        print(_get_version())
        return

    if args.secret:
        # Easter egg - no dependencies needed
        print("The secret compass whispers: 'True north lies within.'")
        return

    if args.list:
        from compass.agents.neo.memory import CodeMemory
        _list_sessions(CodeMemory)
        return

    if args.red_pill is not None:
        _run_red_pill(args.red_pill)
        return

    if args.blue_pill is not None:
        _run_blue_pill(args.blue_pill)
        return

    if args.morpheus is not None:
        _run_morpheus_path(args.morpheus)
        return

    # Code mode - the primary use case when invoked as standalone
    if args.code or args.resume or args.session:
        _run_code_mode(args)
        return

    # Journey/travel modes
    if args.journey:
        from compass.travel.journey import run_journey_mode
        run_journey_mode()
        return

    if args.lodging:
        from compass.travel.journey import run_lodging_mode
        run_lodging_mode()
        return

    if args.online:
        from compass.travel.journey import run_online_mode
        run_online_mode()
        return

    # Default: code mode in current directory
    _run_code_mode(args)


def _list_sessions(CodeMemory):
    """List all code sessions with summary info."""
    sessions = CodeMemory.list_sessions()

    if not sessions:
        print("No code sessions found.")
        return

    print()
    print("CODE SESSIONS")
    print("=" * 70)

    for s in sessions:
        session_id = s["session_id"]
        project = s.get("project_path", "?")
        created = s.get("created_at", "?")[:16] if s.get("created_at") else "?"
        updated = s.get("updated_at", "?")[:16] if s.get("updated_at") else "?"
        plans = f"{s.get('plans_executed', 0)}/{s.get('plans_total', 0)}"
        actions = f"{s.get('actions_successful', 0)}/{s.get('actions_total', 0)}"

        print()
        print(f"  [{session_id}]")
        print(f"  Project: {project}")
        print(f"  Plans:   {plans} executed | Actions: {actions} successful")
        print(f"  Created: {created} | Updated: {updated}")

    print()
    print("=" * 70)
    print(f"  {len(sessions)} session(s)")
    print()
    print("  Resume with: compass --session <id>")
    print()


def _run_code_mode(args):
    """Run code mode - the primary use case."""
    import os

    # Heavy imports deferred until needed
    from compass.llm.oracle import Oracle
    from compass.agents.neo.memory import CodeMemory, generate_session_id
    from compass.agents.neo.index import index_codebase
    from compass.cli import ui
    from compass.cli.input import get_input_with_paste_detection
    from compass.cli.commands import CODE_COMMANDS, set_last_answer
    from compass.agents.neo.state import (
        create_transitions,
        process_request,
    )
    from compass.core.config import ExecutionConfig
    from compass.agents.neo import (
        _execute_request,
        _generate_answer,
        _critic_review,
        _critic_evaluate,
    )

    project_path = os.getcwd()
    _ensure_git_initialized(project_path)

    oracle = Oracle()

    _print_banner()

    # Resolve session
    session_id = getattr(args, 'session', None)
    resume = getattr(args, 'resume', False)
    live = getattr(args, 'live', False)

    memory, is_new = _resolve_session(project_path, session_id, resume, CodeMemory, generate_session_id)
    if memory is None:
        print(f"Session '{session_id}' not found.")
        return

    status = "new" if is_new else "resumed"
    print(f"   session:   {memory.session_id} ({status})")
    if not is_new:
        print()
        print("Tip: /new to start fresh")

    memory.save()

    # Enable telemetry collection for this session
    from compass.core import telemetry
    telemetry._stats = telemetry.NFAStats()

    # Index codebase
    codebase_index = _ensure_indexed(memory, index_codebase, ui)
    print()

    # Main loop
    while True:
        request = get_input_with_paste_detection(memory=memory)

        if request is None or request.lower() in ['quit', 'exit', 'bye', 'done']:
            break

        if not request:
            continue

        if request.startswith('/'):
            cmd = request.split()[0]
            cmd_lower = cmd.lower()
            if cmd_lower in ['/quit', '/exit', '/q']:
                break
            if cmd_lower in ['/new', '/fresh', '/reset']:
                memory = _create_new_session(project_path, CodeMemory, generate_session_id, index_codebase, ui)
                codebase_index = _ensure_indexed(memory, index_codebase, ui)
                continue
            if cmd in CODE_COMMANDS:
                transitions = create_transitions(
                    execute_request_fn=_execute_request,
                    critic_evaluate_fn=_critic_evaluate,
                    critic_review_fn=_critic_review,
                    generate_answer_fn=_generate_answer,
                    set_last_answer_fn=set_last_answer,
                )
                cmd_args = request[len(cmd):].strip()
                CODE_COMMANDS[cmd]["handler"](
                    memory, project_path,
                    oracle=oracle,
                    cmd_args=cmd_args,
                    codebase_index=codebase_index,
                    transitions=transitions,
                )
            else:
                print(f"  Unknown command: {cmd}")
                print("  Type /help for available commands")
            continue

        # Process request
        request = _prepare_request(request, memory)
        memory.add_user_turn(request)

        transitions = create_transitions(
            execute_request_fn=_execute_request,
            critic_evaluate_fn=_critic_evaluate,
            critic_review_fn=_critic_review,
            generate_answer_fn=_generate_answer,
            set_last_answer_fn=set_last_answer,
        )
        config = ExecutionConfig.from_overrides()

        stream_router = _create_stream_router(memory)

        try:
            success = process_request(
                oracle, request, memory, transitions,
                codebase_index=codebase_index,
                config=config,
                stream_router=stream_router,
            )
            if not success:
                print("\n  The Oracle could not complete the request.\n")
        except KeyboardInterrupt:
            ui.stop_spinner()
            if ui.is_auto_approve():
                ui.set_auto_approve(False)
                print(f"\n  Interrupted. Auto-approve disabled.")
            else:
                print(f"\n  Interrupted.")

        memory.save()

        if not live:
            break

    # Cleanup
    telemetry._stats = None  # Clear stats collector
    ui.set_auto_approve(False)
    print("\n  Session saved.")
    print(f"  Session ID: {memory.session_id}")
    print(f"  Path: {memory.save()}")


def _run_red_pill(task: str):
    """Red pill: straight to Neo -- full NFA, one shot."""
    import os
    os.environ.setdefault("COMPASS_FAMILY", "morpheus")
    from compass.llm.oracle import Oracle
    from compass.agents.neo.memory import CodeMemory, generate_session_id
    from compass.agents.neo.state import create_transitions, process_request
    from compass.core.config import ExecutionConfig
    from compass.agents.neo import (
        _execute_request,
        _generate_answer,
        _critic_review,
        _critic_evaluate,
    )

    task = task or input("\n  What do you need? ").strip()
    if not task:
        return

    oracle = Oracle()
    memory = CodeMemory(
        session_id=generate_session_id(),
        project_path=os.getcwd(),
    )

    transitions = create_transitions(
        execute_request_fn=_execute_request,
        critic_evaluate_fn=_critic_evaluate,
        critic_review_fn=_critic_review,
        generate_answer_fn=_generate_answer,
        set_last_answer_fn=lambda a,b: None,
    )

    stream_router = _create_stream_router(memory)

    success = process_request(
        oracle, task, memory, transitions,
        config=ExecutionConfig.from_overrides(),
        stream_router=stream_router,
    )

    print()
    print("=" * 60)
    print(f"  Status: {'success' if success else 'failed'}")
    print("=" * 60)


def _run_blue_pill(task: str):
    """Blue pill: through Trinity. Neo acts (full NFA), she reflects (Claude)."""
    from compass.llm.trinity import Trinity

    task = task or input("\n  What do you need? ").strip()
    if not task:
        return

    trinity = Trinity()
    success, reflection = trinity.red_pill(task)

    print(f"\n  Neo: {'success' if success else 'failed'}")
    print(f"  Trinity: {reflection}")


def _run_morpheus_path(task: str):
    """Morpheus: self-reflection through the Oracle. Same model acts and reflects."""
    import os
    os.environ.setdefault("COMPASS_FAMILY", "morpheus")

    from compass.llm.morpheus import Morpheus

    task = task or input("\n  What do you need? ").strip()
    if not task:
        return

    morpheus = Morpheus()
    result, output = morpheus.red_pill(task)

    print(f"\n  Neo: {result.status.value}")
    print(f"  Morpheus: {output.action.value} -- {output.explanation}")


def _print_banner():
    """Print the compass rose."""
    art = r"""
           *
          /|\
         / | \
        /  *  \
       / / | \ \
      *---*-*---*
       \ \ | / /
        \  *  /
         \ | /
          \|/
           *
    """
    print(art)


def _ensure_git_initialized(project_path: str) -> bool:
    """Ensure git is initialized for change tracking."""
    import os
    import subprocess

    git_dir = os.path.join(project_path, ".git")
    if os.path.isdir(git_dir):
        return True

    try:
        subprocess.run(["git", "init"], cwd=project_path, capture_output=True, check=True)
        subprocess.run(["git", "add", "-A"], cwd=project_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit (auto-created by Compass)"],
            cwd=project_path, capture_output=True, check=False
        )
        print(f"   git:       initialized (for change tracking)")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"   git:       unavailable ({e})")
        return False


def _resolve_session(project_path, session_id, resume, CodeMemory, generate_session_id):
    """Resolve which session to use."""
    if session_id:
        memory = CodeMemory.load(session_id)
        return (memory, False) if memory else (None, False)

    if resume:
        memory = CodeMemory.get_latest()
        if memory and memory.project_path == project_path:
            return memory, False

    latest = CodeMemory.get_latest()
    if latest and latest.project_path == project_path:
        return latest, False

    new_id = generate_session_id()
    memory = CodeMemory(session_id=new_id)
    memory.project_path = project_path
    return memory, True


def _ensure_indexed(memory, index_codebase, ui):
    """Ensure codebase is indexed (structural + RAG)."""
    first_run = not memory.index_summary

    if first_run:
        ui.show_thinking("Indexing codebase")

    codebase_index = index_codebase(memory.project_path)

    if first_run:
        memory.index_summary = codebase_index.summary()
        memory.index_context = codebase_index.get_context(max_chars=8000)
        print(f"Found: {memory.index_summary['file_count']} files, "
              f"{memory.index_summary['function_count']} functions, "
              f"{memory.index_summary['class_count']} classes")
        memory.save()

    # Kick off RAG incremental rebuild in background (non-blocking)
    try:
        from compass.agents.neo.rag import reindex_in_background
        reindex_in_background(memory.project_path, force=first_run)
    except Exception:
        pass  # RAG is optional -- embeddings may not be configured

    return codebase_index


def _create_new_session(project_path, CodeMemory, generate_session_id, index_codebase, ui):
    """Create a fresh session."""
    new_id = generate_session_id()
    memory = CodeMemory(session_id=new_id)
    memory.project_path = project_path

    ui.show_thinking("Indexing codebase")
    codebase_index = index_codebase(project_path)
    memory.index_summary = codebase_index.summary()
    memory.index_context = codebase_index.get_context(max_chars=8000)
    memory.save()

    # Full RAG rebuild in background for new sessions
    try:
        from compass.agents.neo.rag import reindex_in_background
        reindex_in_background(project_path, force=True)
    except Exception:
        pass

    print(f"\nStarted new session: {new_id}")
    print(f"Found: {memory.index_summary['file_count']} files, "
          f"{memory.index_summary['function_count']} functions, "
          f"{memory.index_summary['class_count']} classes")

    return memory


def _prepare_request(request, memory):
    """Prepare request for processing."""
    from compass.cli.input import get_input

    if request.startswith('[Content#') and request.endswith(']'):
        print()
        print(f"  What should I do with this content?")
        message = get_input()
        request = f"{message} (see {request})" if message else f"Analyze this: {request}"

    memory.images = []
    return request


def _create_stream_router(memory=None):
    """Create stream router if logging enabled."""
    from compass.core.stream_subscribers import stream_logging_enabled
    if not stream_logging_enabled():
        return None

    from compass.core.stream_router import StreamRouter
    from compass.core.stream_subscribers import JSONLSubscriber, create_session_jsonl_path
    from compass.agents.neo.memory import get_code_sessions_dir
    from compass.core.reasoning import debug

    session_dir = None
    if memory and hasattr(memory, 'session_id'):
        session_dir = get_code_sessions_dir() / memory.session_id

    router = StreamRouter()
    jsonl_path = create_session_jsonl_path(session_dir)
    router.subscribe(JSONLSubscriber(jsonl_path))
    debug(f"Stream logging to: {jsonl_path}")
    return router


if __name__ == "__main__":
    main()
