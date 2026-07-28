"""
Code session memory.

Like JourneyMemory for travel, CodeMemory persists coding sessions:
- Session info (project path, codebase index)
- Plans created and their approval status
- Actions executed and results
- Conversation history

Sessions are stored in ~/.compass/code/{session_id}/
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from compass.agents.neo.types import LearningType


def get_code_sessions_dir() -> Path:
    """Get the directory for code sessions."""
    base = Path.home() / ".compass" / "code"
    base.mkdir(parents=True, exist_ok=True)
    return base


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def preview_content(content: str, max_lines: int = 10, label: str = "action") -> str:
    """Create a preview showing head. Pure function."""
    from compass.core.content import preview_head_tail
    return preview_head_tail(content, max_lines=max_lines, label=label)


@dataclass
class Plan:
    """A plan created by the Planner."""
    summary: str
    steps: List[str]
    files_affected: List[str]
    risks: Optional[List[str]] = None
    status: str = "pending"  # pending, approved, rejected, executed, partial, failed
    feedback: Optional[str] = None  # User feedback if rejected
    steps_completed: List[int] = field(default_factory=list)  # Indices of completed steps
    steps_failed: List[int] = field(default_factory=list)  # Indices of failed steps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Action:
    """An action executed by the Actor."""
    action_type: str
    target: str
    content: Optional[str] = None
    reasoning: str = ""
    result: Optional[str] = None
    success: bool = True
    executed_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Turn:
    """A conversation turn."""
    role: str  # "user", "planner", "actor"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Attachment:
    """User-pasted content stored separately from messages."""
    id: int
    content: str
    line_count: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ImageAttachment:
    """Image file stored as base64 for vision models."""
    id: int
    data: str  # base64 encoded
    media_type: str  # image/png, image/jpeg, etc.
    filename: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Learning:
    """
    A runtime-verified learning about the environment.

    Uses LearningType enum - no magic strings.
    """
    type: LearningType
    data: Dict[str, Any]
    when: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Convert string type to enum if needed (for deserialization)."""
        if isinstance(self.type, str):
            try:
                self.type = LearningType(self.type)
            except ValueError:
                # Unknown type - default to CORRECTION as safest
                self.type = LearningType.CORRECTION


@dataclass
class ContextFrame:
    """
    A frame in the context stack for isolated operations.

    Push before multi-step operation, pop when done.
    History is discarded, result and learnings are kept.
    """
    history: List[str] = field(default_factory=list)
    result: Optional[str] = None
    learnings: List[Learning] = field(default_factory=list)


@dataclass
class CodeMemory:
    """Persistent memory for a coding session."""

    session_id: str
    project_path: Optional[str] = None

    # Index summary (populated after indexing)
    index_summary: Optional[Dict[str, Any]] = None
    # Full index context (file tree, functions, classes) for Planner visibility
    index_context: Optional[str] = None

    # Plans and actions
    plans: List[Plan] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)

    # Conversation
    conversation: List[Turn] = field(default_factory=list)

    # User-pasted attachments (referenced as [Content#N] in messages)
    attachments: List[Attachment] = field(default_factory=list)

    # Image attachments (referenced as [Image#N] in messages)
    images: List[ImageAttachment] = field(default_factory=list)

    # Context stack for isolated operations (push/pop)
    context_stack: List[ContextFrame] = field(default_factory=list)

    # Runtime-verified learnings (persisted across frames)
    learnings: List[Learning] = field(default_factory=list)

    # Last answer context (for "proceed with next steps")
    last_answer: Optional[str] = None
    last_next_steps: List[str] = field(default_factory=list)

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_plan(self, plan: Plan) -> None:
        """Add a plan to the session."""
        self.plans.append(plan)
        self._touch()

    def add_action(self, action: Action) -> None:
        """Add an executed action to the session."""
        self.actions.append(action)
        self._touch()

    def get_recent_action_results(self, n: int = 20) -> List[str]:
        """Format recent actions as result strings for prompt context.

        Returns last n actions formatted like ctx.action_results but from
        persistent storage. This survives across requests.
        """
        # Actions that already self-limit their output - don't double-truncate
        PASS_THROUGH_ACTIONS = {"AskOracleAction"}
        # These can have meaningful output - truncate generously
        GENEROUS_ACTIONS = {"ReadFileAction", "GrepAction", "SearchAction", "RunCommandAction", "ShellCommandAction"}

        def format_result(a):
            result = a.result or "(no output)"
            if a.action_type in PASS_THROUGH_ACTIONS:
                return result
            if a.action_type in GENEROUS_ACTIONS:
                return preview_content(result, max_lines=52, label=a.action_type)
            return preview_content(result, max_lines=10, label=a.action_type)

        return [
            f"[{'+' if a.success else 'x'}] {a.action_type}: {a.target}\n    -> {format_result(a)}"
            for a in self.actions[-n:]
        ]

    def add_user_turn(self, content: str) -> None:
        """Record user input."""
        self.conversation.append(Turn(role="user", content=content))
        self._touch()

    def add_planner_turn(self, content: str) -> None:
        """Record planner output."""
        self.conversation.append(Turn(role="planner", content=content))
        self._touch()

    def add_actor_turn(self, content: str) -> None:
        """Record actor output."""
        self.conversation.append(Turn(role="actor", content=content))
        self._touch()

    def add_oracle_turn(self, content: str) -> None:
        """Record oracle free-form response (from /ask command)."""
        self.conversation.append(Turn(role="oracle", content=content))
        self._touch()

    def set_last_answer(self, answer: Optional[str], next_steps: Optional[List[str]] = None) -> None:
        """Store last answer for 'proceed with next steps' support.

        Parameters
        ----------
        answer: Optional[str]
            The answer text. If falsy, clears stored answer and next steps.
        next_steps: Optional[List[str]]
            Optional list of next step strings. Stored up to the first 5.
        """
        # Clear if answer is falsy
        if not answer:
            self.last_answer = None
            self.last_next_steps = []
            self._touch()
            return

        self.last_answer = answer

        # Store up to five next steps
        if next_steps:
            self.last_next_steps = next_steps[:5]
        else:
            self.last_next_steps = []

        self._touch()

    def add_attachment(self, content: str) -> int:
        """Store pasted content as attachment, return its ID."""
        next_id = len(self.attachments) + 1
        line_count = content.count('\n') + 1
        self.attachments.append(Attachment(
            id=next_id,
            content=content,
            line_count=line_count,
        ))
        self._touch()
        return next_id

    def get_attachment(self, attachment_id: int) -> Optional[str]:
        """Get attachment content by ID."""
        for att in self.attachments:
            if att.id == attachment_id:
                return att.content
        return None

    def add_image(self, data: str, media_type: str, filename: str) -> int:
        """Store image as base64 attachment, return its ID."""
        next_id = len(self.images) + 1
        self.images.append(ImageAttachment(
            id=next_id,
            data=data,
            media_type=media_type,
            filename=filename,
        ))
        self._touch()
        return next_id

    def get_image(self, image_id: int) -> Optional[ImageAttachment]:
        """Get image attachment by ID."""
        for img in self.images:
            if img.id == image_id:
                return img
        return None

    def get_pending_images(self) -> List[ImageAttachment]:
        """Get all images, converting dicts to ImageAttachment if needed."""
        result = []
        for img in self.images:
            if isinstance(img, dict):
                # Handle JSON-deserialized dicts from session load
                result.append(ImageAttachment(
                    id=img.get("id", 0),
                    data=img.get("data", ""),
                    media_type=img.get("media_type", "image/png"),
                    filename=img.get("filename", "unknown"),
                    created_at=img.get("created_at", ""),
                ))
            else:
                result.append(img)
        return result

    def _touch(self) -> None:
        """Update the modified timestamp."""
        self.updated_at = datetime.now().isoformat()

    # --- Context Stack Operations ---

    def push_context(self) -> None:
        """Push a new context frame for isolated operations."""
        self.context_stack.append(ContextFrame())
        self._touch()

    def pop_context(self) -> Optional[ContextFrame]:
        """Pop context frame, merge learnings into persistent storage."""
        if not self.context_stack:
            return None
        frame = self.context_stack.pop()
        self._merge_learnings(frame.learnings)
        self._touch()
        return frame

    def _merge_learnings(self, new_learnings: List[Learning]) -> None:
        """Merge new learnings into persistent storage, deduping intelligently."""
        for learning in new_learnings:
            # For corrections, dedupe by (type, subject, expected, actual)
            if learning.type == "correction":
                key = (
                    learning.data.get("subject"),
                    learning.data.get("expected"),
                    learning.data.get("actual"),
                )
                existing = [
                    l for l in self.learnings
                    if l.type == "correction" and (
                        l.data.get("subject"),
                        l.data.get("expected"),
                        l.data.get("actual"),
                    ) == key
                ]
                if not existing:
                    self.learnings.append(learning)
            # For file_read, latest range wins; preserve any known full read
            elif learning.type == "file_read":
                file_path = learning.data.get("file")
                existing_idx = None
                for idx, existing in enumerate(self.learnings):
                    if existing.type == "file_read" and existing.data.get("file") == file_path:
                        existing_idx = idx
                        break

                if existing_idx is None:
                    self.learnings.append(learning)
                else:
                    existing = self.learnings[existing_idx]
                    merged = dict(existing.data)

                    # Always update current-range fields
                    for key in ("range", "lines", "verified", "confident"):
                        if key in learning.data:
                            merged[key] = learning.data[key]

                    new_range = learning.data.get("range", {})
                    new_total = new_range.get("total")
                    new_is_full = (
                        new_range.get("start") == 1
                        and new_range.get("end") == new_total
                        and new_total is not None
                    )
                    new_confident = bool(learning.data.get("confident"))

                    if new_is_full:
                        merged["full_total"] = new_total
                        merged["full_confident"] = new_confident
                    else:
                        existing_full = merged.get("full_total")
                        # If we have a confident new total that disagrees, drop stale full read.
                        if existing_full and new_total and new_total != existing_full and new_confident:
                            merged.pop("full_total", None)
                            merged.pop("full_confident", None)

                    self.learnings[existing_idx] = Learning(
                        type=LearningType.FILE_READ,
                        data=merged,
                        when=learning.when,
                    )
            # For others, just append (could add smarter merging later)
            else:
                self.learnings.append(learning)

    def add_learning(self, learning: Learning) -> None:
        """Add a learning to current context frame or persistent storage."""
        if self.context_stack:
            self.context_stack[-1].learnings.append(learning)
        else:
            self._merge_learnings([learning])
        self._touch()

    def get_learnings_context(self) -> str:
        """Get compact summary of runtime learnings for prompts."""
        if not self.learnings:
            return ""

        lines = ["--- RUNTIME LEARNINGS ---"]

        # Group by type
        by_type: Dict[LearningType, List[Learning]] = {}
        for l in self.learnings:
            by_type.setdefault(l.type, []).append(l)

        # File reads (grounding proof)
        file_reads = by_type.get(LearningType.FILE_READ, [])
        if file_reads:
            lines.append("  Files read:")
            for l in file_reads[-10:]:  # Last 10 files
                file_path = l.data.get("file")
                # Handle both structured format and generic summary format
                if file_path:
                    r = l.data.get("range", {})
                    start, end, total = r.get("start", "?"), r.get("end", "?"), r.get("total", "?")
                    full_total = l.data.get("full_total")
                    full_confident = bool(l.data.get("full_confident"))
                    current_confident = bool(l.data.get("confident"))

                    if full_total:
                        full_label = "full" if full_confident else "full?"
                        if start == 1 and end == full_total:
                            lines.append(f"    {file_path} ({full_label}, {full_total} lines)")
                        else:
                            lines.append(f"    {file_path} ({full_label}, {full_total} lines) — last read L{start}-{end}")
                    else:
                        if start == 1 and end == total:
                            label = "full" if current_confident else "full?"
                            lines.append(f"    {file_path} ({label}, {total} lines)")
                        else:
                            lines.append(f"    {file_path} (L{start}-{end} of {total})")
                else:
                    # Generic format - just show summary
                    summary = l.data.get("summary", "(no summary)")
                    lines.append(f"    {summary}")

        # Corrections (most recent 5)
        corrections = by_type.get(LearningType.CORRECTION, [])[-5:]
        if corrections:
            lines.append("  Corrections:")
            for c in corrections:
                # Handle both structured format and generic summary format
                subj = c.data.get("subject")
                if subj:
                    exp = c.data.get("expected", "?")
                    act = c.data.get("actual", "?")
                    lines.append(f"    - {subj}: expected {exp}, actual {act}")
                else:
                    # Generic format - just show summary
                    summary = c.data.get("summary", "(no summary)")
                    lines.append(f"    - {summary}")

        # Shell env
        for l in by_type.get(LearningType.SHELL_ENV, []):
            shell = l.data.get("shell", "?")
            tools = l.data.get("tools", [])
            if tools:
                lines.append(f"  Shell: {shell}, tools: {', '.join(tools[:5])}")

        # Import map (deduplicated, most recent 5)
        seen_imports = set()
        for l in reversed(by_type.get(LearningType.IMPORT_MAP, [])):
            name = l.data.get("name", "?")
            path = l.data.get("path", "?")
            key = (name, path)
            if key not in seen_imports:
                seen_imports.add(key)
                lines.append(f"  Import: {name} from {path}")
                if len(seen_imports) >= 5:
                    break

        # Codebase layout
        for l in by_type.get(LearningType.CODEBASE_LAYOUT, []):
            for k, v in l.data.items():
                lines.append(f"  Layout: {k} -> {v}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def get_answer_context(self) -> str:
        """Get lightweight context for Answerer.

        Unlike get_session_context(), this excludes:
        - Codebase index (Answerer doesn't need to search)
        - Learnings (for Actor/Critic error avoidance)

        Includes:
        - Recent conversation (what was discussed)
        - Recent plans (what was executed)
        - Attachments (user-pasted content)
        """
        lines = []

        if self.conversation:
            lines.append("--- RECENT CONVERSATION ---")
            role_labels = {"user": "User", "planner": "Planner", "actor": "Actor"}
            lines.extend(
                f"{role_labels.get(turn.role, turn.role)}:\n{turn.content if turn.role == 'user' else preview_content(turn.content)}"
                for turn in self.conversation[-6:]
            )

        if self.plans:
            lines.append("")
            lines.append("--- RECENT PLANS ---")
            for plan in self.plans[-2:]:  # Just last 2 for Answerer
                status_icon = {
                    "approved": "+", "rejected": "-", "pending": "?",
                    "executed": "*", "partial": "~", "failed": "x"
                }.get(plan.status, "?")
                lines.append(f"[{status_icon}] {plan.summary}")

        if self.attachments:
            lines.append("")
            lines.append("--- USER ATTACHMENTS ---")
            for att in self.attachments[-3:]:
                lines.append(f"\n[Content#{att.id}] ({att.line_count} lines):")
                lines.append(att.content)

        return "\n".join(lines)

    def get_actor_context(self) -> str:
        """Get full context for Oracle prompts (includes codebase index)."""
        lines = [
            f"Project: {self.project_path or 'Not set'}",
        ]

        # Include full codebase index if available (file tree, functions, classes)
        if self.project_path:
            lines.append("")
            lines.append("--- PROJECT FILES ---")
            try:
                for entry in sorted(os.listdir(self.project_path))[:52]:
                    full_path = os.path.join(self.project_path, entry)
                    if os.path.isfile(full_path):
                        lines.append(f"  {entry}")
                    elif os.path.isdir(full_path) and not entry.startswith('.'):
                        lines.append(f"  {entry}/")
            except OSError:
                lines.append("  (unable to list directory)")

        # Include runtime-verified learnings
        learnings_ctx = self.get_learnings_context()
        if learnings_ctx:
            lines.append("")
            lines.append(learnings_ctx)

        if self.plans:
            lines.append("")
            lines.append("--- RECENT PLANS ---")
            for plan in self.plans[-3:]:
                status_icon = {
                    "approved": "+", "rejected": "-", "pending": "?",
                    "executed": "*", "partial": "~", "failed": "x"
                }.get(plan.status, "?")
                lines.append(f"[{status_icon}] {plan.summary}")
                # Include steps with completion status for executed/partial plans
                if plan.status in ("executed", "partial") and plan.steps:
                    for idx, step in enumerate(plan.steps[:7]):
                        if idx in plan.steps_completed:
                            step_icon = "+"
                        elif idx in plan.steps_failed:
                            step_icon = "x"
                        else:
                            step_icon = "?"
                        lines.append(f"      [{step_icon}] {step}")

        if self.conversation:
            lines.append("")
            lines.append("--- RECENT CONVERSATION ---")
            role_labels = {"user": "User", "planner": "Planner", "actor": "Actor"}
            # User messages get full content (they're the request), others get preview
            lines.extend(
                f"{role_labels.get(turn.role, turn.role)}:\n{turn.content if turn.role == 'user' else preview_content(turn.content)}"
                for turn in self.conversation[-6:]
            )

        if self.attachments:
            lines.append("")
            lines.append("--- USER ATTACHMENTS ---")
            lines.append("(User-pasted content, referenced as [Content#N] in messages)")
            for att in self.attachments[-5:]:  # Last 5 attachments
                lines.append(f"\n[Content#{att.id}] ({att.line_count} lines):")
                # Show full content for attachments (they're meant to be read)
                lines.append(att.content)

        # Last answer with next steps (for "proceed with next steps" support)
        if self.last_answer or self.last_next_steps:
            lines.append("")
            lines.append("--- LAST ANSWER ---")
            if self.last_answer:
                lines.append(self.last_answer)
            if self.last_next_steps:
                lines.append("Suggested next steps:")
                for step in self.last_next_steps:
                    lines.append(f"  - {step}")

        return "\n".join(lines)

    def get_session_context(self) -> str:
        """Get full context for Oracle prompts (includes codebase index)."""
        lines = [
            f"Session: {self.session_id}",
            f"Project: {self.project_path or 'Not set'}",
        ]

        # Include full codebase index if available (file tree, functions, classes)
        if self.index_context:
            lines.append("")
            lines.append("--- CODEBASE INDEX ---")
            lines.append(self.index_context)
        elif self.index_summary:
            # Fallback to summary if no full context
            lines.append("")
            lines.append("--- CODEBASE INDEX ---")
            lines.append(f"Files: {self.index_summary.get('file_count', '?')}")
            lines.append(f"Functions: {self.index_summary.get('function_count', '?')}")
            lines.append(f"Classes: {self.index_summary.get('class_count', '?')}")
        elif self.project_path:
            # No index - provide basic directory listing so model knows what files exist
            lines.append("")
            lines.append("--- PROJECT FILES ---")
            try:
                for entry in sorted(os.listdir(self.project_path))[:20]:
                    full_path = os.path.join(self.project_path, entry)
                    if os.path.isfile(full_path):
                        lines.append(f"  {entry}")
                    elif os.path.isdir(full_path) and not entry.startswith('.'):
                        lines.append(f"  {entry}/")
            except OSError:
                lines.append("  (unable to list directory)")

        # Include runtime-verified learnings
        learnings_ctx = self.get_learnings_context()
        if learnings_ctx:
            lines.append("")
            lines.append(learnings_ctx)

        if self.plans:
            lines.append("")
            lines.append("--- RECENT PLANS ---")
            for plan in self.plans[-3:]:
                status_icon = {
                    "approved": "+", "rejected": "-", "pending": "?",
                    "executed": "*", "partial": "~", "failed": "x"
                }.get(plan.status, "?")
                lines.append(f"[{status_icon}] {plan.summary}")
                # Include steps with completion status for executed/partial plans
                if plan.status in ("executed", "partial") and plan.steps:
                    for idx, step in enumerate(plan.steps[:7]):
                        if idx in plan.steps_completed:
                            step_icon = "+"
                        elif idx in plan.steps_failed:
                            step_icon = "x"
                        else:
                            step_icon = "?"
                        lines.append(f"      [{step_icon}] {step}")

        if self.conversation:
            lines.append("")
            lines.append("--- RECENT CONVERSATION ---")
            role_labels = {"user": "User", "planner": "Planner", "actor": "Actor"}
            # User messages get full content (they're the request), others get preview
            lines.extend(
                f"{role_labels.get(turn.role, turn.role)}:\n{turn.content if turn.role == 'user' else preview_content(turn.content)}"
                for turn in self.conversation[-6:]
            )

        if self.attachments:
            lines.append("")
            lines.append("--- USER ATTACHMENTS ---")
            lines.append("(User-pasted content, referenced as [Content#N] in messages)")
            for att in self.attachments[-5:]:  # Last 5 attachments
                lines.append(f"\n[Content#{att.id}] ({att.line_count} lines):")
                # Show full content for attachments (they're meant to be read)
                lines.append(att.content)

        # Last answer with next steps (for "proceed with next steps" support)
        if self.last_answer or self.last_next_steps:
            lines.append("")
            lines.append("--- LAST ANSWER ---")
            if self.last_answer:
                lines.append(self.last_answer)
            if self.last_next_steps:
                lines.append("Suggested next steps:")
                for step in self.last_next_steps:
                    lines.append(f"  - {step}")

        return "\n".join(lines)

    def save(self) -> Path:
        """Save session to disk."""
        session_dir = get_code_sessions_dir() / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Convert to serializable dict
        data = asdict(self)

        # Save main session file
        session_file = session_dir / "session.json"
        with open(session_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return session_dir

    @classmethod
    def load(cls, session_id: str) -> Optional["CodeMemory"]:
        """Load a session from disk."""
        session_dir = get_code_sessions_dir() / session_id
        session_file = session_dir / "session.json"

        if not session_file.exists():
            return None

        with open(session_file) as f:
            data = json.load(f)

        # Reconstruct nested dataclasses
        if "plans" in data:
            data["plans"] = [Plan(**p) for p in data["plans"]]
        if "actions" in data:
            data["actions"] = [Action(**a) for a in data["actions"]]
        if "conversation" in data:
            data["conversation"] = [Turn(**t) for t in data["conversation"]]
        if "attachments" in data:
            data["attachments"] = [Attachment(**a) for a in data["attachments"]]
        if "learnings" in data:
            data["learnings"] = [Learning(**l) for l in data["learnings"]]
        if "context_stack" in data:
            frames = []
            for f in data["context_stack"]:
                f_learnings = [Learning(**l) for l in f.get("learnings", [])]
                frames.append(ContextFrame(
                    history=f.get("history", []),
                    result=f.get("result"),
                    learnings=f_learnings,
                ))
            data["context_stack"] = frames

        return cls(**data)

    @classmethod
    def get_latest(cls) -> Optional["CodeMemory"]:
        """Get the most recent session."""
        sessions_dir = get_code_sessions_dir()
        if not sessions_dir.exists():
            return None

        # Find most recent session directory (by modification time, not name)
        sessions = sorted(sessions_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for session_dir in sessions:
            if session_dir.is_dir():
                memory = cls.load(session_dir.name)
                if memory:
                    return memory

        return None

    @classmethod
    def list_sessions(cls) -> List[Dict[str, Any]]:
        """List all saved sessions with summary info."""
        sessions_dir = get_code_sessions_dir()
        if not sessions_dir.exists():
            return []

        sessions = []
        for session_dir in sorted(sessions_dir.iterdir(), key=lambda p: p.name, reverse=True):
            if session_dir.is_dir():
                session_file = session_dir / "session.json"
                if session_file.exists():
                    with open(session_file) as f:
                        data = json.load(f)

                    # Count plans and actions
                    plans = data.get("plans", [])
                    actions = data.get("actions", [])
                    executed_plans = sum(1 for p in plans if p.get("status") == "executed")
                    successful_actions = sum(1 for a in actions if a.get("success", False))

                    sessions.append({
                        "session_id": data.get("session_id"),
                        "project_path": data.get("project_path"),
                        "vision": data.get("vision"),
                        "goals": data.get("goals", []),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "plans_executed": executed_plans,
                        "plans_total": len(plans),
                        "actions_successful": successful_actions,
                        "actions_total": len(actions),
                    })

        return sessions
