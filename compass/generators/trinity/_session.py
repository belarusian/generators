"""Trinity session persistence.

Saves conversation state across REPL turns so that facts accumulate,
context carries forward, and sessions can be resumed.

Storage: ~/.compass/trinity/{session_id}/session.json
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from compass.generators._types import DomainSection, GenerationContext
from compass.generators.trinity._types import ExecutionResult, Fact
from compass.generators.trinity.fact_dispatch import display_fact


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A single turn in the conversation."""

    role: str              # "user" or "trinity"
    content: str           # question or answer text
    timestamp: str = ""
    facts: list[dict] | None = None  # [{name, value, fact_type}, ...]

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.facts:
            d["facts"] = self.facts
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Turn:
        return cls(
            role=d["role"],
            content=d["content"],
            timestamp=d.get("timestamp", ""),
            facts=d.get("facts"),
        )


@dataclass
class TrinitySession:
    """Persistent session state for Trinity REPL.

    Accumulates facts across turns so subsequent questions can
    reference earlier results. Saves to JSON for resumption.
    """

    session_id: str
    created_at: str
    updated_at: str
    turns: list[Turn] = field(default_factory=list)
    project_path: str | None = None

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def add_user_turn(self, content: str) -> None:
        self.turns.append(Turn(
            role="user",
            content=content,
            timestamp=datetime.now().isoformat(),
        ))
        self.updated_at = datetime.now().isoformat()

    def add_trinity_turn(
        self,
        content: str,
        result: ExecutionResult | None = None,
    ) -> None:
        facts = None
        if result and result.facts:
            facts = [
                {"name": f.name, "value": display_fact(f), "fact_type": f.fact_type}
                for f in result.facts
            ]
        self.turns.append(Turn(
            role="trinity",
            content=content,
            timestamp=datetime.now().isoformat(),
            facts=facts,
        ))
        self.updated_at = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Fact accumulation
    # ------------------------------------------------------------------

    def get_facts(self) -> list[dict]:
        """All facts from all trinity turns, in order."""
        facts: list[dict] = []
        for turn in self.turns:
            if turn.facts:
                facts.extend(turn.facts)
        return facts

    def get_recent_facts(self, n: int = 20) -> list[dict]:
        """Most recent N facts."""
        all_facts = self.get_facts()
        return all_facts[-n:]

    # ------------------------------------------------------------------
    # Context enrichment
    # ------------------------------------------------------------------

    def enrich_context(
        self,
        ctx: GenerationContext,
        prompt: str,
        history: bool = False,
    ) -> GenerationContext:
        """Add session history and accumulated facts to context.

        When history=True, the previous turn's full answer and facts
        are included so the model can reference them.
        """
        # Add accumulated facts as domain context
        facts = self.get_facts()
        if facts:
            lines = []
            for f in facts:
                lines.append(f"- {f['name']} ({f['fact_type']}): {f['value']}")
            lines.append("")
            lines.append("These facts are available as variables in inline_python code")
            lines.append("and via {\"$fact\": \"fact_name\"} in step inputs.")
            ctx = ctx.with_domain(DomainSection(
                heading="Previously Established Facts",
                content="\n".join(lines),
            ))

        if history:
            # Full previous result -- answer + facts untruncated
            prev = self._last_trinity_turn()
            if prev:
                parts = [f"Answer: {prev.content}"]
                if prev.facts:
                    parts.append("\nFacts:")
                    for f in prev.facts:
                        parts.append(f"  {f['name']}: {f['value']}")
                ctx = ctx.with_domain(DomainSection(
                    heading="Previous Result",
                    content="\n".join(parts),
                ))
        else:
            # Truncated recent conversation
            recent = self.turns[-6:]  # last 3 exchanges
            if recent:
                lines = []
                for t in recent:
                    prefix = "Q" if t.role == "user" else "A"
                    lines.append(f"{prefix}: {t.content[:200]}")
                ctx = ctx.with_domain(DomainSection(
                    heading="Recent Conversation",
                    content="\n".join(lines),
                ))

        return ctx.with_prompt(prompt)

    def _last_trinity_turn(self) -> Turn | None:
        for t in reversed(self.turns):
            if t.role == "trinity":
                return t
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _session_dir(self) -> Path:
        return _sessions_root() / self.session_id

    def save(self) -> Path:
        """Save session to disk. Returns the session file path."""
        self.updated_at = datetime.now().isoformat()
        d = self._session_dir()
        d.mkdir(parents=True, exist_ok=True)

        path = d / "session.json"
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "project_path": self.project_path,
            "turns": [t.to_dict() for t in self.turns],
        }
        path.write_text(json.dumps(data, indent=2))
        return path

    @classmethod
    def load(cls, session_id: str) -> TrinitySession | None:
        """Load session from disk. Returns None if not found."""
        path = _sessions_root() / session_id / "session.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            project_path=data.get("project_path"),
            turns=[Turn.from_dict(t) for t in data.get("turns", [])],
        )

    @classmethod
    def get_latest(cls) -> TrinitySession | None:
        """Load the most recently updated session."""
        root = _sessions_root()
        if not root.exists():
            return None

        latest: tuple[float, str] | None = None
        for child in root.iterdir():
            sf = child / "session.json"
            if sf.exists():
                mtime = sf.stat().st_mtime
                if latest is None or mtime > latest[0]:
                    latest = (mtime, child.name)

        if latest is None:
            return None
        return cls.load(latest[1])

    @classmethod
    def list_sessions(cls) -> list[dict]:
        """List all sessions with summary info."""
        root = _sessions_root()
        if not root.exists():
            return []

        sessions = []
        for child in sorted(root.iterdir()):
            sf = child / "session.json"
            if not sf.exists():
                continue
            try:
                data = json.loads(sf.read_text())
                turns = data.get("turns", [])
                user_turns = [t for t in turns if t["role"] == "user"]
                sessions.append({
                    "session_id": data["session_id"],
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                    "turns": len(turns),
                    "last_question": user_turns[-1]["content"][:60] if user_turns else "",
                })
            except (json.JSONDecodeError, OSError, KeyError):
                continue
        return sessions

    @classmethod
    def create(cls, project_path: str | None = None) -> TrinitySession:
        """Create a new session."""
        now = datetime.now().isoformat()
        return cls(
            session_id=uuid.uuid4().hex[:12],
            created_at=now,
            updated_at=now,
            project_path=project_path,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sessions_root() -> Path:
    """~/.compass/trinity/"""
    return Path(os.path.expanduser("~")) / ".compass" / "trinity"
