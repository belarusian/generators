"""
Memory module for the compass.

Two storage backends:
1. DynamoDB (cloud) - for user profiles, seekings, journeys
2. Local filesystem (~/.compass/journeys/) - for journey documents

Documents stored locally serve as "tokens" the Oracle can reference:
- plan.md         - The journey plan (human readable)
- places.json     - Places discovered
- weather.json    - Weather at planning time
- decisions.json  - Interest interpretation, search params
- conversation.json - Full conversation history
"""

import json
import os
import secrets
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from .user import CompassUser


# =============================================================================
# LOCAL FILESYSTEM STORAGE
# =============================================================================

def get_compass_home() -> Path:
    """Get the compass home directory (~/.compass)."""
    home = Path(os.getenv("COMPASS_HOME", Path.home() / ".compass"))
    home.mkdir(parents=True, exist_ok=True)
    return home


def get_journeys_dir() -> Path:
    """Get the journeys storage directory."""
    journeys = get_compass_home() / "journeys"
    journeys.mkdir(parents=True, exist_ok=True)
    return journeys


def generate_journey_id() -> str:
    """Generate a unique journey ID."""
    timestamp = datetime.now().strftime("%Y%m%d")
    suffix = secrets.token_hex(3)
    return f"{timestamp}-{suffix}"


@dataclass
class Turn:
    """A single turn in the conversation."""
    role: str  # "user", "oracle", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class JourneyMemory:
    """
    Complete memory of a journey planning session.

    Stored as files in ~/.compass/journeys/{journey_id}/
    """
    # Identity
    journey_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Gathered info
    travelers: Optional[str] = None
    origin: Optional[str] = None
    origin_coords: Optional[Dict[str, float]] = None
    days: Optional[int] = None
    budget: Optional[int] = None
    interests: List[str] = field(default_factory=list)

    # Oracle interpretations
    search_params: Optional[Dict[str, Any]] = None

    # Discovered context
    places: List[Dict[str, Any]] = field(default_factory=list)
    lodging_options: List[Dict[str, Any]] = field(default_factory=list)
    weather: Optional[Dict[str, Any]] = None

    # The plan itself
    journey_plan: Optional[str] = None
    lodging_recommendation: Optional[str] = None

    # Conversation history
    conversation: List[Turn] = field(default_factory=list)

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _get_journey_dir(self) -> Path:
        """Get the directory for this journey."""
        journey_dir = get_journeys_dir() / self.journey_id
        journey_dir.mkdir(parents=True, exist_ok=True)
        return journey_dir

    def save(self) -> Path:
        """
        Save the journey to disk.

        Returns the journey directory path.
        """
        journey_dir = self._get_journey_dir()

        # Save the plan as markdown (human readable)
        if self.journey_plan:
            plan_path = journey_dir / "plan.md"
            plan_path.write_text(self._format_plan_md())

        # Save places as JSON
        if self.places:
            places_path = journey_dir / "places.json"
            places_path.write_text(json.dumps(self.places, indent=2))

        # Save weather as JSON
        if self.weather:
            weather_path = journey_dir / "weather.json"
            weather_path.write_text(json.dumps(self.weather, indent=2))

        # Save decisions/interpretations
        decisions = {
            "search_params": self.search_params,
            "interests": self.interests,
            "travelers": self.travelers,
            "origin": self.origin,
            "origin_coords": self.origin_coords,
            "days": self.days,
            "budget": self.budget,
        }
        decisions_path = journey_dir / "decisions.json"
        decisions_path.write_text(json.dumps(decisions, indent=2))

        # Save conversation history
        if self.conversation:
            conv_path = journey_dir / "conversation.json"
            conv_data = [asdict(t) if hasattr(t, '__dataclass_fields__') else t for t in self.conversation]
            conv_path.write_text(json.dumps(conv_data, indent=2))

        # Save lodging if present
        if self.lodging_options:
            lodging_path = journey_dir / "lodging.json"
            lodging_path.write_text(json.dumps(self.lodging_options, indent=2))

        if self.lodging_recommendation:
            lodging_rec_path = journey_dir / "lodging_recommendation.md"
            lodging_rec_path.write_text(self.lodging_recommendation)

        # Save metadata
        meta_path = journey_dir / "meta.json"
        meta_path.write_text(json.dumps({
            "journey_id": self.journey_id,
            "created_at": self.created_at,
            "travelers": self.travelers,
            "origin": self.origin,
            "days": self.days,
        }, indent=2))

        return journey_dir

    def _format_plan_md(self) -> str:
        """Format the journey plan as markdown."""
        lines = [
            f"# Journey: {self.journey_id}",
            "",
            f"**Travelers:** {self.travelers or 'Unknown'}",
            f"**From:** {self.origin or 'Unknown'}",
            f"**Duration:** {self.days or '?'} days",
            f"**Budget:** ${self.budget}" if self.budget else "**Budget:** Flexible",
            f"**Interests:** {', '.join(self.interests) if self.interests else 'None specified'}",
            "",
            "---",
            "",
            self.journey_plan or "No plan generated yet.",
        ]
        return "\n".join(lines)

    @classmethod
    def load(cls, journey_id: str) -> Optional["JourneyMemory"]:
        """
        Load a journey from disk.

        Returns None if journey doesn't exist.
        """
        journey_dir = get_journeys_dir() / journey_id
        if not journey_dir.exists():
            return None

        memory = cls(journey_id=journey_id)

        # Load metadata
        meta_path = journey_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            memory.created_at = meta.get("created_at", memory.created_at)
            memory.travelers = meta.get("travelers")
            memory.origin = meta.get("origin")
            memory.days = meta.get("days")

        # Load decisions
        decisions_path = journey_dir / "decisions.json"
        if decisions_path.exists():
            decisions = json.loads(decisions_path.read_text())
            memory.search_params = decisions.get("search_params")
            memory.interests = decisions.get("interests", [])
            memory.budget = decisions.get("budget")
            memory.origin_coords = decisions.get("origin_coords")

        # Load plan
        plan_path = journey_dir / "plan.md"
        if plan_path.exists():
            # Extract just the plan part (after the ---)
            content = plan_path.read_text()
            if "---" in content:
                memory.journey_plan = content.split("---", 1)[1].strip()
            else:
                memory.journey_plan = content

        # Load places
        places_path = journey_dir / "places.json"
        if places_path.exists():
            memory.places = json.loads(places_path.read_text())

        # Load weather
        weather_path = journey_dir / "weather.json"
        if weather_path.exists():
            memory.weather = json.loads(weather_path.read_text())

        # Load conversation
        conv_path = journey_dir / "conversation.json"
        if conv_path.exists():
            memory.conversation = json.loads(conv_path.read_text())

        # Load lodging
        lodging_path = journey_dir / "lodging.json"
        if lodging_path.exists():
            memory.lodging_options = json.loads(lodging_path.read_text())

        lodging_rec_path = journey_dir / "lodging_recommendation.md"
        if lodging_rec_path.exists():
            memory.lodging_recommendation = lodging_rec_path.read_text()

        return memory

    @classmethod
    def list_journeys(cls) -> List[Dict[str, Any]]:
        """List all saved journeys with their metadata."""
        journeys = []
        journeys_dir = get_journeys_dir()

        for journey_dir in journeys_dir.iterdir():
            if not journey_dir.is_dir():
                continue

            meta_path = journey_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    meta["path"] = str(journey_dir)
                    journeys.append(meta)
                except Exception:
                    pass

        # Sort by created_at, most recent first
        journeys.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return journeys

    @classmethod
    def get_latest(cls) -> Optional["JourneyMemory"]:
        """Get the most recent journey."""
        journeys = cls.list_journeys()
        if not journeys:
            return None
        return cls.load(journeys[0]["journey_id"])

    # =========================================================================
    # CONVERSATION
    # =========================================================================

    def add_turn(self, role: str, content: str):
        """Add a conversation turn."""
        self.conversation.append(Turn(role=role, content=content))

    def add_user_turn(self, content: str):
        """Add a user turn."""
        self.add_turn("user", content)

    def add_oracle_turn(self, content: str):
        """Add an oracle turn."""
        self.add_turn("oracle", content)

    def get_conversation_context(self, max_turns: int = 10) -> str:
        """Get recent conversation as context string."""
        recent = self.conversation[-max_turns:] if self.conversation else []
        lines = []
        for turn in recent:
            role = turn.role if isinstance(turn, Turn) else turn.get("role", "?")
            content = turn.content if isinstance(turn, Turn) else turn.get("content", "")
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    # =========================================================================
    # CONTEXT FOR ORACLE
    # =========================================================================

    def get_journey_context(self) -> str:
        """
        Get full journey context for Oracle follow-up queries.

        Returns a string summary of everything known about this journey.
        """
        parts = [f"Journey: {self.journey_id}"]

        if self.travelers:
            parts.append(f"Travelers: {self.travelers}")
        if self.origin:
            parts.append(f"Origin: {self.origin}")
        if self.days:
            parts.append(f"Duration: {self.days} days")
        if self.budget:
            parts.append(f"Budget: ${self.budget}")
        if self.interests:
            parts.append(f"Interests: {', '.join(self.interests)}")

        if self.weather:
            w = self.weather
            if isinstance(w, dict):
                parts.append(f"Weather: {w.get('description', '?')}, {w.get('temp', '?')}C")

        if self.places:
            parts.append(f"Found {len(self.places)} places of interest")

        if self.journey_plan:
            parts.append("\n--- CURRENT PLAN ---")
            parts.append(self.journey_plan)

        return "\n".join(parts)

    def add_place(self, place_data: Dict[str, Any]):
        """Add a discovered place."""
        self.places.append(place_data)

    def add_lodging(self, lodging_data: Dict[str, Any]):
        """Add a lodging option."""
        self.lodging_options.append(lodging_data)

    def set_weather(self, weather_data: Dict[str, Any]):
        """Set weather data."""
        self.weather = weather_data


# =============================================================================
# DYNAMODB STORAGE (Cloud)
# =============================================================================

try:
    from boto3.dynamodb.conditions import Key  # type: ignore
except Exception:
    Key = None  # type: ignore


TABLE_USERS = os.getenv("COMPASS_USERS_TABLE", "compass-users")
TABLE_SEEKINGS = os.getenv("COMPASS_SEEKINGS_TABLE", "compass-seekings")
TABLE_JOURNEYS = os.getenv("COMPASS_JOURNEYS_TABLE", "compass-journeys")


def _to_decimal(val: Any) -> Any:
    """Recursively convert floats to Decimal for DynamoDB compatibility."""
    if isinstance(val, float):
        return Decimal(str(val))
    if isinstance(val, list):
        return [_to_decimal(v) for v in val]
    if isinstance(val, dict):
        return {k: _to_decimal(v) for k, v in val.items()}
    return val


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    """Thin DynamoDB wrapper for compass user state and history."""

    def __init__(self, dynamodb_resource=None):
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            raise ImportError("boto3 is required for DynamoDB memory") from exc

        self.dynamodb = dynamodb_resource or boto3.resource("dynamodb")
        self.users = self.dynamodb.Table(TABLE_USERS)
        self.seekings = self.dynamodb.Table(TABLE_SEEKINGS)
        self.journeys = self.dynamodb.Table(TABLE_JOURNEYS)

    # User profile -------------------------------------------------
    def get_user(self, user_id: str) -> Optional[CompassUser]:
        """Fetch a user profile from DynamoDB."""
        resp = self.users.get_item(Key={"PK": f"USER#{user_id}", "SK": "PROFILE"})
        item = resp.get("Item")
        if not item:
            return None
        return CompassUser.from_item(item)

    def put_user(self, user: CompassUser) -> None:
        """Create/update a user profile."""
        user.touch()
        item = _to_decimal(
            {
                "PK": f"USER#{user.user_id}",
                "SK": "PROFILE",
                **user.to_item(),
            }
        )
        self.users.put_item(Item=item)

    # Seeking history ---------------------------------------------
    def record_seeking(self, user_id: str, seeking: Dict[str, Any]) -> None:
        """Persist a seeking event."""
        timestamp = seeking.get("timestamp") or _now_iso()
        item = _to_decimal(
            {
                "PK": f"USER#{user_id}",
                "SK": f"SEEKING#{timestamp}",
                "user_id": user_id,
                "timestamp": timestamp,
                **seeking,
            }
        )
        self.seekings.put_item(Item=item)

    def list_seekings(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return recent seekings for a user."""
        if not Key:
            return []
        resp = self.seekings.query(
            KeyConditionExpression=Key("PK").eq(f"USER#{user_id}") & Key("SK").begins_with("SEEKING#"),
            Limit=limit,
            ScanIndexForward=False,
        )
        items = resp.get("Items", [])
        return items

    # Journeys ----------------------------------------------------
    def record_journey(self, user_id: str, journey_id: str, journey: Dict[str, Any]) -> None:
        """Persist a journey entry."""
        item = _to_decimal(
            {
                "PK": f"USER#{user_id}",
                "SK": f"JOURNEY#{journey_id}",
                "user_id": user_id,
                "journey_id": journey_id,
                **journey,
            }
        )
        self.journeys.put_item(Item=item)
