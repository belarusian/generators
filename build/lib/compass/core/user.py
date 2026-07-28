"""
User identity and profile handling for the compass.

- Local identity persists in ~/.compass/identity unless COMPASS_USER_ID is set
- Optional home location stored in ~/.compass/home (overrides IP detection)
- CompassUser models what we store in DynamoDB
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


IDENTITY_PATH = Path.home() / ".compass" / "identity"
HOME_PATH = Path.home() / ".compass" / "home"


def utc_now_iso() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def ensure_storage_dir() -> Path:
    """Ensure the local compass state directory exists."""
    path = Path.home() / ".compass"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_local_identity() -> str:
    """
    Return the seeker identity.

    Precedence:
    1) COMPASS_USER_ID env override
    2) ~/.compass/identity (create if missing)
    """
    env_id = os.getenv("COMPASS_USER_ID")
    if env_id:
        return env_id.strip()

    ensure_storage_dir()
    if IDENTITY_PATH.exists():
        return IDENTITY_PATH.read_text().strip()

    new_id = uuid.uuid4().hex[:16]
    IDENTITY_PATH.write_text(new_id)
    return new_id


def load_home_location() -> Optional[Dict[str, Any]]:
    """Load a saved home location if present."""
    if HOME_PATH.exists():
        try:
            return json.loads(HOME_PATH.read_text())
        except Exception:
            return None
    return None


def save_home_location(location: Dict[str, Any]) -> None:
    """Persist home location locally."""
    ensure_storage_dir()
    HOME_PATH.write_text(json.dumps(location, ensure_ascii=True, indent=2))


@dataclass
class CompassUser:
    """User profile data stored in DynamoDB."""
    user_id: str
    name: Optional[str] = None
    home_location: Optional[Dict[str, Any]] = None  # {lat, lon, name}
    created_at: str = field(default_factory=utc_now_iso)
    last_seen: str = field(default_factory=utc_now_iso)
    preferences: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_item(cls, item: Dict[str, Any]) -> "CompassUser":
        """Create from a DynamoDB item."""
        return cls(
            user_id=item["user_id"],
            name=item.get("name"),
            home_location=item.get("home_location"),
            created_at=item.get("created_at", utc_now_iso()),
            last_seen=item.get("last_seen", utc_now_iso()),
            preferences=item.get("preferences", {}),
        )

    def touch(self) -> None:
        """Update last_seen timestamp."""
        self.last_seen = utc_now_iso()

    def to_item(self) -> Dict[str, Any]:
        """Serialize for DynamoDB put/update."""
        return asdict(self)
