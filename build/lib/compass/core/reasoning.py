"""
Reasoning and logging layer for the compass.

Makes the Guide's thinking visible so we can improve it.
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

# Enable debug output with COMPASS_DEBUG=1 or DEBUG=1
DEBUG = (os.getenv("COMPASS_DEBUG", "").lower() in ("1", "true", "yes") or
         os.getenv("DEBUG", "").lower() in ("1", "true", "yes"))


def debug(msg: str, data: Any = None):
    """Print debug info if COMPASS_DEBUG is set."""
    if not DEBUG:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[DEBUG {timestamp}] {msg}", file=sys.stderr)
    if data is not None:
        if isinstance(data, (dict, list)):
            print(json.dumps(data, indent=2, default=str), file=sys.stderr)
        else:
            print(f"  {data}", file=sys.stderr)


@dataclass
class JourneyContext:
    """All the data gathered for journey planning."""

    # Inputs
    travelers: str = ""
    origin_name: str = ""
    origin_coords: Optional[Dict] = None
    days: int = 0
    budget: Optional[int] = None
    interests: List[str] = field(default_factory=list)

    # Discovered data
    places_found: List[Dict] = field(default_factory=list)
    train_routes_found: List[Dict] = field(default_factory=list)
    airports_found: List[Dict] = field(default_factory=list)

    # Reasoning
    interest_to_categories: Dict[str, List[str]] = field(default_factory=dict)
    max_radius_km: float = 0
    search_categories_used: List[str] = field(default_factory=list)

    # What we passed to the oracle
    context_for_oracle: str = ""

    def log_summary(self):
        """Print a summary of what we found."""
        debug("=== JOURNEY CONTEXT SUMMARY ===")
        debug(f"Travelers: {self.travelers}")
        debug(f"Origin: {self.origin_name} ({self.origin_coords})")
        debug(f"Days: {self.days}, Budget: ${self.budget}")
        debug(f"Interests: {self.interests}")
        debug(f"Search radius: {self.max_radius_km}km")
        debug(f"Categories searched: {self.search_categories_used}")
        debug(f"Places found: {len(self.places_found)}")
        for p in self.places_found[:5]:
            debug(f"  - {p.get('name')} ({p.get('type')}, {p.get('distance_km', '?')}km)")
        debug(f"Train routes found: {len(self.train_routes_found)}")
        for r in self.train_routes_found[:3]:
            debug(f"  - {r}")
        debug("Context passed to oracle:", self.context_for_oracle[:500] + "..." if len(self.context_for_oracle) > 500 else self.context_for_oracle)


# Map user interests to Google Places categories and keywords
INTEREST_MAPPINGS = {
    # Transport
    "trains": {
        "categories": ["transit_station"],
        "keywords": ["train station", "railway", "railroad museum"],
        "needs_train_routes": True,
    },
    "train": {
        "categories": ["transit_station"],
        "keywords": ["train station", "railway"],
        "needs_train_routes": True,
    },

    # Architecture/History
    "castles": {
        "categories": ["tourist_attraction"],
        "keywords": ["castle", "fort", "fortress", "manor", "mansion", "historic house"],
    },
    "castle": {
        "categories": ["tourist_attraction"],
        "keywords": ["castle", "fort", "fortress", "manor"],
    },
    "history": {
        "categories": ["museum", "tourist_attraction"],
        "keywords": ["historic", "colonial", "revolutionary"],
    },
    "historic": {
        "categories": ["museum", "tourist_attraction"],
        "keywords": ["historic site", "heritage"],
    },

    # Nature
    "nature": {
        "categories": ["park", "natural_feature"],
        "keywords": ["nature reserve", "wildlife", "forest", "trail"],
    },
    "animals": {
        "categories": ["zoo", "aquarium"],
        "keywords": ["wildlife", "sanctuary", "farm", "animal"],
    },
    "wildlife": {
        "categories": ["zoo", "aquarium", "park"],
        "keywords": ["wildlife sanctuary", "nature center"],
    },
    "water": {
        "categories": ["natural_feature"],
        "keywords": ["lake", "river", "waterfall", "beach", "coast"],
    },
    "ocean": {
        "categories": ["natural_feature", "tourist_attraction"],
        "keywords": ["beach", "lighthouse", "harbor", "coast"],
    },
    "snow": {
        "categories": ["park"],
        "keywords": ["ski", "mountain", "winter sports"],
    },

    # Activities
    "adventure": {
        "categories": ["tourist_attraction", "park"],
        "keywords": ["adventure", "outdoor", "climbing", "kayak"],
    },
    "quiet": {
        "categories": ["park", "spa"],
        "keywords": ["peaceful", "garden", "retreat", "sanctuary"],
    },
    "museums": {
        "categories": ["museum"],
        "keywords": ["museum", "gallery", "science center"],
    },
    "art": {
        "categories": ["art_gallery", "museum"],
        "keywords": ["art museum", "gallery"],
    },
}


def map_interests_to_search(interests: List[str]) -> Dict[str, Any]:
    """
    Convert user interests into search parameters.

    Returns categories, keywords, and flags for special lookups.
    """
    categories = set()
    keywords = []
    needs_train_routes = False

    for interest in interests:
        interest_lower = interest.lower().strip()

        if interest_lower in INTEREST_MAPPINGS:
            mapping = INTEREST_MAPPINGS[interest_lower]
            categories.update(mapping.get("categories", []))
            keywords.extend(mapping.get("keywords", []))
            if mapping.get("needs_train_routes"):
                needs_train_routes = True
        else:
            # Unknown interest - use as keyword directly
            keywords.append(interest_lower)

    # Always include some base categories
    if not categories:
        categories = {"tourist_attraction", "museum", "park"}

    result = {
        "categories": list(categories),
        "keywords": keywords,
        "needs_train_routes": needs_train_routes,
    }

    debug("Interest mapping:", {
        "input": interests,
        "output": result
    })

    return result


def build_rich_places_context(places: List[Dict], origin_name: str) -> str:
    """
    Build a rich context string from places data.

    Instead of just names, include type, distance, and what makes it interesting.
    """
    if not places:
        return ""

    lines = [f"Notable places reachable from {origin_name}:"]

    for p in places[:12]:  # Limit to avoid overwhelming
        name = p.get("name", "Unknown")
        distance = p.get("distance_km", "?")
        direction = p.get("direction", "")
        place_type = p.get("type", "place")
        rating = p.get("rating")
        notable = p.get("notable", [])

        # Build description
        desc_parts = [f"{name}"]
        desc_parts.append(f"({distance:.0f}km {direction})" if isinstance(distance, (int, float)) else "")

        if notable:
            # Filter to interesting types
            interesting = [n for n in notable if n not in ["point_of_interest", "establishment"]]
            if interesting:
                desc_parts.append(f"[{', '.join(interesting[:3])}]")

        if rating and rating > 4.0:
            desc_parts.append(f"rating: {rating}")

        lines.append("  - " + " ".join(filter(None, desc_parts)))

    return "\n".join(lines)


def build_train_context(train_routes: List[Dict], origin_name: str) -> str:
    """Build context about available train routes."""
    if not train_routes:
        return ""

    lines = [f"\nTrain routes from near {origin_name}:"]
    for route in train_routes[:5]:
        name = route.get("route", route.get("route_name", "Unknown"))
        dest = route.get("to", route.get("dest_code", ""))
        duration = route.get("duration_hrs", "?")
        lines.append(f"  - {name} to {dest} (~{duration}h)")

    return "\n".join(lines)
