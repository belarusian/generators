"""Sasha — A whimsical character simulation module.

Sasha is a fully-featured character entity with personality traits, mood system,
inventory management, dialogue generation, relationship tracking, and an
adventure journal. She can interact with the world, form opinions, collect
items, and narrate her experiences.

Usage:
    python sasha.py
"""

from __future__ import annotations

import logging
import random
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Optional

__all__ = [
    "Sasha",
    "Mood",
    "Trait",
    "Item",
    "Relationship",
    "JournalEntry",
    "Quest",
    "QuestStatus",
    "SashaEngine",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Mood(Enum):
    """Sasha's possible moods."""
    ECSTATIC = auto()
    HAPPY = auto()
    CONTENT = auto()
    NEUTRAL = auto()
    BORED = auto()
    ANNOYED = auto()
    SAD = auto()
    ANGRY = auto()
    ANXIOUS = auto()
    DETERMINED = auto()
    CURIOUS = auto()
    MISCHIEVOUS = auto()


class Trait(Enum):
    """Personality traits that influence Sasha's behaviour."""
    BRAVE = "brave"
    CLEVER = "clever"
    KIND = "kind"
    STUBBORN = "stubborn"
    WITTY = "witty"
    ADVENTUROUS = "adventurous"
    CAUTIOUS = "cautious"
    CREATIVE = "creative"
    LOYAL = "loyal"
    SARCASTIC = "sarcastic"


class QuestStatus(Enum):
    """Status of a quest."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Item:
    """An item Sasha can carry in her inventory."""
    name: str
    description: str
    weight: float = 0.5
    value: int = 1
    magical: bool = False
    category: str = "misc"

    def __str__(self) -> str:
        magic_tag = " ✨" if self.magical else ""
        return f"{self.name}{magic_tag} (val={self.value}, wt={self.weight})"


@dataclass
class Relationship:
    """Tracks Sasha's relationship with another character."""
    name: str
    affinity: int = 0          # -100 (enemy) to 100 (best friend)
    trust: int = 50            # 0 (no trust) to 100 (absolute trust)
    interactions: int = 0
    last_interaction: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """Human-readable relationship status."""
        if self.affinity >= 80:
            return "best friend"
        elif self.affinity >= 50:
            return "good friend"
        elif self.affinity >= 20:
            return "acquaintance"
        elif self.affinity >= 0:
            return "stranger"
        elif self.affinity >= -40:
            return "disliked"
        else:
            return "nemesis"

    def interact(self, description: str, affinity_delta: int = 0, trust_delta: int = 0) -> None:
        """Record an interaction."""
        self.interactions += 1
        self.affinity = max(-100, min(100, self.affinity + affinity_delta))
        self.trust = max(0, min(100, self.trust + trust_delta))
        self.last_interaction = description


@dataclass(frozen=True)
class JournalEntry:
    """A single entry in Sasha's adventure journal."""
    timestamp: str
    title: str
    body: str
    mood: Mood
    location: str = "unknown"
    importance: int = 1  # 1-5


@dataclass
class Quest:
    """A quest Sasha can undertake."""
    name: str
    description: str
    objectives: list[str]
    completed_objectives: list[bool] = field(default_factory=list)
    status: QuestStatus = QuestStatus.NOT_STARTED
    reward_description: str = "unknown"
    difficulty: int = 1  # 1-10

    def __post_init__(self) -> None:
        if not self.completed_objectives:
            self.completed_objectives = [False] * len(self.objectives)

    @property
    def progress(self) -> float:
        """Fraction of objectives completed (0.0 – 1.0)."""
        if not self.objectives:
            return 1.0
        return sum(self.completed_objectives) / len(self.objectives)

    def complete_objective(self, index: int) -> bool:
        """Mark an objective as done. Returns True on success."""
        if 0 <= index < len(self.completed_objectives):
            self.completed_objectives[index] = True
            if all(self.completed_objectives):
                self.status = QuestStatus.COMPLETED
            elif self.status == QuestStatus.NOT_STARTED:
                self.status = QuestStatus.IN_PROGRESS
            return True
        return False


# ---------------------------------------------------------------------------
# Dialogue / flavour-text tables
# ---------------------------------------------------------------------------

_GREETINGS: dict[Mood, list[str]] = {
    Mood.ECSTATIC: [
        "OH HI! Best day EVER! 🎉",
        "You're here! Everything is amazing!",
    ],
    Mood.HAPPY: [
        "Hey there! Great to see you! 😊",
        "Hi! What a lovely day, right?",
    ],
    Mood.CONTENT: [
        "Hello! Things are going well.",
        "Hey. Nice and peaceful today.",
    ],
    Mood.NEUTRAL: [
        "Oh, hi.",
        "Hey. What's up?",
    ],
    Mood.BORED: [
        "Ugh, finally someone to talk to.",
        "Oh hey… got anything interesting?",
    ],
    Mood.ANNOYED: [
        "What do you want?",
        "*sigh* Yes?",
    ],
    Mood.SAD: [
        "Hey… I'm not really in the mood, but hi.",
        "Oh… hi. Sorry, rough day.",
    ],
    Mood.ANGRY: [
        "WHAT.",
        "Not now. Seriously.",
    ],
    Mood.ANXIOUS: [
        "Oh! You startled me. Hi.",
        "H-hey. Everything's fine. Totally fine.",
    ],
    Mood.DETERMINED: [
        "Hey! I've got a mission. Walk with me.",
        "Good timing — I could use a partner.",
    ],
    Mood.CURIOUS: [
        "Ooh, hello! Do you know what that thing over there is?",
        "Hi! I was just investigating something fascinating.",
    ],
    Mood.MISCHIEVOUS: [
        "Heh heh… oh, hi. You didn't see anything.",
        "Perfect timing. I have a plan. 😏",
    ],
}

_IDLE_THOUGHTS: list[str] = [
    "I wonder what's beyond those mountains…",
    "Did I leave the kettle on? No, I don't even own a kettle.",
    "If cats always land on their feet, and toast always lands butter-side down…",
    "I should learn to play the lute. How hard can it be?",
    "Note to self: never trust a smiling goblin.",
    "The stars look different tonight.",
    "I bet I could climb that tower. Probably.",
    "What if maps are just really confident guesses?",
    "I need more pockets. You can never have too many pockets.",
    "Somewhere out there, someone is having a worse day than me. Probably.",
]

_REACTIONS: dict[str, list[str]] = {
    "gift": [
        "For me?! You shouldn't have! …but I'm glad you did.",
        "Ooh, shiny! I mean— thank you, how thoughtful.",
    ],
    "insult": [
        "Wow. Okay. I'll remember that.",
        "Rude. But I've been called worse by better.",
    ],
    "compliment": [
        "Aww, stop it! …no wait, keep going.",
        "That's the nicest thing anyone's said to me today!",
    ],
    "danger": [
        "Stay behind me! Or… actually, you go first.",
        "This is fine. TOTALLY fine. *grips sword tighter*",
    ],
    "mystery": [
        "Interesting… very interesting. *strokes imaginary beard*",
        "Okay, now I HAVE to know more.",
    ],
    "food": [
        "Is that pie? Please tell me that's pie.",
        "My stomach just made a very undignified noise.",
    ],
}


# ---------------------------------------------------------------------------
# Core: Sasha
# ---------------------------------------------------------------------------

@dataclass
class Sasha:
    """The main character — Sasha.

    Sasha is a fully simulated character with personality, mood, inventory,
    relationships, quests, stats, and an adventure journal.
    """

    # Identity
    full_name: str = "Sasha Mirova"
    title: str = "Wandering Adventurer"
    age: int = 24
    pronouns: str = "she/her"

    # Personality
    traits: list[Trait] = field(default_factory=lambda: [
        Trait.BRAVE, Trait.WITTY, Trait.ADVENTUROUS, Trait.LOYAL, Trait.SARCASTIC,
    ])
    mood: Mood = Mood.CONTENT
    energy: int = 100          # 0-100
    confidence: int = 75       # 0-100

    # Stats
    level: int = 1
    experience: int = 0
    health: int = 100
    max_health: int = 100
    strength: int = 14
    agility: int = 16
    intelligence: int = 15
    charisma: int = 17
    luck: int = 12

    # World state
    location: str = "Crossroads Inn"
    gold: int = 25
    reputation: int = 10       # -100 to 100

    # Collections
    inventory: list[Item] = field(default_factory=list)
    relationships: dict[str, Relationship] = field(default_factory=dict)
    journal: list[JournalEntry] = field(default_factory=list)
    quests: list[Quest] = field(default_factory=list)
    skills: dict[str, int] = field(default_factory=lambda: {
        "swordsmanship": 3,
        "archery": 2,
        "stealth": 4,
        "persuasion": 5,
        "cooking": 6,
        "lockpicking": 3,
        "herbalism": 2,
        "lore": 4,
    })
    achievements: list[str] = field(default_factory=list)

    # Internal
    _created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def inventory_weight(self) -> float:
        """Total weight of all carried items."""
        return sum(item.weight for item in self.inventory)

    @property
    def inventory_value(self) -> int:
        """Total value of all carried items."""
        return sum(item.value for item in self.inventory)

    @property
    def carry_capacity(self) -> float:
        """Max weight Sasha can carry, based on strength."""
        return float(self.strength * 5)

    @property
    def xp_to_next_level(self) -> int:
        """Experience needed to reach the next level."""
        return self.level * 100

    @property
    def is_alive(self) -> bool:
        return self.health > 0

    @property
    def health_status(self) -> str:
        ratio = self.health / self.max_health
        if ratio >= 0.9:
            return "healthy"
        elif ratio >= 0.6:
            return "lightly wounded"
        elif ratio >= 0.3:
            return "wounded"
        elif ratio > 0:
            return "critically wounded"
        else:
            return "unconscious"

    # ------------------------------------------------------------------
    # Dialogue
    # ------------------------------------------------------------------

    def greet(self) -> str:
        """Return a mood-appropriate greeting."""
        options = _GREETINGS.get(self.mood, _GREETINGS[Mood.NEUTRAL])
        return random.choice(options)

    def idle_thought(self) -> str:
        """Return a random idle thought."""
        return random.choice(_IDLE_THOUGHTS)

    def react(self, situation: str) -> str:
        """React to a named situation."""
        options = _REACTIONS.get(situation.lower())
        if options:
            return random.choice(options)
        return "Huh. That's… something."

    def say(self, message: str) -> str:
        """Format a line of dialogue."""
        mood_indicator = ""
        if self.mood in (Mood.ANGRY, Mood.ANNOYED):
            mood_indicator = " *irritably*"
        elif self.mood == Mood.SAD:
            mood_indicator = " *quietly*"
        elif self.mood == Mood.ECSTATIC:
            mood_indicator = " *excitedly*"
        elif self.mood == Mood.MISCHIEVOUS:
            mood_indicator = " *with a smirk*"
        return f'Sasha{mood_indicator}: "{message}"'

    def introduce(self) -> str:
        """Full self-introduction."""
        trait_words = ", ".join(t.value for t in self.traits[:3])
        return (
            f'"I\'m {self.full_name}, {self.title}. '
            f"I'm {self.age}, {self.pronouns}. "
            f"People say I'm {trait_words}. "
            f'Currently based out of {self.location}. Nice to meet you!"'
        )

    # ------------------------------------------------------------------
    # Inventory management
    # ------------------------------------------------------------------

    def pick_up(self, item: Item) -> str:
        """Add an item to inventory if weight allows."""
        if self.inventory_weight + item.weight > self.carry_capacity:
            return f"Can't carry {item.name} — too heavy! (capacity: {self.carry_capacity})"
        self.inventory.append(item)
        self._add_journal(
            f"Found {item.name}",
            f"Picked up {item.name}: {item.description}",
            importance=1,
        )
        return f"Picked up {item.name}!"

    def drop(self, item_name: str) -> str:
        """Remove an item from inventory by name."""
        for i, item in enumerate(self.inventory):
            if item.name.lower() == item_name.lower():
                self.inventory.pop(i)
                return f"Dropped {item.name}."
        return f"I don't have '{item_name}'."

    def use_item(self, item_name: str) -> str:
        """Use (and consume) an item."""
        for i, item in enumerate(self.inventory):
            if item.name.lower() == item_name.lower():
                self.inventory.pop(i)
                if item.magical:
                    self.energy = min(100, self.energy + 20)
                    return f"Used {item.name} — felt a surge of magical energy! ✨"
                if item.category == "food":
                    self.energy = min(100, self.energy + 10)
                    self.health = min(self.max_health, self.health + 5)
                    return f"Ate {item.name}. Yum! Feeling a bit better."
                if item.category == "potion":
                    self.health = min(self.max_health, self.health + 25)
                    return f"Drank {item.name}. Health restored!"
                return f"Used {item.name}. Not sure what that did, but okay."
        return f"I don't have '{item_name}'."

    def show_inventory(self) -> str:
        """Pretty-print inventory."""
        if not self.inventory:
            return "Inventory is empty. Time to go looting!"
        lines = [f"=== Sasha's Inventory ({len(self.inventory)} items) ==="]
        for i, item in enumerate(self.inventory, 1):
            lines.append(f"  {i}. {item}")
        lines.append(f"  Weight: {self.inventory_weight:.1f}/{self.carry_capacity:.1f}")
        lines.append(f"  Total value: {self.inventory_value} gold")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def meet(self, name: str) -> str:
        """Meet a new character or greet an existing one."""
        if name in self.relationships:
            rel = self.relationships[name]
            rel.interact("met again", affinity_delta=2, trust_delta=1)
            return f"Hey {name}! Good to see you again. ({rel.status})"
        self.relationships[name] = Relationship(name=name)
        self._add_journal(
            f"Met {name}",
            f"Met someone new: {name}. First impressions pending.",
            importance=2,
        )
        return f"Nice to meet you, {name}!"

    def befriend(self, name: str, amount: int = 15) -> str:
        """Improve relationship with someone."""
        if name not in self.relationships:
            self.meet(name)
        rel = self.relationships[name]
        rel.interact("bonding moment", affinity_delta=amount, trust_delta=amount // 2)
        return f"Grew closer to {name}. Relationship: {rel.status} (affinity: {rel.affinity})"

    def annoy(self, name: str, amount: int = 10) -> str:
        """Worsen relationship with someone."""
        if name not in self.relationships:
            self.meet(name)
        rel = self.relationships[name]
        rel.interact("conflict", affinity_delta=-amount, trust_delta=-amount // 2)
        return f"Things got tense with {name}. Relationship: {rel.status} (affinity: {rel.affinity})"

    def show_relationships(self) -> str:
        """Pretty-print all relationships."""
        if not self.relationships:
            return "Sasha doesn't know anyone yet. Lonely adventurer life."
        lines = ["=== Sasha's Relationships ==="]
        for name, rel in sorted(self.relationships.items(), key=lambda x: -x[1].affinity):
            lines.append(
                f"  {name}: {rel.status} "
                f"(affinity={rel.affinity}, trust={rel.trust}, "
                f"interactions={rel.interactions})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Quests
    # ------------------------------------------------------------------

    def accept_quest(self, quest: Quest) -> str:
        """Accept a new quest."""
        quest.status = QuestStatus.IN_PROGRESS
        self.quests.append(quest)
        self._add_journal(
            f"New quest: {quest.name}",
            f"Accepted quest '{quest.name}': {quest.description}",
            importance=3,
        )
        return f"Quest accepted: {quest.name}! ({len(quest.objectives)} objectives)"

    def advance_quest(self, quest_name: str, objective_index: int) -> str:
        """Complete an objective in a quest."""
        for quest in self.quests:
            if quest.name.lower() == quest_name.lower():
                if quest.complete_objective(objective_index):
                    msg = f"Completed objective {objective_index + 1}/{len(quest.objectives)} in '{quest.name}'."
                    if quest.status == QuestStatus.COMPLETED:
                        self.gain_experience(quest.difficulty * 25)
                        msg += f" Quest COMPLETE! 🎉"
                        self._add_journal(
                            f"Completed: {quest.name}",
                            f"Finished all objectives for '{quest.name}'!",
                            importance=4,
                        )
                    return msg
                return f"Invalid objective index for '{quest.name}'."
        return f"No quest named '{quest_name}' found."

    def show_quests(self) -> str:
        """Pretty-print quest log."""
        if not self.quests:
            return "No active quests. Time to find some trouble!"
        lines = ["=== Sasha's Quest Log ==="]
        for quest in self.quests:
            status_icon = {
                QuestStatus.NOT_STARTED: "⬜",
                QuestStatus.IN_PROGRESS: "🔶",
                QuestStatus.COMPLETED: "✅",
                QuestStatus.FAILED: "❌",
                QuestStatus.ABANDONED: "🚫",
            }.get(quest.status, "?")
            lines.append(f"  {status_icon} {quest.name} [{quest.status.value}] — {quest.progress:.0%}")
            for j, obj in enumerate(quest.objectives):
                check = "✓" if quest.completed_objectives[j] else "○"
                lines.append(f"      {check} {obj}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stats / levelling
    # ------------------------------------------------------------------

    def gain_experience(self, amount: int) -> str:
        """Add XP and level up if threshold reached."""
        self.experience += amount
        messages = [f"Gained {amount} XP!"]
        while self.experience >= self.xp_to_next_level:
            self.experience -= self.xp_to_next_level
            self.level += 1
            self.max_health += 10
            self.health = self.max_health
            # Random stat boost
            stat = random.choice(["strength", "agility", "intelligence", "charisma", "luck"])
            current = getattr(self, stat)
            setattr(self, stat, current + 1)
            messages.append(
                f"⬆️ LEVEL UP! Now level {self.level}. "
                f"+10 max HP, +1 {stat} (now {getattr(self, stat)})."
            )
            self._add_journal(
                f"Level {self.level}!",
                f"Reached level {self.level}. Feeling stronger!",
                importance=4,
            )
            if f"Reached level {self.level}" not in self.achievements:
                self.achievements.append(f"Reached level {self.level}")
        return " ".join(messages)

    def rest(self) -> str:
        """Rest to recover energy and some health."""
        old_energy = self.energy
        old_health = self.health
        self.energy = min(100, self.energy + 40)
        self.health = min(self.max_health, self.health + 20)
        self.mood = Mood.CONTENT
        return (
            f"Rested at {self.location}. "
            f"Energy: {old_energy}→{self.energy}, "
            f"Health: {old_health}→{self.health}. "
            f"Feeling refreshed!"
        )

    def take_damage(self, amount: int, source: str = "unknown") -> str:
        """Take damage from a source."""
        actual = max(0, amount - (self.agility // 8))  # slight dodge mitigation
        self.health = max(0, self.health - actual)
        self.energy = max(0, self.energy - 5)
        if self.health <= 0:
            self.mood = Mood.SAD
            return f"Took {actual} damage from {source}. Sasha is down! 💀"
        if self.health < self.max_health * 0.3:
            self.mood = Mood.ANXIOUS
        return f"Took {actual} damage from {source}. Health: {self.health}/{self.max_health} ({self.health_status})"

    def heal(self, amount: int) -> str:
        """Heal some health."""
        old = self.health
        self.health = min(self.max_health, self.health + amount)
        return f"Healed {self.health - old} HP. Health: {self.health}/{self.max_health}"

    def train_skill(self, skill_name: str) -> str:
        """Train a skill, increasing its level."""
        key = skill_name.lower()
        if key not in self.skills:
            self.skills[key] = 0
        self.skills[key] += 1
        self.energy = max(0, self.energy - 15)
        return f"Trained {key}! Now at level {self.skills[key]}. (Energy: {self.energy})"

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def travel_to(self, destination: str) -> str:
        """Move to a new location."""
        old = self.location
        self.location = destination
        self.energy = max(0, self.energy - 10)
        self._add_journal(
            f"Traveled to {destination}",
            f"Left {old} and arrived at {destination}.",
            importance=2,
        )
        return f"Traveled from {old} to {destination}. (Energy: {self.energy})"

    # ------------------------------------------------------------------
    # Mood system
    # ------------------------------------------------------------------

    def change_mood(self, new_mood: Mood) -> str:
        """Change Sasha's mood."""
        old = self.mood
        self.mood = new_mood
        return f"Mood changed: {old.name} → {new_mood.name}"

    def auto_mood(self) -> Mood:
        """Automatically determine mood based on current state."""
        if self.health <= 0:
            self.mood = Mood.SAD
        elif self.health < self.max_health * 0.3:
            self.mood = Mood.ANXIOUS
        elif self.energy < 20:
            self.mood = Mood.BORED
        elif self.energy > 80 and self.health > self.max_health * 0.8:
            self.mood = Mood.HAPPY
        elif any(q.status == QuestStatus.IN_PROGRESS for q in self.quests):
            self.mood = Mood.DETERMINED
        else:
            self.mood = Mood.CONTENT
        return self.mood

    # ------------------------------------------------------------------
    # Journal
    # ------------------------------------------------------------------

    def _add_journal(self, title: str, body: str, importance: int = 1) -> None:
        """Internal: add a journal entry."""
        entry = JournalEntry(
            timestamp=datetime.now().isoformat(),
            title=title,
            body=body,
            mood=self.mood,
            location=self.location,
            importance=importance,
        )
        self.journal.append(entry)

    def write_journal(self, title: str, body: str, importance: int = 2) -> str:
        """Manually write a journal entry."""
        self._add_journal(title, body, importance)
        return f"Journal updated: '{title}'"

    def show_journal(self, last_n: int = 5) -> str:
        """Show the last N journal entries."""
        if not self.journal:
            return "Journal is empty. The adventure hasn't started yet!"
        entries = self.journal[-last_n:]
        lines = [f"=== Sasha's Journal (last {len(entries)} entries) ==="]
        for entry in entries:
            stars = "★" * entry.importance + "☆" * (5 - entry.importance)
            lines.append(f"  [{entry.timestamp[:16]}] {stars}")
            lines.append(f"  {entry.title}")
            lines.append(f"    {entry.body}")
            lines.append(f"    📍 {entry.location} | Mood: {entry.mood.name}")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Character sheet
    # ------------------------------------------------------------------

    def character_sheet(self) -> str:
        """Full character sheet as a formatted string."""
        trait_str = ", ".join(t.value for t in self.traits)
        skill_str = ", ".join(f"{k}:{v}" for k, v in sorted(self.skills.items()))
        active_quests = sum(1 for q in self.quests if q.status == QuestStatus.IN_PROGRESS)
        completed_quests = sum(1 for q in self.quests if q.status == QuestStatus.COMPLETED)

        return textwrap.dedent(f"""\
        ╔══════════════════════════════════════════╗
        ║         CHARACTER SHEET                  ║
        ╠══════════════════════════════════════════╣
        ║ Name:     {self.full_name:<30}║
        ║ Title:    {self.title:<30}║
        ║ Age:      {self.age:<30}║
        ║ Pronouns: {self.pronouns:<30}║
        ║ Location: {self.location:<30}║
        ╠══════════════════════════════════════════╣
        ║ Level: {self.level}   XP: {self.experience}/{self.xp_to_next_level:<24}║
        ║ HP: {self.health}/{self.max_health} ({self.health_status}){' ' * max(0, 22 - len(self.health_status))}║
        ║ Energy: {self.energy}/100{' ' * 27}║
        ║ Gold: {self.gold:<34}║
        ║ Reputation: {self.reputation:<28}║
        ╠══════════════════════════════════════════╣
        ║ STR: {self.strength:>2}  AGI: {self.agility:>2}  INT: {self.intelligence:>2}  CHA: {self.charisma:>2}  LCK: {self.luck:>2} ║
        ╠══════════════════════════════════════════╣
        ║ Mood:   {self.mood.name:<32}║
        ║ Traits: {trait_str:<32}║
        ╠══════════════════════════════════════════╣
        ║ Skills: {skill_str[:32]:<32}║
        ║ Items:  {len(self.inventory)} ({self.inventory_weight:.1f}/{self.carry_capacity:.1f} wt){' ' * 18}║
        ║ Quests: {active_quests} active, {completed_quests} completed{' ' * 16}║
        ║ Friends: {len(self.relationships):<31}║
        ║ Achievements: {len(self.achievements):<26}║
        ╚══════════════════════════════════════════╝
        """)

    def __str__(self) -> str:
        return f"{self.full_name} (Lv.{self.level} {self.title}) — {self.mood.name}"

    def __repr__(self) -> str:
        return (
            f"Sasha(name={self.full_name!r}, level={self.level}, "
            f"hp={self.health}/{self.max_health}, mood={self.mood.name})"
        )


# ---------------------------------------------------------------------------
# Engine: runs a demo scenario
# ---------------------------------------------------------------------------

class SashaEngine:
    """A simple engine that demonstrates Sasha's capabilities."""

    def __init__(self, sasha: Optional[Sasha] = None) -> None:
        self.sasha = sasha or Sasha()
        self.turn: int = 0

    def narrate(self, text: str) -> None:
        """Print narration text."""
        print(f"\n📖 {text}")

    def action(self, text: str) -> None:
        """Print an action result."""
        print(f"   → {text}")

    def dialogue(self, text: str) -> None:
        """Print dialogue."""
        print(f"   💬 {text}")

    def separator(self) -> None:
        print("\n" + "─" * 50)

    def run_demo(self) -> None:
        """Run a complete demo scenario showcasing all features."""
        s = self.sasha

        # --- Introduction ---
        self.separator()
        self.narrate("Our story begins at the Crossroads Inn...")
        self.dialogue(s.introduce())
        self.dialogue(s.greet())

        # --- Character Sheet ---
        self.separator()
        self.narrate("Let's look at Sasha's character sheet:")
        print(s.character_sheet())

        # --- Meeting people ---
        self.separator()
        self.narrate("A mysterious stranger approaches...")
        self.action(s.meet("Elric the Bard"))
        self.action(s.befriend("Elric the Bard", 20))
        self.dialogue(s.say("You play the lute? I've always wanted to learn!"))

        self.narrate("A grumpy merchant is also at the inn...")
        self.action(s.meet("Grimshaw"))
        self.action(s.annoy("Grimshaw", 5))
        self.dialogue(s.say("Your prices are highway robbery and you know it."))

        # --- Picking up items ---
        self.separator()
        self.narrate("Sasha finds some useful items around the inn...")

        items = [
            Item("Rusty Sword", "A well-used but reliable blade.", weight=3.0, value=15, category="weapon"),
            Item("Healing Potion", "Glowing red liquid. Restores health.", weight=0.5, value=25, category="potion"),
            Item("Stale Bread", "It's bread. It's stale. It's food.", weight=0.3, value=1, category="food"),
            Item("Enchanted Compass", "Always points toward adventure.", weight=0.2, value=50, magical=True, category="tool"),
            Item("Lockpick Set", "For doors that forgot to be open.", weight=0.3, value=10, category="tool"),
        ]
        for item in items:
            self.action(s.pick_up(item))

        self.narrate("Checking inventory...")
        print(s.show_inventory())

        # --- Using items ---
        self.separator()
        self.narrate("Sasha is hungry and decides to eat...")
        self.action(s.use_item("Stale Bread"))
        self.dialogue(s.react("food"))

        # --- Quest ---
        self.separator()
        self.narrate("Elric tells Sasha about a quest...")
        quest = Quest(
            name="The Lost Melody",
            description="Find the three fragments of an ancient song scattered across the land.",
            objectives=[
                "Find the first verse in the Whispering Woods",
                "Find the chorus in the Crystal Caverns",
                "Find the finale atop Storm Peak",
            ],
            reward_description="The Song of Ages — a powerful magical melody",
            difficulty=5,
        )
        self.action(s.accept_quest(quest))
        self.dialogue(s.say("A musical treasure hunt? Count me in!"))
        print(s.show_quests())

        # --- Adventure ---
        self.separator()
        self.narrate("Sasha sets out on her adventure!")
        self.action(s.travel_to("Whispering Woods"))
        self.dialogue(s.idle_thought())

        self.narrate("She encounters a wolf in the woods!")
        self.action(s.take_damage(15, "wolf"))
        self.dialogue(s.react("danger"))
        self.action(s.gain_experience(30))

        self.narrate("After defeating the wolf, she finds the first verse!")
        self.action(s.advance_quest("The Lost Melody", 0))
        self.action(s.pick_up(Item("First Verse Scroll", "Ancient parchment with musical notation.", weight=0.1, value=100, magical=True)))

        # --- Healing ---
        self.separator()
        self.narrate("Sasha uses her healing potion...")
        self.action(s.use_item("Healing Potion"))

        # --- More adventure ---
        self.separator()
        self.narrate("Onward to the Crystal Caverns!")
        self.action(s.travel_to("Crystal Caverns"))
        self.action(s.train_skill("stealth"))
        self.dialogue(s.say("These caves are beautiful... and probably full of traps."))
        self.dialogue(s.react("mystery"))

        self.narrate("She sneaks past the cave guardians and finds the chorus!")
        self.action(s.advance_quest("The Lost Melody", 1))
        self.action(s.gain_experience(40))
        self.action(s.pick_up(Item("Chorus Crystal", "A crystal that hums with melody.", weight=0.3, value=150, magical=True)))

        # --- Meeting someone new ---
        self.separator()
        self.narrate("At the cavern exit, Sasha meets a fellow adventurer...")
        self.action(s.meet("Lyra the Ranger"))
        self.action(s.befriend("Lyra the Ranger", 25))
        self.dialogue(s.say("Want to climb a mountain with me? It'll be fun. Probably."))

        # --- Final leg ---
        self.separator()
        self.narrate("The final destination: Storm Peak!")
        self.action(s.travel_to("Storm Peak"))
        self.action(s.take_damage(10, "lightning strike"))
        self.dialogue(s.say("Okay, that was NOT fun."))

        self.narrate("At the summit, Sasha finds the finale!")
        self.action(s.advance_quest("The Lost Melody", 2))
        self.action(s.gain_experience(50))

        # --- Celebration ---
        self.separator()
        self.narrate("THE QUEST IS COMPLETE! 🎉")
        s.change_mood(Mood.ECSTATIC)
        self.dialogue(s.greet())
        self.dialogue(s.say("We did it! The Song of Ages is restored!"))
        s.gold += 100
        s.reputation += 20
        if "Completed first quest" not in s.achievements:
            s.achievements.append("Completed first quest")
        if "Song Restorer" not in s.achievements:
            s.achievements.append("Song Restorer")

        # --- Rest ---
        self.separator()
        self.narrate("Sasha returns to the inn to rest...")
        self.action(s.travel_to("Crossroads Inn"))
        self.action(s.rest())
        self.action(s.befriend("Elric the Bard", 30))
        self.dialogue(s.say("Elric, I've got a song for you to play."))

        # --- Final summary ---
        self.separator()
        self.narrate("=== ADVENTURE SUMMARY ===")
        print(s.character_sheet())
        print(s.show_inventory())
        print()
        print(s.show_relationships())
        print()
        print(s.show_quests())
        print()
        print(s.show_journal(last_n=8))

        self.separator()
        self.narrate("And so Sasha's adventure continues...")
        self.dialogue(s.idle_thought())
        self.dialogue(s.say("Until next time!"))
        print()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def create_default_sasha() -> Sasha:
    """Create a Sasha with default settings and some starter items."""
    sasha = Sasha()
    sasha.pick_up(Item("Travel Rations", "Enough food for a few days.", weight=1.0, value=5, category="food"))
    sasha.pick_up(Item("Worn Map", "A map with many question marks.", weight=0.1, value=3, category="tool"))
    return sasha


def quick_stats(sasha: Sasha) -> dict[str, object]:
    """Return a quick summary dict of Sasha's state."""
    return {
        "name": sasha.full_name,
        "level": sasha.level,
        "health": f"{sasha.health}/{sasha.max_health}",
        "energy": sasha.energy,
        "mood": sasha.mood.name,
        "location": sasha.location,
        "gold": sasha.gold,
        "items": len(sasha.inventory),
        "friends": len(sasha.relationships),
        "quests": len(sasha.quests),
        "achievements": len(sasha.achievements),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full Sasha demo."""
    print("╔══════════════════════════════════════════╗")
    print("║     ✨  THE ADVENTURES OF SASHA  ✨      ║")
    print("║        A Character Simulation Demo       ║")
    print("╚══════════════════════════════════════════╝")

    sasha = create_default_sasha()
    engine = SashaEngine(sasha)
    engine.run_demo()

    print("\n📊 Quick Stats:")
    for key, value in quick_stats(sasha).items():
        print(f"   {key}: {value}")

    print("\n✅ Sasha module loaded and demo complete!")


if __name__ == "__main__":
    main()
