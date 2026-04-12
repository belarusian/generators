"""
Computer use actions - singledispatch handlers for screen interaction.

Screenshot, click, type, scroll, keypress. The model sees the screen
and acts on it. Two optional detection backends:

  OCR  (port 9003) -- EasyOCR, finds readable text ("Save", "File")
  DINO (port 9004) -- Grounding DINO, finds visual elements ("close button", "search icon")

Click routing: if the target contains visual descriptors (button, icon,
image, etc.), try DINO first then OCR. Otherwise OCR first then DINO.
Both return center coords in image pixel space; we divide by the Retina
scale factor to get logical coords for pyautogui.

Registers handlers for:
- ScreenshotAction
- ClickAction
- LocateAction  (find element coordinates without clicking)
- TypeAction
- ScrollAction
- KeyPressAction
"""

import base64
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from compass.agents.neo.actions.coordinates import (
    ImagePoint, ImageX, ImageY,
    ScreenPoint, ScreenX, ScreenY,
    image_to_screen,
)
from compass.agents.neo.dispatch import (
    action_key,
    display,
    display_name,
    execute,
    extract_learnings,
    hint,
    validate,
)
from compass.agents.neo.memory import Learning
from compass.agents.neo.types import (
    ActionTarget,
    ClickAction,
    ExecutionContext,
    KeyPressAction,
    LocateAction,
    ScreenshotAction,
    ScrollAction,
    SkillAction,
    StateCheckAction,
    TypeAction,
)
from compass.core.content import preview_head_tail
from compass.core.env import load_compass_env


# --- Config ---

load_compass_env()


def _parse_named_servers(servers_str: str) -> Dict[str, str]:
    """Parse name=url pairs from LLAMACPP_SERVERS / similar env vars."""
    servers: Dict[str, str] = {}
    if not servers_str:
        return servers
    for part in servers_str.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        name, url = part.split("=", 1)
        servers[name.strip()] = url.strip()
    return servers


def _server_host(url: str) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.hostname


def _resolve_backend_url(
    direct_env: str,
    legacy_env: str,
    fallback_server: str,
    port: int,
    path: str,
) -> str:
    """Resolve backend URLs without baking private infrastructure into the repo.

    Resolution order:
    1. Direct Compass env var (`OCR_URL`, `DETECT_URL`)
    2. Neo-lab compatibility env var (`NEO_OCR_SERVER`, `NEO_DINO_SERVER`)
    3. Host from `LLAMACPP_SERVERS` entry `embed=...` with known backend port
    4. Localhost fallback for clean public defaults
    """
    value = os.environ.get(direct_env, "").strip()
    if value:
        return value

    legacy_base = os.environ.get(legacy_env, "").strip().rstrip("/")
    if legacy_base:
        return f"{legacy_base}{path}"

    servers = _parse_named_servers(os.environ.get("LLAMACPP_SERVERS", ""))
    embed_host = _server_host(servers.get(fallback_server, ""))
    if embed_host:
        return f"http://{embed_host}:{port}{path}"

    return f"http://localhost:{port}{path}"


OCR_URL = _resolve_backend_url("OCR_URL", "NEO_OCR_SERVER", "embed", 9003, "/ocr")
DETECT_URL = _resolve_backend_url("DETECT_URL", "NEO_DINO_SERVER", "embed", 9004, "/detect")

# Targets containing these words route to DINO first, OCR second.
# Includes colors (red/yellow/green traffic lights), shapes, and UI chrome.
_VISUAL_WORDS = re.compile(
    r"\b(button|icon|image|logo|checkbox|radio|toggle|slider|"
    r"arrow|close|minimize|maximize|tab|thumbnail|avatar|badge|"
    r"red|green|yellow|orange|blue|purple|white|black|grey|gray|"
    r"circle|dot|bar|line|indicator|marker|handle|cursor|"
    r"menu|toolbar|scrollbar|spinner|progress)\b",
    re.IGNORECASE,
)


# --- Helpers ---

def _screenshot():
    """Take a screenshot, return PIL Image."""
    import pyautogui
    return pyautogui.screenshot()


def _screenshot_bytes(img) -> bytes:
    """PIL Image -> PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _scale_factor(img):
    """Physical pixels / logical pixels (2.0 on Retina)."""
    import pyautogui
    logical_w, _ = pyautogui.size()
    return img.size[0] / logical_w


def _ocr(img_bytes: bytes) -> list:
    """Send screenshot to the OCR server (default: localhost:9003).

    Returns: [{"text": str, "center": [cx, cy], "confidence": float}, ...]
    Coordinates are in image pixel space (physical pixels).
    """
    resp = requests.post(OCR_URL, data=img_bytes, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _detect(img_bytes: bytes, query: str) -> list:
    """Send screenshot + query to the detection server (default: localhost:9004).

    Returns: [{"label": str, "center": [cx, cy], "box": [...], "confidence": float}, ...]
    Coordinates are in image pixel space (physical pixels).
    """
    payload = {
        "image": base64.b64encode(img_bytes).decode(),
        "query": query,
    }
    resp = requests.post(DETECT_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _is_visual_target(target: str) -> bool:
    """Does this target describe a visual element (use DINO) vs readable text (use OCR)?"""
    return bool(_VISUAL_WORDS.search(target))


def _find_text(detections: list, query: str) -> list:
    """Find text in OCR detections. Exact match first, then substring."""
    q = query.lower()
    exact = [d for d in detections if d["text"].lower() == q]
    if exact:
        return exact
    partial = [d for d in detections if q in d["text"].lower()]
    return partial


def _find_visual(detections: list, min_confidence: float = 0.3) -> list:
    """Filter DINO detections by confidence threshold."""
    return [d for d in detections if d["confidence"] >= min_confidence]


def _locate_target(img_bytes: bytes, target: str, scale: float) -> tuple:
    """Try to locate a target on screen using OCR and/or DINO.

    Routes based on target string: visual descriptors (colors, shapes,
    UI chrome words) go to DINO first, plain text goes to OCR first.
    Falls back to the other backend if the first finds nothing.

    Returns: (success, screen_pt: ScreenPoint | None, label, conf, method, error)
    """
    use_dino_first = _is_visual_target(target)

    if use_dino_first:
        backends = [("dino", target), ("ocr", target)]
    else:
        backends = [("ocr", target), ("dino", target)]

    all_visible = []

    for i, (method, query) in enumerate(backends):
        is_fallback = i > 0
        try:
            if method == "ocr":
                detections = _ocr(img_bytes)
                matches = _find_text(detections, query)
                all_visible = [d for d in detections if d.get("confidence", 0) > 0.5]
            else:
                # Higher threshold when DINO is fallback -- it hallucinates
                # bounding boxes for arbitrary text queries
                threshold = 0.52 if is_fallback else 0.3
                detections = _detect(img_bytes, query)
                matches = _find_visual(detections, min_confidence=threshold)
                all_visible = detections
        except requests.ConnectionError:
            continue
        except requests.Timeout:
            continue

        if matches:
            best = max(matches, key=lambda d: d["confidence"])
            px, py = best["center"]
            img_pt = ImagePoint(ImageX(int(px)), ImageY(int(py)))
            screen_pt = image_to_screen(img_pt, scale)
            label = best.get("text", best.get("label", target))
            return True, screen_pt, label, best["confidence"], method, ""

    # Both failed -- build error message from whatever we saw
    if all_visible:
        sample_key = "text" if "text" in all_visible[0] else "label"
        sample = ", ".join(f"'{d[sample_key]}'" for d in all_visible[:15])
        error = (
            f"'{target}' not found on screen. "
            f"Visible (top 15): [{sample}]. "
            f"Total: {len(all_visible)} regions detected."
        )
    else:
        error = f"'{target}' not found. No detections from either OCR or DINO."

    return False, None, "", 0, "", error


# ============================================================
# ScreenshotAction
# ============================================================

@display.register(ScreenshotAction)
def _(action: ScreenshotAction) -> ActionTarget:
    return ActionTarget(
        target=action.region,
        display=f"screenshot ({action.region})",
        content=None,
    )


@validate.register(ScreenshotAction)
def _(action: ScreenshotAction, project_path: str = ".", files_read: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
    if action.region not in ("full", "active_window"):
        return False, f"Invalid region '{action.region}'. Use 'full' or 'active_window'."
    return True, None


@execute.register(ScreenshotAction)
def _(action: ScreenshotAction, project_path: str, ctx: ExecutionContext = None) -> Tuple[bool, str]:
    """Capture screen and attach as image for the next LLM turn."""
    try:
        img = _screenshot()
        w, h = img.size
        scale = _scale_factor(img)
        img_bytes = _screenshot_bytes(img)

        # Attach to oracle if available
        if ctx and ctx.oracle:
            ctx.oracle.set_images([img_bytes])

        return True, f"Screenshot captured: {w}x{h} (scale={scale}x). Image attached for your next observation."
    except Exception as e:
        return False, f"Screenshot failed: {e}"


@extract_learnings.register(ScreenshotAction)
def _(action: ScreenshotAction, success: bool, result: str, reflect) -> List[Learning]:
    return []  # Screenshots don't teach


@action_key.register(ScreenshotAction)
def _(action: ScreenshotAction) -> tuple:
    return ("screenshot", action.region)


@hint.register(ScreenshotAction)
def _(action: ScreenshotAction) -> str:
    return "Screen capture. Check display access permissions."


@display_name.register(ScreenshotAction)
def _(action: ScreenshotAction) -> str:
    return "Screenshot"


# ============================================================
# ClickAction
# ============================================================

@display.register(ClickAction)
def _(action: ClickAction) -> ActionTarget:
    target = action.target or f"coords={action.coords}"
    return ActionTarget(
        target=target,
        display=f"click: {target} ({action.button})",
        content=None,
    )


@validate.register(ClickAction)
def _(action: ClickAction, project_path: str = ".", files_read: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
    if not action.target and not action.coords:
        return False, "ClickAction needs either target (text label) or coords. Prefer target."
    if action.button not in ("left", "right", "double"):
        return False, f"Invalid button '{action.button}'. Use 'left', 'right', or 'double'."
    return True, None


@execute.register(ClickAction)
def _(action: ClickAction, project_path: str, ctx: ExecutionContext = None) -> Tuple[bool, str]:
    """Screenshot -> OCR/DINO -> find target -> click at coordinates.

    Routing: visual targets (button, icon, etc.) try DINO first,
    text targets try OCR first. Both fall back to the other.
    """
    import pyautogui

    try:
        # If raw coords provided (fallback), click directly
        if action.coords and not action.target:
            x, y = action.coords
            if action.button == "double":
                pyautogui.doubleClick(x, y)
            else:
                pyautogui.click(x, y, button=action.button)
            return True, f"Clicked at ({x}, {y}) ({action.button})"

        img = _screenshot()
        scale = _scale_factor(img)
        img_bytes = _screenshot_bytes(img)

        found, screen_pt, label, conf, method, error = _locate_target(
            img_bytes, action.target, scale
        )

        if not found:
            return False, error

        if action.button == "double":
            pyautogui.doubleClick(screen_pt.x, screen_pt.y)
        else:
            pyautogui.click(screen_pt.x, screen_pt.y, button=action.button)

        return True, (
            f"Clicked '{label}' at ({screen_pt.x}, {screen_pt.y}) "
            f"(conf={conf}, {action.button}, via {method})"
        )

    except Exception as e:
        return False, f"Click failed: {e}"


@extract_learnings.register(ClickAction)
def _(action: ClickAction, success: bool, result: str, reflect) -> List[Learning]:
    if not success:
        return [reflect(f"Click '{action.target}' failed: {result}\nWhat should we try instead?")]
    return []


@action_key.register(ClickAction)
def _(action: ClickAction) -> tuple:
    return ("click", action.target, action.coords, action.button)


@hint.register(ClickAction)
def _(action: ClickAction) -> str:
    return "Click by text label or visual description. OCR finds text, DINO finds visual elements (buttons, icons). Take a screenshot first."


@display_name.register(ClickAction)
def _(action: ClickAction) -> str:
    return "Click"


# ============================================================
# TypeAction
# ============================================================

@display.register(TypeAction)
def _(action: TypeAction) -> ActionTarget:
    preview = (action.text or "")[:40]
    return ActionTarget(
        target=preview,
        display=f"type: {preview}{'...' if len(action.text or '') > 40 else ''}",
        content=None,
    )


@validate.register(TypeAction)
def _(action: TypeAction, project_path: str = ".", files_read: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
    if not action.text:
        return False, "Missing required field: text"
    return True, None


@execute.register(TypeAction)
def _(action: TypeAction, project_path: str, ctx: ExecutionContext = None) -> Tuple[bool, str]:
    """Type text into focused field."""
    import pyautogui

    try:
        pyautogui.write(action.text, interval=0.02)
        if action.press_enter:
            pyautogui.press("enter")
        return True, f"Typed '{action.text[:40]}{'...' if len(action.text) > 40 else ''}'{' + Enter' if action.press_enter else ''}"
    except Exception as e:
        return False, f"Type failed: {e}"


@extract_learnings.register(TypeAction)
def _(action: TypeAction, success: bool, result: str, reflect) -> List[Learning]:
    return []


@action_key.register(TypeAction)
def _(action: TypeAction) -> tuple:
    return ("type", action.text, action.press_enter)


@hint.register(TypeAction)
def _(action: TypeAction) -> str:
    return "Type text. Make sure the right field is focused first (use ClickAction)."


@display_name.register(TypeAction)
def _(action: TypeAction) -> str:
    return "Type"


# ============================================================
# ScrollAction
# ============================================================

@display.register(ScrollAction)
def _(action: ScrollAction) -> ActionTarget:
    return ActionTarget(
        target=f"{action.direction} {action.amount}",
        display=f"scroll {action.direction} {action.amount}",
        content=None,
    )


@validate.register(ScrollAction)
def _(action: ScrollAction, project_path: str = ".", files_read: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
    if action.direction not in ("up", "down"):
        return False, f"Invalid direction '{action.direction}'. Use 'up' or 'down'."
    if action.amount < 1:
        return False, "Scroll amount must be at least 1."
    return True, None


@execute.register(ScrollAction)
def _(action: ScrollAction, project_path: str, ctx: ExecutionContext = None) -> Tuple[bool, str]:
    """Scroll the active window."""
    import pyautogui

    try:
        clicks = action.amount if action.direction == "up" else -action.amount
        pyautogui.scroll(clicks)
        return True, f"Scrolled {action.direction} {action.amount} clicks"
    except Exception as e:
        return False, f"Scroll failed: {e}"


@extract_learnings.register(ScrollAction)
def _(action: ScrollAction, success: bool, result: str, reflect) -> List[Learning]:
    return []


@action_key.register(ScrollAction)
def _(action: ScrollAction) -> tuple:
    return ("scroll", action.direction, action.amount)


@hint.register(ScrollAction)
def _(action: ScrollAction) -> str:
    return "Scroll. Take a screenshot after to see the result."


@display_name.register(ScrollAction)
def _(action: ScrollAction) -> str:
    return "Scroll"


# ============================================================
# KeyPressAction
# ============================================================

@display.register(KeyPressAction)
def _(action: KeyPressAction) -> ActionTarget:
    return ActionTarget(
        target=action.keys,
        display=f"keypress: {action.keys}",
        content=None,
    )


@validate.register(KeyPressAction)
def _(action: KeyPressAction, project_path: str = ".", files_read: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
    if not action.keys:
        return False, "Missing required field: keys"
    return True, None


@execute.register(KeyPressAction)
def _(action: KeyPressAction, project_path: str, ctx: ExecutionContext = None) -> Tuple[bool, str]:
    """Press a key or key combination."""
    import pyautogui

    try:
        keys = action.keys
        if "+" in keys:
            # Key combination: "cmd+s" -> hotkey("cmd", "s")
            parts = [k.strip() for k in keys.split("+")]
            pyautogui.hotkey(*parts)
        else:
            pyautogui.press(keys)
        return True, f"Pressed {keys}"
    except Exception as e:
        return False, f"KeyPress failed: {e}"


@extract_learnings.register(KeyPressAction)
def _(action: KeyPressAction, success: bool, result: str, reflect) -> List[Learning]:
    return []


@action_key.register(KeyPressAction)
def _(action: KeyPressAction) -> tuple:
    return ("keypress", action.keys)


@hint.register(KeyPressAction)
def _(action: KeyPressAction) -> str:
    return "Key press. Use '+' for combos: 'cmd+s', 'ctrl+c'."


@display_name.register(KeyPressAction)
def _(action: KeyPressAction) -> str:
    return "KeyPress"


# ============================================================
# LocateAction
# ============================================================

@display.register(LocateAction)
def _(action: LocateAction) -> ActionTarget:
    return ActionTarget(
        target=action.target,
        display=f"locate: {action.target}",
        content=None,
    )


@validate.register(LocateAction)
def _(action: LocateAction, project_path: str = ".", files_read: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
    if not action.target:
        return False, "LocateAction needs a target to find."
    return True, None


@execute.register(LocateAction)
def _(action: LocateAction, project_path: str, ctx: ExecutionContext = None) -> Tuple[bool, str]:
    """Screenshot -> OCR/DINO -> find target -> report coordinates (no click).

    Same detection pipeline as ClickAction but read-only.
    """
    try:
        img = _screenshot()
        scale = _scale_factor(img)
        img_bytes = _screenshot_bytes(img)

        found, screen_pt, label, conf, method, error = _locate_target(
            img_bytes, action.target, scale
        )

        if not found:
            return False, error

        return True, (
            f"Found '{label}' at ({screen_pt.x}, {screen_pt.y}) "
            f"(conf={conf:.3f}, via {method})"
        )

    except Exception as e:
        return False, f"Locate failed: {e}"


@extract_learnings.register(LocateAction)
def _(action: LocateAction, success: bool, result: str, reflect) -> List[Learning]:
    return []


@action_key.register(LocateAction)
def _(action: LocateAction) -> tuple:
    return ("locate", action.target)


@hint.register(LocateAction)
def _(action: LocateAction) -> str:
    return "Find an element's screen coordinates without clicking. Use before ClickAction to verify position."


@display_name.register(LocateAction)
def _(action: LocateAction) -> str:
    return "Locate"


# ---------------------------------------------------------------------------
# SkillAction -- run a learned neo-lab skill
# ---------------------------------------------------------------------------

# neo-lab skills dir -- configurable companion workspace
_NEO_LAB = Path(
    os.environ.get(
        "COMPASS_NEO_LAB",
        str(Path.home() / ".compass" / "neo-lab"),
    )
)


def _ensure_neo_lab():
    """Add neo-lab to sys.path if not already there."""
    neo_str = str(_NEO_LAB)
    if neo_str not in sys.path:
        sys.path.insert(0, neo_str)


@display.register(SkillAction)
def _(action: SkillAction) -> str:
    expect = f" (expect: {action.expect})" if action.expect else ""
    return f"Skill: {action.skill}{expect}"


@validate.register(SkillAction)
def _(action: SkillAction) -> Optional[str]:
    if not action.skill:
        return "SkillAction requires a skill name"
    skill_path = _NEO_LAB / "skills" / f"{action.skill}.yaml"
    if not skill_path.exists():
        return f"Skill '{action.skill}' not found at {skill_path}"
    return None


@execute.register(SkillAction)
def _(action: SkillAction, ctx: ExecutionContext) -> Tuple[bool, str]:
    _ensure_neo_lab()
    try:
        from neo.skill import replay, validate_screen
    except ImportError:
        return False, f"Cannot import neo.skill from {_NEO_LAB}"

    try:
        success, facts = replay(action.skill, pause=0.2)
    except FileNotFoundError:
        return False, f"Skill '{action.skill}' not found"
    except Exception as e:
        return False, f"Skill '{action.skill}' failed: {e}"

    if not success:
        return False, f"Skill '{action.skill}' did not complete"

    # Post-skill validation
    if action.expect:
        valid, found = validate_screen(action.expect, timeout=10)
        if not valid:
            return False, (
                f"Skill '{action.skill}' completed but "
                f"expected '{action.expect}' not found on screen"
            )

    fact_str = ", ".join(f"{k}={v}" for k, v in facts.items()) if facts else "done"
    return True, f"Skill '{action.skill}' completed. Facts: {fact_str}"


@extract_learnings.register(SkillAction)
def _(action: SkillAction, success: bool, result: str, reflect) -> List[Learning]:
    return []


@action_key.register(SkillAction)
def _(action: SkillAction) -> tuple:
    return ("skill", action.skill)


@hint.register(SkillAction)
def _(action: SkillAction) -> str:
    return (
        "Run a learned skill -- a validated sequence of screen actions "
        "taught via neo-lab. Skills compose: each can invoke sub-skills. "
        "Use for multi-step flows like login, navigation, data export."
    )


@display_name.register(SkillAction)
def _(action: SkillAction) -> str:
    return "Skill"


# ---------------------------------------------------------------------------
# StateCheckAction -- identify current screen state via neo-lab state graph
# ---------------------------------------------------------------------------

@display.register(StateCheckAction)
def _(action: StateCheckAction) -> str:
    if action.state:
        return f"State check: am I at '{action.state}'?"
    return "State check: where am I?"


@validate.register(StateCheckAction)
def _(action: StateCheckAction) -> Optional[str]:
    states_path = _NEO_LAB / "states.yaml"
    if not states_path.exists():
        return f"No state graph found at {states_path}"
    if action.state:
        import yaml
        with open(states_path) as f:
            graph = yaml.safe_load(f) or {}
        if action.state not in graph.get("states", {}):
            known = list(graph.get("states", {}).keys())
            return f"State '{action.state}' not registered. Known: {known}"
    return None


@execute.register(StateCheckAction)
def _(action: StateCheckAction, ctx: ExecutionContext) -> Tuple[bool, str]:
    _ensure_neo_lab()
    try:
        from neo.state_graph import where_am_i, identify
    except ImportError:
        return False, f"Cannot import neo.state_graph from {_NEO_LAB}"

    if action.state:
        # Specific check: am I at this state?
        current, conf = where_am_i()
        if current == action.state:
            return True, f"Yes -- currently at '{action.state}' ({conf:.0%} confidence)"
        elif current:
            return False, f"No -- at '{current}' ({conf:.0%}), not '{action.state}'"
        else:
            return False, f"Cannot identify current state (no match above threshold)"
    else:
        # General query: what state am I in?
        matches = identify()
        if not matches:
            return True, "No known state matches current screen"
        lines = []
        for name, ratio, found, total, details in matches[:5]:
            parts = []
            if details.get("text", "-") != "-":
                parts.append(f"text {details['text']}")
            if details.get("visual", "-") != "-":
                parts.append(f"visual {details['visual']}")
            lines.append(f"  {name}: {ratio:.0%} ({found}/{total}) [{', '.join(parts)}]")
        return True, "State matches:\n" + "\n".join(lines)


@extract_learnings.register(StateCheckAction)
def _(action: StateCheckAction, success: bool, result: str, reflect) -> List[Learning]:
    return []


@action_key.register(StateCheckAction)
def _(action: StateCheckAction) -> tuple:
    return ("state_check", action.state or "*")


@hint.register(StateCheckAction)
def _(action: StateCheckAction) -> str:
    return (
        "Check what screen state Neo is looking at using the state graph. "
        "States are identified by OCR text markers and DINO visual markers "
        "taught via neo-lab. Use to verify navigation or find current position."
    )


@display_name.register(StateCheckAction)
def _(action: StateCheckAction) -> str:
    return "State Check"
