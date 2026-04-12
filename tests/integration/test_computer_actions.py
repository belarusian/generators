"""Integration tests: computer use actions.

Tests the full action dispatch pipeline for Screenshot, Click, Type,
Scroll, KeyPress. Tests that hit the OCR server require it to be
reachable at localhost:9003.

No mocks. These test real dispatch, real validation, real execution.
"""

import os
import functools

import pytest
import requests

from compass.core.env import load_compass_env
from compass.agents.neo.types import (
    ActionTarget,
    ExecutionContext,
    ScreenshotAction,
    ClickAction,
    TypeAction,
    ScrollAction,
    KeyPressAction,
)
from compass.agents.neo.dispatch import (
    action_key,
    display,
    display_name,
    execute,
    hint,
    validate,
)

# Load shared/local env before resolving OCR_URL defaults.
load_compass_env()

from compass.agents.neo.actions.computer import OCR_URL


@functools.cache
def _has_display() -> bool:
    """Check if a GUI display is available."""
    try:
        import pyautogui
        pyautogui.size()
        return True
    except Exception:
        return False


@functools.cache
def _has_ocr_server() -> bool:
    """Check if the OCR server is reachable."""
    try:
        resp = requests.get(OCR_URL.replace("/ocr", "/health"), timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ============================================================
# Dispatch registration tests (no hardware needed)
# ============================================================

class TestComputerActionDispatch:
    """Verify all 5 actions have singledispatch registrations."""

    def test_screenshot_display(self):
        action = ScreenshotAction(region="full")
        result = display(action)
        assert isinstance(result, ActionTarget)
        assert "screenshot" in result.display

    def test_click_display(self):
        action = ClickAction(target="Save", button="left")
        result = display(action)
        assert isinstance(result, ActionTarget)
        assert "Save" in result.display

    def test_type_display(self):
        action = TypeAction(text="hello@example.com")
        result = display(action)
        assert isinstance(result, ActionTarget)
        assert "hello@example.com" in result.display

    def test_scroll_display(self):
        action = ScrollAction(direction="down", amount=3)
        result = display(action)
        assert isinstance(result, ActionTarget)
        assert "down" in result.display

    def test_keypress_display(self):
        action = KeyPressAction(keys="cmd+s")
        result = display(action)
        assert isinstance(result, ActionTarget)
        assert "cmd+s" in result.display

    def test_display_names(self):
        assert display_name(ScreenshotAction()) == "Screenshot"
        assert display_name(ClickAction(target="x")) == "Click"
        assert display_name(TypeAction(text="x")) == "Type"
        assert display_name(ScrollAction()) == "Scroll"
        assert display_name(KeyPressAction(keys="x")) == "KeyPress"

    def test_hints_nonempty(self):
        assert hint(ScreenshotAction())
        assert hint(ClickAction(target="x"))
        assert hint(TypeAction(text="x"))
        assert hint(ScrollAction())
        assert hint(KeyPressAction(keys="x"))

    def test_action_keys_unique(self):
        keys = [
            action_key(ScreenshotAction()),
            action_key(ClickAction(target="Save")),
            action_key(TypeAction(text="hello")),
            action_key(ScrollAction()),
            action_key(KeyPressAction(keys="enter")),
        ]
        assert len(set(keys)) == 5


# ============================================================
# Validation tests (no hardware needed)
# ============================================================

class TestComputerActionValidation:
    """Validate field constraints."""

    def test_screenshot_valid_regions(self):
        ok, err = validate(ScreenshotAction(region="full"))
        assert ok
        ok, err = validate(ScreenshotAction(region="active_window"))
        assert ok

    def test_screenshot_invalid_region(self):
        ok, err = validate(ScreenshotAction(region="left_half"))
        assert not ok
        assert "Invalid region" in err

    def test_click_needs_target_or_coords(self):
        ok, err = validate(ClickAction())
        assert not ok
        assert "target" in err.lower() or "coords" in err.lower()

    def test_click_valid_with_target(self):
        ok, err = validate(ClickAction(target="Save"))
        assert ok

    def test_click_valid_with_coords(self):
        ok, err = validate(ClickAction(coords=(100, 200)))
        assert ok

    def test_click_invalid_button(self):
        ok, err = validate(ClickAction(target="Save", button="middle"))
        assert not ok
        assert "middle" in err

    def test_type_needs_text(self):
        ok, err = validate(TypeAction())
        assert not ok
        assert "text" in err.lower()

    def test_type_valid(self):
        ok, err = validate(TypeAction(text="hello"))
        assert ok

    def test_scroll_valid(self):
        ok, err = validate(ScrollAction(direction="up", amount=5))
        assert ok

    def test_scroll_invalid_direction(self):
        ok, err = validate(ScrollAction(direction="left"))
        assert not ok
        assert "left" in err

    def test_scroll_invalid_amount(self):
        ok, err = validate(ScrollAction(amount=0))
        assert not ok

    def test_keypress_needs_keys(self):
        ok, err = validate(KeyPressAction())
        assert not ok
        assert "keys" in err.lower()

    def test_keypress_valid(self):
        ok, err = validate(KeyPressAction(keys="enter"))
        assert ok


# ============================================================
# Execute tests (require display + OCR server)
# ============================================================

@pytest.mark.skipif(not _has_display(), reason="No display available")
class TestScreenshotExecute:
    """Screenshot execution -- requires a display."""

    def test_screenshot_captures(self):
        action = ScreenshotAction(region="full")
        ctx = ExecutionContext()
        success, result = execute(action, ".", ctx)
        if not success and "could not create image from display" in result.lower():
            pytest.skip("Screenshot backend cannot capture the current display")
        assert success, result
        assert "Screenshot captured" in result
        assert "scale=" in result


@pytest.mark.skipif(
    not (_has_display() and _has_ocr_server()),
    reason="Needs display + OCR server",
)
class TestClickExecute:
    """Click execution -- requires display + OCR server."""

    def test_click_finds_text_on_screen(self):
        """OCR should find text that's visible. We don't actually click."""
        # Just verify the OCR round-trip works via the action
        from compass.agents.neo.actions.computer import _screenshot, _screenshot_bytes, _ocr

        img = _screenshot()
        detections = _ocr(_screenshot_bytes(img))
        assert len(detections) > 0, "OCR returned no detections"
        # At least some text should be on screen
        texts = [d["text"] for d in detections]
        assert any(len(t) > 0 for t in texts)

    def test_click_missing_text_shows_alternatives(self):
        """Clicking nonexistent text should list what IS visible."""
        action = ClickAction(target="ZZZYYYXXX_NONEXISTENT_12345")
        ctx = ExecutionContext()
        success, result = execute(action, ".", ctx)
        assert not success
        assert "not found" in result.lower()
        assert "Visible" in result or "No detections" in result
