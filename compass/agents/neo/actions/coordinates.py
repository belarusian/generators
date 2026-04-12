"""
Typed coordinate spaces and translation for computer-use actions.

Three coordinate frames:

  ImagePoint   -- raw pixels from screenshot (physical; what OCR/DINO return)
  ScreenPoint  -- logical desktop pixels (what pyautogui expects)
  ClientPoint  -- relative to a window's content area

Prevents accidentally passing image-space coords to pyautogui (which
expects logical coords) or vice versa.  The conversion functions make
the scale factor explicit.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import NewType, Optional


# -- Opaque scalar types (prevent int mix-ups at call sites) --

ImageX  = NewType("ImageX",  int)
ImageY  = NewType("ImageY",  int)
ScreenX = NewType("ScreenX", int)
ScreenY = NewType("ScreenY", int)
ClientX = NewType("ClientX", int)
ClientY = NewType("ClientY", int)


# -- Point types --

@dataclass(frozen=True)
class ImagePoint:
    """Coordinates in screenshot pixel space (physical pixels)."""
    x: ImageX
    y: ImageY


@dataclass(frozen=True)
class ScreenPoint:
    """Coordinates in logical desktop space (pyautogui units)."""
    x: ScreenX
    y: ScreenY


@dataclass(frozen=True)
class ClientPoint:
    """Coordinates relative to a window's client area."""
    x: ClientX
    y: ClientY


# -- Window geometry --

@dataclass
class WindowGeometry:
    """Bounding box and client insets for a window, in screen coords."""
    left: int
    top: int
    right: int
    bottom: int
    client_left: int = 0
    client_top: int = 0
    client_right: int = 0
    client_bottom: int = 0
    dpi_scale: float = 1.0
    timestamp: Optional[datetime] = None


@dataclass
class WindowContext:
    """Snapshot of a window's geometry and state at capture time."""
    handle: object = None
    pid: Optional[int] = None
    geometry: Optional[WindowGeometry] = None
    is_minimized: bool = False
    is_visible: bool = True
    scale_factor: float = 2.0   # physical / logical (Retina default)
    capture_time: Optional[datetime] = None


# -- Translation functions --

def image_to_screen(pt: ImagePoint, scale: float) -> ScreenPoint:
    """Image pixel coords -> logical screen coords (divide by scale)."""
    return ScreenPoint(
        ScreenX(int(pt.x / scale)),
        ScreenY(int(pt.y / scale)),
    )


def screen_to_image(pt: ScreenPoint, scale: float) -> ImagePoint:
    """Logical screen coords -> image pixel coords (multiply by scale)."""
    return ImagePoint(
        ImageX(int(pt.x * scale)),
        ImageY(int(pt.y * scale)),
    )


def screen_to_client(pt: ScreenPoint, ctx: WindowContext) -> ClientPoint:
    """Logical screen coords -> window client-area coords."""
    if not ctx.geometry:
        raise ValueError("WindowContext has no geometry")
    g = ctx.geometry
    return ClientPoint(
        ClientX(pt.x - g.left + g.client_left),
        ClientY(pt.y - g.top + g.client_top),
    )


def client_to_screen(pt: ClientPoint, ctx: WindowContext) -> ScreenPoint:
    """Window client-area coords -> logical screen coords."""
    if not ctx.geometry:
        raise ValueError("WindowContext has no geometry")
    g = ctx.geometry
    return ScreenPoint(
        ScreenX(pt.x - g.client_left + g.left),
        ScreenY(pt.y - g.client_top + g.top),
    )
