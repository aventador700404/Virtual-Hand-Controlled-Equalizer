"""
Ambient Conductor — On-screen EQ Panel Layout
===============================================
One shared, adjustable layout for the right-hand EQ faders. The gesture
code (hit-testing a pinch) and the HUD (drawing) both read it, so moving
or resizing the panel keeps them in sync.

  • Drag the panel with the mouse to move it, mouse wheel (or + / -) to resize.
  • The layout is saved to layout.json in the project folder and restored
    on the next run.

All coordinates are normalized to the frame (x: 0 = left, y: 0 = top).
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

# Geometry at scale 1.0
_BASE_SPACING = 0.09      # distance between neighbouring faders
_BASE_HEIGHT = 0.40       # fader track length
_BASE_GRAB_X = 0.045      # how close (x) a pinch must be to grab a fader
_PAD_X = 0.045            # panel padding left/right of the outer faders
_PAD_TOP = 0.10           # room above the track for title, names, values
_PAD_BOTTOM = 0.036       # room below the track

MIN_SCALE, MAX_SCALE = 0.6, 1.6

LAYOUT_FILE = Path(__file__).resolve().parent.parent / "layout.json"


@dataclass
class EqLayout:
    cx: float = 0.30       # x of the middle fader (left side: your left hand works the EQ)
    cy: float = 0.42       # vertical centre of the fader tracks
    scale: float = 1.0

    # ── Derived geometry ──────────────────────────────────────────────────
    @property
    def fader_x(self) -> list:
        s = _BASE_SPACING * self.scale
        return [self.cx - s, self.cx, self.cx + s]

    @property
    def top(self) -> float:
        return self.cy - _BASE_HEIGHT * self.scale / 2

    @property
    def bottom(self) -> float:
        return self.cy + _BASE_HEIGHT * self.scale / 2

    @property
    def grab_radius_x(self) -> float:
        return max(0.03, _BASE_GRAB_X * self.scale)

    def panel_rect(self) -> tuple:
        """(x0, y0, x1, y1) of the drawn panel, normalized."""
        xs = self.fader_x
        return (xs[0] - _PAD_X * self.scale, self.top - _PAD_TOP * self.scale,
                xs[-1] + _PAD_X * self.scale, self.bottom + _PAD_BOTTOM * self.scale)

    def contains(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.panel_rect()
        return x0 <= x <= x1 and y0 <= y <= y1

    # ── Editing ───────────────────────────────────────────────────────────
    def _clamp(self):
        """Keep the whole panel on screen."""
        x0, y0, x1, y1 = self.panel_rect()
        if x0 < 0:
            self.cx -= x0
        if x1 > 1:
            self.cx -= x1 - 1
        if y0 < 0:
            self.cy -= y0
        if y1 > 1:
            self.cy -= y1 - 1

    def move_to(self, cx: float, cy: float):
        self.cx, self.cy = cx, cy
        self._clamp()

    def zoom(self, factor: float):
        self.scale = min(MAX_SCALE, max(MIN_SCALE, self.scale * factor))
        self._clamp()

    def reset(self):
        default = EqLayout()
        self.cx, self.cy, self.scale = default.cx, default.cy, default.scale

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path: Path = LAYOUT_FILE):
        try:
            path.write_text(json.dumps({"eq": asdict(self)}, indent=2), encoding="utf-8")
        except OSError:
            pass  # a layout that can't be saved is not worth crashing over

    @classmethod
    def load(cls, path: Path = LAYOUT_FILE) -> "EqLayout":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))["eq"]
            layout = cls(cx=float(data["cx"]), cy=float(data["cy"]),
                         scale=float(data["scale"]))
            layout.scale = min(MAX_SCALE, max(MIN_SCALE, layout.scale))
            layout._clamp()
            return layout
        except (OSError, ValueError, KeyError, TypeError):
            return cls()


class EqPanelMouse:
    """
    OpenCV mouse handler: drag the EQ panel to move it, wheel to resize.
    Register with cv2.setMouseCallback(window, handler.on_mouse).
    """

    def __init__(self, layout: EqLayout, frame_size_getter):
        self.layout = layout
        self._frame_size = frame_size_getter  # () -> (w, h) of the shown frame
        self._dragging = False
        self._offset = (0.0, 0.0)

    def on_mouse(self, event, x, y, flags, param=None):
        import cv2

        w, h = self._frame_size()
        if not w or not h:
            return
        nx, ny = x / w, y / h

        if event == cv2.EVENT_LBUTTONDOWN and self.layout.contains(nx, ny):
            self._dragging = True
            self._offset = (self.layout.cx - nx, self.layout.cy - ny)
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self.layout.move_to(nx + self._offset[0], ny + self._offset[1])
        elif event == cv2.EVENT_LBUTTONUP and self._dragging:
            self._dragging = False
            self.layout.save()
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = _wheel_delta(flags)
            if delta:
                self.layout.zoom(1.1 if delta > 0 else 1 / 1.1)
                self.layout.save()


def _wheel_delta(flags: int) -> int:
    """Signed wheel delta from OpenCV mouse flags (same as cv2.getMouseWheelDelta)."""
    d = (flags >> 16) & 0xFFFF
    return d - 0x10000 if d >= 0x8000 else d
