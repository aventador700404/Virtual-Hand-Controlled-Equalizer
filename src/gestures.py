"""
Ambient Conductor — Gesture Detection
=======================================
Two-hand operation (mirrored view). Internally the roles follow MediaPipe's
labels: "Left" = mixer, "Right" = EQ. On this setup that is:

  YOUR RIGHT hand (mixer) — raised fingers add stems; a fist mutes (the "drop").
                       Filter + echo work like a clutch: PINCH to grab them
                       (the pinch point becomes the neutral reference), move
                       left/right for low-pass/high-pass and up for more echo,
                       RELEASE to fix the values where they are.
  YOUR LEFT hand  (EQ)    — pinch thumb + index on one of the on-screen
                       LOW / MID / HIGH faders and move up or down to drag
                       its knob. Release to leave it there. Bottom = kill.

Includes a temporal debounce buffer for finger counts and hysteresis on the
pinch so it doesn't flicker at the threshold.
"""

import math
from dataclasses import dataclass
from typing import Optional

from config import (
    THUMB_TIP,
    THUMB_IP,
    INDEX_TIP,
    MIDDLE_MCP,
    PINKY_MCP,
    WRIST,
    FINGER_TIPS,
    FINGER_PIPS,
    FINGER_COUNT_DEBOUNCE_FRAMES,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    EQ_BANDS,
    EQ_MAX_DB,
    EQ_MIN_DB,
    EQ_KILL_POS,
    EQ_DETENT_DB,
    PINCH_ON_RATIO,
    PINCH_OFF_RATIO,
    PINCH_CURSOR_SMOOTHING,
    FX_X_TRAVEL,
    FX_Y_TRAVEL,
    PINCH_FREEZE_FRAMES,
    FX_DEAD_ZONE,
)
from layout import EqLayout
from vision import HandData, HandState

# Fader position (0 = bottom, 1 = top) that sits exactly at 0 dB
EQ_UNITY_POS = (0.0 - EQ_MIN_DB) / (EQ_MAX_DB - EQ_MIN_DB)

# Filter position (0 = full low-pass, 0.5 = bypass, 1 = full high-pass)
FILTER_BYPASS_ZONE = (0.43, 0.57)


def fader_to_db(pos: float) -> float:
    """Fader position (0–1) → gain in dB. The bottom slot is a full kill (-inf)."""
    if pos <= EQ_KILL_POS:
        return float("-inf")
    db = EQ_MIN_DB + (EQ_MAX_DB - EQ_MIN_DB) * pos
    return 0.0 if abs(db) < EQ_DETENT_DB else db


def _dead_zone(d: float, size: float = FX_DEAD_ZONE) -> float:
    """Ignore tiny hand jitter around the pinch anchor."""
    return 0.0 if abs(d) <= size else d - math.copysign(size, d)


def filter_from_pos(x: float) -> tuple:
    """Filter position (0–1) → (mode, cutoff Hz). Same zones as DualModeFilter."""
    lo, hi = FILTER_BYPASS_ZONE
    if x < lo:
        return "low", 100.0 * (200.0 ** (x / lo))
    if x > hi:
        return "high", 20.0 * (250.0 ** ((x - hi) / (1.0 - hi)))
    return "bypass", 20000.0


@dataclass
class ConductorState:
    """The high-level musical control state derived from hand gestures."""
    finger_count: int = 0              # 0–5, from the mixer (left) hand
    filter_cutoff_norm: float = 0.5    # 0 = full LP, 0.5 = bypass, 1 = full HP
    filter_cutoff_hz: float = 20000.0  # Mapped Hz value
    filter_mode: str = "bypass"        # "low", "high", "bypass"
    effect_wet_norm: float = 0.0       # 0.0–1.0 echo amount
    tempo_factor: float = 1.0          # 0.5–1.7x playback speed factor
    left_detected: bool = False        # mixer hand visible
    right_detected: bool = False       # EQ hand visible
    mixer_active: bool = False         # = left_detected (fist-mute only applies then)
    # ── FX clutch (left hand) ──
    fx_grab: bool = False              # filter/echo currently held by a pinch
    fx_anchor: Optional[tuple] = None  # (x, y) where the pinch started
    fx_cursor: Optional[tuple] = None  # current left-hand pinch point
    # ── EQ (right hand) ──
    eq_pos: tuple = (EQ_UNITY_POS,) * 3                 # fader positions 0–1
    eq_db: tuple = (0.0, 0.0, 0.0)                      # dB per band (-inf = kill)
    eq_grab: int = -1                  # band being held (-1 = none)
    eq_hover: int = -1                 # band under the pinch cursor
    eq_pinching: bool = False
    eq_cursor: Optional[tuple] = None  # (x, y) normalized frame coords


def count_fingers(landmarks, handedness: str) -> int:
    """
    Count the number of extended fingers on a hand.

    For index, middle, ring, pinky: a finger is extended if
    the fingertip's Y is higher than the PIP joint's Y.
    (Since Y is inverted in our pipeline, higher = larger Y value,
    but landmarks are in raw MediaPipe coords where lower Y = higher
    in the frame. We compare raw landmark coords here.)

    For the thumb: the tip must be farther from the palm than the IP joint
    (orientation-independent; `handedness` is kept for API compatibility).

    Args:
        landmarks: List of 21 MediaPipe hand landmarks (raw, normalized).
        handedness: "Left" or "Right" (from MediaPipe's perspective,
                    already corrected for mirror flip).

    Returns:
        Integer 0–5 representing the number of raised fingers.
    """
    if not landmarks or len(landmarks) < 21:
        return 0

    count = 0

    # ── Thumb ─────────────────────────────────────────────────────────────
    # Extended = the tip sits farther from the palm (pinky knuckle) than the
    # IP joint does. Unlike comparing X coordinates, this works for either
    # hand and whether the palm or the back of the hand faces the camera,
    # so a fist with the thumb wrapped over the fingers reads as 0.
    aspect = CAMERA_WIDTH / CAMERA_HEIGHT

    def _dist(a, b):
        return math.hypot((a.x - b.x) * aspect, a.y - b.y)

    palm = landmarks[PINKY_MCP]
    if _dist(landmarks[THUMB_TIP], palm) > _dist(landmarks[THUMB_IP], palm):
        count += 1

    # ── Index, Middle, Ring, Pinky ────────────────────────────────────────
    # A finger is extended when the tip Y is ABOVE (lower value in raw coords)
    # the PIP joint Y.
    for tip_id, pip_id in zip(FINGER_TIPS, FINGER_PIPS):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            count += 1

    return count


class PinchTracker:
    """Thumb–index pinch with hysteresis and a smoothed cursor (one per hand)."""

    def __init__(self):
        self.pinching = False
        self.cursor: Optional[tuple] = None

    @staticmethod
    def ratio(lm) -> float:
        """Thumb–index tip distance relative to hand size (aspect-corrected)."""
        aspect = CAMERA_WIDTH / CAMERA_HEIGHT

        def dist(a, b):
            return math.hypot((a.x - b.x) * aspect, a.y - b.y)

        hand_size = dist(lm[WRIST], lm[MIDDLE_MCP])
        return dist(lm[THUMB_TIP], lm[INDEX_TIP]) / max(hand_size, 1e-6)

    def update(self, hand: Optional[HandData]) -> tuple:
        """Returns (pinching, just_started, just_released)."""
        was = self.pinching
        if hand is None or not hand.landmarks or len(hand.landmarks) < 21:
            self.pinching = False
            self.cursor = None
            return False, False, was
        lm = hand.landmarks
        self.pinching = self.ratio(lm) < (PINCH_OFF_RATIO if was else PINCH_ON_RATIO)
        raw = ((lm[THUMB_TIP].x + lm[INDEX_TIP].x) / 2,
               (lm[THUMB_TIP].y + lm[INDEX_TIP].y) / 2)
        if self.cursor is None or (self.pinching and not was):
            self.cursor = raw   # snap on a new pinch so the anchor is exactly where you pinched
        else:
            a = PINCH_CURSOR_SMOOTHING
            self.cursor = (self.cursor[0] + a * (raw[0] - self.cursor[0]),
                           self.cursor[1] + a * (raw[1] - self.cursor[1]))
        return self.pinching, self.pinching and not was, was and not self.pinching


class GestureProcessor:
    """
    Processes raw HandState into a debounced ConductorState.
    Keeps state across frames: stable finger count, the held filter/echo
    values and the EQ faders (which stay where you leave them, like knobs).
    """

    def __init__(self, layout: Optional[EqLayout] = None):
        self.layout = layout or EqLayout()
        self._count_history: list[int] = []
        self._stable_count: int = 0
        self._freeze = 0                    # frames left with the count frozen
        # Left hand: filter / echo clutch
        self._fx_pinch = PinchTracker()
        self._filter_pos = 0.5
        self._wet = 0.0
        self._fx_anchor: Optional[tuple] = None
        self._fx_base = (0.5, 0.0)
        # Right hand: EQ faders
        self._eq_pinch = PinchTracker()
        self._eq_pos = [EQ_UNITY_POS] * len(EQ_BANDS)
        self._grab = -1
        self._grab_y = 0.0       # cursor y when the fader was grabbed
        self._grab_pos = 0.0     # fader position when it was grabbed

    def reset(self):
        """Everything back to neutral: filter bypass, no echo, EQ at 0 dB."""
        self._filter_pos, self._wet = 0.5, 0.0
        self._fx_anchor = None
        self._eq_pos = [EQ_UNITY_POS] * len(EQ_BANDS)
        self._grab = -1

    def _debounce_count(self, raw_count: int) -> int:
        """
        Only update the stable finger count if the same value has
        been seen for FINGER_COUNT_DEBOUNCE_FRAMES consecutive frames.
        This eliminates flicker when fingers are at borderline angles.
        """
        self._count_history.append(raw_count)

        # Keep the buffer trimmed
        if len(self._count_history) > FINGER_COUNT_DEBOUNCE_FRAMES:
            self._count_history = self._count_history[-FINGER_COUNT_DEBOUNCE_FRAMES:]

        # Check if all recent values agree
        if len(self._count_history) >= FINGER_COUNT_DEBOUNCE_FRAMES:
            if all(c == raw_count for c in self._count_history):
                self._stable_count = raw_count

        return self._stable_count

    # ── Left hand: mixer ──────────────────────────────────────────────────

    def _process_mixer(self, hand: Optional[HandData], state: ConductorState):
        pinching, started, released = self._fx_pinch.update(hand)

        if hand is not None and self._fx_pinch.cursor is not None:
            state.left_detected = True
            state.mixer_active = True

            # Finger count → stems. A pinch changes the hand shape, so the
            # count is frozen while pinching and for a few frames after.
            if pinching:
                self._freeze = PINCH_FREEZE_FRAMES
                self._count_history.clear()
            elif self._freeze > 0:
                self._freeze -= 1
            else:
                raw_count = count_fingers(hand.landmarks, hand.handedness)
                self._debounce_count(raw_count)

            # Filter / echo clutch: pinch = move, release = fix
            cx, cy = self._fx_pinch.cursor
            if started:
                self._fx_anchor = (cx, cy)
                self._fx_base = (self._filter_pos, self._wet)
            if pinching and self._fx_anchor is not None:
                ax, ay = self._fx_anchor
                dx, dy = _dead_zone(cx - ax), _dead_zone(ay - cy)
                self._filter_pos = min(1.0, max(0.0, self._fx_base[0] + dx / FX_X_TRAVEL))
                self._wet = min(1.0, max(0.0, self._fx_base[1] + dy / FX_Y_TRAVEL))
                state.fx_grab = True
                state.fx_anchor = self._fx_anchor
            state.fx_cursor = (cx, cy)

        if released or (hand is None and self._fx_anchor is not None):
            # Snap small leftovers to neutral so "almost bypass" is bypass
            lo, hi = FILTER_BYPASS_ZONE
            if lo <= self._filter_pos <= hi:
                self._filter_pos = 0.5
            if self._wet < 0.05:
                self._wet = 0.0
            self._fx_anchor = None

        state.finger_count = self._stable_count
        state.filter_cutoff_norm = self._filter_pos
        state.filter_mode, state.filter_cutoff_hz = filter_from_pos(self._filter_pos)
        state.effect_wet_norm = self._wet
        state.tempo_factor = 1.0

    # ── Right hand: EQ faders ─────────────────────────────────────────────

    def _band_under(self, x: float, y: float) -> int:
        lay = self.layout
        margin = 0.06 * lay.scale
        if not (lay.top - margin <= y <= lay.bottom + margin):
            return -1
        best, best_dx = -1, lay.grab_radius_x
        for i, fx in enumerate(lay.fader_x):
            dx = abs(x - fx)
            if dx <= best_dx:
                best, best_dx = i, dx
        return best

    def _process_eq(self, hand: Optional[HandData], state: ConductorState):
        pinching, started, _ = self._eq_pinch.update(hand)
        if hand is None or self._eq_pinch.cursor is None:
            self._grab = -1     # hand gone → let go of the fader (its value stays)
        else:
            state.right_detected = True
            cx, cy = self._eq_pinch.cursor
            hover = self._band_under(cx, cy)

            if started:
                # Grab only if the pinch started on a fader column. The fader
                # then follows the hand's vertical movement (relative drag),
                # so grabbing never makes the value jump.
                self._grab = hover
                if hover >= 0:
                    self._grab_y = cy
                    self._grab_pos = self._eq_pos[hover]
            elif not pinching:
                self._grab = -1

            if self._grab >= 0:
                travel = (self._grab_y - cy) / (self.layout.bottom - self.layout.top)
                self._eq_pos[self._grab] = min(1.0, max(0.0, self._grab_pos + travel))

            state.eq_hover = hover
            state.eq_cursor = (cx, cy)

        state.eq_pinching = pinching
        state.eq_grab = self._grab
        state.eq_pos = tuple(self._eq_pos)
        state.eq_db = tuple(fader_to_db(p) for p in self._eq_pos)

    def process(self, hand_state: HandState) -> ConductorState:
        """Convert a HandState from the vision module into a ConductorState."""
        state = ConductorState()
        self._process_mixer(hand_state.left, state)
        self._process_eq(hand_state.right, state)
        return state
