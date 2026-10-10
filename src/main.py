"""
Ambient Conductor — Main Entry Point
=======================================
Ties together the vision loop, gesture processor, and audio engine
into a single real-time interactive application.

Usage:
    python src/main.py [--stems-dir PATH] [--camera INDEX] [--no-audio] [--bg]
    python src/main.py --song "my song.mp3"     # split once with Demucs, then play
    python src/main.py --choose                 # pick from the songs/ folder

Controls:
    Right hand  → Raised fingers control stem mixing (0-4+ fingers), fist = drop
                → Pinch to grab filter/echo: move left/right = LP/HP,
                  up = more echo; release to fix the values
    Left hand   → Pinch (thumb + index) a LOW / MID / HIGH fader and drag it
                  up/down; release to leave it. Bottom of a fader = kill.
    Mouse       → Drag the EQ panel to move it, wheel to resize
    R           → Reset filter, echo and EQ to neutral
    + / - / 0   → Resize the EQ panel / reset its position
    B           → Toggle webcam view / illustration background
    Q / Esc     → Quit
"""

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

# Ensure src/ is in the path
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CAMERA_INDEX,
    STEMS_DIR,
    STEM_NAMES,
    TARGET_FPS,
    HUD_PANEL_ALPHA,
    HUD_PANEL_RADIUS,
    HUD_BG_COLOR,
    HUD_BG_COLOR_2,
    HUD_TEXT_COLOR,
    HUD_TEXT_DIM,
    HUD_BORDER_COLOR,
    HUD_ACCENT_CYAN,
    HUD_ACCENT_MAGENTA,
    HUD_ACCENT_PURPLE,
    HUD_ACCENT_ORANGE,
    HUD_ACTIVE_COLOR,
    HUD_INACTIVE_COLOR,
    HUD_FILTER_BAR_START,
    HUD_FILTER_BAR_END,
    HUD_FX_BAR_START,
    HUD_FX_BAR_END,
    HUD_STEM_COLORS,
    HUD_LERP_SPEED,
    HUD_EQ_COLORS,
    HUD_EQ_HAND_COLOR,
    EQ_BANDS,
)
from vision import VisionTracker, HAND_CONNECTIONS, HandState
from gestures import GestureProcessor, ConductorState, EQ_UNITY_POS
from layout import EqLayout, EqPanelMouse
from audio_engine import AudioEngine
from generate_stems import generate_all_stems


# ─── HUD Drawing Utilities ───────────────────────────────────────────────────

def _lerp(current: float, target: float, speed: float = HUD_LERP_SPEED) -> float:
    """Smooth linear interpolation for HUD values."""
    return current + speed * (target - current)


def _draw_rounded_rect(frame, pt1, pt2, color, radius, thickness=-1, alpha=1.0):
    """
    Draw a rounded rectangle on the frame.
    If alpha < 1.0, blends with the existing frame content.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    if alpha < 1.0:
        overlay = frame.copy()
        target = overlay
    else:
        target = frame

    # Draw filled rounded rectangle using polylines + circles
    if thickness == -1:
        # Fill the interior rectangle regions
        cv2.rectangle(target, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(target, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
        # Fill corner circles
        cv2.circle(target, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(target, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(target, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(target, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
    else:
        # Draw border only
        pts = np.array([
            [x1 + r, y1], [x2 - r, y1],
            [x2, y1 + r], [x2, y2 - r],
            [x2 - r, y2], [x1 + r, y2],
            [x1, y2 - r], [x1, y1 + r],
        ], np.int32)
        cv2.polylines(target, [pts], True, color, thickness, cv2.LINE_AA)
        cv2.ellipse(target, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(target, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(target, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(target, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)

    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _draw_gradient_bar_h(frame, x, y, w, h, fill_ratio, color_start, color_end, bg_color):
    """Draw a horizontal gradient progress bar with rounded ends."""
    # Background track
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1, cv2.LINE_AA)
    # Rounded ends on background
    r = h // 2
    cv2.circle(frame, (x + r, y + r), r, bg_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x + w - r, y + r), r, bg_color, -1, cv2.LINE_AA)

    # Fill
    fill_w = int(w * max(0.0, min(1.0, fill_ratio)))
    if fill_w > h:  # Only draw if wider than the rounded end
        # Draw gradient by blending colors across the fill width
        for px in range(fill_w):
            t = px / max(w, 1)
            c = tuple(int(color_start[i] + t * (color_end[i] - color_start[i])) for i in range(3))
            cv2.line(frame, (x + px, y), (x + px, y + h), c, 1)
        # Round the left end
        cv2.circle(frame, (x + r, y + r), r, color_start, -1, cv2.LINE_AA)
        # Round the right end of fill
        if fill_w > r * 2:
            t_end = fill_w / max(w, 1)
            c_end = tuple(int(color_start[i] + t_end * (color_end[i] - color_start[i])) for i in range(3))
            cv2.circle(frame, (x + fill_w - r, y + r), r, c_end, -1, cv2.LINE_AA)


def _draw_gradient_bar_v(frame, x, y, w, h, fill_ratio, color_start, color_end, bg_color):
    """Draw a vertical gradient progress bar (fills from bottom up) with rounded ends."""
    r = w // 2
    # Background track
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x + r, y + r), r, bg_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x + r, y + h - r), r, bg_color, -1, cv2.LINE_AA)

    # Fill from bottom up
    fill_h = int(h * max(0.0, min(1.0, fill_ratio)))
    if fill_h > w:
        fill_top = y + h - fill_h
        for py in range(fill_h):
            actual_y = fill_top + py
            t = py / max(fill_h, 1)
            c = tuple(int(color_start[i] + t * (color_end[i] - color_start[i])) for i in range(3))
            cv2.line(frame, (x, actual_y), (x + w, actual_y), c, 1)
        # Round bottom end
        cv2.circle(frame, (x + r, y + h - r), r, color_start, -1, cv2.LINE_AA)
        # Round top end of fill
        if fill_h > r * 2:
            cv2.circle(frame, (x + r, fill_top + r), r, color_end, -1, cv2.LINE_AA)


# ─── HUD State (smooth interpolation) ────────────────────────────────────────

class HUDState:
    """Maintains smoothly interpolated HUD display values."""
    def __init__(self):
        self.filter_norm = 0.5
        self.fx_wet = 0.0
        self.stem_glows = [0.0, 0.0, 0.0, 0.0]  # Glow intensity per stem
        self.eq_pos = [EQ_UNITY_POS] * len(EQ_BANDS)
        self._frame_count = 0

    def update(self, state: ConductorState, hand_state: HandState):
        """Smoothly interpolate towards the target values."""
        self.filter_norm = _lerp(self.filter_norm, state.filter_cutoff_norm)
        self.fx_wet = _lerp(self.fx_wet, state.effect_wet_norm)
        self._frame_count += 1

        for i in range(4):
            target = 1.0 if (i == 0 or state.finger_count >= i) else 0.0
            self.stem_glows[i] = _lerp(self.stem_glows[i], target, speed=0.12)

        for i, pos in enumerate(state.eq_pos):
            self.eq_pos[i] = _lerp(self.eq_pos[i], pos, speed=0.45)

    @property
    def pulse(self) -> float:
        """Slow breathing pulse for active elements."""
        return 0.7 + 0.3 * math.sin(self._frame_count * 0.08)


# ─── HUD Skeleton drawing helper ──────────────────────────────────────────────

def _draw_hand_skeleton(frame, hand_data, is_fist, pulse_val, base_color=None):
    h, w = frame.shape[:2]
    points = []
    for lm in hand_data.landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        points.append((px, py))

    # Base color: Red/Orange for fist, Cyan for active
    if is_fist:
        # Pulsing intense orange-red BGR
        color = (20, int(80 + 50 * pulse_val), int(220 + 35 * pulse_val))
        glow_color = (10, 20, 110)
    elif base_color is not None:
        color = base_color
        glow_color = tuple(c // 2 for c in base_color)
    else:
        color = (230, 200, 50)  # Cyan BGR
        glow_color = (115, 100, 25)

    # Connections with glow
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            # Outer glow
            cv2.line(frame, points[start_idx], points[end_idx], glow_color, 4, cv2.LINE_AA)
            # Inner line
            cv2.line(frame, points[start_idx], points[end_idx], color, 2, cv2.LINE_AA)

    # Landmark dots
    for i, (px, py) in enumerate(points):
        is_tip = i in (4, 8, 12, 16, 20)
        if is_tip:
            cv2.circle(frame, (px, py), 9, glow_color, 1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (px, py), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)


# ─── EQ faders (your left hand) ───────────────────────────────────────────────

def _format_db(db: float) -> str:
    if db == float("-inf"):
        return "KILL"
    if db == 0.0:
        return "0 dB"
    return f"{db:+.1f}"


def _draw_eq(frame, state: ConductorState, hud_state: "HUDState", layout: EqLayout):
    """Three vertical faders the EQ hand (your left) can reach for and pinch.
    Position/size come from the shared layout (movable with the mouse).
    Drawn before the hand skeletons so the hand appears on top of them."""
    h, w = frame.shape[:2]
    top, bottom = int(layout.top * h), int(layout.bottom * h)
    span = bottom - top
    xs = [int(fx * w) for fx in layout.fader_x]
    px0, py0, px1, py1 = layout.panel_rect()
    x0, y0, x1, y1 = int(px0 * w), int(py0 * h), int(px1 * w), int(py1 * h)

    _draw_rounded_rect(frame, (x0, y0), (x1, y1), HUD_BG_COLOR, HUD_PANEL_RADIUS,
                       thickness=-1, alpha=0.6)
    _draw_rounded_rect(frame, (x0, y0), (x1, y1), HUD_BORDER_COLOR, HUD_PANEL_RADIUS,
                       thickness=1)
    cv2.putText(frame, "EQ", (x0 + 14, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_TEXT_COLOR, 1, cv2.LINE_AA)
    hint = "PINCH + DRAG"
    hint_w = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0]
    cv2.putText(frame, hint, (x1 - hint_w - 14, y0 + 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, HUD_TEXT_DIM, 1, cv2.LINE_AA)
    mouse_hint = "MOUSE: MOVE / WHEEL: SIZE"
    mw = cv2.getTextSize(mouse_hint, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0][0]
    cv2.putText(frame, mouse_hint, ((x0 + x1 - mw) // 2, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, HUD_TEXT_DIM, 1, cv2.LINE_AA)

    unity_y = bottom - int(EQ_UNITY_POS * span)
    kill_color = (60, 60, 230)

    for i, (name, x) in enumerate(zip(EQ_BANDS, xs)):
        color = HUD_EQ_COLORS[i]
        held = state.eq_grab == i
        hover = state.eq_hover == i and not held
        killed = state.eq_db[i] == float("-inf")
        knob_y = bottom - int(hud_state.eq_pos[i] * span)

        # Track, unity (0 dB) tick, and fill from 0 dB to the knob
        cv2.line(frame, (x, top), (x, bottom), HUD_INACTIVE_COLOR, 6, cv2.LINE_AA)
        cv2.line(frame, (x - 14, unity_y), (x + 14, unity_y), HUD_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.line(frame, (x, unity_y), (x, knob_y), kill_color if killed else color, 6, cv2.LINE_AA)

        # Knob
        if held:
            cv2.circle(frame, (x, knob_y), 22, color, 2, cv2.LINE_AA)
        elif hover:
            cv2.circle(frame, (x, knob_y), 19, color, 1, cv2.LINE_AA)
        cv2.circle(frame, (x, knob_y), 13 if held else 10,
                   kill_color if killed else color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, knob_y), 4, (255, 255, 255), -1, cv2.LINE_AA)

        # Band name and value above the track (the hand covers the bottom)
        name_color = color if (held or hover) else HUD_TEXT_COLOR
        tw = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.putText(frame, name, (x - tw // 2, top - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, name_color, 1, cv2.LINE_AA)
        value = _format_db(state.eq_db[i])
        vw = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        cv2.putText(frame, value, (x - vw // 2, top - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    kill_color if killed else HUD_TEXT_DIM, 1, cv2.LINE_AA)


def _draw_eq_cursor(frame, state: ConductorState):
    """Pinch cursor (ring = open hand, dot = pinching), drawn over the hand."""
    h, w = frame.shape[:2]
    if state.eq_cursor is not None:
        cx, cy = int(state.eq_cursor[0] * w), int(state.eq_cursor[1] * h)
        if state.eq_pinching:
            cv2.circle(frame, (cx, cy), 8, HUD_EQ_HAND_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), 12, HUD_EQ_HAND_COLOR, 2, cv2.LINE_AA)


def _draw_fx_grab(frame, state: ConductorState):
    """Left-hand filter/echo clutch: anchor ring + line to the pinch point
    while held, and a small cursor so you can see where the pinch is."""
    h, w = frame.shape[:2]
    color = HUD_ACCENT_CYAN
    if state.fx_grab and state.fx_anchor is not None and state.fx_cursor is not None:
        ax, ay = int(state.fx_anchor[0] * w), int(state.fx_anchor[1] * h)
        cx, cy = int(state.fx_cursor[0] * w), int(state.fx_cursor[1] * h)
        cv2.circle(frame, (ax, ay), 16, color, 1, cv2.LINE_AA)
        cv2.circle(frame, (ax, ay), 3, color, -1, cv2.LINE_AA)
        cv2.line(frame, (ax, ay), (cx, cy), color, 2, cv2.LINE_AA)
        for text, (dx, dy) in (("LP", (-58, 5)), ("HP", (36, 5)), ("ECHO", (-18, -26))):
            cv2.putText(frame, text, (ax + dx, ay + dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, HUD_TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 8, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
    elif state.fx_cursor is not None:
        cx, cy = int(state.fx_cursor[0] * w), int(state.fx_cursor[1] * h)
        cv2.circle(frame, (cx, cy), 12, color, 2, cv2.LINE_AA)


# ─── HUD Overlay ─────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, state: ConductorState, fps: float,
             hud_state: HUDState, hand_state: HandState,
             layout: EqLayout = None) -> np.ndarray:
    """
    Draw a premium heads-up display overlay on the camera frame.

    Features:
      - Rounded semi-transparent panels
      - Gradient progress bars
      - Pulsing glow on active stems
      - Smooth value interpolation
      - Hand detection status with label
    """
    h, w = frame.shape[:2]
    layout = layout or EqLayout()

    # Update smooth HUD values
    hud_state.update(state, hand_state)

    # MediaPipe "Left" = mixer (your right hand), "Right" = EQ (your left hand)
    mixer_hand = hand_state.left
    eq_hand = hand_state.right
    hands = [hd for hd in (mixer_hand, eq_hand) if hd is not None]

    is_fist = mixer_hand is not None and state.finger_count == 0

    # ── Edge Proximity Warning Vignette ──────────────────────────────────
    edge_prox = max((getattr(hd, 'edge_proximity', 0.0) for hd in hands), default=0.0)
    if edge_prox > 0.0:
        alpha = min(edge_prox * 0.35, 0.35)  # Max 35% opacity
        border_thickness = int(30 * edge_prox)

        if border_thickness >= 2:
            overlay = frame.copy()
            color = (50, 50, 220)  # Red BGR
            cv2.rectangle(overlay, (0, 0), (w, border_thickness), color, -1)
            cv2.rectangle(overlay, (0, h - border_thickness), (w, h), color, -1)
            cv2.rectangle(overlay, (0, 0), (border_thickness, h), color, -1)
            cv2.rectangle(overlay, (w - border_thickness, 0), (w, h), color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # ── Clenched Fist Drop Vignette (orange/red warning vignette) ──────────
    if is_fist:
        pulse_val = hud_state.pulse
        alpha = min(0.35 * pulse_val, 0.35)
        border_thickness = int(25 * pulse_val)

        if border_thickness >= 2:
            overlay = frame.copy()
            # Orange/red BGR color
            color = (15, 60, 220)
            cv2.rectangle(overlay, (0, 0), (w, border_thickness), color, -1)
            cv2.rectangle(overlay, (0, h - border_thickness), (w, h), color, -1)
            cv2.rectangle(overlay, (0, 0), (border_thickness, h), color, -1)
            cv2.rectangle(overlay, (w - border_thickness, 0), (w, h), color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)



    # ── EQ faders (your left hand) ──────────────────────────────────────
    _draw_eq(frame, state, hud_state, layout)

    # ── Draw Hand Skeletons ──────────────────────────────────────────────
    if mixer_hand is not None and mixer_hand.landmarks:
        _draw_hand_skeleton(frame, mixer_hand, is_fist, hud_state.pulse)
    if eq_hand is not None and eq_hand.landmarks:
        _draw_hand_skeleton(frame, eq_hand, False, hud_state.pulse,
                            base_color=HUD_EQ_HAND_COLOR)
    _draw_eq_cursor(frame, state)
    _draw_fx_grab(frame, state)

    # ── Bottom panel (rounded, semi-transparent) ──────────────────────────
    panel_h = 120
    panel_margin = 16
    panel_y = h - panel_h - panel_margin
    _draw_rounded_rect(
        frame,
        (panel_margin, panel_y),
        (w - panel_margin, h - panel_margin),
        HUD_BG_COLOR,
        HUD_PANEL_RADIUS,
        thickness=-1,
        alpha=HUD_PANEL_ALPHA,
    )
    # Subtle border
    _draw_rounded_rect(
        frame,
        (panel_margin, panel_y),
        (w - panel_margin, h - panel_margin),
        HUD_BORDER_COLOR,
        HUD_PANEL_RADIUS,
        thickness=1,
    )

    # ── Top bar (rounded, semi-transparent) ───────────────────────────────
    top_bar_h = 42
    _draw_rounded_rect(
        frame,
        (panel_margin, panel_margin),
        (w - panel_margin, panel_margin + top_bar_h),
        HUD_BG_COLOR,
        HUD_PANEL_RADIUS // 2,
        thickness=-1,
        alpha=HUD_PANEL_ALPHA,
    )
    _draw_rounded_rect(
        frame,
        (panel_margin, panel_margin),
        (w - panel_margin, panel_margin + top_bar_h),
        HUD_BORDER_COLOR,
        HUD_PANEL_RADIUS // 2,
        thickness=1,
    )

    # ── Title (top bar, centered) ─────────────────────────────────────────
    if is_fist:
        pulse_val = hud_state.pulse
        title = ">>  DROP ACTIVE  <<"
        title_color = (20, int(80 + 50 * pulse_val), int(220 + 35 * pulse_val))
    else:
        title = "AMBIENT CONDUCTOR"
        title_color = HUD_ACCENT_CYAN

    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    title_x = (w - title_size[0]) // 2
    cv2.putText(frame, title, (title_x, panel_margin + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, title_color, 1, cv2.LINE_AA)

    # ── Hand detection status (top bar, left): one dot per hand role ──────
    dot_x = panel_margin + 20
    dot_y = panel_margin + 22
    for hand_label, present in (("MIX (R)", state.left_detected), ("EQ (L)", state.right_detected)):
        if present:
            hand_color = (20, 100, 240) if (is_fist and hand_label.startswith("MIX")) else HUD_ACTIVE_COLOR
            cv2.circle(frame, (dot_x, dot_y), 7, hand_color, 1, cv2.LINE_AA)  # glow ring
        else:
            hand_color = HUD_INACTIVE_COLOR
        cv2.circle(frame, (dot_x, dot_y), 4, hand_color, -1, cv2.LINE_AA)
        cv2.putText(frame, hand_label, (dot_x + 14, dot_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, hand_color, 1, cv2.LINE_AA)
        dot_x += 105

    # ── FPS counter (top bar, right) ──────────────────────────────────────
    fps_text = f"{fps:.0f} FPS"
    fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    cv2.putText(frame, fps_text, (w - panel_margin - fps_size[0] - 12, panel_margin + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, HUD_TEXT_DIM, 1, cv2.LINE_AA)

    # ── Finger count (bottom panel, left section) ─────────────────────────
    count_text = str(state.finger_count)
    count_x = panel_margin + 30
    count_y = panel_y + 70
    count_color = (20, 100, 240) if is_fist else HUD_ACCENT_CYAN
    shadow_color = (10, 30, 80) if is_fist else (80, 60, 15)

    # Large number with glow
    cv2.putText(frame, count_text, (count_x - 1, count_y + 1),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, shadow_color, 4, cv2.LINE_AA)
    cv2.putText(frame, count_text, (count_x, count_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, count_color, 3, cv2.LINE_AA)

    # Label beneath
    cv2.putText(frame, "FINGERS", (count_x - 5, count_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, HUD_TEXT_DIM, 1, cv2.LINE_AA)

    # ── Stem indicators (bottom panel, center section) ────────────────────
    stem_labels = ["PAD", "BASS", "DRUMS", "MELODY"]
    stem_start_x = 180
    stem_y = panel_y + 35

    for i, label in enumerate(stem_labels):
        x = stem_start_x + i * 95
        glow = hud_state.stem_glows[i]
        color = HUD_STEM_COLORS[i]

        if glow > 0.1:
            # Active state: pulsing glow
            pulse = hud_state.pulse
            glow_radius = int(12 + 3 * pulse * glow)
            dim_color = tuple(int(c * 0.3 * glow) for c in color)

            # Outer glow ring
            cv2.circle(frame, (x, stem_y), glow_radius, dim_color, 2, cv2.LINE_AA)
            # Inner filled circle
            cv2.circle(frame, (x, stem_y), 7, color, -1, cv2.LINE_AA)
            # Bright center dot
            cv2.circle(frame, (x, stem_y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            # Inactive state
            cv2.circle(frame, (x, stem_y), 7, HUD_INACTIVE_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, stem_y), 7, HUD_BORDER_COLOR, 1, cv2.LINE_AA)

        # Label
        label_color = color if glow > 0.1 else HUD_TEXT_DIM
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0]
        cv2.putText(frame, label, (x - label_size[0] // 2, stem_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, label_color, 1, cv2.LINE_AA)

    # ── Filter cutoff bar (bottom panel, horizontal) ──────────────────────
    bar_x = 180
    bar_y = panel_y + 70
    bar_w = 360
    bar_h = 10

    _draw_gradient_bar_h(
        frame, bar_x, bar_y, bar_w, bar_h,
        hud_state.filter_norm,
        HUD_FILTER_BAR_START, HUD_FILTER_BAR_END,
        HUD_INACTIVE_COLOR,
    )

    # Label with dynamic Filter Mode and colors
    cutoff_hz = state.filter_cutoff_hz
    mode = state.filter_mode

    if mode == "low":
        dot_color = HUD_ACCENT_CYAN
        if cutoff_hz >= 1000:
            cutoff_label = f"LOW-PASS FILTER  {cutoff_hz / 1000:.1f} kHz"
        else:
            cutoff_label = f"LOW-PASS FILTER  {cutoff_hz:.0f} Hz"
    elif mode == "high":
        dot_color = HUD_ACCENT_MAGENTA
        if cutoff_hz >= 1000:
            cutoff_label = f"HIGH-PASS FILTER  {cutoff_hz / 1000:.1f} kHz"
        else:
            cutoff_label = f"HIGH-PASS FILTER  {cutoff_hz:.0f} Hz"
    else:
        dot_color = HUD_TEXT_DIM
        cutoff_label = "FILTER  BYPASSED"

    cv2.putText(frame, cutoff_label, (bar_x, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, HUD_TEXT_DIM, 1, cv2.LINE_AA)

    # Clutch status: MOVE while the mixer hand holds the FX, HOLD when fixed
    if state.fx_grab:
        fx_status, fx_status_color = "MOVE", HUD_ACCENT_CYAN
    elif mode != "bypass" or state.effect_wet_norm > 0:
        fx_status, fx_status_color = "HOLD  (pinch to move, R = reset)", HUD_ACCENT_ORANGE
    else:
        fx_status, fx_status_color = "PINCH RIGHT HAND TO MOVE", HUD_TEXT_DIM
    status_w = cv2.getTextSize(fx_status, cv2.FONT_HERSHEY_SIMPLEX, 0.33, 1)[0][0]
    cv2.putText(frame, fx_status, (bar_x + bar_w - status_w, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, fx_status_color, 1, cv2.LINE_AA)

    # Position dot on the bar
    dot_pos_x = bar_x + int(bar_w * hud_state.filter_norm)
    cv2.circle(frame, (dot_pos_x, bar_y + bar_h // 2), 6, dot_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (dot_pos_x, bar_y + bar_h // 2), 3, (255, 255, 255), -1, cv2.LINE_AA)

    # ── FX wet bar (bottom panel, right side, vertical) ───────────────────
    vbar_x = w - panel_margin - 55
    vbar_y = panel_y + 15
    vbar_w = 12
    vbar_h = panel_h - 45

    _draw_gradient_bar_v(
        frame, vbar_x, vbar_y, vbar_w, vbar_h,
        hud_state.fx_wet,
        HUD_FX_BAR_START, HUD_FX_BAR_END,
        HUD_INACTIVE_COLOR,
    )

    # Label
    wet_pct = int(hud_state.fx_wet * 100)
    fx_label = f"FX {wet_pct}%"
    cv2.putText(frame, fx_label, (vbar_x - 8, vbar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, HUD_TEXT_DIM, 1, cv2.LINE_AA)

    # Position dot on the vertical bar
    dot_pos_y = vbar_y + vbar_h - int(vbar_h * hud_state.fx_wet)
    cv2.circle(frame, (vbar_x + vbar_w // 2, dot_pos_y), 6, HUD_ACCENT_MAGENTA, -1, cv2.LINE_AA)
    cv2.circle(frame, (vbar_x + vbar_w // 2, dot_pos_y), 3, (255, 255, 255), -1, cv2.LINE_AA)

    return frame


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ambient Conductor — Gesture-controlled music")
    parser.add_argument("--stems-dir", default=STEMS_DIR, help="Path to stems directory")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--no-audio", action="store_true", help="Run vision only (no audio)")
    parser.add_argument("--bg", action="store_true",
                        help="Start with the illustration background instead of the webcam (toggle with B)")
    parser.add_argument("--song", help="Song file (MP3/WAV/...) to split into stems and play")
    parser.add_argument("--choose", action="store_true",
                        help="Pick a song from the songs/ folder in a menu")
    parser.add_argument("--split-model", default="htdemucs",
                        help="Demucs model used to split songs (htdemucs_ft = better, ~4x slower)")
    args = parser.parse_args()

    # ── Resolve paths relative to project root ────────────────────────────
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stems_path = os.path.join(project_root, args.stems_dir)

    # ── Song mode: MP3 → stems (split once, then cached) ──────────────────
    if args.song or args.choose:
        from song_prep import choose_song, prepare_song
        song = args.song or choose_song(project_root, model=args.split_model)
        if song:
            stems_path = str(prepare_song(song, project_root, model=args.split_model))

    # ── Generate stems if they don't exist ────────────────────────────────
    stems_exist = all(
        os.path.exists(os.path.join(stems_path, f"{name}.wav"))
        for name in STEM_NAMES
    )
    if not stems_exist:
        print("📁 Stems not found. Generating placeholder stems...")
        generate_all_stems()

    # ── Banner ────────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("   🎶  A M B I E N T   C O N D U C T O R  🎶")
    print("=" * 56)
    print()
    print("  Right hand → Raised fingers: add stems | fist: drop")
    print("             → Pinch + move: filter (left/right), echo (up) | release: fix")
    print("  Left hand  → Pinch a LOW/MID/HIGH fader and drag up/down")
    print("  Mouse      → Drag the EQ panel to move it, wheel to resize")
    print("  Keys       → R: reset FX/EQ | + - 0: EQ panel size/reset | B: background | Q/Esc: quit")
    print()

    # ── Initialize components ─────────────────────────────────────────────
    audio_engine = None
    if not args.no_audio:
        print("🔊 Loading audio engine...")
        audio_engine = AudioEngine(stems_dir=stems_path)
        audio_engine.start()

    print("📷 Starting vision tracker...")
    tracker = VisionTracker(camera_index=args.camera)
    layout = EqLayout.load()               # EQ panel position/size (layout.json)
    gesture_proc = GestureProcessor(layout)
    hud_state = HUDState()

    # ── Optional illustration background (webcam view is the default) ─────
    bg_filename = "pngtree-musical-illustration-featuring-colorful-swirling-waves-of-music-notes-picture-image_16449283.jpg.png"
    bg_img_path = os.path.join(project_root, bg_filename)
    bg_img = cv2.imread(bg_img_path) if os.path.exists(bg_img_path) else None
    show_bg = args.bg and bg_img is not None
    print(f"🖼  View: {'illustration background' if show_bg else 'webcam'}"
          + ("  (press B to switch)" if bg_img is not None else ""))

    # Window + mouse: drag the EQ panel to move it, wheel to resize
    window = "Ambient Conductor"
    cv2.namedWindow(window)
    shown_size = [0, 0]                      # (w, h) of the last frame shown
    panel_mouse = EqPanelMouse(layout, lambda: tuple(shown_size))
    cv2.setMouseCallback(window, panel_mouse.on_mouse)

    # Frame timing for FPS cap
    frame_interval = 1.0 / TARGET_FPS

    print("✅ All systems go! Show your hand to the camera.\n")

    try:
        while True:
            frame_start = time.monotonic()

            # ── Capture & process frame ───────────────────────────────────
            hand_state = tracker.process_frame()
            if hand_state is None:
                print("⚠ Camera read failed. Retrying...")
                time.sleep(0.1)
                continue

            # ── Extract gesture state ─────────────────────────────────────
            conductor_state = gesture_proc.process(hand_state)

            # ── Update audio engine ───────────────────────────────────────
            if audio_engine is not None:
                audio_engine.update_state(conductor_state)

            # ── Draw HUD overlay ──────────────────────────────────────────
            if hand_state.frame is not None:
                if show_bg:
                    h_cap, w_cap = hand_state.frame.shape[:2]
                    frame_base = cv2.resize(bg_img, (w_cap, h_cap))
                else:
                    frame_base = hand_state.frame  # mirrored webcam image

                frame = draw_hud(frame_base, conductor_state,
                                 hand_state.fps, hud_state, hand_state, layout)
                shown_size[0], shown_size[1] = frame.shape[1], frame.shape[0]
                cv2.imshow(window, frame)

            # ── Check for quit ────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or Esc
                break
            if key in (ord('b'), ord('B')) and bg_img is not None:
                show_bg = not show_bg
            elif key in (ord('r'), ord('R')):
                gesture_proc.reset()
            elif key in (ord('+'), ord('=')):
                layout.zoom(1.1)
                layout.save()
            elif key in (ord('-'), ord('_')):
                layout.zoom(1 / 1.1)
                layout.save()
            elif key == ord('0'):
                layout.reset()
                layout.save()

            # ── FPS cap ───────────────────────────────────────────────────
            elapsed = time.monotonic() - frame_start
            remaining = frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n\n⚡ Interrupted by user.")

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────
        print("\n🧹 Cleaning up...")
        if audio_engine is not None:
            audio_engine.stop()
        tracker.release()
        cv2.destroyAllWindows()
        print("👋 Goodbye!\n")


if __name__ == "__main__":
    main()
