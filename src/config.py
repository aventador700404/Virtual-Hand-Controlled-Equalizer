"""
Ambient Conductor — Configuration Constants
=============================================
Central configuration for sample rates, buffer sizes, camera settings,
MediaPipe landmark IDs, audio mapping ranges, and HUD design system.
"""

# ─── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024          # ~23 ms latency at 44.1 kHz
CHANNELS = 1               # Mono output (stems are mixed to mono)
STEM_DURATION_SEC = 16     # Length of generated placeholder stems

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TARGET_FPS = 30            # Cap the main loop at ~30 FPS

# ─── MediaPipe Hands ──────────────────────────────────────────────────────────
MAX_HANDS = 2              # your right hand = mixer (FX), your left hand = EQ
MIN_DETECTION_CONFIDENCE = 0.5    # Lowered from 0.7 for more robust tracking
MIN_TRACKING_CONFIDENCE = 0.4     # Lowered from 0.5 for fewer hand drops

# Grace period: keep reporting last hand position for this many frames
# after tracking is lost, to prevent single-frame drops from killing the state.
HAND_LOST_GRACE_FRAMES = 8        # ~260ms at 30 FPS

# Landmark IDs (from MediaPipe hand model)
WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Finger tip/pip pairs for counting (excluding thumb)
FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

# ─── Coordinate Smoothing ────────────────────────────────────────────────────
SMOOTHING_FACTOR = 0.25    # EMA alpha (lower = smoother but laggier)

# ─── Audio Mapping Ranges ────────────────────────────────────────────────────
MIN_CUTOFF_HZ = 300.0      # Hand at far left
MAX_CUTOFF_HZ = 20000.0    # Hand at far right

# Delay / reverb-like effect
DELAY_TIME_MS = 250.0      # Fixed delay line length
DELAY_FEEDBACK = 0.4       # Feedback amount (0.0 – 1.0, keep < 0.7 to avoid runaway)

# ─── Mixer-Hand FX (filter + echo) — pinch to move, release to fix ──────────
# While pinching, the pinch point is the neutral reference; moving away from
# it changes the effect. These are how far (in frame widths / heights) the hand
# must travel to sweep the full range.
FX_X_TRAVEL = 0.5          # full low-pass ← → high-pass sweep
FX_Y_TRAVEL = 0.5          # echo 0 → 100 %
PINCH_FREEZE_FRAMES = 5    # keep the finger count frozen this long after a pinch
FX_DEAD_ZONE = 0.015       # ignore hand jitter this close to the pinch point

# ─── Right-Hand DJ EQ (3-band isolator) ──────────────────────────────────────
EQ_BANDS = ["LOW", "MID", "HIGH"]
EQ_LOW_CROSSOVER_HZ = 250.0        # LOW  <  250 Hz
EQ_HIGH_CROSSOVER_HZ = 2500.0      # HIGH > 2.5 kHz, MID in between
EQ_MAX_DB = 6.0                    # Fader top
EQ_MIN_DB = -30.0                  # Just above the kill zone
EQ_KILL_POS = 0.04                 # Fader position at/below this = band muted
EQ_DETENT_DB = 1.5                 # Snap to 0 dB when this close (easy reset)

# On-screen fader position/size: see layout.py (movable with the mouse,
# saved to layout.json)

# Pinch = thumb tip ↔ index tip distance relative to hand size (hysteresis)
PINCH_ON_RATIO = 0.30
PINCH_OFF_RATIO = 0.45
PINCH_CURSOR_SMOOTHING = 0.5       # EMA alpha for the pinch cursor

# ─── Gesture Debounce ────────────────────────────────────────────────────────
FINGER_COUNT_DEBOUNCE_FRAMES = 3   # Consecutive frames required before state change

# ─── Stem File Names ─────────────────────────────────────────────────────────
STEM_NAMES = ["pad", "bass", "drums", "melody"]
STEMS_DIR = "stems"

# ─── HUD Design System (BGR for OpenCV) ──────────────────────────────────────

# Panel styling
HUD_PANEL_ALPHA = 0.75         # Panel transparency
HUD_PANEL_RADIUS = 18          # Rounded corner radius

# Core palette — dark mode with vibrant accents
HUD_BG_COLOR = (15, 15, 20)           # Near-black panel background
HUD_BG_COLOR_2 = (25, 25, 35)         # Slightly lighter for depth
HUD_TEXT_COLOR = (210, 210, 215)       # Soft white text
HUD_TEXT_DIM = (120, 120, 130)         # Dimmed secondary text
HUD_BORDER_COLOR = (50, 50, 60)       # Subtle panel border

# Accent colors (BGR)
HUD_ACCENT_CYAN = (230, 200, 50)      # Cyan / teal (primary accent)
HUD_ACCENT_MAGENTA = (200, 50, 230)   # Magenta / pink (secondary accent)
HUD_ACCENT_PURPLE = (235, 100, 160)   # Soft purple
HUD_ACCENT_ORANGE = (60, 140, 255)    # Warm orange

# Status colors (BGR)
HUD_ACTIVE_COLOR = (180, 245, 80)     # Bright lime green
HUD_INACTIVE_COLOR = (55, 55, 60)     # Dark muted gray

# Gradient bar colors — start and end (BGR)
HUD_FILTER_BAR_START = (230, 180, 30)   # Cyan
HUD_FILTER_BAR_END = (200, 50, 230)     # Magenta
HUD_FX_BAR_START = (235, 100, 160)      # Purple
HUD_FX_BAR_END = (60, 140, 255)         # Orange

# Glow effect
HUD_GLOW_RADIUS = 20
HUD_GLOW_ALPHA = 0.3

# Stem indicator colors when active (BGR) — each stem gets its own color
HUD_STEM_COLORS = [
    (230, 200, 50),    # PAD:    Cyan
    (200, 50, 230),    # BASS:   Magenta
    (60, 200, 255),    # DRUMS:  Orange/Gold
    (180, 245, 80),    # MELODY: Lime green
]

# Hand landmark rendering
HUD_HAND_SKELETON_COLOR = (230, 200, 50)   # Cyan skeleton
HUD_HAND_JOINT_COLOR = (255, 255, 255)     # White joints
HUD_HAND_GLOW_COLOR = (230, 200, 50)       # Glow around joints
HUD_EDGE_WARNING_COLOR = (50, 50, 220)     # Red warning at frame edges

# Smooth interpolation speed for HUD values (0.0–1.0, higher = faster)
HUD_LERP_SPEED = 0.15

# EQ fader colors (BGR): LOW, MID, HIGH
HUD_EQ_COLORS = [
    (200, 50, 230),    # LOW:  Magenta
    (60, 200, 255),    # MID:  Gold
    (230, 200, 50),    # HIGH: Cyan
]
HUD_EQ_HAND_COLOR = (235, 100, 160)   # Right (EQ) hand skeleton: soft purple
