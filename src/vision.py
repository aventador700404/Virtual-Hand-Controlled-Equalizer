"""
Ambient Conductor — Vision Module
===================================
Captures webcam frames via OpenCV, runs MediaPipe Hand Landmarker
(Tasks API) to detect a single hand, and returns normalized,
smoothed wrist coordinates along with full landmark data for
gesture processing.

Includes a grace period to prevent single-frame tracking drops
from disrupting the musical performance.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    SMOOTHING_FACTOR,
    WRIST,
    HAND_LOST_GRACE_FRAMES,
)

# ─── MediaPipe Tasks API imports ──────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand connections for drawing (21 landmarks, standard topology)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


@dataclass
class HandData:
    """Normalized data for a single detected hand."""
    wrist_x: float = 0.5
    wrist_y: float = 0.5       # Already Y-inverted (up = 1.0)
    landmarks: list = field(default_factory=list)  # All 21 landmark objects
    handedness: str = "Unknown"
    edge_proximity: float = 0.0  # 0.0 = center, 1.0 = at the very edge


@dataclass
class HandState:
    """Combined state for both hands in a single frame."""
    left: Optional[HandData] = None
    right: Optional[HandData] = None
    frame: Optional[np.ndarray] = None   # Annotated camera frame for display
    fps: float = 0.0


def _find_model_path() -> str:
    """Locate the hand_landmarker.task model file."""
    # Check relative to this file's directory (src/)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)

    candidates = [
        os.path.join(project_root, "models", "hand_landmarker.task"),
        os.path.join(project_root, "hand_landmarker.task"),
        os.path.join(src_dir, "hand_landmarker.task"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find 'hand_landmarker.task' model file. "
        "Download it with:\n"
        "  curl -L -o models/hand_landmarker.task --create-dirs "
        '"https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
        'hand_landmarker/float16/latest/hand_landmarker.task"'
    )


def _compute_edge_proximity(wrist_x: float, wrist_y: float) -> float:
    """
    Compute how close the hand is to the edge of the frame.
    Returns 0.0 when centered, approaches 1.0 at the very edge.
    """
    # Distance from center (0.5, 0.5) in each axis, doubled to normalize to 0-1
    dx = abs(wrist_x - 0.5) * 2.0  # 0.0 at center, 1.0 at edge
    dy = abs(wrist_y - 0.5) * 2.0
    # Use the maximum axis distance
    proximity = max(dx, dy)
    # Make it ramp up sharply near the edges (last 20% of frame)
    proximity = max(0.0, (proximity - 0.6) / 0.4)
    return min(1.0, proximity)


class VisionTracker:
    """
    Manages the webcam capture and MediaPipe Hand Landmarker pipeline.
    Call `process_frame()` each iteration of your main loop.

    Includes a grace period: if tracking is lost for fewer than
    HAND_LOST_GRACE_FRAMES consecutive frames, the last known
    hand position is reported to prevent single-frame tracking drops.
    """

    def __init__(self, camera_index: int = CAMERA_INDEX):
        # ── Camera setup ──────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}. "
                "Check that your webcam is connected and camera permissions "
                "are granted for your terminal app."
            )

        # Log actual camera resolution (may differ from requested)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  ✓ Camera opened: {actual_w}×{actual_h} "
              f"(requested {CAMERA_WIDTH}×{CAMERA_HEIGHT})")

        # ── MediaPipe Hand Landmarker (Tasks API) ─────────────────────────
        model_path = _find_model_path()
        print(f"  ✓ Loading model: {model_path}")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # ── Real monotonic timestamp tracking (required for VIDEO mode) ───
        self._start_time_ns = time.monotonic_ns()
        self._last_timestamp_ms = 0  # Track last to ensure monotonic increase

        # ── EMA smoothing state ───────────────────────────────────────────
        self._smooth_left_x: float = 0.5
        self._smooth_left_y: float = 0.5
        self._smooth_right_x: float = 0.5
        self._smooth_right_y: float = 0.5

        # ── FPS tracking ─────────────────────────────────────────────────
        self._prev_time: float = 0.0

        # ── Grace period state ────────────────────────────────────────────
        self._frames_since_left_lost: int = HAND_LOST_GRACE_FRAMES + 1
        self._frames_since_right_lost: int = HAND_LOST_GRACE_FRAMES + 1
        self._last_left_data: Optional[HandData] = None
        self._last_right_data: Optional[HandData] = None

    def _get_timestamp_ms(self) -> int:
        """Get a real monotonic timestamp in milliseconds for MediaPipe."""
        elapsed_ns = time.monotonic_ns() - self._start_time_ns
        timestamp_ms = elapsed_ns // 1_000_000
        # Ensure strictly monotonically increasing
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _ema(self, prev: float, new: float, alpha: float = SMOOTHING_FACTOR) -> float:
        """Exponential moving average for jitter reduction."""
        return prev + alpha * (new - prev)

    def _draw_landmarks(self, frame: np.ndarray, landmarks, is_left: bool):
        """
        Draw hand landmarks and connections on the frame with
        a glow effect for a more polished visual appearance.
        """
        h, w = frame.shape[:2]

        # Convert normalized landmarks to pixel coordinates
        points = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))

        # Base color — cyan tint
        color = (230, 200, 50)  # Cyan (BGR)

        # Draw connections with glow
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                # Outer glow (thicker, semi-transparent)
                cv2.line(frame, points[start_idx], points[end_idx],
                         (115, 100, 25), 4, cv2.LINE_AA)
                # Inner line
                cv2.line(frame, points[start_idx], points[end_idx],
                         color, 2, cv2.LINE_AA)

        # Draw landmark dots with glow
        for i, (px, py) in enumerate(points):
            is_tip = i in (4, 8, 12, 16, 20)
            if is_tip:
                # Fingertips: larger with glow ring
                cv2.circle(frame, (px, py), 10, (115, 100, 25), 1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 6, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # Regular joints
                cv2.circle(frame, (px, py), 4, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_edge_warning(self, frame: np.ndarray, edge_proximity: float):
        """
        Draw a subtle red vignette on the frame edges when the
        hand is approaching the edge of the camera frame.
        """
        if edge_proximity <= 0.0:
            return

        h, w = frame.shape[:2]
        alpha = min(edge_proximity * 0.35, 0.35)  # Max 35% opacity
        border_thickness = int(30 * edge_proximity)

        if border_thickness < 2:
            return

        overlay = frame.copy()
        # Draw red borders on all sides
        color = (50, 50, 220)  # Red-ish (BGR)
        cv2.rectangle(overlay, (0, 0), (w, border_thickness), color, -1)
        cv2.rectangle(overlay, (0, h - border_thickness), (w, h), color, -1)
        cv2.rectangle(overlay, (0, 0), (border_thickness, h), color, -1)
        cv2.rectangle(overlay, (w - border_thickness, 0), (w, h), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    def process_frame(self) -> Optional[HandState]:
        """
        Capture one frame, run hand detection, and return a HandState.
        Returns None if the camera read fails.

        Implements a grace period: if a hand was tracked recently but
        is lost for fewer than HAND_LOST_GRACE_FRAMES frames, the last
        known position is returned to prevent momentary tracking drops.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Flip horizontally so it feels like a mirror
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── FPS calculation ───────────────────────────────────────────────
        current_time = cv2.getTickCount()
        if self._prev_time > 0:
            elapsed = (current_time - self._prev_time) / cv2.getTickFrequency()
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
        else:
            fps = 0.0
        self._prev_time = current_time

        # ── Run MediaPipe Hand Landmarker ─────────────────────────────────
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = self._get_timestamp_ms()
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # ── Parse results ─────────────────────────────────────────────────
        state = HandState(fps=fps)
        left_data: Optional[HandData] = None
        right_data: Optional[HandData] = None
        detected: list[HandData] = []

        if results.hand_landmarks and results.handedness:
            for hand_landmarks, hand_info in zip(
                results.hand_landmarks, results.handedness
            ):
                label = hand_info[0].category_name  # "Left" or "Right"

                # Draw landmarks on the frame (moved to draw_hud for fist coloring)
                is_left = (label == "Left")
                # self._draw_landmarks(frame, hand_landmarks, is_left)

                # Extract wrist position (normalized 0–1)
                wrist = hand_landmarks[WRIST]
                raw_x = wrist.x
                raw_y = 1.0 - wrist.y  # Invert Y: up = higher value

                # Clamp to [0, 1]
                raw_x = max(0.0, min(1.0, raw_x))
                raw_y = max(0.0, min(1.0, raw_y))

                # Compute edge proximity for warning overlay
                edge_prox = _compute_edge_proximity(wrist.x, wrist.y)

                hand_data = HandData(
                    wrist_x=raw_x,
                    wrist_y=raw_y,
                    landmarks=hand_landmarks,
                    handedness=label,
                    edge_proximity=edge_prox,
                )

                detected.append(hand_data)

        # Roles come from MediaPipe's label: "Left" → mixer, "Right" → EQ.
        # On this mirrored setup "Left" is your RIGHT hand (on the right of
        # the screen) and "Right" is your LEFT hand. If both hands get the
        # same label, split them by screen position the same way.
        if len(detected) == 2 and detected[0].handedness == detected[1].handedness:
            detected.sort(key=lambda d: d.wrist_x)
            detected[0].handedness, detected[1].handedness = "Right", "Left"

        for hand_data in detected:
            if hand_data.handedness == "Left":
                left_data = hand_data
            else:
                right_data = hand_data

        # ── Grace period logic ────────────────────────────────────────────
        if left_data is not None:
            self._frames_since_left_lost = 0
            self._last_left_data = left_data
        else:
            self._frames_since_left_lost += 1
            if (self._frames_since_left_lost <= HAND_LOST_GRACE_FRAMES
                    and self._last_left_data is not None):
                left_data = self._last_left_data

        if right_data is not None:
            self._frames_since_right_lost = 0
            self._last_right_data = right_data
        else:
            self._frames_since_right_lost += 1
            if (self._frames_since_right_lost <= HAND_LOST_GRACE_FRAMES
                    and self._last_right_data is not None):
                right_data = self._last_right_data

        # ── Apply EMA smoothing ───────────────────────────────────────────
        if left_data:
            self._smooth_left_x = self._ema(self._smooth_left_x, left_data.wrist_x)
            self._smooth_left_y = self._ema(self._smooth_left_y, left_data.wrist_y)
            left_data.wrist_x = self._smooth_left_x
            left_data.wrist_y = self._smooth_left_y
            state.left = left_data

        if right_data:
            self._smooth_right_x = self._ema(self._smooth_right_x, right_data.wrist_x)
            self._smooth_right_y = self._ema(self._smooth_right_y, right_data.wrist_y)
            right_data.wrist_x = self._smooth_right_x
            right_data.wrist_y = self._smooth_right_y
            state.right = right_data

        # ── Draw edge warning if hand is near the boundary ────────────────
        active_hand = right_data or left_data
        # Commented out here, drawn in draw_hud on the background image instead
        # if active_hand and hasattr(active_hand, 'edge_proximity'):
        #     self._draw_edge_warning(frame, active_hand.edge_proximity)

        state.frame = frame
        return state

    def release(self):
        """Release camera and MediaPipe resources."""
        self.cap.release()
        self.landmarker.close()
