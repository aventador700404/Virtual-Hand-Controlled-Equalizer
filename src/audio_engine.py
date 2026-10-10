"""
Ambient Conductor — Audio Engine
==================================
Loads audio stems, plays them in sync via a sounddevice callback,
and applies real-time gain control and DSP effects based on the
ConductorState provided by the gesture processor.

Lock-free design: the vision thread writes a ConductorState reference;
the audio callback reads it. Python's GIL ensures reference assignment
is effectively atomic for this use case.
"""

import os
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import (
    SAMPLE_RATE,
    BLOCK_SIZE,
    CHANNELS,
    STEMS_DIR,
    STEM_NAMES,
)
from dsp import DualModeFilter, FeedbackDelay, ThreeBandEQ, db_to_gain
from gestures import ConductorState


class AudioEngine:
    """
    Multi-stem real-time audio mixer with DSP effects.

    Loads stems from .wav files, plays them in a continuous loop,
    and adjusts gain + filters based on a shared ConductorState.
    """

    def __init__(self, stems_dir: str = STEMS_DIR):
        # ── Load stems ────────────────────────────────────────────────────
        self.stems: list[np.ndarray] = []
        self.stem_lengths: list[int] = []
        self._load_stems(stems_dir)

        if not self.stems:
            raise RuntimeError(
                f"No stems found in '{stems_dir}/'. "
                "Run `python src/generate_stems.py` first to create placeholder stems."
            )

        # Ensure all stems are the same length (pad shorter ones with silence)
        max_len = max(self.stem_lengths)
        for i in range(len(self.stems)):
            if len(self.stems[i]) < max_len:
                padding = np.zeros(max_len - len(self.stems[i]), dtype=np.float32)
                self.stems[i] = np.concatenate([self.stems[i], padding])
        self.loop_length = max_len

        # ── Playback state ────────────────────────────────────────────────
        self._play_head = 0

        # ── Per-stem gain (current and target, for smooth ramping) ────────
        self._num_stems = len(self.stems)
        self._current_gains = np.zeros(self._num_stems, dtype=np.float32)
        self._target_gains = np.zeros(self._num_stems, dtype=np.float32)
        # Start with the pad stem on
        self._target_gains[0] = 0.8

        # ── DSP effects ───────────────────────────────────────────────────
        self.eq = ThreeBandEQ()
        self.filter = DualModeFilter()
        self.delay = FeedbackDelay()

        # ── Lock-free conductor state ─────────────────────────────────────
        # Python's GIL makes simple reference assignment atomic.
        # The vision thread writes a new ConductorState object;
        # the audio callback reads the current reference.
        self._conductor_state = ConductorState()

        # ── Audio stream ──────────────────────────────────────────────────
        self._stream: Optional[sd.OutputStream] = None

    def _load_stems(self, stems_dir: str):
        """Load stem .wav files in the order defined by STEM_NAMES."""
        for name in STEM_NAMES:
            filepath = os.path.join(stems_dir, f"{name}.wav")
            if os.path.exists(filepath):
                data, sr = sf.read(filepath, dtype='float32')

                # Convert stereo to mono if needed
                if data.ndim == 2:
                    data = data.mean(axis=1)

                # Resample if sample rate doesn't match
                if sr != SAMPLE_RATE:
                    try:
                        from scipy.signal import resample
                        new_length = int(len(data) * SAMPLE_RATE / sr)
                        print(
                            f"  ⟳ Resampling '{name}.wav': {sr} Hz → {SAMPLE_RATE} Hz "
                            f"({len(data)} → {new_length} samples)..."
                        )
                        data = resample(data, new_length).astype(np.float32)
                        sr = SAMPLE_RATE
                    except ImportError:
                        print(
                            f"⚠ Warning: '{filepath}' has sample rate {sr}, "
                            f"expected {SAMPLE_RATE}. Install scipy for auto-resampling."
                        )

                self.stems.append(data)
                self.stem_lengths.append(len(data))
                print(f"  ✓ Loaded stem: {name}.wav ({len(data) / sr:.1f}s)")
            else:
                print(f"  ✗ Stem not found: {filepath}")

    def update_state(self, state: ConductorState):
        """
        Update the shared conductor state (called from the vision thread).
        Lock-free: simple reference assignment is atomic under the GIL.
        """
        self._conductor_state = state

    def _compute_target_gains(self, finger_count: int, hand_active: bool):
        """
        Set target gains based on finger count and hand detection.
        If a hand is active and finger count is 0 (clenched fist), mute all stems.
        Otherwise, Stem 0 (Pad) is on at 0.8, and other stems activate with fingers.
        """
        targets = np.zeros(self._num_stems, dtype=np.float32)

        if hand_active and finger_count == 0:
            # Clenched fist: mute everything
            self._target_gains = targets
            return

        targets[0] = 0.8  # Pad always on

        stem_gains = [0.8, 0.75, 0.7, 0.65]  # Decreasing slightly to prevent clipping
        for i in range(1, min(self._num_stems, 4)):
            if finger_count >= i:
                targets[i] = stem_gains[i]

        # Handle extra stems beyond 4 (if any)
        for i in range(4, self._num_stems):
            if finger_count >= i:
                targets[i] = 0.6

        self._target_gains = targets

    def _audio_callback(self, outdata, frames, time_info, status):
        """
        The real-time audio callback invoked by sounddevice.
        Runs on a high-priority PortAudio thread.

        - Reads the current block from each stem.
        - Applies smoothed per-stem gain.
        - Sums the stems.
        - Applies the 3-band EQ, DJ filter and delay effect.
        - Writes to the output buffer.
        """
        if status:
            print(f"Audio status: {status}")

        # ── Read the current conductor state (lock-free snapshot) ─────────
        state = self._conductor_state

        # ── Update targets from gesture ───────────────────────────────────
        # Fist-mute only applies while the mixer (left) hand is in view
        hand_active = state.mixer_active
        self._compute_target_gains(state.finger_count, hand_active)

        # ── Update DSP parameters ─────────────────────────────────────────
        self.filter.update_filter(state.filter_cutoff_norm)
        if hand_active and state.finger_count == 0:
            # Clenched fist: maximize reverb/delay tail
            self.delay.update_wet_mix(0.85)
        else:
            self.delay.update_wet_mix(state.effect_wet_norm)

        # ── Read blocks from each stem with dynamic tempo/resampling ──────
        T = state.tempo_factor

        # Calculate lookup indices inside the circular stem buffer
        indices = (self._play_head + np.arange(frames, dtype=np.float64) * T) % self.loop_length
        idx_floor = np.floor(indices).astype(np.int32)
        idx_ceil = (idx_floor + 1) % self.loop_length
        frac = (indices - idx_floor).astype(np.float32)

        mix = np.zeros(frames, dtype=np.float32)

        for i in range(self._num_stems):
            # Linearly interpolate between adjacent sample indices to resample on-the-fly
            chunk = (1.0 - frac) * self.stems[i][idx_floor] + frac * self.stems[i][idx_ceil]

            # ── Smooth gain ramping (linear interpolation over the block) ─
            start_gain = self._current_gains[i]
            end_gain = self._target_gains[i]

            if abs(start_gain - end_gain) < 0.001:
                # No significant change — just multiply
                mix += chunk * end_gain
                self._current_gains[i] = end_gain
            else:
                # Linear ramp from start_gain to end_gain
                ramp = np.linspace(start_gain, end_gain, frames, dtype=np.float32)
                mix += chunk * ramp
                self._current_gains[i] = end_gain

        # ── Advance play head (loop) ──────────────────────────────────────
        self._play_head = (self._play_head + frames * T) % self.loop_length

        # ── Apply DSP chain ───────────────────────────────────────────────
        mix = self.eq.process(mix, [db_to_gain(db) for db in state.eq_db])
        mix = self.filter.process(mix)
        mix = self.delay.process(mix)

        # ── Soft clipping to prevent harsh distortion ─────────────────────
        mix = np.tanh(mix)

        # ── Write to output buffer ────────────────────────────────────────
        outdata[:, 0] = mix

    def start(self):
        """Start the audio stream."""
        print("\n🔊 Starting audio engine...")
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype='float32',
            callback=self._audio_callback,
            latency='low',
        )
        self._stream.start()
        print("  ✓ Audio stream active")

    def stop(self):
        """Stop the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            print("🔇 Audio engine stopped")
