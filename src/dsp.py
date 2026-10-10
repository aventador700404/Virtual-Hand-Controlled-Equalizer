"""
Ambient Conductor — DSP Module
================================
Low-pass Butterworth filter and feedback delay effect for
real-time audio processing inside the sounddevice callback.

Both classes maintain internal state across buffer boundaries
to prevent clicks and discontinuities.
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from config import (
    SAMPLE_RATE, BLOCK_SIZE, DELAY_TIME_MS, DELAY_FEEDBACK,
    EQ_LOW_CROSSOVER_HZ, EQ_HIGH_CROSSOVER_HZ,
)


class DualModeFilter:
    """
    Real-time dual-mode filter (Low-Pass on the left, High-Pass on the right).

    Supports low-pass, high-pass, and bypass modes. Maintains filter state
    across boundary sweeps to prevent pop/click artifacts.
    """

    def __init__(self, order: int = 4):
        self.order = order
        self.mode = 'bypass'
        self.current_cutoff = 20000.0
        self._min_change_hz = 50.0  # Minimum change to trigger redesign

        # Default open filter (low-pass at 20kHz)
        self.sos = self._design(20000.0, 'low')
        self._zi = sosfilt_zi(self.sos)

    def _design(self, cutoff_hz: float, btype: str) -> np.ndarray:
        """Design SOS coefficients for low-pass or high-pass filter."""
        nyquist = SAMPLE_RATE / 2.0
        if btype == 'low':
            cutoff_hz = max(40.0, min(cutoff_hz, nyquist - 100.0))
        else:
            cutoff_hz = max(20.0, min(cutoff_hz, 8000.0))
        return butter(self.order, cutoff_hz, btype=btype, fs=SAMPLE_RATE, output='sos')

    def update_filter(self, x_norm: float):
        """
        Determine target mode and cutoff frequency based on normalized X coordinate.
        [0.0, 0.43] -> Low-pass (100 Hz to 20000 Hz)
        [0.43, 0.57] -> Bypass (fully open, no processing)
        [0.57, 1.0] -> High-pass (20 Hz to 5000 Hz)
        """
        if x_norm < 0.43:
            target_mode = 'low'
            # Exponential sweep: 100 Hz at x=0.0, 20000 Hz at x=0.43
            t = x_norm / 0.43
            target_cutoff = 100.0 * (200.0 ** t)
        elif x_norm > 0.57:
            target_mode = 'high'
            # Exponential sweep: 20 Hz at x=0.57, 5000 Hz at x=1.0
            t = (x_norm - 0.57) / 0.43
            target_cutoff = 20.0 * (250.0 ** t)
        else:
            target_mode = 'bypass'
            target_cutoff = 20000.0

        mode_changed = (target_mode != self.mode)
        cutoff_changed = (abs(target_cutoff - self.current_cutoff) > self._min_change_hz)

        if mode_changed or (target_mode != 'bypass' and cutoff_changed):
            self.mode = target_mode
            self.current_cutoff = target_cutoff

            if target_mode == 'low':
                new_sos = self._design(target_cutoff, 'low')
            elif target_mode == 'high':
                new_sos = self._design(target_cutoff, 'high')
            else:
                # Bypass mode: use a wide-open low-pass
                new_sos = self._design(20000.0, 'low')

            new_zi = sosfilt_zi(new_sos)
            # Carry forward state if structural shape matches (same order filter)
            if self._zi.shape == new_zi.shape:
                pass
            else:
                self._zi = new_zi
            self.sos = new_sos

    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """Filter a block of audio samples using the active filter mode."""
        if self.mode == 'bypass':
            return audio_block.copy()  # Return copy to preserve block interface

        filtered, self._zi = sosfilt(self.sos, audio_block, zi=self._zi)
        return filtered.astype(np.float32)


class FeedbackDelay:
    """
    Vectorized feedback delay line that creates a reverb-like wash
    at short delay times. Uses a circular buffer with NumPy array
    operations instead of a Python for-loop.

    Parameters:
        delay_time_ms: Length of the delay in milliseconds.
        feedback: How much of the delayed signal is fed back (0.0–1.0).
                  Keep below 0.7 to prevent runaway feedback.
    """

    def __init__(
        self,
        delay_time_ms: float = DELAY_TIME_MS,
        feedback: float = DELAY_FEEDBACK,
    ):
        self.feedback = min(feedback, 0.95)  # Safety clamp

        # Calculate delay length in samples
        self.delay_samples = int(SAMPLE_RATE * delay_time_ms / 1000.0)

        # Circular buffer (pre-allocated, zero-filled)
        self._buffer = np.zeros(self.delay_samples, dtype=np.float32)
        self._write_pos = 0

        # Current wet/dry mix (0.0 = fully dry, 1.0 = fully wet)
        self.wet_mix = 0.0

    def update_wet_mix(self, wet_norm: float):
        """Update the wet/dry mix from a normalized value (0.0–1.0)."""
        self.wet_mix = max(0.0, min(1.0, wet_norm))

    def process(self, audio_block: np.ndarray) -> np.ndarray:
        """
        Process a block of audio through the delay line.
        Fully vectorized — no Python for-loop over samples.
        Returns the wet/dry mixed output.
        """
        block_len = len(audio_block)

        # Build the array of read indices into the circular buffer
        read_indices = (self._write_pos + np.arange(block_len)) % self.delay_samples

        # Read delayed samples (vectorized)
        delayed = self._buffer[read_indices]

        # Mix: output = dry * (1 - wet*0.5) + delayed * wet
        output = audio_block * (1.0 - self.wet_mix * 0.5) + delayed * self.wet_mix

        # Write new samples + feedback into the buffer (vectorized)
        new_values = audio_block + delayed * self.feedback
        self._buffer[read_indices] = new_values

        # Advance write position
        self._write_pos = (self._write_pos + block_len) % self.delay_samples

        return output.astype(np.float32)


def db_to_gain(db: float) -> float:
    """dB → linear gain. -inf dB (a killed band) → 0.0."""
    if db == float("-inf"):
        return 0.0
    return float(10.0 ** (db / 20.0))


def _lr4(cutoff_hz: float, btype: str) -> np.ndarray:
    """4th-order Linkwitz-Riley = two cascaded 2nd-order Butterworth sections."""
    sos = butter(2, cutoff_hz, btype=btype, fs=SAMPLE_RATE, output='sos')
    return np.vstack([sos, sos])


class ThreeBandEQ:
    """
    DJ-style 3-band isolator EQ (LOW / MID / HIGH).

    The signal is split with Linkwitz-Riley crossovers, each band gets its own
    gain, and the bands are summed back. LR4 bands sum to a flat magnitude
    response, so at 0 dB everywhere the EQ is transparent, and a gain of 0
    removes the band with a steep 24 dB/oct slope (a real DJ "kill").

        low  = LP(f1) → allpass(f2)        (phase-aligned with the others)
        mid  = HP(f1) → LP(f2)
        high = HP(f1) → HP(f2)

    Gains are ramped across each block to avoid zipper noise.
    """

    def __init__(self, low_hz: float = EQ_LOW_CROSSOVER_HZ,
                 high_hz: float = EQ_HIGH_CROSSOVER_HZ):
        lp1, hp1 = _lr4(low_hz, 'low'), _lr4(low_hz, 'high')
        lp2, hp2 = _lr4(high_hz, 'low'), _lr4(high_hz, 'high')
        # Each filter instance keeps its own state across blocks
        self._sos = {
            'lp1': lp1, 'hp1': hp1,
            'mid_lp2': lp2, 'high_hp2': hp2,
            'ap_lp2': lp2, 'ap_hp2': hp2,
        }
        self._zi = {k: np.zeros((sos.shape[0], 2)) for k, sos in self._sos.items()}
        self._gains = np.ones(3, dtype=np.float32)

    def _filt(self, key: str, x: np.ndarray) -> np.ndarray:
        y, self._zi[key] = sosfilt(self._sos[key], x, zi=self._zi[key])
        return y

    def process(self, audio_block: np.ndarray, target_gains) -> np.ndarray:
        """Apply the EQ. `target_gains` = linear gains for (LOW, MID, HIGH)."""
        x = audio_block.astype(np.float64)
        low = self._filt('lp1', x)
        rest = self._filt('hp1', x)
        mid = self._filt('mid_lp2', rest)
        high = self._filt('high_hp2', rest)
        low = self._filt('ap_lp2', low) + self._filt('ap_hp2', low)  # allpass(f2)

        n = len(x)
        out = np.zeros(n, dtype=np.float64)
        for i, band in enumerate((low, mid, high)):
            start, end = float(self._gains[i]), float(target_gains[i])
            if abs(start - end) < 1e-4:
                out += band * end
            else:
                out += band * np.linspace(start, end, n)
            self._gains[i] = end
        return out.astype(np.float32)
