"""
Ambient Conductor — Stem Generator
=====================================
Generates 4 synthetic placeholder audio stems using NumPy:
  1. pad.wav    — Lush, evolving pad (layered detuned sine waves + filtered noise)
  2. bass.wav   — Deep bass pattern (sine sub-bass with rhythmic envelope)
  3. drums.wav  — Drum pattern (kick + snare + hi-hat at ~100 BPM)
  4. melody.wav — Ethereal pentatonic melody with vibrato

All stems are:
  - Mono, 44100 Hz, float32
  - Exactly STEM_DURATION_SEC seconds long
  - Loopable (fade in/out at boundaries)
  - Musically coherent when layered together (key of C minor)

Usage:
    python src/generate_stems.py
"""

import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

# Add parent directory so we can import config
sys.path.insert(0, os.path.dirname(__file__))
from config import SAMPLE_RATE, STEM_DURATION_SEC, STEMS_DIR


# ─── Utility Functions ───────────────────────────────────────────────────────

def note_freq(note_name: str) -> float:
    """Convert a note name (e.g., 'C3', 'G#4') to frequency in Hz."""
    notes = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
    }
    # Parse note name and octave
    if len(note_name) >= 3 and note_name[1] in ('#', 'b'):
        name = note_name[:2]
        octave = int(note_name[2:])
    else:
        name = note_name[0]
        octave = int(note_name[1:])

    semitone = notes[name] + (octave + 1) * 12
    return 440.0 * (2.0 ** ((semitone - 69) / 12.0))


def fade_in_out(audio: np.ndarray, fade_ms: float = 50.0) -> np.ndarray:
    """Apply a short fade-in and fade-out to prevent loop clicks."""
    fade_samples = int(SAMPLE_RATE * fade_ms / 1000.0)
    fade_samples = min(fade_samples, len(audio) // 4)

    audio = audio.copy()
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    return audio


def normalize(audio: np.ndarray, peak: float = 0.85) -> np.ndarray:
    """Normalize audio to a target peak level."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio * (peak / max_val)
    return audio


def lowpass(audio: np.ndarray, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth low-pass filter."""
    sos = butter(order, cutoff_hz, btype='low', fs=SAMPLE_RATE, output='sos')
    return sosfilt(sos, audio).astype(np.float32)


# ─── Stem Generators ─────────────────────────────────────────────────────────

def generate_pad(duration: float) -> np.ndarray:
    """
    Lush ambient pad: layered detuned sines forming a Cm7 chord
    with a slow LFO modulating amplitude, plus filtered noise wash.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False, dtype=np.float64)
    pad = np.zeros_like(t)

    # Cm7 chord: C3, Eb3, G3, Bb3
    chord_freqs = [note_freq('C3'), note_freq('Eb3'), note_freq('G3'), note_freq('Bb3')]

    for i, freq in enumerate(chord_freqs):
        # Slight detuning for richness
        detune = 1.0 + (i - 1.5) * 0.002
        # Main sine
        pad += 0.15 * np.sin(2 * np.pi * freq * detune * t)
        # Octave above, quieter
        pad += 0.07 * np.sin(2 * np.pi * freq * 2.0 * detune * t)

    # Slow amplitude LFO (breathing)
    lfo = 0.7 + 0.3 * np.sin(2 * np.pi * 0.15 * t)
    pad *= lfo

    # Add filtered noise wash for texture
    noise = np.random.randn(len(t)).astype(np.float64) * 0.03
    noise_filtered = lowpass(noise.astype(np.float32), 800.0)
    pad += noise_filtered

    pad = normalize(pad.astype(np.float32), peak=0.7)
    return fade_in_out(pad)


def generate_bass(duration: float) -> np.ndarray:
    """
    Deep sub-bass pattern: C2 → G2 → Ab2 → F2, repeating every 4 bars.
    Each note is a sine wave with a subtle pluck envelope.
    """
    bpm = 100
    beat_duration = 60.0 / bpm
    bar_duration = beat_duration * 4
    total_samples = int(SAMPLE_RATE * duration)
    bass = np.zeros(total_samples, dtype=np.float64)

    # 4-bar pattern, each note lasts 1 bar
    pattern = ['C2', 'C2', 'G2', 'G2', 'Ab2', 'Ab2', 'F2', 'F2']
    note_duration = bar_duration / 2  # Each note is half a bar (2 beats)

    pattern_duration = len(pattern) * note_duration
    num_repeats = int(np.ceil(duration / pattern_duration))

    sample_pos = 0
    for _ in range(num_repeats):
        for note_name in pattern:
            freq = note_freq(note_name)
            note_samples = int(SAMPLE_RATE * note_duration)

            if sample_pos + note_samples > total_samples:
                note_samples = total_samples - sample_pos
            if note_samples <= 0:
                break

            t_note = np.arange(note_samples, dtype=np.float64) / SAMPLE_RATE

            # Sine sub-bass with soft pluck envelope
            envelope = np.exp(-t_note * 1.5)  # Gentle decay
            envelope = np.clip(envelope, 0.3, 1.0)  # Don't decay too much

            note_audio = 0.6 * np.sin(2 * np.pi * freq * t_note) * envelope
            # Add a faint second harmonic for presence
            note_audio += 0.1 * np.sin(2 * np.pi * freq * 2 * t_note) * envelope

            bass[sample_pos:sample_pos + note_samples] += note_audio[:note_samples]
            sample_pos += note_samples

    bass = lowpass(bass.astype(np.float32), 250.0)  # Deep low-pass
    bass = normalize(bass, peak=0.75)
    return fade_in_out(bass)


def generate_drums(duration: float) -> np.ndarray:
    """
    Simple drum pattern at 100 BPM:
    - Kick on beats 1 and 3
    - Snare on beats 2 and 4
    - Hi-hat on every 8th note
    """
    bpm = 100
    beat_duration = 60.0 / bpm
    total_samples = int(SAMPLE_RATE * duration)
    drums = np.zeros(total_samples, dtype=np.float64)

    eighth_note = beat_duration / 2
    num_eighths = int(duration / eighth_note)

    for i in range(num_eighths):
        pos = int(i * eighth_note * SAMPLE_RATE)
        beat_in_bar = (i % 8)  # 0–7 (eighth notes per bar)

        # ── Kick: beats 1 and 3 (eighth positions 0 and 4) ───────────────
        if beat_in_bar in (0, 4):
            kick_len = int(0.15 * SAMPLE_RATE)
            t_kick = np.arange(kick_len, dtype=np.float64) / SAMPLE_RATE
            # Pitch-dropping sine (starts at 150 Hz, drops to 50 Hz)
            freq_sweep = 150.0 * np.exp(-t_kick * 20.0) + 50.0
            phase = np.cumsum(freq_sweep) / SAMPLE_RATE
            kick = 0.8 * np.sin(2 * np.pi * phase) * np.exp(-t_kick * 15.0)

            end = min(pos + kick_len, total_samples)
            drums[pos:end] += kick[:end - pos]

        # ── Snare: beats 2 and 4 (eighth positions 2 and 6) ──────────────
        if beat_in_bar in (2, 6):
            snare_len = int(0.12 * SAMPLE_RATE)
            t_snare = np.arange(snare_len, dtype=np.float64) / SAMPLE_RATE
            # Noise burst + low sine body
            noise = np.random.randn(snare_len) * 0.35 * np.exp(-t_snare * 20.0)
            body = 0.25 * np.sin(2 * np.pi * 200.0 * t_snare) * np.exp(-t_snare * 30.0)
            snare = noise + body

            end = min(pos + snare_len, total_samples)
            drums[pos:end] += snare[:end - pos]

        # ── Hi-hat: every eighth note ─────────────────────────────────────
        hat_len = int(0.04 * SAMPLE_RATE)
        t_hat = np.arange(hat_len, dtype=np.float64) / SAMPLE_RATE
        hat = np.random.randn(hat_len) * 0.12 * np.exp(-t_hat * 80.0)
        # High-pass by just using high-freq noise (already mostly high freq)

        end = min(pos + hat_len, total_samples)
        drums[pos:end] += hat[:end - pos]

    drums = normalize(drums.astype(np.float32), peak=0.75)
    return fade_in_out(drums)


def generate_melody(duration: float) -> np.ndarray:
    """
    Ethereal pentatonic melody in C minor, higher register.
    Simple sine waves with vibrato and an ADSR-ish envelope.
    """
    bpm = 100
    beat_duration = 60.0 / bpm
    total_samples = int(SAMPLE_RATE * duration)
    melody = np.zeros(total_samples, dtype=np.float64)

    # C minor pentatonic: C, Eb, F, G, Bb (octave 4 and 5)
    scale = ['C5', 'Eb5', 'F5', 'G5', 'Bb5', 'C6', 'Bb5', 'G5',
             'F5', 'Eb5', 'C5', 'Eb5', 'G5', 'F5', 'Eb5', 'C5']

    # Each note is 1 beat, with some rests
    note_pattern = []
    for i, note in enumerate(scale):
        note_pattern.append(note)
        # Insert occasional rests (None)
        if i % 4 == 3:
            note_pattern.append(None)

    pattern_duration = len(note_pattern) * beat_duration
    num_repeats = int(np.ceil(duration / pattern_duration))

    sample_pos = 0
    for _ in range(num_repeats):
        for note_name in note_pattern:
            note_samples = int(SAMPLE_RATE * beat_duration)

            if sample_pos + note_samples > total_samples:
                note_samples = total_samples - sample_pos
            if note_samples <= 0:
                break

            if note_name is None:
                # Rest
                sample_pos += note_samples
                continue

            freq = note_freq(note_name)
            t_note = np.arange(note_samples, dtype=np.float64) / SAMPLE_RATE

            # Vibrato (subtle pitch modulation)
            vibrato_rate = 5.0  # Hz
            vibrato_depth = 3.0  # Hz deviation
            freq_modulated = freq + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t_note)

            # Generate with phase accumulation for smooth vibrato
            phase = np.cumsum(freq_modulated) / SAMPLE_RATE
            tone = 0.35 * np.sin(2 * np.pi * phase)

            # ADSR-ish envelope: fast attack, gentle decay, sustain, release
            attack = int(0.02 * SAMPLE_RATE)
            release = int(0.08 * SAMPLE_RATE)
            envelope = np.ones(note_samples, dtype=np.float64)
            if attack < note_samples:
                envelope[:attack] = np.linspace(0, 1, attack)
            if release < note_samples:
                envelope[-release:] = np.linspace(1, 0, release)
            # Slight decay after attack
            decay_start = min(attack, note_samples)
            decay_env = np.exp(-np.arange(note_samples - decay_start, dtype=np.float64)
                               / (SAMPLE_RATE * 0.8))
            envelope[decay_start:] *= decay_env * 0.7 + 0.3

            tone *= envelope

            melody[sample_pos:sample_pos + note_samples] += tone[:note_samples]
            sample_pos += note_samples

    # Soften with a low-pass
    melody = lowpass(melody.astype(np.float32), 4000.0)
    melody = normalize(melody, peak=0.6)
    return fade_in_out(melody)


# ─── Main ─────────────────────────────────────────────────────────────────────

def generate_all_stems():
    """Generate all 4 placeholder stems and save to the stems directory."""
    # Resolve stems directory relative to project root
    project_root = os.path.dirname(os.path.dirname(__file__))
    stems_path = os.path.join(project_root, STEMS_DIR)
    os.makedirs(stems_path, exist_ok=True)

    duration = float(STEM_DURATION_SEC)

    generators = {
        "pad": generate_pad,
        "bass": generate_bass,
        "drums": generate_drums,
        "melody": generate_melody,
    }

    print(f"\n🎵 Generating {len(generators)} stems ({duration:.0f}s each, "
          f"{SAMPLE_RATE} Hz, mono)...\n")

    for name, gen_func in generators.items():
        print(f"  ⏳ Generating {name}...", end=" ", flush=True)
        audio = gen_func(duration)
        filepath = os.path.join(stems_path, f"{name}.wav")

        # Save as 16-bit PCM WAV
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(filepath, SAMPLE_RATE, audio_int16)

        file_size_kb = os.path.getsize(filepath) / 1024
        print(f"✓ ({file_size_kb:.0f} KB)")

    print(f"\n✅ All stems saved to '{stems_path}/'")
    print("   You can replace these with real stems — just keep the same filenames.\n")


if __name__ == "__main__":
    generate_all_stems()
