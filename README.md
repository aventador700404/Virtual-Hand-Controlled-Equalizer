# 🎛️ Virtual Hand-Controlled Equalizer

**DJ with your hands and a webcam.** Drop in any song — AI splits it into stems — then mix it live: fingers layer the instruments, a pinch grabs the filter and echo, and on-screen faders give you a 3-band DJ EQ with kills.

> Based on [optimusbuilder/Ambient-Conductor](https://github.com/optimusbuilder/Ambient-Conductor), extended with automatic Demucs stem splitting, two-hand control, a Linkwitz-Riley DJ EQ, pinch-to-move / release-to-hold FX, a movable EQ panel and a one-click Windows launcher.
> The original project's demo stems and background image are not included here.

---

## How It Works

```
┌──────────────────┐     Normalized Coordinates     ┌─────────────────────┐
│   Vision Loop    │ ────────────────────────────▶  │    Audio Engine     │
│ (MediaPipe Hands)│   R fingers → Stem Mixer      │ (sounddevice +      │
│                  │   R pinch X → Filter Cutoff   │  scipy DSP)         │
│                  │   R pinch Y → Delay/Echo      │                     │
│                  │   L pinch   → 3-band EQ       │                     │
└──────────────────┘                                └─────────────────────┘
```

### Controls

Two hands: your **right hand mixes**, your **left hand EQs** (the window is mirrored; the EQ panel starts on the left).

| Hand | Gesture / Position | Effect |
|------|------|--------|
| Right (MIX) | **Fist (0 fingers)** | Everything mutes, only the echo tail rings out (the "drop") |
| Right (MIX) | **1 raised finger** | Pad + bass line |
| Right (MIX) | **2 raised fingers** | + Drums |
| Right (MIX) | **3+ raised fingers** | + Melody / vocals |
| Right (MIX) | **Pinch (thumb + index), then move left ↔ right** | DJ filter: low-pass ← bypass → high-pass, measured from where you pinched |
| Right (MIX) | **Pinch, then move up ↕ down** | Delay/echo wet mix (up = more echo) |
| Right (MIX) | **Release the pinch** | Filter and echo stay where you left them (HOLD) |
| Left (EQ) | **Pinch (thumb + index) on a fader, move up/down** | 3-band DJ EQ — LOW (<250 Hz), MID, HIGH (>2.5 kHz), +6 dB to −30 dB |
| Left (EQ) | **Release the pinch** | The fader stays where you left it |
| Left (EQ) | **Drag a fader to the bottom** | Kill that band completely |

The EQ is a Linkwitz-Riley isolator: flat when all faders sit at 0 dB, with 24 dB/oct kills.

Roles follow MediaPipe's hand label: its "Left" (your right hand in the mirrored view) is the mixer, its "Right" (your left hand) is the EQ.

| Mouse / Key | Effect |
|------|--------|
| **Drag the EQ panel** | Move it anywhere (saved to `layout.json`) |
| **Mouse wheel** / **+ −** | Resize the EQ panel · **0** resets its position |
| **R** | Reset filter, echo and EQ to neutral |

> [!NOTE]
> If a hand leaves the frame, the music continues with your last finger count, and the filter, echo and EQ keep their values.

The window shows your mirrored webcam with the HUD on top. **B** switches to an optional background image if one is present.

Press **Q** or **Esc** to quit.

---

## Quick Start

### Windows (one click)

1. Install Python 3.10+ from python.org.
2. Double-click **`run.bat`**. The first run sets everything up (virtual environment, packages, PyTorch + Demucs, models) — later runs start right away.
3. Put songs (`.mp3`, `.wav`, `.flac`, `.ogg`) in `songs/` and pick one from the menu, or drag a song onto `run.bat`.

With no song, synthetic placeholder stems are generated in `stems/`.

### Manual (any OS)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
curl -L -o models/hand_landmarker.task --create-dirs \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
python src/main.py --choose       # or: python src/main.py --song "path/to/song.mp3"
```

---

## Custom Stems

To use your own audio, export 4 separate `.wav` files from your song:

1. **`stems/pad.wav`** — Ambient pad / synth layer (always playing)
2. **`stems/bass.wav`** — Bass line
3. **`stems/drums.wav`** — Drum track
4. **`stems/melody.wav`** — Vocals / lead melody

**Requirements:**
- All stems must be the same length (start at the same point)
- WAV format, 44100 Hz sample rate
- Mono or stereo (stereo will be mixed to mono)

---

## Play Your Own Song (MP3)

Drop any song in and it is split into the 4 stems automatically with [Demucs](https://github.com/adefossez/demucs) — once per song, then cached in `stems/cache/`.

**Windows:** drag an `.mp3` onto `run.bat`, or put songs in `songs/` and double-click `run.bat` to pick one. The first run installs everything once (PyTorch with CUDA if an NVIDIA GPU is found, otherwise CPU) and downloads the models; later runs skip setup and start right away. In the song menu, press **a** to pre-split every song in `songs/` in one go.

**Any OS:**

```bash
pip install torch "demucs>=4.1"
python src/main.py --song "path/to/song.mp3"   # or --choose to pick from songs/
python src/song_prep.py --all                  # pre-split everything in songs/
```

Stem mapping: `other → pad`, `bass → bass`, `drums → drums`, `vocals → melody`. Use `--split-model htdemucs_ft` for cleaner stems (about 4× slower).

---

## CLI Options

```bash
python src/main.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--stems-dir` | `stems` | Path to stems directory |
| `--camera` | `0` | Camera index (try `1` for external webcam) |
| `--no-audio` | off | Run vision-only mode (no sound) |
| `--song` | — | Song file to split into stems and play |
| `--choose` | off | Pick a song from `songs/` in a menu |
| `--split-model` | `htdemucs` | Demucs model used for splitting |
| `--bg` | off | Start with the optional background image instead of the webcam |

---

## Project Structure

```
├── requirements.txt        # Python dependencies
├── stems/                  # Audio stem .wav files
│   ├── pad.wav
│   ├── bass.wav
│   ├── drums.wav
│   └── melody.wav
└── src/
    ├── main.py             # Entry point
    ├── vision.py           # OpenCV + MediaPipe hand tracking
    ├── gestures.py         # Finger counting + gesture mapping
    ├── audio_engine.py     # Real-time stem mixing via sounddevice
    ├── dsp.py              # Butterworth filter + feedback delay
    ├── generate_stems.py   # Synthetic stem generator utility
    └── config.py           # Central configuration constants
```

---

## Tech Stack

- **[MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)** — 21-landmark hand tracking at 30+ FPS
- **[OpenCV](https://opencv.org/)** — Webcam capture and frame display
- **[sounddevice](https://python-sounddevice.readthedocs.io/)** — Low-latency callback-based audio I/O (PortAudio)
- **[SciPy](https://scipy.org/)** — Butterworth filter design and real-time IIR filtering
- **[NumPy](https://numpy.org/)** — Array math for audio and coordinate processing
