"""
Ambient Conductor — Song Preparation
======================================
Turns any song file (MP3, WAV, FLAC, OGG...) into the 4 stems the audio
engine plays, using Meta's Demucs (htdemucs) through its Python API.

  • Each song is split once; the stems are cached in stems/cache/<song>_<id>/
    so the next run starts instantly. The menu (or `--all`) can split every
    song in songs/ ahead of time, loading the model only once.
  • Uses the GPU automatically when PyTorch can see one (CUDA), else the CPU.
  • Reads audio with Demucs' own decoder (sphn), so no FFmpeg / torchcodec
    install is needed on Windows.

Command line (run from the project folder):
    python src/song_prep.py --all              # pre-split everything in songs/
    python src/song_prep.py --download-model   # just fetch the Demucs model

Demucs source → Ambient Conductor stem (same mapping as split_song.py):
    other  → pad      (always on)
    bass   → bass     (1 finger)
    drums  → drums    (2 fingers)
    vocals → melody   (3+ fingers)
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from config import SAMPLE_RATE, STEM_NAMES

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
DEMUCS_TO_STEM = {"other": "pad", "bass": "bass", "drums": "drums", "vocals": "melody"}
DEFAULT_MODEL = "htdemucs"
INFO_FILE = "info.json"  # written last → marks a complete cache entry


# ─── Paths ────────────────────────────────────────────────────────────────────

def songs_dir(project_root) -> Path:
    """Drop-in folder for songs: <project>/songs/"""
    return Path(project_root) / "songs"


def cache_dir_for(song: Path, project_root, model: str = DEFAULT_MODEL) -> Path:
    """Stable per-song cache folder. The id changes with the file size and the
    split model, so a replaced song or a different model gets a fresh split."""
    key = hashlib.sha1(f"{song.name}|{song.stat().st_size}|{model}".encode("utf-8")).hexdigest()[:8]
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", song.stem).strip(" .")[:60] or "song"
    return Path(project_root) / "stems" / "cache" / f"{safe}_{key}"


def is_cached(out_dir: Path) -> bool:
    return (out_dir / INFO_FILE).exists() and all(
        (out_dir / f"{name}.wav").exists() for name in STEM_NAMES
    )


# ─── Song picker ──────────────────────────────────────────────────────────────

def _list_songs(project_root) -> list:
    """Songs in the songs/ folder, newest first."""
    folder = songs_dir(project_root)
    folder.mkdir(exist_ok=True)
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def choose_song(project_root, model: str = DEFAULT_MODEL) -> Optional[Path]:
    """
    Console menu over the songs/ folder (newest first).
    Returns the chosen song path, or None to play the default stems.
    Option "a" pre-splits every song that isn't ready yet, then shows the menu again.
    """
    while True:
        songs = _list_songs(project_root)
        if not songs:
            print(f"\n🎵 songs 폴더가 비어 있어요 → {songs_dir(project_root)}")
            print("   여기에 mp3를 넣거나, mp3 파일을 run.bat 위로 끌어다 놓으면 그 곡으로 시작해요.")
            print("   지금은 기본 음악으로 시작할게요.\n")
            return None

        ready = [is_cached(cache_dir_for(s, project_root, model)) for s in songs]
        n_todo = ready.count(False)

        print("\n🎵 어떤 곡으로 할까요?  (Enter = 1번)\n")
        print("   0) 기본 음악")
        for i, (song, ok) in enumerate(zip(songs, ready), 1):
            print(f"   {i}) {song.name}{'  ✓ 준비됨' if ok else ''}")
        if n_todo:
            print(f"   a) 아직 준비 안 된 {n_todo}곡 전부 미리 분리해 두기")
        print()

        while True:
            try:
                raw = input("   번호 입력 > ").strip().lower()
            except EOFError:  # no console (e.g. piped) → newest song
                raw = ""
            if raw == "":
                return songs[0]
            if raw == "a" and n_todo:
                prepare_all(project_root, model)
                break  # show the menu again, now with everything ready
            if raw.isdigit() and 0 <= int(raw) <= len(songs):
                return None if raw == "0" else songs[int(raw) - 1]
            print(f"   0~{len(songs)} 사이 숫자{' 또는 a' if n_todo else ''}를 입력해 주세요.")


# ─── Splitting ────────────────────────────────────────────────────────────────

class SplitError(Exception):
    """A user-facing problem while splitting (message is ready to print)."""


class _Splitter:
    """Loads the Demucs model once and splits any number of songs with it."""

    def __init__(self, model: str = DEFAULT_MODEL):
        # Quiet the HuggingFace "symlinks not supported on Windows" warning
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            import torch
            from demucs.api import Separator, LoadAudioError
        except ImportError:
            raise SplitError(
                "❌ 곡을 분리하려면 PyTorch와 Demucs가 필요해요.\n"
                "   run.bat으로 실행하면 자동으로 설치되고, 직접 설치하려면:\n"
                '   pip install torch "demucs>=4.1"'
            )
        self._load_error = LoadAudioError
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        where = f"GPU ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else "CPU"
        print(f"\n🤖 분리 모델 준비 중 — {model}, {where}")
        print("   (모델은 처음 한 번만 ~80MB 내려받고, 이후엔 저장된 걸 써요)")
        try:
            self._sep = Separator(model=model, device=self.device, progress=True)
        except Exception as err:
            raise SplitError(
                f"❌ 분리 모델({model})을 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n   {err!r}"
            )

    def split(self, song: Path, out_dir: Path):
        print(f"\n✂  '{song.name}' → 4개 스템 (CPU면 곡당 1~3분, GPU면 몇 초)")
        started = time.time()
        try:
            _, sources = self._sep.separate_audio_file(song)
        except self._load_error as err:
            raise SplitError(
                f"❌ 이 파일은 읽을 수 없어요: {song.name}\n{err}\n"
                "   mp3나 wav로 변환해서 다시 넣어 주세요."
            )
        sr = self._sep.samplerate

        # Mono (the engine plays mono), resampled to the engine's rate if needed
        stems = {}
        for src, dst in DEMUCS_TO_STEM.items():
            mono = sources[src].mean(dim=0).cpu().numpy().astype(np.float32)
            if sr != SAMPLE_RATE:
                from scipy.signal import resample_poly
                g = np.gcd(int(sr), SAMPLE_RATE)
                mono = resample_poly(mono, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
            stems[dst] = mono

        # One shared gain so no stem clips in 16-bit, keeping their relative balance
        peak = max(float(np.abs(s).max()) for s in stems.values())
        scale = 0.99 / peak if peak > 0.99 else 1.0

        out_dir.mkdir(parents=True, exist_ok=True)
        for name, data in stems.items():
            sf.write(str(out_dir / f"{name}.wav"), data * scale, SAMPLE_RATE, subtype="PCM_16")

        info = {
            "source": str(song),
            "model": self.model,
            "device": self.device,
            "seconds": round(len(stems["pad"]) / SAMPLE_RATE, 1),
            "split_time_sec": round(time.time() - started, 1),
        }
        (out_dir / INFO_FILE).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ 완료 ({info['split_time_sec']}초) → {out_dir}")


def prepare_song(song, project_root, model: str = DEFAULT_MODEL) -> Path:
    """Return a stems directory for `song`, splitting it with Demucs if needed."""
    song = Path(song).expanduser().resolve()
    if not song.is_file():
        sys.exit(f"❌ 파일을 찾을 수 없어요: {song}")

    out_dir = cache_dir_for(song, project_root, model)
    if is_cached(out_dir):
        print(f"🎵 '{song.name}' — 이미 분리해 둔 스템을 사용해요.")
        return out_dir

    try:
        _Splitter(model).split(song, out_dir)
    except SplitError as err:
        sys.exit(str(err))
    return out_dir


def prepare_all(project_root, model: str = DEFAULT_MODEL) -> int:
    """Split every song in songs/ that isn't cached yet (model loaded once).
    Unreadable files are skipped. Returns how many songs were split."""
    songs = _list_songs(project_root)
    todo = [s for s in songs if not is_cached(cache_dir_for(s, project_root, model))]
    if not todo:
        print(f"\n✅ songs 폴더의 {len(songs)}곡은 전부 준비돼 있어요.")
        return 0

    print(f"\n🎵 {len(todo)}곡을 미리 분리해 둘게요. 한 번 해두면 다음부터는 바로 시작돼요.")
    try:
        splitter = _Splitter(model)
    except SplitError as err:
        print(err)
        return 0

    done = 0
    for i, song in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}]", end="")
        try:
            splitter.split(song, cache_dir_for(song, project_root, model))
            done += 1
        except SplitError as err:
            print(err)
    print(f"\n🎉 {done}/{len(todo)}곡 준비 완료!")
    return done


def download_model(model: str = DEFAULT_MODEL) -> bool:
    """Fetch the Demucs model ahead of time so the first split doesn't wait."""
    try:
        _Splitter(model)
    except SplitError as err:
        print(err)
        return False
    print("✅ 분리 모델 준비 완료")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare songs for Ambient Conductor")
    parser.add_argument("--all", action="store_true",
                        help="Split every song in songs/ that isn't cached yet")
    parser.add_argument("--download-model", action="store_true",
                        help="Only download the Demucs model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Demucs model name")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    ok = True
    if args.download_model:
        ok = download_model(args.model)
    if args.all:
        prepare_all(root, args.model)
    if not (args.download_model or args.all):
        parser.print_help()
    sys.exit(0 if ok else 1)
