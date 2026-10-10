"""
Ambient Conductor — AI Stem Splitting Utility
===============================================
Splits any input song file (MP3, WAV, FLAC, etc.) into 4 stems using Meta's Demucs.
Converts the output stems to mono 44100Hz WAV format and saves them to the stems/ folder.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import soundfile as sf

def main():
    parser = argparse.ArgumentParser(description="AI Stem Splitting for Ambient Conductor")
    parser.add_argument("file", help="Path to the audio file to split (MP3, WAV, etc.)")
    parser.add_argument("--output-dir", default="stems", help="Where to output the processed stems")
    args = parser.parse_args()

    input_file = os.path.abspath(args.file)
    if not os.path.exists(input_file):
        print(f"❌ Error: File not found: {input_file}")
        sys.exit(1)

    print(f"🎵 Song to split: {input_file}")

    # Create temporary directory for Demucs output
    temp_dir = tempfile.mkdtemp()
    print(f"📂 Created temporary directory: {temp_dir}")

    # Formulate Demucs command
    # Using sys.executable to ensure we run demucs in the same virtualenv python process
    cmd = [
        sys.executable,
        "-m", "demucs",
        "-o", temp_dir,
        input_file
    ]

    print("\n🤖 Starting AI stem splitting (Demucs)...")
    print("👉 Note: The first run will download Meta's pre-trained model (approx. 100 MB).")
    print("👉 This might take 1-3 minutes depending on your Mac's CPU/GPU performance.")
    print(f"Running command: {' '.join(cmd)}\n")

    try:
        # Run Demucs
        subprocess.run(cmd, check=True)
        print("\n✅ Demucs processing complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running Demucs: {e}")
        shutil.rmtree(temp_dir)
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Error: python-demucs not found in current environment. Run pip install -r requirements.txt first.")
        shutil.rmtree(temp_dir)
        sys.exit(1)

    # Locate output files
    # Demucs structure: {temp_dir}/htdemucs/{song_name}/{stem_name}.wav
    htdemucs_dir = os.path.join(temp_dir, "htdemucs")
    if not os.path.exists(htdemucs_dir):
        print("❌ Error: Demucs output directory not found.")
        shutil.rmtree(temp_dir)
        sys.exit(1)

    subdirs = [d for d in os.listdir(htdemucs_dir) if os.path.isdir(os.path.join(htdemucs_dir, d))]
    if not subdirs:
        print("❌ Error: No processed song folder found.")
        shutil.rmtree(temp_dir)
        sys.exit(1)

    song_dir = os.path.join(htdemucs_dir, subdirs[0])
    print(f"📂 Located Demucs output in: {song_dir}")

    # Mapping of Demucs outputs to Ambient Conductor stems
    mapping = {
        "other.wav": "pad.wav",
        "bass.wav": "bass.wav",
        "drums.wav": "drums.wav",
        "vocals.wav": "melody.wav"
    }

    # Ensure output stems directory exists (resolve relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_stems_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_stems_dir, exist_ok=True)

    print("\n🎙 Processing stems (converting to mono)...")

    for src_name, dest_name in mapping.items():
        src_path = os.path.join(song_dir, src_name)
        dest_path = os.path.join(output_stems_dir, dest_name)

        if not os.path.exists(src_path):
            print(f"⚠ Warning: Expected stem '{src_name}' was not generated.")
            continue

        # Load file using soundfile
        data, sr = sf.read(src_path, dtype='float32')

        # Convert to mono if stereo
        if data.ndim == 2:
            data = data.mean(axis=1)

        # Write out to target path
        sf.write(dest_path, data, sr, format='WAV', subtype='PCM_16')
        print(f"  ✓ Processed and saved: {dest_name} ({sr} Hz, mono)")

    # Normalize stem lengths — ensure all 4 are exactly the same length
    print("\n📏 Normalizing stem lengths...")
    stem_lengths = {}
    for dest_name in mapping.values():
        dest_path = os.path.join(output_stems_dir, dest_name)
        if os.path.exists(dest_path):
            data, sr = sf.read(dest_path, dtype='float32')
            stem_lengths[dest_name] = len(data)

    if stem_lengths:
        min_len = min(stem_lengths.values())
        for dest_name, length in stem_lengths.items():
            if length > min_len:
                dest_path = os.path.join(output_stems_dir, dest_name)
                data, sr = sf.read(dest_path, dtype='float32')
                data = data[:min_len]
                sf.write(dest_path, data, sr, format='WAV', subtype='PCM_16')
                print(f"  ✓ Trimmed {dest_name}: {length} → {min_len} samples")
        print(f"  ✓ All stems normalized to {min_len} samples ({min_len / 44100:.1f}s)")

    # Cleanup temporary directory
    print("\n🧹 Cleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("\n🎉 Success! Stems are ready. Run `python src/main.py` to start conducting!")

if __name__ == "__main__":
    main()
