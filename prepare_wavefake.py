"""
EchoVox — WaveFake Dataset Preparation
=======================================
Extracts audio from WaveFake parquet files into data/real/ and data/fake/

WaveFake parquet structure:
  - audio:        dict with 'bytes' key containing raw WAV bytes
  - audio_id:     filename string (e.g. LJ001-0001)
  - real_or_fake: label string ('real' for real, 'WF1'-'WF6' for fake vocoders)

WaveFake vocoder labels:
  WF1 = MelGAN
  WF2 = Full-band MelGAN
  WF3 = Multi-band MelGAN
  WF4 = HiFi-GAN
  WF5 = Parallel WaveGAN
  WF6 = WaveGlow

Run:
  python prepare_wavefake.py --wavefake_dir D:/wavefake/data --max_files 5000
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ── Label mapping ──────────────────────────────────────────────────────────────
FAKE_LABELS = {"WF1", "WF2", "WF3", "WF4", "WF5", "WF6"}
REAL_LABEL  = "real"

# Vocoder names for logging
VOCODER_NAMES = {
    "WF1": "MelGAN",
    "WF2": "Full-band MelGAN",
    "WF3": "Multi-band MelGAN",
    "WF4": "HiFi-GAN",
    "WF5": "Parallel WaveGAN",
    "WF6": "WaveGlow",
}


def extract_wavefake(wavefake_dir: str, output_real: str, output_fake: str,
                     max_real: int, max_fake: int, max_fake_per_vocoder: int):
    """
    Extract WAV files from WaveFake parquet files into output directories.
    """
    wavefake_dir = Path(wavefake_dir)
    real_dir     = Path(output_real)
    fake_dir     = Path(output_fake)

    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(wavefake_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: No parquet files found in {wavefake_dir}")
        print("Make sure --wavefake_dir points to the folder with partition*.parquet files")
        return

    print(f"Found {len(parquet_files)} parquet files in {wavefake_dir}")
    print(f"Target: up to {max_real} real, {max_fake} fake ({max_fake_per_vocoder} per vocoder)")
    print()

    real_count   = 0
    fake_counts  = {k: 0 for k in FAKE_LABELS}
    skipped      = 0
    total_fake   = 0

    for pf in tqdm(parquet_files, desc="Processing partitions"):
        # Stop early if we have enough
        if real_count >= max_real and total_fake >= max_fake:
            break

        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            print(f"  Warning: could not read {pf.name}: {e}")
            continue

        for _, row in df.iterrows():
            label    = str(row["real_or_fake"]).strip()
            audio_id = str(row["audio_id"]).strip()

            # Get raw audio bytes
            audio_field = row["audio"]
            if isinstance(audio_field, dict):
                audio_bytes = audio_field.get("bytes", b"")
            elif isinstance(audio_field, bytes):
                audio_bytes = audio_field
            else:
                skipped += 1
                continue

            if not audio_bytes:
                skipped += 1
                continue

            # ── Real voice ─────────────────────────────────────────────────────
            if label == REAL_LABEL:
                if real_count >= max_real:
                    continue
                out_path = real_dir / f"wavefake_real_{audio_id}.wav"
                if out_path.exists():
                    real_count += 1
                    continue
                try:
                    out_path.write_bytes(audio_bytes)
                    real_count += 1
                except Exception as e:
                    print(f"  Warning: could not write {out_path.name}: {e}")

            # ── Fake voice ─────────────────────────────────────────────────────
            elif label in FAKE_LABELS:
                if fake_counts[label] >= max_fake_per_vocoder:
                    continue
                if total_fake >= max_fake:
                    continue
                vocoder  = VOCODER_NAMES.get(label, label)
                out_path = fake_dir / f"wavefake_{label}_{audio_id}.wav"
                if out_path.exists():
                    fake_counts[label] += 1
                    total_fake += 1
                    continue
                try:
                    out_path.write_bytes(audio_bytes)
                    fake_counts[label] += 1
                    total_fake += 1
                except Exception as e:
                    print(f"  Warning: could not write {out_path.name}: {e}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  WaveFake Extraction Complete")
    print("=" * 55)
    print(f"  Real voices extracted  : {real_count:,}")
    print(f"  Fake voices extracted  : {total_fake:,}")
    print()
    print("  Fake breakdown by vocoder:")
    for label, count in fake_counts.items():
        if count > 0:
            print(f"    {VOCODER_NAMES[label]:<22} ({label}): {count:,}")
    if skipped:
        print(f"\n  Skipped (bad data)     : {skipped:,}")
    print()
    print(f"  Real files → {real_dir}")
    print(f"  Fake files → {fake_dir}")
    print()
    print("  Next steps:")
    print("  1. python balance_dataset.py")
    print("  2. del features.npz  (or rm features.npz on Linux)")
    print("  3. python extract_features.py --aug_factor 2 --batch_size 8")
    print("  4. python train.py --epochs 30")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Prepare WaveFake dataset for EchoVox")
    parser.add_argument(
        "--wavefake_dir", type=str, default=r"D:\wavefake\data",
        help="Path to WaveFake parquet files (default: D:/wavefake/data)"
    )
    parser.add_argument(
        "--real_dir", type=str, default="data/real",
        help="Output directory for real voices (default: data/real)"
    )
    parser.add_argument(
        "--fake_dir", type=str, default="data/fake",
        help="Output directory for fake voices (default: data/fake)"
    )
    parser.add_argument(
        "--max_real", type=int, default=5000,
        help="Max real voice files to extract (default: 5000)"
    )
    parser.add_argument(
        "--max_fake", type=int, default=15000,
        help="Max total fake files to extract (default: 15000)"
    )
    parser.add_argument(
        "--max_fake_per_vocoder", type=int, default=2500,
        help="Max fake files per vocoder (default: 2500, 6 vocoders = 15000)"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  EchoVox — WaveFake Dataset Preparation")
    print("=" * 55)
    print(f"  Source  : {args.wavefake_dir}")
    print(f"  Real →  : {args.real_dir}")
    print(f"  Fake →  : {args.fake_dir}")
    print("=" * 55)
    print()

    extract_wavefake(
        wavefake_dir         = args.wavefake_dir,
        output_real          = args.real_dir,
        output_fake          = args.fake_dir,
        max_real             = args.max_real,
        max_fake             = args.max_fake,
        max_fake_per_vocoder = args.max_fake_per_vocoder,
    )


if __name__ == "__main__":
    main()
