"""
EchoVox — Dataset Balancer
======================================
Usage:
    python balance_dataset.py
    python balance_dataset.py --fake_dir data/fake --target_legacy 15000 --target_modern_pct 0.20
"""

import os
import shutil
import random
import argparse
from pathlib import Path

random.seed(42)


def main(fake_dir: str, target_legacy: int, target_modern_pct: float):
    fake_path = Path(fake_dir)

    # ── Separate legacy (WAV = ASVspoof) from modern (MP3 = ElevenLabs/TTS) ──
  
    all_files  = list(fake_path.iterdir())
    legacy     = [f for f in all_files if f.suffix.lower() == ".wav"]
    modern     = [f for f in all_files if f.suffix.lower() == ".mp3"]

    print(f"\n{'='*55}")
    print(f"  EchoVox — Dataset Balancer")
    print(f"{'='*55}")
    print(f"\n  Found {len(legacy):,} legacy WAV fakes (ASVspoof)")
    print(f"  Found {len(modern):,} modern MP3 fakes (ElevenLabs/TTS)")

    if not modern:
        print("\n  [ERROR] No MP3 files found in fake/")
        print("  Make sure your ElevenLabs/TTS fakes are copied into data/fake/")
        return

    if not legacy:
        print("\n  [ERROR] No WAV files found in fake/")
        return

    # ── Step 1: Calculate targets ──────────────────────────────────────────────

    target_modern = int(target_legacy * target_modern_pct / (1 - target_modern_pct))

    print(f"\n  Target legacy fakes  : {target_legacy:,}")
    print(f"  Target modern fakes  : {target_modern:,}  ({target_modern_pct*100:.0f}% of total)")
    print(f"  Total fakes after    : {target_legacy + target_modern:,}")
    print(f"  Modern % of fakes    : {target_modern/(target_legacy+target_modern)*100:.1f}%")

    # ── Step 2: Downsample legacy WAVs ────────────────────────────────────────
  
    backup_dir = fake_path.parent / "fake_legacy_backup"
    backup_dir.mkdir(exist_ok=True)

    if len(legacy) > target_legacy:
        to_remove = random.sample(legacy, len(legacy) - target_legacy)
        print(f"\n  Moving {len(to_remove):,} legacy files to fake_legacy_backup/ ...")
        for f in to_remove:
            shutil.move(str(f), str(backup_dir / f.name))
        print(f"  Done. Legacy files remaining: {target_legacy:,}")
        print(f"  (Backup at: {backup_dir}  — safe to restore if needed)")
    else:
        print(f"\n  Legacy count ({len(legacy):,}) already <= target ({target_legacy:,}), skipping downsample")

    # ── Step 3: Oversample modern MP3s via copying ────────────────────────────
  
    if len(modern) >= target_modern:
        print(f"\n  Modern count ({len(modern):,}) already >= target ({target_modern:,}), skipping oversample")
    else:
        copies_needed = target_modern - len(modern)
        print(f"\n  Oversampling {len(modern):,} modern fakes → {target_modern:,} (+{copies_needed:,} copies) ...")

        copy_pool = modern.copy()
        copy_idx  = 0
        added     = 0

        while added < copies_needed:
            src  = copy_pool[copy_idx % len(copy_pool)]
            stem = src.stem
            ext  = src.suffix
            dst  = fake_path / f"{stem}_copy{added:04d}{ext}"
            shutil.copy2(str(src), str(dst))
            added    += 1
            copy_idx += 1

        print(f"  Done. Modern files now: {target_modern:,}")

    # ── Final count ────────────────────────────────────────────────────────────
  
    final_legacy = len(list(fake_path.glob("*.wav")))
    final_modern = len(list(fake_path.glob("*.mp3")))
    total_fake   = final_legacy + final_modern
    total_real   = len(list((fake_path.parent / "real").glob("*.wav")))

    print(f"\n{'='*55}")
    print(f"  Dataset balanced!")
    print(f"  Real files     : {total_real:,}")
    print(f"  Legacy fakes   : {final_legacy:,}")
    print(f"  Modern fakes   : {final_modern:,}")
    print(f"  Total fakes    : {total_fake:,}")
    print(f"  Modern %       : {final_modern/total_fake*100:.1f}%")
    print(f"{'='*55}")
    print(f"\n  Next steps:")
    print(f"  1. del features.npz")
    print(f"  2. python extract_features.py --aug_factor 2 --batch_size 8")
    print(f"  3. python train.py --epochs 30")
    print(f"\n  To restore original ASVspoof files:")
    print(f"  copy {backup_dir}\\*.wav data\\fake\\")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_dir",          default="data/fake",  help="Path to fake audio folder")
    ap.add_argument("--target_legacy",     default=15000, type=int, help="Target ASVspoof WAV count")
    ap.add_argument("--target_modern_pct", default=0.20,  type=float, help="Target modern fake % of total")
    args = ap.parse_args()
    main(args.fake_dir, args.target_legacy, args.target_modern_pct)
