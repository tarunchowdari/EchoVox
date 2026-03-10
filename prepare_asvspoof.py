"""
VeriVoice Sentinel - ASVspoof 2019 LA Dataset Preparer
Reads the official protocol files, converts FLAC → WAV, and organizes
files into data/real/ and data/fake/ for extract_features.py.

Usage:
    # If ASVspoof is in the default location:
    python prepare_asvspoof.py

    # Custom paths:
    python prepare_asvspoof.py \
        --asv_root  /path/to/ASVspoof2019_LA \
        --data_dir  data \
        --splits    train dev \
        --max_fake  20000

Directory structure expected:
    ASVspoof2019_LA/
    ├── ASVspoof2019_LA_train/flac/
    ├── ASVspoof2019_LA_dev/flac/
    ├── ASVspoof2019_LA_eval/flac/
    └── ASVspoof2019_LA_cm_protocols/
            ASVspoof2019.LA.cm.train.trn.txt
            ASVspoof2019.LA.cm.dev.trl.txt
            ASVspoof2019.LA.cm.eval.trl.txt

Protocol file columns:
    SPEAKER_ID  FILE_ID  -  SYSTEM_ID  LABEL
    where LABEL is 'bonafide' (real) or 'spoof' (fake)
"""

import os
import argparse
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

TARGET_SR = 16_000

# Map split name → (flac subfolder, protocol filename)
SPLIT_MAP = {
    "train": ("ASVspoof2019_LA_train/flac", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev":   ("ASVspoof2019_LA_dev/flac",   "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval":  ("ASVspoof2019_LA_eval/flac",  "ASVspoof2019.LA.cm.eval.trl.txt"),
}


def parse_protocol(protocol_path: Path) -> dict[str, str]:
    """Return {file_id: 'bonafide'|'spoof'} from an ASVspoof protocol file."""
    labels = {}
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id = parts[1]
            label   = parts[4]          # 'bonafide' or 'spoof'
            labels[file_id] = label
    return labels


def convert_flac_to_wav(src: Path, dst: Path) -> bool:
    """Convert one FLAC file to 16kHz mono WAV using torchaudio (fast, GPU-friendly)."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(src))
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        waveform = waveform.mean(dim=0, keepdim=True)   # mono
        torchaudio.save(str(dst), waveform, TARGET_SR)
        return True
    except Exception as e:
        print(f"  [WARN] Could not convert {src.name}: {e}")
        return False


def prepare(asv_root: str, data_dir: str, splits: list[str],
            max_real: int, max_fake: int, convert: bool):

    asv_path  = Path(asv_root)
    proto_dir = asv_path / "ASVspoof2019_LA_cm_protocols"
    real_dir  = Path(data_dir) / "real"
    fake_dir  = Path(data_dir) / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    total_real = total_fake = 0
    skipped = 0

    for split in splits:
        if split not in SPLIT_MAP:
            print(f"  [WARN] Unknown split '{split}', skipping")
            continue

        flac_subdir, proto_name = SPLIT_MAP[split]
        flac_dir   = asv_path / flac_subdir
        proto_path = proto_dir / proto_name

        if not proto_path.exists():
            print(f"  [ERROR] Protocol file not found: {proto_path}")
            continue
        if not flac_dir.exists():
            print(f"  [ERROR] FLAC directory not found: {flac_dir}")
            continue

        print(f"\nProcessing split: {split}")
        labels = parse_protocol(proto_path)
        print(f"  Protocol entries: {len(labels):,}")

        real_count = sum(1 for v in labels.values() if v == "bonafide")
        fake_count = sum(1 for v in labels.values() if v == "spoof")
        print(f"  Bonafide (real): {real_count:,}  |  Spoof (fake): {fake_count:,}")

        # Balance cap per split
        split_max_real = max_real // len(splits) if max_real else None
        split_max_fake = max_fake // len(splits) if max_fake else None
        added_real = added_fake = 0

        files = sorted(labels.items())
        pbar = tqdm(files, unit="file", dynamic_ncols=True)

        for file_id, label in pbar:
            is_real = (label == "bonafide")

            # Cap check
            if is_real and split_max_real and added_real >= split_max_real:
                continue
            if not is_real and split_max_fake and added_fake >= split_max_fake:
                continue

            src_flac = flac_dir / f"{file_id}.flac"
            if not src_flac.exists():
                # Some ASVspoof downloads use .wav already
                src_wav = flac_dir / f"{file_id}.wav"
                if src_wav.exists():
                    src_flac = src_wav
                else:
                    skipped += 1
                    continue

            dst_dir  = real_dir if is_real else fake_dir
            dst_name = f"{split}_{file_id}.wav"
            dst_path = dst_dir / dst_name

            if dst_path.exists():
                # Already converted
                pass
            elif src_flac.suffix == ".wav" and not convert:
                shutil.copy2(src_flac, dst_path)
            else:
                if not convert_flac_to_wav(src_flac, dst_path):
                    skipped += 1
                    continue

            if is_real:
                added_real += 1
                total_real += 1
            else:
                added_fake += 1
                total_fake += 1

            pbar.set_postfix(real=added_real, fake=added_fake)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Dataset preparation complete!")
    print(f"  Real files  → {real_dir}  ({total_real:,})")
    print(f"  Fake files  → {fake_dir}  ({total_fake:,})")
    print(f"  Skipped     : {skipped:,}")
    print(f"{'='*55}")

    if total_real == 0 or total_fake == 0:
        print("\n  [ERROR] No files were prepared. Check --asv_root path.")
        return

    ratio = total_fake / max(total_real, 1)
    if ratio > 5:
        print(f"\n  [NOTE] Class imbalance: {ratio:.1f}x more fake than real.")
        print(f"         Consider using --max_fake {total_real * 4} to reduce imbalance.")
        print(f"         Or train.py handles this via stratified splits automatically.")

    print(f"\nNext step: python extract_features.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeriVoice Sentinel - Prepare ASVspoof 2019 LA")
    parser.add_argument("--asv_root",  default="ASVspoof2019_LA",
                        help="Root folder of the downloaded ASVspoof 2019 LA dataset")
    parser.add_argument("--data_dir",  default="data",
                        help="Output data/ folder (will create real/ and fake/ inside)")
    parser.add_argument("--splits",    nargs="+", default=["train", "dev"],
                        choices=["train", "dev", "eval"],
                        help="Which splits to include (default: train dev)")
    parser.add_argument("--max_real",  default=0, type=int,
                        help="Cap on real files (0 = no cap, use all ~12k)")
    parser.add_argument("--max_fake",  default=0, type=int,
                        help="Cap on fake files (0 = no cap; ASVspoof has ~108k fake)")
    parser.add_argument("--no_convert", action="store_true",
                        help="Skip FLAC→WAV conversion (if files are already WAV)")
    args = parser.parse_args()

    prepare(
        asv_root  = args.asv_root,
        data_dir  = args.data_dir,
        splits    = args.splits,
        max_real  = args.max_real,
        max_fake  = args.max_fake,
        convert   = not args.no_convert,
    )
