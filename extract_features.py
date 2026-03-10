"""
VeriVoice Sentinel — Step 1: Feature Extraction (with Augmentation)
====================================================================
Walks data/real/ and data/fake/, extracts 768-dim Wav2Vec2 embeddings.
Each file is processed MULTIPLE times with different augmentations to
make the model robust to real-world audio conditions.

Augmentations applied for real-world robustness:
  - Gaussian noise       (simulates mic/background noise)
  - Volume variation     (simulates different recording levels)
  - Speed perturbation   (simulates different speaking rates)
  - High-pass filter     (simulates phone call / compression)
  - Random cropping      (forces model to work on any segment)
  - Combined             (noise + volume together)

Usage:
    python extract_features.py
    python extract_features.py --data_dir data --batch_size 16 --aug_factor 3
"""

import os
import argparse
import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

TARGET_SR   = 16_000
MODEL_NAME  = "facebook/wav2vec2-base"
AUDIO_EXTS  = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
MIN_SAMPLES = TARGET_SR * 1   # skip clips shorter than 1 second


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU : {name}  ({vram:.1f} GB VRAM)")
        return torch.device("cuda")
    print("  CPU : CUDA not found — extraction will be slow")
    return torch.device("cpu")


# ── Audio loading ──────────────────────────────────────────────────────────────

def load_audio(path: Path) -> torch.Tensor | None:
    """Load audio → 1-D float32 tensor at TARGET_SR, peak-normalised."""
    try:
        waveform, sr = torchaudio.load(str(path))
        if sr != TARGET_SR:
            waveform = AF.resample(waveform, sr, TARGET_SR)
        waveform = waveform.mean(dim=0)
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        if len(waveform) < MIN_SAMPLES:
            return None
        # Truncate to max 10 seconds to prevent VRAM OOM on long files
        MAX_SAMPLES = TARGET_SR * 10
        if len(waveform) > MAX_SAMPLES:
            waveform = waveform[:MAX_SAMPLES]
        return waveform
    except Exception as e:
        print(f"\n  [WARN] Cannot load {path.name}: {e}")
        return None


# ── Augmentations ──────────────────────────────────────────────────────────────

def aug_noise(wave: torch.Tensor) -> torch.Tensor:
    """Add Gaussian noise — simulates mic/background noise."""
    strength = np.random.uniform(0.001, 0.025)
    return (wave + torch.randn_like(wave) * strength).clamp(-1.0, 1.0)


def aug_volume(wave: torch.Tensor) -> torch.Tensor:
    """Random volume scaling — simulates different recording levels."""
    return (wave * np.random.uniform(0.4, 1.0)).clamp(-1.0, 1.0)


def aug_speed(wave: torch.Tensor) -> torch.Tensor:
    """Speed perturbation ±10% via resampling trick."""
    factor   = np.random.uniform(0.90, 1.10)
    orig_len = len(wave)
    fake_sr  = int(TARGET_SR * factor)
    out = AF.resample(wave.unsqueeze(0), TARGET_SR, fake_sr).squeeze(0)
    if len(out) < orig_len:
        out = torch.nn.functional.pad(out, (0, orig_len - len(out)))
    return out[:orig_len]


def aug_highpass(wave: torch.Tensor) -> torch.Tensor:
    """High-pass filter — simulates phone call / lossy compression."""
    cutoff   = float(np.random.randint(200, 800))
    filtered = AF.highpass_biquad(wave.unsqueeze(0), TARGET_SR, cutoff).squeeze(0)
    peak = filtered.abs().max()
    return (filtered / peak) if peak > 0 else filtered


def aug_crop(wave: torch.Tensor) -> torch.Tensor:
    """Random crop to 60–100% length then zero-pad back."""
    n      = len(wave)
    keep   = int(n * np.random.uniform(0.6, 1.0))
    start  = np.random.randint(0, n - keep + 1)
    out    = wave[start : start + keep]
    return torch.nn.functional.pad(out, (0, n - len(out)))


def aug_combined(wave: torch.Tensor) -> torch.Tensor:
    """Noise + volume — most common real-world combination."""
    return aug_noise(aug_volume(wave))


ALL_AUGS = [aug_noise, aug_volume, aug_speed, aug_highpass, aug_crop, aug_combined]


def augment(wave: torch.Tensor, n: int) -> list[torch.Tensor]:
    """Return n randomly chosen augmented copies. Always includes noise."""
    chosen = [aug_noise]
    pool   = [f for f in ALL_AUGS if f is not aug_noise]
    extra  = np.random.choice(len(pool), size=min(n - 1, len(pool)), replace=False)
    chosen += [pool[i] for i in extra]
    np.random.shuffle(chosen)
    results = []
    for fn in chosen[:n]:
        try:
            results.append(fn(wave.clone()))
        except Exception:
            results.append(aug_noise(wave.clone()))
    return results


# ── Embedding ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_batch(
    waveforms: list[torch.Tensor],
    processor: Wav2Vec2Processor,
    model: Wav2Vec2Model,
    device: torch.device,
) -> np.ndarray:
    inputs = processor(
        [w.numpy() for w in waveforms],
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out    = model(**inputs)
    return out.last_hidden_state.mean(dim=1).cpu().numpy()


# ── File collection ────────────────────────────────────────────────────────────

def collect_files(data_dir: Path) -> tuple[list[Path], list[int]]:
    paths, labels = [], []
    for lbl, name in [(0, "real"), (1, "fake")]:
        folder = data_dir / name
        if not folder.exists():
            print(f"  [WARN] {folder} not found — skipping")
            continue
        found = sorted(f for f in folder.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
        print(f"  {name:>4}/  {len(found):>6,} files")
        paths.extend(found)
        labels.extend([lbl] * len(found))
    return paths, labels


# ── Main ───────────────────────────────────────────────────────────────────────

def main(data_dir: str, batch_size: int, output: str, aug_factor: int) -> None:
    print("\n" + "=" * 55)
    print("  VeriVoice Sentinel — Feature Extraction")
    print("=" * 55)

    device = get_device()

    print(f"\n  Loading {MODEL_NAME} …")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model     = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("  Model ready.\n")

    data_path = Path(data_dir)
    print("  Scanning dataset …")
    paths, labels = collect_files(data_path)

    if not paths:
        print(f"\n  [ERROR] No audio files found.")
        return

    total     = len(paths)
    aug_total = total * (1 + aug_factor)
    print(f"\n  Original files   : {total:,}")
    print(f"  Aug factor       : {aug_factor}x  →  {aug_total:,} total embeddings")
    print(f"  Batch size       : {batch_size}")
    print(f"  Augmentations    : noise, volume, speed, highpass, crop, combined")
    print("  Starting …\n")

    all_embeddings: list[np.ndarray] = []
    all_labels:     list[int]        = []
    wave_buf:       list[torch.Tensor] = []
    label_buf:      list[int]        = []
    skipped = 0

    def flush():
        if wave_buf:
            all_embeddings.append(embed_batch(wave_buf, processor, model, device))
            all_labels.extend(label_buf)
            wave_buf.clear()
            label_buf.clear()

    def push(w: torch.Tensor, lbl: int):
        wave_buf.append(w)
        label_buf.append(lbl)
        if len(wave_buf) >= batch_size:
            flush()

    bar = tqdm(zip(paths, labels), total=total, unit="file", dynamic_ncols=True)
    for path, lbl in bar:
        bar.set_postfix(skip=skipped, file=path.name[:25])
        wave = load_audio(path)
        if wave is None:
            skipped += 1
            continue
        push(wave, lbl)
        if aug_factor > 0:
            for aug_wave in augment(wave, aug_factor):
                push(aug_wave, lbl)

    flush()

    if not all_embeddings:
        print("\n  [ERROR] No embeddings produced.")
        return

    X = np.vstack(all_embeddings).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)
    np.savez_compressed(output, X=X, y=y)

    print(f"\n{'=' * 55}")
    print(f"  Done!  Original: {total:,}  Skipped: {skipped:,}")
    print(f"  Total embeddings (with aug): {len(y):,}")
    print(f"  Real (0): {(y==0).sum():,}   Fake (1): {(y==1).sum():,}")
    print(f"  Shape: {X.shape}")
    print(f"  Saved → {output}")
    print(f"{'=' * 55}")
    print("\n  Next: python train.py\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   default="data",         help="Root data folder")
    ap.add_argument("--batch_size", default=16,  type=int,  help="GPU batch size")
    ap.add_argument("--output",     default="features.npz", help="Output .npz path")
    ap.add_argument("--aug_factor", default=3,   type=int,
                    help="Augmented copies per file (default 3 = 4x total data)")
    args = ap.parse_args()
    main(args.data_dir, args.batch_size, args.output, args.aug_factor) 