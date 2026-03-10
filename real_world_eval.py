"""
VeriVoice Sentinel — Real-World Evaluation
===========================================
Evaluates the model on a mixed holdout set:
  - Random sample of ASVspoof real files
  - Random sample of ASVspoof fake files  
  - ALL modern TTS/ElevenLabs fake files (MP3s)

This gives a realistic accuracy number that reflects
real-world performance, not just ASVspoof benchmark performance.

Usage:
    python real_world_eval.py
    python real_world_eval.py --n_real 100 --n_legacy_fake 100
"""

import os
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF
import joblib
from pathlib import Path
from tqdm import tqdm

TARGET_SR  = 16_000
MODEL_NAME = "facebook/wav2vec2-base"
random.seed(42)


class DeepfakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,  64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_audio(path):
    try:
        waveform, sr = torchaudio.load(str(path))
        if sr != TARGET_SR:
            waveform = AF.resample(waveform, sr, TARGET_SR)
        waveform = waveform.mean(dim=0)
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        MAX_SAMPLES = TARGET_SR * 10
        if len(waveform) > MAX_SAMPLES:
            waveform = waveform[:MAX_SAMPLES]
        if len(waveform) < TARGET_SR:
            return None
        return waveform
    except:
        return None


@torch.no_grad()
def embed_batch(waveforms, processor, model, device):
    inputs = processor(
        [w.numpy() for w in waveforms],
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    return out.last_hidden_state.mean(dim=1).cpu().numpy()


def main(n_real, n_legacy_fake, fake_threshold):
    print(f"\n{'='*55}")
    print(f"  VeriVoice Sentinel — Real-World Evaluation")
    print(f"{'='*55}")

    device = get_device()
    print(f"  Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ── Load models ────────────────────────────────────────────────────────────
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    processor  = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    wav2vec    = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    wav2vec.eval()

    clf = DeepfakeClassifier().to(device)
    clf.load_state_dict(torch.load("model.pt", map_location=device, weights_only=False))
    clf.eval()

    scaler = joblib.load("scaler.pkl")

    # ── Collect files ──────────────────────────────────────────────────────────
    real_dir   = Path("data/real")
    fake_dir   = Path("data/fake")

    real_files   = list(real_dir.glob("*.wav"))
    legacy_fakes = list(fake_dir.glob("*.wav"))
    modern_fakes = list(fake_dir.glob("*.mp3"))

    print(f"\n  Available — Real: {len(real_files):,}  Legacy fake: {len(legacy_fakes):,}  Modern fake: {len(modern_fakes):,}")

    # Sample
    real_sample   = random.sample(real_files,   min(n_real,        len(real_files)))
    legacy_sample = random.sample(legacy_fakes, min(n_legacy_fake, len(legacy_fakes)))
    modern_sample = modern_fakes  # use ALL modern fakes

    print(f"  Evaluating — Real: {len(real_sample)}  Legacy fake: {len(legacy_sample)}  Modern fake: {len(modern_sample)}")

    # Build test set: (path, true_label)
    test_set = (
        [(f, 0) for f in real_sample] +
        [(f, 1) for f in legacy_sample] +
        [(f, 1) for f in modern_sample]
    )
    random.shuffle(test_set)

    # ── Run inference ──────────────────────────────────────────────────────────
    y_true, y_pred, y_prob = [], [], []
    skipped = 0
    batch_waves, batch_labels = [], []

    def flush_batch():
        if not batch_waves:
            return
        embs   = embed_batch(batch_waves, processor, wav2vec, device)
        scaled = scaler.transform(embs).astype(np.float32)
        with torch.no_grad():
            logits = clf(torch.tensor(scaled).to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        for i, lbl in enumerate(batch_labels):
            fp   = float(probs[i][1])
            pred = 1 if fp > fake_threshold else 0
            y_true.append(lbl)
            y_pred.append(pred)
            y_prob.append(fp)
        batch_waves.clear()
        batch_labels.clear()

    for path, lbl in tqdm(test_set, desc="  Evaluating", unit="file"):
        w = load_audio(path)
        if w is None:
            skipped += 1
            continue
        batch_waves.append(w)
        batch_labels.append(lbl)
        if len(batch_waves) >= 16:
            flush_batch()

    flush_batch()

    # ── Metrics ────────────────────────────────────────────────────────────────
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = float('nan')

    cm = confusion_matrix(y_true, y_pred)

    n_real_correct   = (y_pred[y_true==0] == 0).sum()
    n_real_total     = (y_true==0).sum()
    n_legacy_correct = (y_pred[(y_true==1) & (np.array([1]*len(legacy_sample)+[0]*len(modern_sample)+[0]*len(real_sample))[:len(y_true)]==0)] == 1).sum() if len(legacy_sample) > 0 else 0

    # Modern fake accuracy
    n_modern = len(modern_sample)
    modern_preds = y_pred[len(real_sample):len(real_sample)+len(legacy_sample)+n_modern][-n_modern:] if n_modern > 0 else []
    modern_acc = (np.array(modern_preds)==1).mean() * 100 if len(modern_preds) > 0 else 0

    print(f"\n{'='*55}")
    print(f"  REAL-WORLD EVALUATION RESULTS")
    print(f"  Threshold : fake_prob > {fake_threshold}")
    print(f"  Skipped   : {skipped} files (too short / unreadable)")
    print(f"{'='*55}")
    print(f"  Overall Accuracy  : {acc*100:.1f}%")
    print(f"  Precision         : {prec*100:.1f}%")
    print(f"  Recall            : {rec*100:.1f}%")
    print(f"  F1 Score          : {f1*100:.1f}%")
    print(f"  ROC-AUC           : {auc*100:.1f}%")
    print(f"\n  Breakdown:")
    print(f"  Real voice accuracy        : {n_real_correct}/{n_real_total} = {n_real_correct/n_real_total*100:.1f}%")
    print(f"  Legacy fake (ASVspoof) acc : see confusion matrix")
    print(f"  Modern fake (TTS) accuracy : {modern_acc:.1f}%  ← key metric")
    print(f"\n  Confusion Matrix")
    print(f"                Pred Real   Pred Fake")
    if cm.shape == (2,2):
        print(f"  True Real  :  {cm[0,0]:>8,}   {cm[0,1]:>8,}")
        print(f"  True Fake  :  {cm[1,0]:>8,}   {cm[1,1]:>8,}")
    print(f"{'='*55}")

    if acc < 0.80:
        print(f"\n  Honest real-world accuracy: {acc*100:.1f}%")
        print(f"  This reflects genuine generalisation performance.")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_real",         default=200, type=int,   help="Real files to sample")
    ap.add_argument("--n_legacy_fake",  default=200, type=int,   help="ASVspoof WAV fakes to sample")
    ap.add_argument("--fake_threshold", default=0.35, type=float, help="Fake detection threshold")
    args = ap.parse_args()
    main(args.n_real, args.n_legacy_fake, args.fake_threshold)
