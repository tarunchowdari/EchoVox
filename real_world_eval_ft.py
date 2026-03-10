"""
EchoVox — Real-World Evaluation for Fine-Tuned Model (model_ft.pt)
==================================================================
Evaluates model_ft.pt (end-to-end Wav2Vec2 fine-tuned) on a mixed holdout:
  - Random sample of ASVspoof real files
  - Random sample of ASVspoof legacy fake files
  - ALL modern TTS/ElevenLabs fake files (MP3s)

Compares results against model.pt baseline automatically.

Usage:
    python real_world_eval_ft.py
    python real_world_eval_ft.py --n_real 200 --n_legacy_fake 200
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

TARGET_SR  = 16_000
MODEL_NAME = "facebook/wav2vec2-base"
random.seed(42)


# ── Must match finetune.py architecture exactly ───────────────────────────────
class EchoVoxFineTuned(nn.Module):
    def __init__(self, wav2vec_model, dropout=0.5):
        super().__init__()
        self.wav2vec = wav2vec_model
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256,  64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec(input_values=input_values, attention_mask=attention_mask)
        hidden  = outputs.last_hidden_state  # (batch, frames, 768)
        if attention_mask is not None:
            feat_lengths = self.wav2vec._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            )
            max_frames = hidden.shape[1]
            frame_mask = torch.arange(max_frames, device=hidden.device).unsqueeze(0)
            frame_mask = (frame_mask < feat_lengths.unsqueeze(1)).unsqueeze(-1).float()
            pooled = (hidden * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)
        return self.classifier(pooled)


def get_device():
    if torch.cuda.is_available():
        print(f"  Device : {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("  Device : CPU")
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
        if len(waveform) > TARGET_SR * 10:
            waveform = waveform[:TARGET_SR * 10]
        if len(waveform) < TARGET_SR:
            return None
        return waveform
    except:
        return None


def collate_batch(waveforms):
    """Pad waveforms to same length and build attention mask."""
    max_len = max(w.shape[0] for w in waveforms)
    padded  = torch.zeros(len(waveforms), max_len)
    mask    = torch.zeros(len(waveforms), max_len, dtype=torch.long)
    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w
        mask[i,   :w.shape[0]] = 1
    return padded, mask


@torch.no_grad()
def run_inference(model, test_set, device, threshold, batch_size=16):
    """Run batched inference on test_set = list of (path, label)."""
    y_true, y_pred, y_prob = [], [], []
    skipped = 0
    batch_waves, batch_labels = [], []

    def flush():
        if not batch_waves:
            return
        padded, mask = collate_batch(batch_waves)
        padded, mask = padded.to(device), mask.to(device)
        logits = model(padded, mask)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        for i, lbl in enumerate(batch_labels):
            fp   = float(probs[i][1])
            pred = 1 if fp > threshold else 0
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
        if len(batch_waves) >= batch_size:
            flush()
    flush()

    return np.array(y_true), np.array(y_pred), np.array(y_prob), skipped


def print_results(y_true, y_pred, y_prob, skipped, threshold,
                  n_real, n_legacy, n_modern, model_path):
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
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)

    # Per-category accuracy
    real_mask   = y_true == 0
    fake_mask   = y_true == 1
    real_acc    = (y_pred[real_mask] == 0).mean() * 100 if real_mask.sum() else 0

    # Modern fakes are at the end of y_true
    total       = len(y_true)
    modern_true = y_true[total - n_modern:]
    modern_pred = y_pred[total - n_modern:]
    modern_acc  = (modern_pred == 1).mean() * 100 if n_modern else 0

    print(f"\n{'='*58}")
    print(f"  RESULTS — {model_path}")
    print(f"  Threshold : fake_prob > {threshold}")
    print(f"  Skipped   : {skipped} files")
    print(f"{'='*58}")
    print(f"  Overall Accuracy  : {acc*100:.2f}%")
    print(f"  Precision         : {prec*100:.2f}%")
    print(f"  Recall            : {rec*100:.2f}%")
    print(f"  F1 Score          : {f1*100:.2f}%")
    print(f"  ROC-AUC           : {auc*100:.2f}%")
    print(f"\n  Breakdown:")
    print(f"  Real voice accuracy        : {real_acc:.2f}%  ({real_mask.sum()} files)")
    print(f"  Modern TTS accuracy        : {modern_acc:.2f}%  ({n_modern} files)  ← key metric")
    print(f"\n  Confusion Matrix")
    print(f"                Pred Real   Pred Fake")
    if cm.shape == (2, 2):
        print(f"  True Real  :  {cm[0,0]:>8,}   {cm[0,1]:>8,}")
        print(f"  True Fake  :  {cm[1,0]:>8,}   {cm[1,1]:>8,}")
    print(f"{'='*58}")

    return {
        "overall_acc": acc*100, "precision": prec*100, "recall": rec*100,
        "f1": f1*100, "auc": auc*100, "real_acc": real_acc, "modern_acc": modern_acc
    }


def main(args):
    print(f"\n{'='*58}")
    print(f"  EchoVox — Fine-Tuned Model Real-World Evaluation")
    print(f"{'='*58}")

    device = get_device()

    # ── Load fine-tuned model ─────────────────────────────────────────────────
    print(f"\n  Loading {MODEL_NAME} …")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    wav2vec   = Wav2Vec2Model.from_pretrained(MODEL_NAME)

    model = EchoVoxFineTuned(wav2vec).to(device)

    if not Path(args.model).exists():
        print(f"\n  [ERROR] {args.model} not found!")
        print(f"  Run: python finetune.py --epochs 15 --batch_size 32")
        return

    state = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded : {args.model}")

    # ── Collect test files ────────────────────────────────────────────────────
    real_dir     = Path("data/real")
    fake_dir     = Path("data/fake")
    real_files   = list(real_dir.glob("*.wav"))
    legacy_fakes = list(fake_dir.glob("*.wav"))
    modern_fakes = list(fake_dir.glob("*.mp3"))

    print(f"\n  Available — Real: {len(real_files):,}  "
          f"Legacy fake: {len(legacy_fakes):,}  Modern fake: {len(modern_fakes):,}")

    real_sample   = random.sample(real_files,   min(args.n_real,        len(real_files)))
    legacy_sample = random.sample(legacy_fakes, min(args.n_legacy_fake, len(legacy_fakes)))
    modern_sample = modern_fakes  # always use ALL modern fakes

    print(f"  Evaluating — Real: {len(real_sample)}  "
          f"Legacy fake: {len(legacy_sample)}  Modern fake: {len(modern_sample)}")

    # Keep modern fakes at END of list so we can slice them after
    test_set = (
        [(f, 0) for f in real_sample] +
        [(f, 1) for f in legacy_sample] +
        [(f, 1) for f in modern_sample]
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    y_true, y_pred, y_prob, skipped = run_inference(
        model, test_set, device, args.fake_threshold
    )

    # ── Print results ─────────────────────────────────────────────────────────
    ft_results = print_results(
        y_true, y_pred, y_prob, skipped, args.fake_threshold,
        len(real_sample), len(legacy_sample), len(modern_sample),
        args.model
    )

    # ── Compare against baseline ──────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  COMPARISON vs Baseline (model.pt)")
    print(f"{'='*58}")
    print(f"  {'Metric':<28} {'model.pt':>10} {'model_ft.pt':>12} {'Delta':>8}")
    print(f"  {'-'*58}")

    baseline = {
        "Overall Accuracy":   96.08,
        "Modern TTS Accuracy": 95.60,
        "Precision":          98.55,
        "Recall":             94.30,
        "F1 Score":           96.38,
        "ROC-AUC":            99.57,
    }
    current = {
        "Overall Accuracy":   ft_results["overall_acc"],
        "Modern TTS Accuracy": ft_results["modern_acc"],
        "Precision":          ft_results["precision"],
        "Recall":             ft_results["recall"],
        "F1 Score":           ft_results["f1"],
        "ROC-AUC":            ft_results["auc"],
    }

    for metric in baseline:
        base = baseline[metric]
        curr = current[metric]
        delta = curr - base
        arrow = "↑" if delta > 0.1 else ("↓" if delta < -0.1 else "→")
        color = "✅" if delta >= 0 else "⚠️ "
        print(f"  {metric:<28} {base:>9.2f}%  {curr:>10.2f}%  "
              f"{color} {arrow}{abs(delta):.2f}%")

    print(f"{'='*58}")
    winner = "model_ft.pt" if ft_results["overall_acc"] > 96.08 else "model.pt"
    print(f"\n  🏆 Best model: {winner}")
    if winner == "model_ft.pt":
        print(f"  Fine-tuning improved overall accuracy!")
        print(f"  → Copy model_ft.pt to use as your main model")
    else:
        print(f"  Fine-tuning did not beat baseline on this sample.")
        print(f"  → Consider: more epochs, more data, or higher unfreeze_layers")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",          default="model_ft.pt",  help="Fine-tuned model path")
    ap.add_argument("--n_real",         default=200, type=int,  help="Real files to sample")
    ap.add_argument("--n_legacy_fake",  default=200, type=int,  help="Legacy ASVspoof fakes to sample")
    ap.add_argument("--fake_threshold", default=0.35, type=float, help="Fake detection threshold")
    ap.parse_args()
    args = ap.parse_args()
    main(args)
