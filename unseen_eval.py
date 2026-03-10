"""
EchoVox — Unseen Data Evaluation
=================================
Tests BOTH model.pt and model_ft.pt on data they have NEVER seen:

  Dataset 1 — Your own recordings (unseen_test/real/)
  Dataset 2 — Mendeley: ElevenLabs + Respeecher fakes (Fake_ElevenLabs_Respeecher/)
  Dataset 3 — In-The-Wild: real + fake celebrity voices (release_in_the_wild/)

This is the honest test. No overlap with training data.

Usage:
    python unseen_eval.py
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF
import joblib
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

TARGET_SR  = 16_000
MODEL_NAME = "facebook/wav2vec2-base"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
YOUR_REAL_DIR       = Path("unseen_test/real")
MENDELEY_FAKE_DIR   = Path("Fake_ElevenLabs_Respeecher")
ITW_REAL_DIR        = Path("release_in_the_wild/real")
ITW_FAKE_DIR        = Path("release_in_the_wild/fake")


# ── Baseline model (frozen Wav2Vec2 + MLP + scaler) ──────────────────────────
class DeepfakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,  64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)


# ── Fine-tuned model (end-to-end Wav2Vec2 + MLP) ─────────────────────────────
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
        hidden  = outputs.last_hidden_state
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


# ── Audio utils ───────────────────────────────────────────────────────────────
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
        if len(waveform) < TARGET_SR // 2:
            return None
        return waveform
    except:
        return None


def collate_batch(waveforms):
    max_len = max(w.shape[0] for w in waveforms)
    padded  = torch.zeros(len(waveforms), max_len)
    mask    = torch.zeros(len(waveforms), max_len, dtype=torch.long)
    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w
        mask[i,   :w.shape[0]] = 1
    return padded, mask


def collect_files(folder, label, max_files=None):
    if not folder.exists():
        print(f"  [WARN] Folder not found: {folder}")
        return [], []
    files = [f for f in folder.rglob("*") if f.suffix.lower() in AUDIO_EXTS]
    if max_files and len(files) > max_files:
        files = random.sample(files, max_files)
    return files, [label] * len(files)


# ── Inference: Fine-tuned model ───────────────────────────────────────────────
@torch.no_grad()
def infer_ft(model, test_set, device, threshold=0.35, batch_size=16):
    y_true, y_pred, y_prob = [], [], []
    skipped = 0
    batch_waves, batch_labels = [], []

    def flush():
        if not batch_waves:
            return
        padded, mask = collate_batch(batch_waves)
        logits = model(padded.to(device), mask.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        for i, lbl in enumerate(batch_labels):
            fp = float(probs[i][1])
            y_true.append(lbl)
            y_pred.append(1 if fp > threshold else 0)
            y_prob.append(fp)
        batch_waves.clear()
        batch_labels.clear()

    for path, lbl in test_set:
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


# ── Inference: Baseline model ─────────────────────────────────────────────────
@torch.no_grad()
def infer_baseline(clf, wav2vec, processor, scaler, test_set, device,
                   threshold=0.35, batch_size=16):
    y_true, y_pred, y_prob = [], [], []
    skipped = 0
    batch_waves, batch_labels = [], []

    def flush():
        if not batch_waves:
            return
        inputs = processor(
            [w.numpy() for w in batch_waves],
            sampling_rate=TARGET_SR, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embs   = wav2vec(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        scaled = scaler.transform(embs).astype(np.float32)
        logits = clf(torch.tensor(scaled).to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        for i, lbl in enumerate(batch_labels):
            fp = float(probs[i][1])
            y_true.append(lbl)
            y_pred.append(1 if fp > threshold else 0)
            y_prob.append(fp)
        batch_waves.clear()
        batch_labels.clear()

    for path, lbl in test_set:
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


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
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
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc, cm=cm)


def print_section(title, base_m, ft_m, n_real, n_fake, base_skip, ft_skip):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"  Files: {n_real} real  +  {n_fake} fake")
    print(f"{'='*62}")
    print(f"  {'Metric':<22} {'model.pt':>10} {'model_ft.pt':>12} {'Delta':>10}")
    print(f"  {'-'*56}")

    metrics = [
        ("Accuracy",  base_m['acc']*100,  ft_m['acc']*100),
        ("Precision", base_m['prec']*100, ft_m['prec']*100),
        ("Recall",    base_m['rec']*100,  ft_m['rec']*100),
        ("F1 Score",  base_m['f1']*100,   ft_m['f1']*100),
        ("ROC-AUC",   base_m['auc']*100,  ft_m['auc']*100),
    ]
    for name, base_val, ft_val in metrics:
        delta = ft_val - base_val
        flag  = "✅" if delta >= 0 else "⚠️ "
        arrow = "↑" if delta > 0.1 else ("↓" if delta < -0.1 else "→")
        print(f"  {name:<22} {base_val:>9.2f}%  {ft_val:>10.2f}%  "
              f"{flag} {arrow}{abs(delta):.2f}%")

    cm_b = base_m['cm']
    cm_f = ft_m['cm']
    print(f"\n  Confusion Matrix        model.pt          model_ft.pt")
    if cm_b.shape == (2,2) and cm_f.shape == (2,2):
        print(f"  True Real  : pred R/F   {cm_b[0,0]:>6,} / {cm_b[0,1]:<6,}    "
              f"{cm_f[0,0]:>6,} / {cm_f[0,1]:<6,}")
        print(f"  True Fake  : pred R/F   {cm_b[1,0]:>6,} / {cm_b[1,1]:<6,}    "
              f"{cm_f[1,0]:>6,} / {cm_f[1,1]:<6,}")
    print(f"  Skipped: model.pt={base_skip}  model_ft.pt={ft_skip}")


def print_final_verdict(results):
    print(f"\n{'='*62}")
    print(f"  FINAL HONEST VERDICT — UNSEEN DATA ONLY")
    print(f"{'='*62}")
    print(f"  {'Dataset':<30} {'model.pt':>10} {'model_ft.pt':>12} {'Winner':>8}")
    print(f"  {'-'*62}")
    ft_wins = 0
    for name, base_acc, ft_acc in results:
        winner = "ft ✅" if ft_acc > base_acc else ("base ⚠️" if base_acc > ft_acc else "tie")
        if ft_acc > base_acc:
            ft_wins += 1
        print(f"  {name:<30} {base_acc:>9.2f}%  {ft_acc:>10.2f}%  {winner:>8}")
    print(f"{'='*62}")
    if ft_wins >= 2:
        print(f"\n  🏆 model_ft.pt wins on {ft_wins}/3 datasets")
        print(f"  Fine-tuning genuinely improved real-world performance.")
        print(f"  → Use model_ft.pt as your production model")
    else:
        print(f"\n  ⚠️  model.pt holds up better on unseen data")
        print(f"  Fine-tuned model may have overfit to training distribution.")
        print(f"  → Consider more diverse training data or higher regularization")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*62}")
    print(f"  EchoVox — Honest Unseen Data Evaluation")
    print(f"  Testing BOTH models on data never seen during training")
    print(f"{'='*62}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ── Load Wav2Vec2 (shared) ─────────────────────────────────────────────────
    print(f"\n  Loading {MODEL_NAME} …")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    # ── Load baseline model ────────────────────────────────────────────────────
    print(f"  Loading model.pt (baseline) …")
    wav2vec_base = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    wav2vec_base.eval()
    clf = DeepfakeClassifier().to(device)
    clf.load_state_dict(torch.load("model.pt", map_location=device, weights_only=False))
    clf.eval()
    scaler = joblib.load("scaler.pkl")

    # ── Load fine-tuned model ──────────────────────────────────────────────────
    print(f"  Loading model_ft.pt (fine-tuned) …")
    wav2vec_ft = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    ft_model   = EchoVoxFineTuned(wav2vec_ft).to(device)
    ft_model.load_state_dict(torch.load("model_ft.pt", map_location=device, weights_only=False))
    ft_model.eval()

    verdict_rows = []

    # ══════════════════════════════════════════════════════════════════════════
    # DATASET 1 — Your own recordings vs Mendeley fakes
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  ── Dataset 1: Your recordings + Mendeley fakes ──")
    real_p, real_l = collect_files(YOUR_REAL_DIR,     label=0)
    fake_p, fake_l = collect_files(MENDELEY_FAKE_DIR, label=1)

    if not real_p:
        print(f"  [SKIP] No files found in {YOUR_REAL_DIR}")
    elif not fake_p:
        print(f"  [SKIP] No files found in {MENDELEY_FAKE_DIR}")
    else:
        test_set = list(zip(real_p + fake_p, real_l + fake_l))
        print(f"  {len(real_p)} real  +  {len(fake_p)} fake  =  {len(test_set)} total")

        print(f"  Running model.pt …")
        bt, bp, bpr, bsk = infer_baseline(clf, wav2vec_base, processor, scaler,
                                          test_set, device)
        print(f"  Running model_ft.pt …")
        ft, fp, fpr, fsk = infer_ft(ft_model, test_set, device)

        base_m = compute_metrics(bt, bp, bpr)
        ft_m   = compute_metrics(ft, fp, fpr)

        print_section("Dataset 1 — Your Recordings + Mendeley (ElevenLabs/Respeecher)",
                      base_m, ft_m, len(real_p), len(fake_p), bsk, fsk)
        verdict_rows.append(("Your clips + Mendeley fakes",
                             base_m['acc']*100, ft_m['acc']*100))

    # ══════════════════════════════════════════════════════════════════════════
    # DATASET 2 — In-The-Wild (real celebrity voices vs deepfakes)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n  ── Dataset 2: In-The-Wild dataset ──")
    itw_real_p, itw_real_l = collect_files(ITW_REAL_DIR, label=0, max_files=500)
    itw_fake_p, itw_fake_l = collect_files(ITW_FAKE_DIR, label=1, max_files=500)

    if not itw_real_p:
        print(f"  [SKIP] No files in {ITW_REAL_DIR}")
    elif not itw_fake_p:
        print(f"  [SKIP] No files in {ITW_FAKE_DIR}")
    else:
        test_set = list(zip(itw_real_p + itw_fake_p, itw_real_l + itw_fake_l))
        print(f"  {len(itw_real_p)} real  +  {len(itw_fake_p)} fake  =  {len(test_set)} total")

        print(f"  Running model.pt …")
        bt, bp, bpr, bsk = infer_baseline(clf, wav2vec_base, processor, scaler,
                                          tqdm(test_set, desc="  baseline"), device)
        print(f"  Running model_ft.pt …")
        ft, fp, fpr, fsk = infer_ft(ft_model,
                                    tqdm(test_set, desc="  fine-tuned"), device)

        base_m = compute_metrics(bt, bp, bpr)
        ft_m   = compute_metrics(ft, fp, fpr)

        print_section("Dataset 2 — In-The-Wild (Celebrity Voices)",
                      base_m, ft_m, len(itw_real_p), len(itw_fake_p), bsk, fsk)
        verdict_rows.append(("In-The-Wild celebrities",
                             base_m['acc']*100, ft_m['acc']*100))

    # ══════════════════════════════════════════════════════════════════════════
    # DATASET 3 — Combined (everything together)
    # ══════════════════════════════════════════════════════════════════════════
    if len(verdict_rows) == 2:
        combined_base = sum(r[1] for r in verdict_rows) / len(verdict_rows)
        combined_ft   = sum(r[2] for r in verdict_rows) / len(verdict_rows)
        verdict_rows.append(("Combined average", combined_base, combined_ft))

    # ── Final verdict ──────────────────────────────────────────────────────────
    if verdict_rows:
        print_final_verdict(verdict_rows)


if __name__ == "__main__":
    main()
