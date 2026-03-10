"""
EchoVox — Phase 2: End-to-End Wav2Vec2 Fine-tuning
====================================================
Trains Wav2Vec2 + MLP classifier together end-to-end.
Unfreezes the last N transformer layers of Wav2Vec2 while keeping
early layers frozen (preserves learned speech representations).

Architecture:
  Wav2Vec2-base (last 4 layers unfrozen) → mean pooling → 768-dim
  → Linear(768, 256) → BN → ReLU → Dropout(0.5)
  → Linear(256, 64)  → BN → ReLU → Dropout(0.5)
  → Linear(64, 2)    → softmax

Why this works better than frozen features:
  - Frozen: Wav2Vec2 extracts generic speech features, MLP learns to classify
  - Fine-tuned: Wav2Vec2 learns to extract deepfake-specific features
  - Expected improvement: +3-8% on modern TTS detection

Usage:
    python finetune.py
    python finetune.py --epochs 20 --unfreeze_layers 4 --batch_size 8 --lr 2e-5
"""

import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler   # ← fp16 mixed precision (updated API)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_SR   = 16_000
MODEL_NAME  = "facebook/wav2vec2-base"
AUDIO_EXTS  = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
MIN_SAMPLES = TARGET_SR * 1
MAX_SAMPLES = TARGET_SR * 8   # 8 sec max to fit VRAM


# ── Model ──────────────────────────────────────────────────────────────────────

class EchoVoxFineTuned(nn.Module):
    """
    Wav2Vec2 (partially unfrozen) + MLP classifier.
    Only the last `unfreeze_layers` transformer blocks are trainable.
    """
    def __init__(self, wav2vec_model: Wav2Vec2Model,
                 unfreeze_layers: int = 4, dropout: float = 0.5):
        super().__init__()
        self.wav2vec = wav2vec_model

        # ── Freeze everything first ────────────────────────────────────────────
        for param in self.wav2vec.parameters():
            param.requires_grad = False

        # ── Unfreeze last N transformer layers ────────────────────────────────
        total_layers = len(self.wav2vec.encoder.layers)
        unfreeze_from = total_layers - unfreeze_layers
        for i, layer in enumerate(self.wav2vec.encoder.layers):
            if i >= unfreeze_from:
                for param in layer.parameters():
                    param.requires_grad = True

        # ── Always unfreeze layer norm at top ─────────────────────────────────
        for param in self.wav2vec.encoder.layer_norm.parameters():
            param.requires_grad = True

        # ── MLP classifier ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 2),
        )

    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        outputs = self.wav2vec(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        # Mean pool over time dimension → (batch, 768)
        hidden = outputs.last_hidden_state  # (batch, frames, 768)
        if attention_mask is not None:
            # Convert raw sample mask → frame-level mask using Wav2Vec2 subsampling
            # Wav2Vec2 downsamples audio ~320x through CNN feature extractor
            feat_lengths = self.wav2vec._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            )  # (batch,) — number of valid frames per sample
            # Build frame-level mask matching hidden state sequence length
            max_frames = hidden.shape[1]
            frame_mask = torch.arange(max_frames, device=hidden.device).unsqueeze(0)  # (1, frames)
            frame_mask = (frame_mask < feat_lengths.unsqueeze(1)).unsqueeze(-1).float()  # (batch, frames, 1)
            pooled = (hidden * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)
        return self.classifier(pooled)


# ── Dataset ────────────────────────────────────────────────────────────────────

class AudioDataset(Dataset):
    def __init__(self, paths: list, labels: list,
                 processor: Wav2Vec2Processor, augment: bool = False):
        self.paths     = paths
        self.labels    = labels
        self.processor = processor
        self.augment   = augment

    def __len__(self):
        return len(self.paths)

    def _load(self, path: Path):
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
            if len(waveform) > MAX_SAMPLES:
                waveform = waveform[:MAX_SAMPLES]
            return waveform
        except Exception:
            return None

    def _augment(self, wave: torch.Tensor) -> torch.Tensor:
        choice = np.random.randint(4)
        if choice == 0:   # noise
            wave = (wave + torch.randn_like(wave) * np.random.uniform(0.001, 0.02)).clamp(-1, 1)
        elif choice == 1: # volume
            wave = (wave * np.random.uniform(0.5, 1.0)).clamp(-1, 1)
        elif choice == 2: # speed
            factor  = np.random.uniform(0.92, 1.08)
            fake_sr = int(TARGET_SR * factor)
            out = AF.resample(wave.unsqueeze(0), TARGET_SR, fake_sr).squeeze(0)
            n   = len(wave)
            out = torch.nn.functional.pad(out, (0, max(0, n - len(out))))[:n]
            wave = out
        # choice 3 = no aug
        return wave

    def __getitem__(self, idx):
        wave = self._load(self.paths[idx])
        if wave is None:
            # Return silent audio as fallback
            wave = torch.zeros(TARGET_SR)

        if self.augment:
            wave = self._augment(wave)

        inputs = self.processor(
            wave.numpy(),
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=False,
        )
        return {
            "input_values": inputs.input_values.squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    """Pad variable-length sequences in batch."""
    input_values = [item["input_values"] for item in batch]
    labels       = torch.stack([item["label"] for item in batch])

    max_len = max(x.shape[0] for x in input_values)
    padded  = torch.zeros(len(input_values), max_len)
    mask    = torch.zeros(len(input_values), max_len, dtype=torch.long)

    for i, x in enumerate(input_values):
        padded[i, :x.shape[0]] = x
        mask[i, :x.shape[0]]   = 1

    return {"input_values": padded, "attention_mask": mask, "labels": labels}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("  CPU : No GPU — fine-tuning will be very slow")
    return torch.device("cpu")


def collect_files(data_dir: Path):
    paths, labels = [], []
    for lbl, name in [(0, "real"), (1, "fake")]:
        folder = data_dir / name
        if not folder.exists():
            print(f"  [WARN] {folder} not found")
            continue
        found = sorted(f for f in folder.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
        print(f"  {name:>4}/  {len(found):>6,} files")
        paths.extend(found)
        labels.extend([lbl] * len(found))
    return paths, labels


def run_epoch(model, loader, criterion, optimizer, device, train: bool, scaler=None):
    model.train(train)
    total_loss = correct = n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            iv   = batch["input_values"].to(device)
            am   = batch["attention_mask"].to(device)
            y    = batch["labels"].to(device)
            with autocast("cuda", enabled=(scaler is not None)):
                logits = model(iv, am)
                loss   = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            n          += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    targets, preds, probs = [], [], []
    for batch in loader:
        iv  = batch["input_values"].to(device)
        am  = batch["attention_mask"].to(device)
        y   = batch["labels"]
        logits = model(iv, am)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        targets.extend(y.numpy())
        preds.extend(logits.argmax(1).cpu().numpy())
        probs.extend(p)
    return np.array(targets), np.array(preds), np.array(probs)


def print_metrics(y_true, y_pred, y_prob, split="Test"):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*55}")
    print(f"  {split} Results")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print(f"  ROC-AUC   : {auc*100:.2f}%")
    print(f"\n  Confusion Matrix")
    print(f"                Pred Real   Pred Fake")
    print(f"  True Real  :  {cm[0,0]:>8,}   {cm[0,1]:>8,}")
    print(f"  True Fake  :  {cm[1,0]:>8,}   {cm[1,1]:>8,}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Real','Fake'])}")
    print(f"{'='*55}")
    return acc


def save_plot(trl, vll, tra, vaa):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(trl)+1)
    ax1.plot(ep, trl, label="Train"); ax1.plot(ep, vll, label="Val")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep, [a*100 for a in tra], label="Train")
    ax2.plot(ep, [a*100 for a in vaa], label="Val")
    ax2.set_title("Accuracy (%)"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle("EchoVox Fine-tuning History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("finetune_history.png", dpi=130)
    plt.close()
    print("  Plot saved → finetune_history.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    print("\n" + "="*55)
    print("  EchoVox — Wav2Vec2 Fine-tuning")
    print("="*55)

    device = get_device()

    # ── Load Wav2Vec2 ──────────────────────────────────────────────────────────
    print(f"\n  Loading {MODEL_NAME} …")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    wav2vec   = Wav2Vec2Model.from_pretrained(MODEL_NAME)

    model = EchoVoxFineTuned(
        wav2vec_model   = wav2vec,
        unfreeze_layers = args.unfreeze_layers,
        dropout         = args.dropout,
    ).to(device)

    # Count params
    total_params    = sum(p.numel() for p in model.parameters())
    trainable       = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen          = total_params - trainable
    print(f"\n  Wav2Vec2 layers  : 12 total, last {args.unfreeze_layers} unfrozen")
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Frozen params    : {frozen:,}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\n  Scanning {args.data_dir} …")
    paths, labels = collect_files(Path(args.data_dir))

    if not paths:
        print("  [ERROR] No audio files found.")
        return

    # Limit dataset size to speed up fine-tuning (optional)
    if args.max_samples and len(paths) > args.max_samples:
        idx = np.random.choice(len(paths), args.max_samples, replace=False)
        paths  = [paths[i]  for i in idx]
        labels = [labels[i] for i in idx]
        print(f"  Sampled {args.max_samples:,} files for faster fine-tuning")

    # ── Split ──────────────────────────────────────────────────────────────────
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        paths, labels, test_size=0.15, stratify=labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15/0.85, stratify=y_tmp, random_state=42)

    print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = AudioDataset(X_train, y_train, processor, augment=True)
    val_ds   = AudioDataset(X_val,   y_val,   processor, augment=False)
    test_ds  = AudioDataset(X_test,  y_test,  processor, augment=False)

    # num_workers=2: parallel CPU data loading while GPU trains
    nw = 2 if os.name != "nt" else 0   # Windows safe default
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=nw, pin_memory=(device.type=="cuda"),
                              persistent_workers=(nw>0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=nw, pin_memory=(device.type=="cuda"),
                              persistent_workers=(nw>0))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=nw, pin_memory=(device.type=="cuda"),
                              persistent_workers=(nw>0))

    # ── Loss & optimizer ──────────────────────────────────────────────────────
    counts = np.bincount(y_train)
    base_w = [len(y_train) / (2 * c) for c in counts]
    base_w[0] *= 2.0
    weights   = torch.tensor(base_w, dtype=torch.float32).to(device)
    print(f"  Class weights — Real: {weights[0]:.3f}  Fake: {weights[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=weights)

    # Two param groups: lower LR for Wav2Vec2, higher for classifier
    wav2vec_params    = [p for p in model.wav2vec.parameters()    if p.requires_grad]
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": wav2vec_params,    "lr": args.lr,        "weight_decay": 1e-2},
        {"params": classifier_params, "lr": args.lr * 10,   "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    print(f"\n  Epochs : {args.epochs}  Batch : {args.batch_size}")
    print(f"  LR (Wav2Vec2): {args.lr:.0e}  LR (classifier): {args.lr*10:.0e}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc      = 0.0
    best_epoch        = 0
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    header = f"{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  {'LR':>9}"
    print(header)
    print("-" * len(header))

    # ── fp16 GradScaler — ~2x faster on RTX 4060 ─────────────────────────────
    use_fp16 = (device.type == "cuda")
    scaler   = GradScaler("cuda") if use_fp16 else None
    if use_fp16:
        print("  fp16 mixed precision: ENABLED (faster training)")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True,  scaler=scaler)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      device, train=False, scaler=None)
        scheduler.step(va_loss)

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc);    val_accs.append(va_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_acc*100:>6.2f}%  "
              f"{va_loss:>8.4f}  {va_acc*100:>6.2f}%  {lr_now:>9.2e}")

        if va_acc > best_val_acc:
            best_val_acc      = va_acc
            best_epoch        = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.model)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {args.early_stop} epochs)")
                break

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min  |  "
          f"Best val acc : {best_val_acc*100:.2f}% @ epoch {best_epoch}")
    print(f"  Model saved → {args.model}")

    save_plot(train_losses, val_losses, train_accs, val_accs)

    # ── Test evaluation ────────────────────────────────────────────────────────
    print("\n  Loading best checkpoint for test evaluation …")
    model.load_state_dict(torch.load(args.model, map_location=device))
    y_true, y_pred, y_prob = predict_all(model, test_loader, device)
    print_metrics(y_true, y_pred, y_prob, split="Test")

    print("\n  Next: python real_world_eval_ft.py  (run real-world eval on fine-tuned model)\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",        default="data",              help="Root data folder")
    ap.add_argument("--model",           default="model_ft.pt",       help="Output model path")
    ap.add_argument("--epochs",          default=15,   type=int)
    ap.add_argument("--batch_size",      default=8,    type=int,      help="Keep low — audio loaded on-the-fly")
    ap.add_argument("--lr",              default=2e-5, type=float,    help="LR for Wav2Vec2 layers")
    ap.add_argument("--dropout",         default=0.5,  type=float)
    ap.add_argument("--unfreeze_layers", default=2,    type=int,      help="Last N transformer layers to unfreeze (2=faster, 4=more powerful)")
    ap.add_argument("--early_stop",      default=5,    type=int,      help="Early stopping patience")
    ap.add_argument("--max_samples",     default=6000, type=int,      help="Max files to use. 6000=balanced coverage, 10000=full")
    args = ap.parse_args()
    main(args)