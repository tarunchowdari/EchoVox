"""
EchoVox — Step 2: Train MLP Classifier
==================================================
Loads features.npz, normalises with StandardScaler, trains a 3-layer
MLP (768 → 256 → 64 → 2) on GPU, prints full metrics, saves model.pt
and scaler.pkl.

Usage:
    python train.py
    python train.py --features features.npz --epochs 30 --lr 1e-4 --batch_size 256
"""

import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Model ──────────────────────────────────────────────────────────────────────

class DeepfakeClassifier(nn.Module):
    """
    3-layer MLP with BatchNorm and Dropout.
    Architecture: 768 → 256 → 64 → 2
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("  CPU : No GPU found")
    return torch.device("cpu")


def make_loader(X, y, batch_size: int, shuffle: bool, pin: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=pin)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    model.train(train)
    total_loss = correct = n = 0
    with torch.set_grad_enabled(train):
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_b)
            correct    += (logits.argmax(1) == y_b).sum().item()
            n          += len(y_b)
    return total_loss / n, correct / n


@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    targets, preds, probs = [], [], []
    for X_b, y_b in loader:
        logits = model(X_b.to(device))
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        targets.extend(y_b.numpy())
        preds.extend(logits.argmax(1).cpu().numpy())
        probs.extend(p)
    return np.array(targets), np.array(preds), np.array(probs)


def print_metrics(y_true, y_pred, y_prob, split: str = "Test") -> float:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 55}")
    print(f"  {split} Results")
    print(f"{'=' * 55}")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec * 100:.2f}%")
    print(f"  F1 Score  : {f1 * 100:.2f}%")
    print(f"  ROC-AUC   : {auc * 100:.2f}%")
    print(f"\n  Confusion Matrix")
    print(f"                Pred Real   Pred Fake")
    print(f"  True Real  :  {cm[0, 0]:>8,}   {cm[0, 1]:>8,}")
    print(f"  True Fake  :  {cm[1, 0]:>8,}   {cm[1, 1]:>8,}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Real', 'Fake'])}")
    print(f"{'=' * 55}")
    return acc


def save_training_plot(train_losses, val_losses, train_accs, val_accs) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(train_losses) + 1)

    ax1.plot(ep, train_losses, label="Train")
    ax1.plot(ep, val_losses,   label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep, [a * 100 for a in train_accs], label="Train")
    ax2.plot(ep, [a * 100 for a in val_accs],   label="Val")
    ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("EchoVox — Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=130)
    plt.close()
    print("  Plot saved → training_history.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args) -> None:
    print("\n" + "=" * 55)
    print("  EchoVox — Training")
    print("=" * 55)

    device = get_device()

    # ── Load features ──────────────────────────────────────────────────────────
    print(f"\n  Loading {args.features} …")
    data = np.load(args.features)
    X    = data["X"].astype(np.float32)
    y    = data["y"].astype(np.int64)
    print(f"  Shape : {X.shape}   Real : {(y==0).sum():,}   Fake : {(y==1).sum():,}")

    # ── Split 70 / 15 / 15 ────────────────────────────────────────────────────
    X_tmp,  X_test,  y_tmp,  y_test  = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val   = train_test_split(
        X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=42)

    print(f"  Train : {len(y_train):,}   Val : {len(y_val):,}   Test : {len(y_test):,}")

    # ── Normalise ──────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    joblib.dump(scaler, args.scaler)
    print(f"  Scaler saved → {args.scaler}")

    # ── DataLoaders ────────────────────────────────────────────────────────────
    pin = device.type == "cuda"
    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True,  pin=pin)
    val_loader   = make_loader(X_val,   y_val,   args.batch_size, shuffle=False, pin=pin)
    test_loader  = make_loader(X_test,  y_test,  args.batch_size, shuffle=False, pin=pin)

    # ── Model + loss with class-weight for imbalance ───────────────────────────
    model = DeepfakeClassifier(dropout=args.dropout).to(device)

    counts  = np.bincount(y_train)
    # Boost real weight extra to counter heavy fake imbalance (3.6:1 fake:real)
    base_weights = [len(y_train) / (2 * c) for c in counts]
    # Apply 2x multiplier to real class weight
    base_weights[0] *= 2.0
    weights = torch.tensor(base_weights, dtype=torch.float32).to(device)
    print(f"  Class weights — Real : {weights[0]:.3f}   Fake : {weights[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    early_stop_patience = args.early_stop
    epochs_no_improve   = 0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model params : {n_params:,}")
    print(f"  Epochs : {args.epochs}   LR : {args.lr}   Batch : {args.batch_size}\n")

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    best_epoch     = 0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    header = f"{'Epoch':>6}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  {'LR':>9}"
    print(header)
    print("-" * len(header))

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      device, train=False)
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
            if epochs_no_improve >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed / 60:.1f} min  |  "
          f"Best val acc : {best_val_acc * 100:.2f}% @ epoch {best_epoch}")
    print(f"  Model saved → {args.model}")

    save_training_plot(train_losses, val_losses, train_accs, val_accs)

    # ── Test evaluation ────────────────────────────────────────────────────────
    print("\n  Loading best checkpoint for final test evaluation …")
    model.load_state_dict(torch.load(args.model, map_location=device))
    y_true, y_pred, y_prob = predict_all(model, test_loader, device)
    test_acc = print_metrics(y_true, y_pred, y_prob, split="Test")

    if test_acc >= 0.92:
        print(f"  ✓ Target accuracy (92 %) ACHIEVED — {test_acc*100:.2f}%\n")
    else:
        print(f"  ✗ Accuracy {test_acc*100:.1f}% below 92% target.\n"
              f"    Try --epochs 40, or add more training data.\n")

    print("  Next: python predict.py --audio path/to/file.wav\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",   default="features.npz",  help="Input .npz file")
    ap.add_argument("--model",      default="model.pt",       help="Output model path")
    ap.add_argument("--scaler",     default="scaler.pkl",     help="Output scaler path")
    ap.add_argument("--epochs",     default=30,  type=int)
    ap.add_argument("--batch_size", default=256, type=int)
    ap.add_argument("--lr",         default=1e-4, type=float)
    ap.add_argument("--dropout",    default=0.5,  type=float)
    ap.add_argument("--early_stop", default=7,    type=int,   help="Early stopping patience epochs")
    args = ap.parse_args()
    main(args)