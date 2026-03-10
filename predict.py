"""
VeriVoice Sentinel — Step 3: Inference
=======================================
Load model.pt + scaler.pkl, analyse any audio file, return a structured
result dict used by both CLI and app.py.

CLI usage:
    python predict.py --audio bank_call.wav
    python predict.py --audio suspicious.mp3 --json
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import joblib

TARGET_SR  = 16_000
MODEL_NAME = "facebook/wav2vec2-base"


# ── Model — must match train.py exactly ───────────────────────────────────────

class DeepfakeClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3):
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


# ── Audio helpers ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audio(path: str) -> tuple[torch.Tensor, float]:
    """Return (mono waveform tensor, duration_seconds) at TARGET_SR."""
    waveform, sr = torchaudio.load(path)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    waveform = waveform.mean(dim=0)                         # → 1-D
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    duration = len(waveform) / TARGET_SR
    return waveform, round(duration, 2)


@torch.no_grad()
def wav2vec_embed(
    waveform: torch.Tensor,
    processor,
    wav2vec_model,
    device: torch.device,
) -> np.ndarray:
    """Extract 768-dim embedding via Wav2Vec2."""
    inputs = processor(
        waveform.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = wav2vec_model(**inputs)
    return out.last_hidden_state.mean(dim=1).cpu().numpy()   # (1, 768)


# ── Anomaly detection ─────────────────────────────────────────────────────────

def detect_anomalies(wave: np.ndarray, fake_prob: float) -> list[str]:
    """Heuristic acoustic checks that explain the verdict to the user."""
    flags = []

    # Zero-crossing rate — too-low ZCR → unnatural pitch consistency
    zcr = (np.diff(np.sign(wave)) != 0).mean()
    if zcr < 0.08:
        flags.append(
            f"Unnatural pitch consistency — ZCR {zcr:.3f} (threshold < 0.08)"
        )

    # Amplitude flatness (std of absolute amplitude)
    flatness = float(np.abs(wave).std())
    if flatness > 0.15:
        flags.append(
            f"Repetitive spectral artifacts — flatness {flatness:.3f} (threshold > 0.15)"
        )

    # Silence ratio — real speech has breathing pauses
    silence_ratio = float((np.abs(wave) < 0.02).mean())
    if silence_ratio < 0.05:
        flags.append(
            f"Lack of natural breathing pauses — silence ratio {silence_ratio:.3f}"
        )

    # Frame-level energy variance — TTS is too consistent
    hop = TARGET_SR // 10                           # 100 ms frames
    energies = [
        float(np.mean(wave[i : i + hop] ** 2))
        for i in range(0, len(wave) - hop, hop)
    ]
    if energies:
        energy_std = float(np.std(energies))
        if energy_std < 0.005 and fake_prob > 0.5:
            flags.append(
                f"Uniform frame energy — std {energy_std:.5f} (too consistent for real speech)"
            )

    if not flags and fake_prob > 0.5:
        flags.append(
            "Neural embedding pattern inconsistent with natural human speech"
        )

    return flags


# ── Public API (called by app.py) ─────────────────────────────────────────────

def analyse(
    audio_path: str,
    model_path: str  = "model.pt",
    scaler_path: str = "scaler.pkl",
) -> dict:
    """
    Full inference pipeline.

    Returns
    -------
    dict with keys:
        file, duration, verdict, is_fake, confidence,
        real_prob, fake_prob, anomalies,
        waveform (list[float]), mfcc (list[list[float]]), sample_rate
    """
    for p, label in [(model_path, "model.pt"), (scaler_path, "scaler.pkl")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label} not found at '{p}'. Run train.py first.")

    device = get_device()

    # Load models
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    processor     = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    wav2vec_model = Wav2Vec2Model.from_pretrained(MODEL_NAME).to(device)
    wav2vec_model.eval()

    clf    = DeepfakeClassifier().to(device)
    clf.load_state_dict(torch.load(model_path, map_location=device))
    clf.eval()

    scaler = joblib.load(scaler_path)

    # Audio
    waveform, duration = load_audio(audio_path)
    wave_np = waveform.numpy()

    # Embed → scale → classify
    emb        = wav2vec_embed(waveform, processor, wav2vec_model, device)
    emb_scaled = scaler.transform(emb).astype(np.float32)

    with torch.no_grad():
        logits = clf(torch.tensor(emb_scaled).to(device))
        temperature = 4.0
        probs  = torch.softmax(logits / temperature, dim=1).cpu().numpy()[0]

    real_prob  = float(probs[0])
    fake_prob  = float(probs[1])
    is_fake    = fake_prob > 0.35
    verdict    = "DEEPFAKE VOICE" if is_fake else "REAL VOICE"
    confidence = max(real_prob, fake_prob) * 100

    # MFCC (13 coefficients) — for visualisation only
    mfcc_t = torchaudio.transforms.MFCC(
        sample_rate=TARGET_SR,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40},
    )
    mfcc = mfcc_t(waveform.unsqueeze(0)).squeeze(0).numpy()   # (13, T)

    return {
        "file":        os.path.basename(audio_path),
        "duration":    duration,
        "verdict":     verdict,
        "is_fake":     is_fake,
        "confidence":  round(confidence, 1),
        "real_prob":   round(real_prob * 100, 1),
        "fake_prob":   round(fake_prob * 100, 1),
        "anomalies":   detect_anomalies(wave_np, fake_prob),
        "waveform":    wave_np.tolist(),
        "mfcc":        mfcc.tolist(),
        "sample_rate": TARGET_SR,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="VeriVoice Sentinel — Predict")
    ap.add_argument("--audio",  required=True,        help="Audio file to analyse")
    ap.add_argument("--model",  default="model.pt",   help="Trained model checkpoint")
    ap.add_argument("--scaler", default="scaler.pkl", help="Saved StandardScaler")
    ap.add_argument("--json",   action="store_true",  help="Print raw JSON (no arrays)")
    args = ap.parse_args()

    result = analyse(args.audio, args.model, args.scaler)

    if args.json:
        slim = {k: v for k, v in result.items() if k not in ("waveform", "mfcc")}
        print(json.dumps(slim, indent=2))
        return

    colour_on  = "\033[91m" if result["is_fake"] else "\033[92m"
    colour_off = "\033[0m"

    print(f"\n{'=' * 55}")
    print(f"  File       : {result['file']} ({result['duration']} s)")
    print(f"\n  {colour_on}PREDICTION : {result['verdict']}{colour_off}")
    print(f"  CONFIDENCE : {result['confidence']:.1f} %")
    print(f"  DEEPFAKE P : {result['fake_prob']:.1f} %")

    if result["anomalies"]:
        print(f"\n  ANOMALIES DETECTED :")
        for a in result["anomalies"]:
            print(f"    • {a}")

    if result["is_fake"]:
        print("\n  RECOMMENDATION : Flag call for manual review.")
        print("  Do not authorise any financial transaction based on this voice.")
    else:
        print("\n  RECOMMENDATION : Voice appears authentic.")

    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
