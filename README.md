# EchoVox

An AI-powered deepfake voice detection system that distinguishes real human speech from AI-generated audio using fine-tuned Wav2Vec2 + MLP classification.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

---

## What It Does

EchoVox analyzes audio files and determines whether the voice is real or AI-generated (TTS/voice cloning). It works with:
- ElevenLabs voice clones
- Respeecher synthesized speech
- ASVspoof spoofed audio
- Other modern TTS systems

---

## Architecture

```
Audio Input
    |
Wav2Vec2-base (facebook/wav2vec2-base)
    | mean pooling → 768-dim embeddings
MLP Classifier
    Linear(768 → 256) → BatchNorm → ReLU → Dropout
    Linear(256 → 64)  → BatchNorm → ReLU → Dropout
    Linear(64 → 2)    → Softmax
    |
Real / Fake
```

Two models are available:
- **model.pt** — Frozen Wav2Vec2 feature extractor + trained MLP (fast inference)
- **model_ft.pt** — End-to-end fine-tuned Wav2Vec2 + MLP (higher accuracy)

---

## Performance

| Metric | model.pt | model_ft.pt |
|--------|----------|-------------|
| Test Accuracy | 94.32% | 90.67% (val) |
| Precision | 96.99% | — |
| Recall | 92.18% | — |
| F1 Score | 94.52% | — |
| ROC-AUC | 99.02% | — |

**Unseen data evaluation (honest):**

| Dataset | model.pt | model_ft.pt |
|---------|----------|-------------|
| Mendeley (ElevenLabs + Respeecher) | 91.10% | 92.92% |
| In-The-Wild (celebrity voices) | 63.30% | 69.90% |

---

## Training Data

| Dataset | Type | Files |
|---------|------|-------|
| ASVspoof 2019 | Real + Legacy fake WAV | ~30,000 |
| CommonVoice Spontaneous Speech | Real MP3 | 1,368 |
| ElevenLabs / Respeecher (Mendeley) | Modern fake WAV | 600 |
| WaveFake | Modern fake WAV | ~30GB |

Total embeddings after augmentation: 140,804

---

## Setup

### Prerequisites
- Python 3.12
- NVIDIA GPU (recommended — tested on RTX 4060 8GB)
- ffmpeg

### Install

```bash
git clone https://github.com/tarunchowdari/EchoVox.git
cd EchoVox
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Web UI (Streamlit)

```bash
streamlit run app.py
```
Open http://localhost:8501 and upload an audio file.

### Command Line

```bash
python predict.py --audio path/to/audio.wav
```

### Evaluate on your own data

```bash
python unseen_eval.py
```

---

## Training Pipeline

### Step 1 — Extract features
```bash
python extract_features.py
```

### Step 2 — Train MLP classifier
```bash
python train.py
```

### Step 3 — Fine-tune Wav2Vec2 end-to-end
```bash
python finetune.py --epochs 15 --batch_size 32
```

### Step 4 — Evaluate
```bash
python real_world_eval.py        # baseline model
python real_world_eval_ft.py     # fine-tuned model
python unseen_eval.py            # honest unseen data test
```

---

## Project Structure

```
EchoVox/
├── app.py                  # Streamlit web UI
├── predict.py              # Single file inference
├── train.py                # MLP training
├── finetune.py             # End-to-end Wav2Vec2 fine-tuning
├── extract_features.py     # Wav2Vec2 feature extraction
├── real_world_eval.py      # Evaluation (baseline model)
├── real_world_eval_ft.py   # Evaluation (fine-tuned model)
├── unseen_eval.py          # Honest unseen data evaluation
├── balance_dataset.py      # Dataset balancing utility
├── requirements.txt        # Dependencies
└── data/
    ├── real/               # Real voice audio files
    └── fake/               # Fake/synthetic audio files
```

---

## Honest Limitations

- Real voice detection on truly unseen data: 50-70% (work in progress)
- Model performs best on audio similar to ASVspoof 2019 distribution
- Background noise can affect classification confidence
- Ongoing work: adding more diverse real voice data and noise augmentation

---

## Roadmap

- [x] Phase 1 — Frozen Wav2Vec2 + MLP baseline
- [x] Phase 2 — End-to-end Wav2Vec2 fine-tuning
- [x] Phase 3 — CommonVoice + data augmentation
- [ ] Phase 4 — WaveFake dataset integration
- [ ] Phase 5 — Real-time microphone input
- [ ] Phase 6 — Multi-language support

---

## Acknowledgements

- [ASVspoof 2019](https://www.asvspoof.org/) — spoofed speech dataset
- [Mozilla CommonVoice](https://commonvoice.mozilla.org/) — real voice dataset
- [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) — pretrained model
- [Mendeley Fake Audio Dataset](https://data.mendeley.com/datasets/79g59sp69z/1) — ElevenLabs + Respeecher fakes
- [In-The-Wild Dataset](https://deepfake-total.com/in_the_wild) — celebrity deepfake audio
