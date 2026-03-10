"""
EchoVox — Streamlit Web App
Run: streamlit run app.py
"""

import io
import os
import tempfile
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

st.set_page_config(
    page_title="EchoVox",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background-color: #0a0d12; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }
  [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #1a2030; }
  .sidebar-logo { display: flex; align-items: center; gap: 12px; padding: 8px 0 24px 0; }
  .sidebar-logo-icon { width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #0d2d2d, #0a3d3d); border: 1px solid #1a4a4a; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; }
  .sidebar-logo-name { font-size: 1.05rem; font-weight: 700; color: #e6edf3; letter-spacing: 0.3px; }
  .sidebar-logo-sub { font-size: 0.65rem; font-weight: 600; color: #00b4b4; letter-spacing: 2px; text-transform: uppercase; }
  .sidebar-section { font-size: 0.65rem; font-weight: 600; color: #4a5568; letter-spacing: 2px; text-transform: uppercase; margin: 24px 0 12px 0; }
  .sidebar-step { display: flex; gap: 14px; align-items: flex-start; padding: 6px 0; color: #8b949e; font-size: 0.85rem; }
  .sidebar-step-num { font-size: 0.7rem; font-weight: 700; color: #00b4b4; min-width: 20px; padding-top: 1px; }
  .main-title { display: flex; align-items: center; gap: 12px; justify-content: center; margin-bottom: 4px; }
  .main-title-text { font-size: 2.4rem; font-weight: 800; color: #e6edf3; letter-spacing: -0.5px; }
  .main-subtitle { text-align: center; color: #4a5568; font-size: 0.95rem; margin-bottom: 28px; }
  .upload-zone { background: #0d1117; border: 1px solid #1a2030; border-radius: 16px; padding: 48px 24px; text-align: center; margin-bottom: 8px; }
  .upload-icon-wrap { width: 64px; height: 64px; border-radius: 16px; background: #0d2020; border: 1px solid #1a3a3a; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px auto; font-size: 1.8rem; }
  .upload-title { font-size: 1.1rem; font-weight: 600; color: #e6edf3; margin-bottom: 4px; }
  .upload-sub { font-size: 0.85rem; color: #4a5568; }
  .format-badges { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-top: 16px; }
  .format-badge { background: #0d1117; border: 1px solid #1a2030; color: #8b949e; font-size: 0.75rem; font-weight: 500; padding: 4px 12px; border-radius: 20px; }
  .verdict-real { background: linear-gradient(135deg, #0a1f0f, #0f2a18); border: 1px solid #1a4a2a; border-radius: 14px; padding: 20px 24px; display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }
  .verdict-fake { background: linear-gradient(135deg, #1f0a0a, #2a0f0f); border: 1px solid #4a1a1a; border-radius: 14px; padding: 20px 24px; display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }
  .verdict-icon-box { width: 52px; height: 52px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; flex-shrink: 0; }
  .verdict-icon-real { background: #0f3d1a; border: 1px solid #1a5a28; }
  .verdict-icon-fake { background: #3d0f0f; border: 1px solid #5a1a1a; }
  .verdict-label { font-size: 1.6rem; font-weight: 800; letter-spacing: 1px; margin: 0; }
  .verdict-meta { font-size: 0.82rem; color: #4a5568; margin: 2px 0 0 0; }
  .tile { background: #0d1117; border: 1px solid #1a2030; border-radius: 14px; padding: 20px; }
  .tile-icon { font-size: 1.1rem; margin-bottom: 10px; }
  .tile-value { font-size: 2rem; font-weight: 700; color: #e6edf3; margin: 0 0 4px 0; line-height: 1; }
  .tile-label { font-size: 0.78rem; color: #4a5568; margin: 0; }
  .prob-section { background: #0d1117; border: 1px solid #1a2030; border-radius: 14px; padding: 20px 24px; margin: 16px 0; }
  .prob-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .prob-title { font-size: 0.95rem; font-weight: 600; color: #e6edf3; }
  .risk-badge { font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 20px; color: #8b949e; }
  .prob-bar-bg { background: #1a2030; border-radius: 8px; height: 8px; position: relative; margin: 8px 0; }
  .prob-bar-fill { height: 8px; border-radius: 8px; }
  .prob-labels { display: flex; justify-content: space-between; font-size: 0.72rem; color: #4a5568; margin-top: 6px; }
  .viz-box { background: #0d1117; border: 1px solid #1a2030; border-radius: 14px; padding: 16px; margin: 8px 0; }
  .anomaly-item { background: #0f1318; border: 1px solid #1a2030; border-radius: 10px; padding: 13px 16px; margin: 8px 0; display: flex; align-items: center; gap: 12px; font-size: 0.88rem; color: #8b949e; }
  .section-title { font-size: 1rem; font-weight: 600; color: #e6edf3; margin: 20px 0 10px 0; }
  .stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; }
  .stTabs [data-baseweb="tab"] { background: #0d1117; border: 1px solid #1a2030; border-radius: 20px; color: #8b949e; padding: 6px 18px; font-size: 0.85rem; }
  .stTabs [aria-selected="true"] { background: #0d2d2d !important; color: #00b4b4 !important; border-color: #1a4a4a !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading EchoVox models…")
def load_pipeline(model_path="model.pt", scaler_path="scaler.pkl"):
    import torch, torch.nn as nn, joblib
    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    class DeepfakeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(768,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 2),
            )
        def forward(self, x): return self.net(x)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    wav2vec   = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    wav2vec.eval()
    clf = DeepfakeClassifier().to(device)
    clf.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    clf.eval()
    scaler = joblib.load(scaler_path)
    return processor, wav2vec, clf, scaler, device


def run_inference(audio_bytes, suffix):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes); tmp = f.name
    try:
        from predict import analyse
        return analyse(tmp)
    finally:
        os.unlink(tmp)


FIG_BG = "#0d1117"
AX_BG  = "#0a0d12"

def _fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); return buf.getvalue()

def plot_waveform(wave, sr):
    wave_np = np.array(wave)
    t = np.linspace(0, len(wave_np)/sr, len(wave_np))
    fig, ax = plt.subplots(figsize=(10,3), facecolor=FIG_BG)
    ax.set_facecolor(AX_BG)
    ax.plot(t, wave_np, color="#00d4d4", linewidth=0.7, alpha=0.95)
    ax.fill_between(t, wave_np, alpha=0.15, color="#00d4d4")
    ax.axhline(0, color="#1a2030", linewidth=0.5)
    ax.set_xlabel("Time (s)", color="#4a5568", fontsize=9)
    ax.set_ylabel("Amplitude", color="#4a5568", fontsize=9)
    ax.tick_params(colors="#4a5568", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2030")
    plt.tight_layout(pad=0.8)
    return _fig_to_png(fig)

def plot_spectrogram(audio_bytes, suffix):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes); tmp = f.name
    try:
        y, sr = librosa.load(tmp, sr=16_000, mono=True)
    finally:
        os.unlink(tmp)
    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10,3.2), facecolor=FIG_BG)
    ax.set_facecolor(AX_BG)
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="cool", fmax=8000)
    ax.tick_params(colors="#4a5568", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2030")
    cb = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cb.ax.yaxis.set_tick_params(color="#4a5568", labelsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#4a5568")
    plt.tight_layout(pad=0.8)
    return _fig_to_png(fig)



def plot_confusion_matrix() -> bytes:
    cm = np.array([[2211, 58], [163, 8077]])
    fig, ax = plt.subplots(figsize=(3.5, 2.8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    im = ax.imshow(cm, cmap="YlGn", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Real", "Pred Fake"], color="#8b949e", fontsize=8)
    ax.set_yticklabels(["True Real", "True Fake"], color="#8b949e", fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="#0a0d12" if cm[i,j] > 2000 else "#e6edf3",
                    fontsize=9, fontweight="700")
    ax.set_title("Confusion Matrix", color="#e6edf3", fontsize=9, pad=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout(pad=0.5)
    return _fig_to_png(fig)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div class="sidebar-logo">
  <div class="sidebar-logo-icon">🎙️</div>
  <div>
    <div class="sidebar-logo-name">EchoVox</div>
    <div class="sidebar-logo-sub">Voice Forensics</div>
  </div>
</div>
<div class="sidebar-section">How It Works</div>
<div class="sidebar-step"><span class="sidebar-step-num">01</span><span>Upload a voice audio sample</span></div>
<div class="sidebar-step"><span class="sidebar-step-num">02</span><span>AI extracts acoustic features and embeddings</span></div>
<div class="sidebar-step"><span class="sidebar-step-num">03</span><span>Deep classifier detects synthetic artifacts</span></div>
<div class="sidebar-step"><span class="sidebar-step-num">04</span><span>Forensic report generated with confidence scores</span></div>
<div class="sidebar-section" style="margin-top:32px;">Model Status</div>
""", unsafe_allow_html=True)
    model_ok  = os.path.exists("model.pt")
    scaler_ok = os.path.exists("scaler.pkl")
    st.markdown(f"{'✅' if model_ok  else '❌'} model.pt")
    st.markdown(f"{'✅' if scaler_ok else '❌'} scaler.pkl")
    if not (model_ok and scaler_ok):
        st.error("Run `python train.py` first.")
    st.markdown('<div class="sidebar-section" style="margin-top:24px;">Model Performance</div>', unsafe_allow_html=True)
    st.image(plot_confusion_matrix(), use_container_width=True)
    st.markdown("""
<div style="font-size:0.72rem; color:#4a5568; margin-top:4px;">
  Accuracy: 97.9% &nbsp;|&nbsp; AUC: 99.0%<br>
  Real-world: 94.4% &nbsp;|&nbsp; Modern TTS: 58%
</div>""", unsafe_allow_html=True)



# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-title">
  <span style="font-size:2rem;">🎙️</span>
  <span class="main-title-text">EchoVox</span>
</div>
<div class="main-subtitle">AI Voice Authenticity Detection</div>
""", unsafe_allow_html=True)

if not (os.path.exists("model.pt") and os.path.exists("scaler.pkl")):
    st.warning("⚠️ Model not ready. Run `python train.py` then restart.")
    st.stop()

# ── Upload zone ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="upload-zone">
  <div class="upload-icon-wrap">⬆️</div>
  <div class="upload-title">Drop audio file here or click to upload</div>
  <div class="upload-sub">Upload a voice sample to analyse authenticity.</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("audio", type=["wav","mp3","m4a","flac","ogg"], label_visibility="collapsed")

st.markdown("""
<div class="format-badges">
  <span class="format-badge">WAV</span><span class="format-badge">MP3</span>
  <span class="format-badge">FLAC</span><span class="format-badge">M4A</span>
  <span class="format-badge">OGG</span>
</div>
""", unsafe_allow_html=True)

if not uploaded:
    st.stop()

audio_bytes = uploaded.read()
suffix      = "." + uploaded.name.rsplit(".", 1)[-1].lower()
fname       = uploaded.name

col_play, _ = st.columns([1, 2])
with col_play:
    st.audio(audio_bytes, format=f"audio/{suffix.lstrip('.')}")

with st.spinner("🔍 Analysing voice authenticity…"):
    try:
        result = run_inference(audio_bytes, suffix)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

is_fake    = result["is_fake"]
confidence = result["confidence"]
fake_prob  = result["fake_prob"]
real_prob  = result["real_prob"]
anomalies  = result["anomalies"]
duration   = result["duration"]

try:
    secs = float(duration)
    dur_fmt = f"0:{int(secs):02d}"
except:
    dur_fmt = f"{duration}s"

# ── Verdict ────────────────────────────────────────────────────────────────────
if is_fake:
    st.markdown(f"""
<div class="verdict-fake">
  <div class="verdict-icon-box verdict-icon-fake">⚠️</div>
  <div>
    <p class="verdict-label" style="color:#ff4d4d;">DEEPFAKE VOICE</p>
    <p class="verdict-meta">{fname} · {dur_fmt}</p>
  </div>
</div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
<div class="verdict-real">
  <div class="verdict-icon-box verdict-icon-real">🛡️</div>
  <div>
    <p class="verdict-label" style="color:#00e676;">REAL VOICE</p>
    <p class="verdict-meta">{fname} · {dur_fmt}</p>
  </div>
</div>""", unsafe_allow_html=True)

# ── 2x2 tiles ──────────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.markdown(f'<div class="tile"><div class="tile-icon" style="color:#00b4b4;">🛡️</div><p class="tile-value">{confidence:.0f}%</p><p class="tile-label">Confidence</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="tile"><div class="tile-icon" style="color:#ff4d4d;">⚠️</div><p class="tile-value">{fake_prob:.0f}%</p><p class="tile-label">Deepfake Probability</p></div>', unsafe_allow_html=True)

st.markdown("&nbsp;", unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    st.markdown(f'<div class="tile"><div class="tile-icon" style="color:#00e676;">✅</div><p class="tile-value">{real_prob:.0f}%</p><p class="tile-label">Real Voice Probability</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="tile"><div class="tile-icon" style="color:#9b59b6;">🕐</div><p class="tile-value">{dur_fmt}</p><p class="tile-label">Duration</p></div>', unsafe_allow_html=True)

# ── Probability bar ────────────────────────────────────────────────────────────
if fake_prob < 35:
    risk_label, bar_color = "Low Risk", "#00e676"
elif fake_prob < 65:
    risk_label, bar_color = "Medium Risk", "#f39c12"
elif fake_prob < 85:
    risk_label, bar_color = "High Risk", "#e67e22"
else:
    risk_label, bar_color = "Critical Risk", "#ff4d4d"

st.markdown(f"""
<div class="prob-section">
  <div class="prob-header">
    <span class="prob-title">Deepfake Probability</span>
    <span class="risk-badge">{risk_label}</span>
  </div>
  <div class="prob-bar-bg">
    <div class="prob-bar-fill" style="width:{fake_prob:.1f}%; background:{bar_color};"></div>
  </div>
  <div class="prob-labels">
    <span>0%</span>
    <span style="color:{bar_color}; font-weight:600;">{fake_prob:.0f}%</span>
    <span>100%</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Visualisations ─────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Waveform", "Mel Spectrogram"])
with tab1:
    st.markdown('<div class="viz-box">', unsafe_allow_html=True)
    st.image(plot_waveform(result["waveform"], result["sample_rate"]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with tab2:
    st.markdown('<div class="viz-box">', unsafe_allow_html=True)
    st.image(plot_spectrogram(audio_bytes, suffix), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Anomalies ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Detected Anomalies</div>', unsafe_allow_html=True)
if anomalies:
    for a in anomalies:
        st.markdown(f'<div class="anomaly-item"><span>⚠️</span><span>{a}</span></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="anomaly-item"><span>✅</span><span>No anomalies detected — acoustic patterns appear natural.</span></div>', unsafe_allow_html=True)