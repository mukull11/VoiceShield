import streamlit as st
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import tempfile
import warnings

warnings.filterwarnings("ignore")

from config import (
    SAMPLE_RATE,
    TARGET_LENGTH,
    INFERENCE_THRESHOLD,
    BEST_MODEL_PATH,
    DEMO_REAL,
    DEMO_FAKE_ELEVENLABS,
    DEMO_FAKE_RVC,
    DEVICE,
)
from src.model import create_model, load_model_checkpoint
from src.preprocess import preprocess_for_inference
from src.features import extract_features_for_inference, visualize_waveform_and_spectrogram

st.set_page_config(
    page_title="VoiceShield - AI Voice Clone Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model():
    """Load the trained model."""
    device = torch.device(DEVICE)
    model = create_model().to(device)
    model = load_model_checkpoint(model, str(BEST_MODEL_PATH), device)
    return model, device

def preprocess_audio(audio_bytes, sr=None):
    """Preprocess uploaded audio file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name
    
    try:
        audio, sample_rate = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        
        if sample_rate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        
        audio = preprocess_for_inference(audio, SAMPLE_RATE)
        
        return audio, SAMPLE_RATE
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def predict_audio(model, device, audio):
    """Run inference on preprocessed audio."""
    spec = extract_features_for_inference(audio, SAMPLE_RATE)
    spec_tensor = torch.from_numpy(spec).float().unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        logit = model(spec_tensor)
        prob = torch.sigmoid(logit).item()
    
    is_spoof = prob > INFERENCE_THRESHOLD
    label = "AI-GENERATED ⚠️" if is_spoof else "REAL ✓"
    confidence = abs(0.5 - prob) * 200
    
    return label, prob, confidence, spec, is_spoof

def plot_waveform(audio, sr=SAMPLE_RATE):
    """Create waveform plot."""
    fig, ax = plt.subplots(figsize=(10, 2))
    time_axis = np.arange(len(audio)) / sr
    ax.plot(time_axis, audio, color='#1f77b4', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_spectrogram(spec):
    """Create spectrogram plot."""
    if spec.ndim == 3:
        spec = spec[0]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spec, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(im, ax=ax, format='%+2.0f dB')
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Mel Bands')
    ax.set_title('Mel-Spectrogram')
    plt.tight_layout()
    return fig

def create_confidence_gauge(confidence, is_spoof):
    """Create a confidence gauge visualization."""
    color = "#ff4444" if is_spoof else "#00cc66"
    
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh([0], [confidence], color=color, height=0.5)
    ax.barh([0], [100 - confidence], left=[confidence], color='#e0e0e0', height=0.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Confidence (%)')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    st.title("🛡️ VoiceShield - AI Voice Clone Detector")
    st.markdown("**Upload an audio clip to detect if it's a real human voice or AI-generated**")
    
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **VoiceShield** detects AI-generated and cloned voices using deep learning.
        
        **Model:** EfficientNet-B0  
        **Training Data:** ASVspoof 2021 LA (Logical Access)  
        **Input:** 4-second audio clips at 16kHz  
        **Features:** Mel-spectrograms (128 mel bands)
        
        **Classes:**
        - ✅ **REAL** - Bonafide human speech
        - ⚠️ **AI-GENERATED** - Spoofed/cloned speech
        """)
        
        st.header("📊 Model Info")
        st.markdown("""
        - **Architecture:** EfficientNet-B0 (Transfer Learning)
        - **Loss:** BCEWithLogitsLoss
        - **Optimizer:** Adam (lr=1e-4)
        - **Class Weights:** Bonafide=1.0, Spoof=3.0
        - **Threshold:** 0.5
        """)
    
    tab1, tab2 = st.tabs(["🎤 Upload & Predict", "🎵 Demo Examples"])
    
    with tab1:
        st.header("Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
            help="Supported formats: WAV, MP3, FLAC, OGG, M4A"
        )
        
        if uploaded_file is not None:
            with st.spinner("Processing audio..."):
                audio_bytes = uploaded_file.read()
                audio, sr = preprocess_audio(audio_bytes)
                
                if len(audio) == 0:
                    st.error("Failed to process audio file. Please try another file.")
                    return
                
                model, device = load_model()
                
                label, prob, confidence, spec, is_spoof = predict_audio(model, device, audio)
            
            st.success("Analysis complete!")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 Visualizations")
                
                wave_fig = plot_waveform(audio)
                st.pyplot(wave_fig)
                
                spec_fig = plot_spectrogram(spec)
                st.pyplot(spec_fig)
            
            with col2:
                st.subheader("🎯 Result")
                
                if is_spoof:
                    st.error(f"## {label}")
                else:
                    st.success(f"## {label}")
                
                st.metric("AI Probability", f"{prob*100:.1f}%")
                st.metric("Confidence", f"{confidence:.1f}%")
                
                gauge_fig = create_confidence_gauge(confidence, is_spoof)
                st.pyplot(gauge_fig)
                
                st.markdown("---")
                st.markdown("**Interpretation:**")
                if is_spoof:
                    st.markdown("""
                    ⚠️ **This audio appears to be AI-generated or cloned.**
                    
                    The model detected characteristics consistent with synthetic speech.
                    Exercise caution if this audio is used for authentication or verification.
                    """)
                else:
                    st.markdown("""
                    ✅ **This audio appears to be from a real human.**
                    
                    The model detected natural speech characteristics.
                    However, no detector is 100% accurate - always verify through multiple channels.
                    """)
    
    with tab2:
        st.header("Demo Examples")
        st.markdown("Click any button to test with pre-loaded samples")
        
        demo_files = {
            "Real Human Voice": DEMO_REAL,
            "AI-Generated (ElevenLabs)": DEMO_FAKE_ELEVENLABS,
            "AI-Generated (RVC)": DEMO_FAKE_RVC,
        }
        
        model, device = load_model()
        
        cols = st.columns(3)
        
        for idx, (name, path) in enumerate(demo_files.items()):
            with cols[idx]:
                st.subheader(name)
                
                if path.exists():
                    if st.button(f"Analyze {name}", key=f"demo_{idx}"):
                        with st.spinner(f"Analyzing {name}..."):
                            audio, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
                            audio = preprocess_for_inference(audio, sr)
                            
                            label, prob, confidence, spec, is_spoof = predict_audio(model, device, audio)
                        
                        st.success("Done!")
                        
                        wave_fig = plot_waveform(audio)
                        st.pyplot(wave_fig)
                        
                        spec_fig = plot_spectrogram(spec)
                        st.pyplot(spec_fig)
                        
                        if is_spoof:
                            st.error(f"## {label}")
                        else:
                            st.success(f"## {label}")
                        
                        st.metric("AI Probability", f"{prob*100:.1f}%")
                        st.metric("Confidence", f"{confidence:.1f}%")
                else:
                    st.warning(f"Demo file not found: {path.name}")
                    st.info("Add demo samples to `demo_samples/` folder")

if __name__ == "__main__":
    main()