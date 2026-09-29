# 🛡️ VoiceShield - AI Voice Clone & Synthetic Speech Detector

> **Detect AI-generated and cloned voices in real-time using deep learning**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Problem Statement

India faces a rising tide of **voice fraud** — from AI-powered vishing (voice phishing) calls impersonating bank officials, to deepfake audio used in social engineering attacks. With tools like ElevenLabs, RVC, and Voice.ai making voice cloning accessible to anyone, traditional voice authentication is no longer secure.

**VoiceShield** addresses this by providing an **AI-powered detector** that distinguishes real human speech from synthetic/cloned voices with high accuracy.

---

## 💡 Solution Overview

VoiceShield is an **end-to-end ML pipeline** that:
1. **Preprocesses** audio to 16kHz, 4-second fixed-length segments
2. **Extracts** mel-spectrograms (128 mel bands) as input features
3. **Classifies** using EfficientNet-B0 (transfer learning from ImageNet)
4. **Deploys** via Streamlit for real-time inference

---

## 📊 Dataset: ASVspoof 2021 LA (Logical Access)

| Split | Samples | Bonafide | Spoof |
|-------|---------|----------|-------|
| Train | ~25,000 | ~2,500   | ~22,500 |
| Dev   | ~25,000 | ~2,500   | ~22,500 |
| Eval  | ~70,000 | ~7,000   | ~63,000 |

**Spoof Types:** A07-A19 (various TTS and VC systems)

---

## 🏗️ Architecture

```
Audio (16kHz, 4s) 
    │
    ▼
┌─────────────────────┐
│  Preprocessing      │  ← Trim silence, pad/trim to 64k samples
│  (librosa)          │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Mel-Spectrogram    │  ← 128 mel bands, 400 FFT, 160 hop
│  (128 × 251)        │     Normalized to zero mean, unit variance
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  EfficientNet-B0    │  ← Pretrained on ImageNet
│  (Transfer Learning)│     Frozen backbone + new classifier
│  Output: 1 logit    │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Sigmoid → Prob     │  ← P(spoof) > 0.5 → "AI-GENERATED"
└─────────────────────┘
```

---

## 📈 Results (After Training)

| Metric | Value |
|--------|-------|
| **Accuracy** | XX.XX% |
| **F1-Score** | XX.XX% |
| **Precision** | XX.XX% |
| **Recall** | XX.XX% |
| **Specificity** | XX.XX% |
| **ROC-AUC** | XX.XX% |
| **EER** | X.XX% |

> *Fill in after training completes*

---

## 📁 Folder Structure

```
voiceshield/
├── src/
│   ├── preprocess.py      # Audio loading, resampling, trimming, fixed length
│   ├── features.py        # Mel-spectrogram extraction + save .npy
│   ├── dataset.py         # PyTorch Dataset class
│   ├── model.py           # CNN architecture (EfficientNet-B0)
│   ├── train.py           # Training loop + save best weights
│   └── evaluate.py        # Accuracy, EER, confusion matrix, ROC curve
├── data/
│   ├── raw/               (gitignored - ASVspoof dataset here)
│   ├── processed/         (gitignored - .npy spectrograms)
│   └── splits/            (gitignored - train/val/test indices)
├── models/
│   └── best_model.pth     (gitignored - saved weights)
├── demo_samples/
│   ├── real_voice.wav
│   ├── fake_elevenlabs.wav
│   └── fake_rvc.wav
├── app.py                 # Streamlit UI for inference
├── requirements.txt       # All dependencies
├── config.py              # Paths & hyperparameters
├── README.md              # This file
└── .gitignore             # Ignore data/models/processed files
```

---

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd voiceshield
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download ASVspoof 2021 LA Dataset
Download from: https://www.asvspoof.org/asvspoof2021/asvspoof2021.html

Place in Google Drive (for Colab) or local `data/raw/`:
```
data/raw/
├── ASVspoof2019_LA/
│   ├── train/flac/
│   └── dev/flac/
├── ASVspoof2021_LA/
│   └── eval/flac/
└── keys/LA/CM/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2021.LA.cm.eval.trl.txt
```

### 4. Update Config Paths
Edit `config.py` to match your dataset location:
```python
DATA_ROOT = Path("/content/drive/MyDrive/voiceshield_data")  # Colab
# OR
DATA_ROOT = Path("./data/raw")  # Local
```

---

## 🏃 How to Use

### Training (Google Colab Recommended)
```bash
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Run preprocessing
python src/preprocess.py

# 3. Extract features
python src/features.py

# 4. Train model
python src/train.py --epochs 10 --batch-size 32

# 5. Evaluate
python src/evaluate.py
```

### Inference (Local)
```bash
# Run Streamlit app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Command Line Options
```bash
# Train with custom settings
python src/train.py --model efficientnet-b0 --epochs 15 --batch-size 64 --lr 5e-5

# Resume training
python src/train.py --resume models/best_model.pth --epochs 10

# Evaluate with custom model
python src/evaluate.py --model models/best_model.pth
```

---

## 🎬 Demo Samples

Add your own test samples to `demo_samples/`:
- `real_voice.wav` - Genuine human speech
- `fake_elevenlabs.wav` - Generated with ElevenLabs
- `fake_rvc.wav` - Generated with RVC (Retrieval-based Voice Conversion)

The Streamlit app includes a **Demo Examples** tab to test these instantly.

---

## 🔮 Future Improvements

- [ ] **Data Augmentation**: SpecAugment, noise injection, pitch shifting
- [ ] **Ensemble Models**: Combine EfficientNet + ResNet + RawNet
- [ ] **Raw Waveform Models**: RawNet2, Wav2Vec2 fine-tuning
- [ ] **Multi-language Support**: Test on non-English datasets
- [ ] **Real-time Streaming**: WebRTC integration for live call analysis
- [ ] **Explainability**: Grad-CAM visualizations for model decisions
- [ ] **Mobile Deployment**: TensorFlow Lite / ONNX export
- [ ] **API Service**: FastAPI wrapper for production deployment

---

## 📚 References

1. **ASVspoof 2021 Challenge**: https://www.asvspoof.org/asvspoof2021/
2. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019
3. **RawNet2**: Tak et al., "RawNet2: Improved RawNet with Multi-scale Feature Aggregation", Interspeech 2021
4. **SpecAugment**: Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR", Interspeech 2019
5. **Voice Cloning Detection**: Various recent papers on anti-spoofing

---

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## ⚠️ Disclaimer

VoiceShield is a research tool for educational and defensive purposes. It should not be used as the sole mechanism for security-critical authentication. Always implement defense-in-depth strategies including multi-factor authentication.

---

**Built with ❤️ for voice security in the AI era**