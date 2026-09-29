import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

# =============================================================================
# AUDIO CONFIGURATION
# =============================================================================
SAMPLE_RATE = 16000
TARGET_DURATION = 4.0  # seconds
TARGET_LENGTH = int(SAMPLE_RATE * TARGET_DURATION)  # 64000 samples
PAD_MODE = "constant"  # pad with zeros
TRIM_SILENCE_THRESHOLD = 0.02

# =============================================================================
# MEL-SPECTROGRAM CONFIGURATION
# =============================================================================
N_MELS = 128
N_FFT = 400
HOP_LENGTH = 160
F_MIN = 50.0
F_MAX = 7600.0
# Expected output shape: (1, 128, 251) after channel expand
SPEC_HEIGHT = N_MELS
SPEC_WIDTH = 251

# =============================================================================
# DATASET PATHS (Update these for your environment)
# =============================================================================
# For Google Colab with Google Drive mounted at /content/drive
DATA_ROOT = Path("/content/drive/MyDrive/voiceshield_data")

# ASVspoof 2019 LA (for train/dev)
TRAIN_AUDIO_DIR = DATA_ROOT / "ASVspoof2019_LA" / "train" / "flac"
DEV_AUDIO_DIR = DATA_ROOT / "ASVspoof2019_LA" / "dev" / "flac"

# ASVspoof 2021 LA (for eval)
EVAL_AUDIO_DIR = DATA_ROOT / "ASVspoof2021_LA" / "eval" / "flac"

# Protocol file (contains labels for ALL splits: train/dev/eval)
LABELS_FILE = DATA_ROOT / "keys" / "LA" / "CM" / "ASVspoof2021.LA.cm.train.trg.txt"

# Local fallback paths (for local development)
LOCAL_DATA_ROOT = BASE_DIR / "data" / "raw"
LOCAL_TRAIN_AUDIO_DIR = LOCAL_DATA_ROOT / "ASVspoof2019_LA" / "train" / "flac"
LOCAL_DEV_AUDIO_DIR = LOCAL_DATA_ROOT / "ASVspoof2019_LA" / "dev" / "flac"
LOCAL_EVAL_AUDIO_DIR = LOCAL_DATA_ROOT / "ASVspoof2021_LA" / "eval" / "flac"
LOCAL_LABELS_FILE = LOCAL_DATA_ROOT / "keys" / "LA" / "CM" / "ASVspoof2021.LA.cm.train.trg.txt"

# =============================================================================
# PROCESSED DATA PATHS
# =============================================================================
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
TRAIN_SPECS_DIR = PROCESSED_DATA_DIR / "train_specs"
DEV_SPECS_DIR = PROCESSED_DATA_DIR / "dev_specs"
EVAL_SPECS_DIR = PROCESSED_DATA_DIR / "eval_specs"

SPLITS_DIR = BASE_DIR / "data" / "splits"
TRAIN_METADATA_CSV = SPLITS_DIR / "train_metadata.csv"
DEV_METADATA_CSV = SPLITS_DIR / "dev_metadata.csv"
EVAL_METADATA_CSV = SPLITS_DIR / "eval_metadata.csv"

# =============================================================================
# MODEL PATHS
# =============================================================================
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
TRAINING_LOGS_CSV = MODELS_DIR / "training_logs.csv"

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
MODEL_NAME = "efficientnet-b0"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
OPTIMIZER = "adam"
LOSS_FUNCTION = "bce_with_logits"
CLASS_WEIGHTS = {"bonafide": 1.0, "spoof": 3.0}  # Handle class imbalance
DROPOUT_RATE = 0.3
FREEZE_LAYERS = True  # Freeze early layers for transfer learning
PATIENCE = 3  # Early stopping patience

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu"

# =============================================================================
# DEMO SAMPLES
# =============================================================================
DEMO_SAMPLES_DIR = BASE_DIR / "demo_samples"
DEMO_REAL = DEMO_SAMPLES_DIR / "real_voice.wav"
DEMO_FAKE_ELEVENLABS = DEMO_SAMPLES_DIR / "fake_elevenlabs.wav"
DEMO_FAKE_RVC = DEMO_SAMPLES_DIR / "fake_rvc.wav"

# =============================================================================
# RESULTS & VISUALIZATION
# =============================================================================
RESULTS_DIR = BASE_DIR / "results"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = RESULTS_DIR / "roc_curve.png"
TRAINING_PLOTS_DIR = RESULTS_DIR / "training_plots"

# =============================================================================
# INFERENCE THRESHOLD
# =============================================================================
INFERENCE_THRESHOLD = 0.5

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_audio_dirs(split: str):
    """Get audio directory and labels file for a given split."""
    if split == "train":
        return TRAIN_AUDIO_DIR, LABELS_FILE
    elif split == "dev":
        return DEV_AUDIO_DIR, LABELS_FILE
    elif split == "eval":
        return EVAL_AUDIO_DIR, LABELS_FILE
    else:
        raise ValueError(f"Unknown split: {split}")

def get_local_audio_dirs(split: str):
    """Get local audio directory and labels file for a given split."""
    if split == "train":
        return LOCAL_TRAIN_AUDIO_DIR, LOCAL_TRAIN_LABELS_FILE
    elif split == "dev":
        return LOCAL_DEV_AUDIO_DIR, LOCAL_DEV_LABELS_FILE
    elif split == "eval":
        return LOCAL_EVAL_AUDIO_DIR, LOCAL_EVAL_LABELS_FILE
    else:
        raise ValueError(f"Unknown split: {split}")

def get_specs_dir(split: str):
    """Get spectrograms directory for a given split."""
    if split == "train":
        return TRAIN_SPECS_DIR
    elif split == "dev":
        return DEV_SPECS_DIR
    elif split == "eval":
        return EVAL_SPECS_DIR
    else:
        raise ValueError(f"Unknown split: {split}")

def get_metadata_csv(split: str):
    """Get metadata CSV path for a given split."""
    if split == "train":
        return TRAIN_METADATA_CSV
    elif split == "dev":
        return DEV_METADATA_CSV
    elif split == "eval":
        return EVAL_METADATA_CSV
    else:
        raise ValueError(f"Unknown split: {split}")

def ensure_dirs():
    """Create all necessary directories."""
    dirs = [
        PROCESSED_DATA_DIR,
        TRAIN_SPECS_DIR,
        DEV_SPECS_DIR,
        EVAL_SPECS_DIR,
        SPLITS_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        TRAINING_PLOTS_DIR,
        DEMO_SAMPLES_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)