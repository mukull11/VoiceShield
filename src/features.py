import librosa
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from config import (
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    F_MIN,
    F_MAX,
    SPEC_HEIGHT,
    SPEC_WIDTH,
)

warnings.filterwarnings("ignore", category=UserWarning)


def extract_melspectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> np.ndarray:
    """
    Extract mel-spectrogram from audio signal.
    
    Args:
        audio: Input audio signal (1D array)
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length between frames
        f_min: Minimum frequency
        f_max: Maximum frequency
        
    Returns:
        Normalized mel-spectrogram of shape (1, n_mels, time_steps)
    """
    if len(audio) == 0:
        return np.zeros((1, n_mels, SPEC_WIDTH), dtype=np.float32)
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=f_min,
        fmax=f_max,
        power=2.0,
    )
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    mean = mel_spec_db.mean()
    std = mel_spec_db.std() + 1e-8
    mel_spec_norm = (mel_spec_db - mean) / std
    
    if mel_spec_norm.shape[1] < SPEC_WIDTH:
        pad_width = SPEC_WIDTH - mel_spec_norm.shape[1]
        mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_spec_norm.shape[1] > SPEC_WIDTH:
        mel_spec_norm = mel_spec_norm[:, :SPEC_WIDTH]
    
    mel_spec_norm = mel_spec_norm[np.newaxis, ...]
    
    return mel_spec_norm.astype(np.float32)


def visualize_spectrogram(
    spec: np.ndarray,
    title: str = "Mel-Spectrogram",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4)
) -> None:
    """
    Plot mel-spectrogram heatmap.
    
    Args:
        spec: Spectrogram array of shape (1, n_mels, time) or (n_mels, time)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
    """
    if spec.ndim == 3:
        spec = spec[0]
    
    plt.figure(figsize=figsize)
    plt.imshow(spec, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Bands')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_waveform_and_spectrogram(
    audio: np.ndarray,
    spec: np.ndarray,
    label: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot waveform and spectrogram side by side.
    
    Args:
        audio: Audio signal
        spec: Spectrogram array
        label: Label for title
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    time_axis = np.arange(len(audio)) / SAMPLE_RATE
    axes[0].plot(time_axis, audio, color='steelblue', linewidth=0.5)
    axes[0].set_title(f'Waveform - {label}')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    if spec.ndim == 3:
        spec = spec[0]
    
    im = axes[1].imshow(spec, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title(f'Mel-Spectrogram - {label}')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('Mel Bands')
    plt.colorbar(im, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def parse_protocol_file(protocol_path: Path) -> pd.DataFrame:
    """
    Parse ASVspoof protocol file.
    
    Format: SPEAKER_ID FILE_ID SYS_ID KEY
    Example: LA_0069 LA_D_9424274 - bonafide
    
    Args:
        protocol_path: Path to protocol .txt file
        
    Returns:
        DataFrame with columns: file_id, label, sys_id
    """
    records = []
    
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                speaker_id = parts[0]
                file_id = parts[1]
                sys_id = parts[2]
                key = parts[3]
                
                records.append({
                    'file_id': file_id,
                    'speaker_id': speaker_id,
                    'sys_id': sys_id,
                    'label': key,
                })
    
    return pd.DataFrame(records)


def extract_dataset_features(
    audio_dir: Path,
    labels_file: Path,
    output_dir: Path,
    split: str = "train",
    preprocessed: bool = True,
) -> pd.DataFrame:
    """
    Extract mel-spectrograms for entire dataset split.
    
    Args:
        audio_dir: Directory containing audio files (.flac)
        labels_file: Path to protocol file
        output_dir: Directory to save .npy spectrograms
        split: Dataset split name (train/dev/eval)
        preprocessed: Whether audio is already preprocessed (.npy) or raw (.flac)
        
    Returns:
        Metadata DataFrame with columns: file_id, audio_path, spec_path, label, split
    """
    audio_dir = Path(audio_dir)
    labels_file = Path(labels_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    protocol_df = parse_protocol_file(labels_file)
    
    metadata_records = []
    
    for _, row in tqdm(protocol_df.iterrows(), total=len(protocol_df), desc=f"Extracting {split} features"):
        file_id = row['file_id']
        label = row['label']
        
        if preprocessed:
            audio_path = audio_dir / f"{file_id}.npy"
            if not audio_path.exists():
                audio_path = audio_dir / f"{file_id}.flac"
        else:
            audio_path = audio_dir / f"{file_id}.flac"
        
        spec_filename = f"{file_id}_{label}.npy"
        spec_path = output_dir / spec_filename
        
        try:
            if audio_path.suffix == '.npy':
                audio = np.load(audio_path)
            else:
                audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
            
            spec = extract_melspectrogram(audio)
            np.save(spec_path, spec)
            
            metadata_records.append({
                'file_id': file_id,
                'audio_path': str(audio_path),
                'spec_path': str(spec_path),
                'label': label,
                'split': split,
                'sys_id': row['sys_id'],
            })
            
        except Exception as e:
            warnings.warn(f"Failed to process {file_id}: {e}")
            continue
    
    metadata_df = pd.DataFrame(metadata_records)
    return metadata_df


def save_features(spec: np.ndarray, file_path: Path) -> None:
    """Save spectrogram to .npy file."""
    np.save(file_path, spec)


def load_features(file_path: Path) -> np.ndarray:
    """Load spectrogram from .npy file."""
    return np.load(file_path)


def extract_features_for_inference(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract mel-spectrogram for inference.
    
    Args:
        audio: Preprocessed audio signal
        sr: Sample rate
        
    Returns:
        Spectrogram of shape (1, 128, 251)
    """
    return extract_melspectrogram(audio, sr=sr)


if __name__ == "__main__":
    from config import ensure_dirs, LOCAL_TRAIN_AUDIO_DIR, LOCAL_DEV_AUDIO_DIR, LOCAL_EVAL_AUDIO_DIR
    from config import LOCAL_TRAIN_LABELS_FILE, LOCAL_DEV_LABELS_FILE, LOCAL_EVAL_LABELS_FILE
    from config import TRAIN_SPECS_DIR, DEV_SPECS_DIR, EVAL_SPECS_DIR
    from config import TRAIN_METADATA_CSV, DEV_METADATA_CSV, EVAL_METADATA_CSV
    from config import SPLITS_DIR
    
    ensure_dirs()
    
    print("Starting feature extraction...")
    
    splits_config = [
        ("train", LOCAL_TRAIN_AUDIO_DIR, LOCAL_TRAIN_LABELS_FILE, TRAIN_SPECS_DIR, TRAIN_METADATA_CSV),
        ("dev", LOCAL_DEV_AUDIO_DIR, LOCAL_DEV_LABELS_FILE, DEV_SPECS_DIR, DEV_METADATA_CSV),
        ("eval", LOCAL_EVAL_AUDIO_DIR, LOCAL_EVAL_LABELS_FILE, EVAL_SPECS_DIR, EVAL_METADATA_CSV),
    ]
    
    for split, audio_dir, labels_file, specs_dir, metadata_csv in splits_config:
        print(f"\nProcessing {split} split...")
        if audio_dir.exists() and labels_file.exists():
            metadata_df = extract_dataset_features(audio_dir, labels_file, specs_dir, split, preprocessed=True)
            metadata_df.to_csv(metadata_csv, index=False)
            print(f"  Extracted features for {len(metadata_df)} samples")
            print(f"  Metadata saved to {metadata_csv}")
        else:
            print(f"  Skipping {split}: directory or labels file not found")
    
    print("\nFeature extraction complete!")