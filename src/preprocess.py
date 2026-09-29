import librosa
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Optional

from config import (
    SAMPLE_RATE,
    TARGET_LENGTH,
    TRIM_SILENCE_THRESHOLD,
    PAD_MODE,
)

warnings.filterwarnings("ignore", category=UserWarning)


def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load audio file using librosa.
    
    Args:
        file_path: Path to the audio file (.flac, .wav, etc.)
        sr: Target sample rate
        
    Returns:
        Tuple of (audio_signal, sample_rate)
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        return audio, sample_rate
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {e}")


def resample_audio(audio: np.ndarray, sr_original: int, sr_target: int = SAMPLE_RATE) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Input audio signal
        sr_original: Original sample rate
        sr_target: Target sample rate
        
    Returns:
        Resampled audio signal
    """
    if sr_original == sr_target:
        return audio
    return librosa.resample(audio, orig_sr=sr_original, target_sr=sr_target)


def trim_silence(audio: np.ndarray, threshold: float = TRIM_SILENCE_THRESHOLD) -> np.ndarray:
    """
    Remove leading and trailing silence from audio.
    
    Args:
        audio: Input audio signal
        threshold: Amplitude threshold below which is considered silence
        
    Returns:
        Trimmed audio signal
    """
    if len(audio) == 0:
        return audio
    
    non_silent_indices = np.where(np.abs(audio) > threshold)[0]
    
    if len(non_silent_indices) == 0:
        return np.array([], dtype=audio.dtype)
    
    start_idx = non_silent_indices[0]
    end_idx = non_silent_indices[-1] + 1
    
    return audio[start_idx:end_idx]


def pad_or_trim(audio: np.ndarray, target_length: int = TARGET_LENGTH) -> np.ndarray:
    """
    Pad or trim audio to exact target length.
    
    Args:
        audio: Input audio signal
        target_length: Target number of samples
        
    Returns:
        Audio signal of exactly target_length samples
    """
    current_length = len(audio)
    
    if current_length == target_length:
        return audio
    elif current_length < target_length:
        padding = target_length - current_length
        return np.pad(audio, (0, padding), mode=PAD_MODE)
    else:
        return audio[:target_length]


def preprocess_single(
    file_path: str,
    sr: int = SAMPLE_RATE,
    target_length: int = TARGET_LENGTH
) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single audio file.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        target_length: Target number of samples
        
    Returns:
        Preprocessed audio signal of exactly target_length samples
    """
    try:
        audio, sample_rate = load_audio(file_path, sr)
        
        if sample_rate != sr:
            audio = resample_audio(audio, sample_rate, sr)
        
        audio = trim_silence(audio)
        
        if len(audio) == 0:
            warnings.warn(f"Audio file {file_path} is empty after silence trimming")
            return np.zeros(target_length, dtype=np.float32)
        
        audio = pad_or_trim(audio, target_length)
        
        return audio.astype(np.float32)
        
    except Exception as e:
        warnings.warn(f"Error preprocessing {file_path}: {e}")
        return np.zeros(target_length, dtype=np.float32)


def preprocess_dataset(
    audio_dir: Path,
    output_dir: Path,
    sr: int = SAMPLE_RATE,
    target_length: int = TARGET_LENGTH,
    file_pattern: str = "*.flac"
) -> List[str]:
    """
    Preprocess all audio files in a directory.
    
    Args:
        audio_dir: Directory containing audio files
        output_dir: Directory to save processed audio as .npy files
        sr: Target sample rate
        target_length: Target number of samples
        file_pattern: Glob pattern for audio files
        
    Returns:
        List of successfully processed file paths
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    audio_files = list(audio_dir.glob(file_pattern))
    
    if not audio_files:
        warnings.warn(f"No audio files found in {audio_dir} with pattern {file_pattern}")
        return []
    
    successful_files = []
    
    for audio_file in tqdm(audio_files, desc=f"Preprocessing {audio_dir.name}"):
        try:
            audio = preprocess_single(str(audio_file), sr, target_length)
            
            output_file = output_dir / f"{audio_file.stem}.npy"
            np.save(output_file, audio)
            
            successful_files.append(str(audio_file))
            
        except Exception as e:
            warnings.warn(f"Failed to process {audio_file}: {e}")
            continue
    
    return successful_files


def preprocess_for_inference(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Preprocess audio array for inference (real-time).
    
    Args:
        audio: Raw audio signal
        sr: Sample rate of input audio
        
    Returns:
        Preprocessed audio ready for feature extraction
    """
    if sr != SAMPLE_RATE:
        audio = resample_audio(audio, sr, SAMPLE_RATE)
    
    audio = trim_silence(audio)
    
    if len(audio) == 0:
        return np.zeros(TARGET_LENGTH, dtype=np.float32)
    
    audio = pad_or_trim(audio, TARGET_LENGTH)
    
    return audio.astype(np.float32)


if __name__ == "__main__":
    from config import ensure_dirs, LOCAL_TRAIN_AUDIO_DIR, LOCAL_DEV_AUDIO_DIR, LOCAL_EVAL_AUDIO_DIR
    from config import TRAIN_SPECS_DIR, DEV_SPECS_DIR, EVAL_SPECS_DIR
    
    ensure_dirs()
    
    print("Starting dataset preprocessing...")
    
    for split_name, audio_dir, specs_dir in [
        ("train", LOCAL_TRAIN_AUDIO_DIR, TRAIN_SPECS_DIR),
        ("dev", LOCAL_DEV_AUDIO_DIR, DEV_SPECS_DIR),
        ("eval", LOCAL_EVAL_AUDIO_DIR, EVAL_SPECS_DIR),
    ]:
        print(f"\nProcessing {split_name} split...")
        if audio_dir.exists():
            processed = preprocess_dataset(audio_dir, specs_dir)
            print(f"  Successfully processed: {len(processed)} files")
        else:
            print(f"  Directory not found: {audio_dir}")
    
    print("\nPreprocessing complete!")