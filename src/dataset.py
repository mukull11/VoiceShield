import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, List


class ASVspoofDataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2021 LA dataset.
    
    Loads precomputed mel-spectrograms and labels from metadata CSV.
    """
    
    LABEL_MAP = {
        'bonafide': 0,
        'spoof': 1,
    }
    
    def __init__(
        self,
        metadata_csv: str,
        specs_dir: str,
        transform: Optional[Callable] = None,
        label_col: str = 'label',
        path_col: str = 'spec_path',
    ):
        """
        Initialize dataset.
        
        Args:
            metadata_csv: Path to metadata CSV file
            specs_dir: Directory containing .npy spectrogram files
            transform: Optional transform to apply to spectrograms
            label_col: Column name for labels in metadata
            path_col: Column name for spectrogram paths in metadata
        """
        self.metadata_csv = Path(metadata_csv)
        self.specs_dir = Path(specs_dir)
        self.transform = transform
        self.label_col = label_col
        self.path_col = path_col
        
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")
        
        self.df = pd.read_csv(self.metadata_csv)
        
        self.data = []
        for _, row in self.df.iterrows():
            spec_path = Path(row[path_col])
            if not spec_path.is_absolute():
                spec_path = self.specs_dir / spec_path.name
            
            if spec_path.exists():
                self.data.append({
                    'file_id': row['file_id'],
                    'spec_path': str(spec_path),
                    'label': row[label_col],
                })
            else:
                print(f"Warning: Spectrogram not found: {spec_path}")
        
        print(f"Loaded {len(self.data)} samples from {metadata_csv}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (spectrogram_tensor, label_tensor)
            - spectrogram: shape (1, 128, 251)
            - label: scalar tensor (0 for bonafide, 1 for spoof)
        """
        item = self.data[idx]
        
        spec = np.load(item['spec_path'])
        
        if spec.ndim == 2:
            spec = spec[np.newaxis, ...]
        
        spec_tensor = torch.from_numpy(spec).float()
        
        label_str = item['label']
        label = self.LABEL_MAP.get(label_str.lower(), 1)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
        
        return spec_tensor, label_tensor
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get count of samples per class."""
        labels = [self.LABEL_MAP.get(item['label'].lower(), 1) for item in self.data]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def get_class_weights(self) -> Dict[int, float]:
        """Calculate class weights for imbalanced datasets."""
        dist = self.get_label_distribution()
        total = sum(dist.values())
        weights = {cls: total / (len(dist) * count) for cls, count in dist.items()}
        return weights


class InferenceDataset(Dataset):
    """
    Dataset for inference on raw audio files.
    """
    
    def __init__(self, audio_paths: List[str], preprocess_fn: Callable):
        self.audio_paths = audio_paths
        self.preprocess_fn = preprocess_fn
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        audio_path = self.audio_paths[idx]
        spec = self.preprocess_fn(audio_path)
        spec_tensor = torch.from_numpy(spec).float()
        return spec_tensor, audio_path


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader with standard settings.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_datasets_and_loaders(
    train_csv: str,
    dev_csv: str,
    eval_csv: str,
    train_specs_dir: str,
    dev_specs_dir: str,
    eval_specs_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform=None,
) -> Tuple[Dict[str, Dataset], Dict[str, torch.utils.data.DataLoader]]:
    """
    Create all datasets and dataloaders for train/dev/eval splits.
    
    Returns:
        Tuple of (datasets dict, dataloaders dict)
    """
    datasets = {
        'train': ASVspoofDataset(train_csv, train_specs_dir, transform=transform),
        'dev': ASVspoofDataset(dev_csv, dev_specs_dir, transform=None),
        'eval': ASVspoofDataset(eval_csv, eval_specs_dir, transform=None),
    }
    
    dataloaders = {
        'train': create_dataloader(datasets['train'], batch_size, shuffle=True, num_workers=num_workers),
        'dev': create_dataloader(datasets['dev'], batch_size, shuffle=False, num_workers=num_workers),
        'eval': create_dataloader(datasets['eval'], batch_size, shuffle=False, num_workers=num_workers),
    }
    
    return datasets, dataloaders


if __name__ == "__main__":
    from config import (
        TRAIN_METADATA_CSV, DEV_METADATA_CSV, EVAL_METADATA_CSV,
        TRAIN_SPECS_DIR, DEV_SPECS_DIR, EVAL_SPECS_DIR,
    )
    
    print("Testing dataset loading...")
    
    for name, csv_path, specs_dir in [
        ("train", TRAIN_METADATA_CSV, TRAIN_SPECS_DIR),
        ("dev", DEV_METADATA_CSV, DEV_SPECS_DIR),
        ("eval", EVAL_METADATA_CSV, EVAL_SPECS_DIR),
    ]:
        if Path(csv_path).exists():
            print(f"\nTesting {name} dataset...")
            dataset = ASVspoofDataset(csv_path, specs_dir)
            print(f"  Length: {len(dataset)}")
            print(f"  Label distribution: {dataset.get_label_distribution()}")
            
            spec, label = dataset[0]
            print(f"  Sample shape: {spec.shape}, Label: {label.item()}")
        else:
            print(f"\n{name} metadata not found: {csv_path}")