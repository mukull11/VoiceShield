import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import warnings
import time

warnings.filterwarnings("ignore")

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    CLASS_WEIGHTS,
    PATIENCE,
    BEST_MODEL_PATH,
    TRAINING_LOGS_CSV,
    DEVICE,
    MODELS_DIR,
    TRAIN_METADATA_CSV,
    DEV_METADATA_CSV,
    TRAIN_SPECS_DIR,
    DEV_SPECS_DIR,
    ensure_dirs,
)
from src.dataset import ASVspoofDataset, create_dataloader
from src.model import create_model, count_parameters


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (avg_loss, accuracy, f1_score)
    """
    model.train()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for specs, labels in tqdm(dataloader, desc="Training", leave=False):
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        logits = model(specs)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * specs.size(0)
        
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, f1


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Validate model.
    
    Returns:
        Tuple of (avg_loss, accuracy, f1_score, roc_auc)
    """
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for specs, labels in tqdm(dataloader, desc="Validation", leave=False):
            specs = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
            
            logits = model(specs)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * specs.size(0)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.flatten())
    
    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    
    return avg_loss, accuracy, f1, roc_auc


def train_model(
    model_type: str = 'efficientnet-b0',
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    device: str = DEVICE,
    model_path: Path = BEST_MODEL_PATH,
    logs_path: Path = TRAINING_LOGS_CSV,
) -> nn.Module:
    """
    Main training function.
    
    Args:
        model_type: Model architecture to use
        batch_size: Training batch size
        learning_rate: Learning rate
        epochs: Number of epochs
        patience: Early stopping patience
        device: Device to train on
        model_path: Path to save best model
        logs_path: Path to save training logs
        
    Returns:
        Trained model
    """
    ensure_dirs()
    
    device = torch.device(device)
    print(f"Using device: {device}")
    
    print("\nLoading datasets...")
    train_dataset = ASVspoofDataset(TRAIN_METADATA_CSV, TRAIN_SPECS_DIR)
    dev_dataset = ASVspoofDataset(DEV_METADATA_CSV, DEV_SPECS_DIR)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")
    print(f"Train label distribution: {train_dataset.get_label_distribution()}")
    print(f"Dev label distribution: {dev_dataset.get_label_distribution()}")
    
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = create_dataloader(dev_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nCreating model: {model_type}")
    model = create_model(
        model_type=model_type,
        dropout_rate=0.3,
        freeze_backbone=True,
    ).to(device)
    
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    class_weights_tensor = torch.tensor(
        [CLASS_WEIGHTS['bonafide'], CLASS_WEIGHTS['spoof']],
        dtype=torch.float32,
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1:2])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    
    training_logs = []
    
    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs")
    print(f"{'='*60}")
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, class_weights_tensor
        )
        
        val_loss, val_acc, val_f1, val_roc_auc = validate(
            model, dev_loader, criterion, device
        )
        
        scheduler.step(val_f1)
        
        epoch_time = time.time() - epoch_start
        
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1': val_f1,
            'val_roc_auc': val_roc_auc,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time,
        }
        training_logs.append(log_entry)
        
        print(
            f"[Epoch {epoch:2d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} | Val AUC: {val_roc_auc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_accuracy': val_acc,
                'val_roc_auc': val_roc_auc,
            }, model_path)
            
            print(f"  >>> New best model saved! Val F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
    
    logs_df = pd.DataFrame(training_logs)
    logs_df.to_csv(logs_path, index=False)
    print(f"\nTraining logs saved to {logs_path}")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best Val F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Best model saved to: {model_path}")
    print(f"{'='*60}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def resume_training(
    checkpoint_path: Path,
    model_type: str = 'efficientnet-b0',
    additional_epochs: int = 10,
) -> nn.Module:
    """
    Resume training from checkpoint.
    """
    device = torch.device(DEVICE)
    
    train_dataset = ASVspoofDataset(TRAIN_METADATA_CSV, TRAIN_SPECS_DIR)
    dev_dataset = ASVspoofDataset(DEV_METADATA_CSV, DEV_SPECS_DIR)
    
    train_loader = create_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = create_dataloader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = create_model(model_type=model_type).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_f1 = checkpoint.get('best_f1', 0.0)
    
    class_weights_tensor = torch.tensor(
        [CLASS_WEIGHTS['bonafide'], CLASS_WEIGHTS['spoof']],
        dtype=torch.float32,
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1:2])
    
    print(f"Resuming from epoch {start_epoch}, best F1: {best_f1:.4f}")
    
    for epoch in range(start_epoch + 1, start_epoch + additional_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, class_weights_tensor
        )
        
        val_loss, val_acc, val_f1, val_roc_auc = validate(
            model, dev_loader, criterion, device
        )
        
        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | Val AUC: {val_roc_auc:.4f}"
        )
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, BEST_MODEL_PATH)
            print(f"  >>> New best model saved!")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train VoiceShield model")
    parser.add_argument('--model', type=str, default='efficientnet-b0', help='Model type')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    if args.resume:
        resume_training(Path(args.resume), args.model, args.epochs)
    else:
        train_model(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )