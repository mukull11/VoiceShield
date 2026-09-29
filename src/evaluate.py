import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from config import (
    EVAL_METADATA_CSV,
    EVAL_SPECS_DIR,
    BEST_MODEL_PATH,
    INFERENCE_THRESHOLD,
    RESULTS_DIR,
    CONFUSION_MATRIX_PATH,
    ROC_CURVE_PATH,
    DEVICE,
    BATCH_SIZE,
    ensure_dirs,
)
from src.dataset import ASVspoofDataset, create_dataloader
from src.model import create_model, load_model_checkpoint


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on dataloader.
    
    Returns:
        Tuple of (predictions, true_labels, probabilities)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for specs, labels in tqdm(dataloader, desc="Evaluating"):
            specs = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            logits = model(specs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > INFERENCE_THRESHOLD).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Compute comprehensive evaluation metrics.
    
    Returns:
        Dictionary with all metrics
    """
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    roc_auc = roc_auc_score(true_labels, probabilities) if len(set(true_labels)) > 1 else 0.0
    
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'roc_auc': roc_auc,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
    }


def compute_eer(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    n_thresholds: int = 1000,
) -> float:
    """
    Compute Equal Error Rate (EER).
    
    EER is the threshold where False Acceptance Rate (FAR) = False Rejection Rate (FRR)
    
    Returns:
        EER value (0-1)
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    
    far_values = []
    frr_values = []
    
    for thresh in thresholds:
        preds = (probabilities > thresh).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(true_labels, preds, labels=[0, 1]).ravel()
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        far_values.append(far)
        frr_values.append(frr)
    
    far_values = np.array(far_values)
    frr_values = np.array(frr_values)
    
    diff = np.abs(far_values - frr_values)
    eer_idx = np.argmin(diff)
    
    eer = (far_values[eer_idx] + frr_values[eer_idx]) / 2
    
    return eer


def plot_confusion_matrix(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    save_path: Path = CONFUSION_MATRIX_PATH,
    class_names: list = ['Bonafide', 'Spoof'],
) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    save_path: Path = ROC_CURVE_PATH,
) -> None:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1 - FRR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_det_curve(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    save_path: Path = None,
) -> None:
    """Plot Detection Error Tradeoff (DET) curve."""
    from sklearn.metrics import det_curve
    
    fpr, fnr, thresholds = det_curve(true_labels, probabilities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, fnr, color='darkorange', lw=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('False Negative Rate (FRR)')
    plt.title('DET Curve')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_score_distribution(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    save_path: Path = None,
) -> None:
    """Plot score distribution for bonafide vs spoof."""
    bonafide_scores = probabilities[true_labels == 0]
    spoof_scores = probabilities[true_labels == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(bonafide_scores, bins=50, alpha=0.5, label='Bonafide', density=True, color='green')
    plt.hist(spoof_scores, bins=50, alpha=0.5, label='Spoof', density=True, color='red')
    plt.axvline(INFERENCE_THRESHOLD, color='black', linestyle='--', label=f'Threshold ({INFERENCE_THRESHOLD})')
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_waveform_and_spectrogram(
    audio: np.ndarray,
    spec: np.ndarray,
    label: str,
    save_path: Path = None,
) -> None:
    """Plot waveform and spectrogram for a sample."""
    from src.features import visualize_waveform_and_spectrogram
    visualize_waveform_and_spectrogram(audio, spec, label, save_path)


def run_full_evaluation(
    model_path: Path = BEST_MODEL_PATH,
    eval_csv: Path = EVAL_METADATA_CSV,
    eval_specs_dir: Path = EVAL_SPECS_DIR,
    batch_size: int = BATCH_SIZE,
    device: str = DEVICE,
) -> dict:
    """
    Run complete evaluation pipeline.
    
    Returns:
        Dictionary with all metrics
    """
    ensure_dirs()
    
    device = torch.device(device)
    print(f"Using device: {device}")
    
    print("\nLoading evaluation dataset...")
    eval_dataset = ASVspoofDataset(eval_csv, eval_specs_dir)
    eval_loader = create_dataloader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Label distribution: {eval_dataset.get_label_distribution()}")
    
    print(f"\nLoading model from {model_path}...")
    model = create_model().to(device)
    model = load_model_checkpoint(model, str(model_path), device)
    
    print("\nRunning evaluation...")
    predictions, true_labels, probabilities = evaluate_model(model, eval_loader, device)
    
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, true_labels, probabilities)
    
    eer = compute_eer(predictions, true_labels, probabilities)
    metrics['eer'] = eer
    
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:    {metrics['accuracy']*100:.2f}%")
    print(f"F1-Score:    {metrics['f1_score']*100:.2f}%")
    print(f"Precision:   {metrics['precision']*100:.2f}%")
    print(f"Recall:      {metrics['recall']*100:.2f}%")
    print(f"Specificity: {metrics['specificity']*100:.2f}%")
    print(f"Sensitivity: {metrics['sensitivity']*100:.2f}%")
    print(f"ROC-AUC:     {metrics['roc_auc']*100:.2f}%")
    print(f"EER:         {eer*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['true_positives']}  FP: {metrics['false_positives']}")
    print(f"  FN: {metrics['false_negatives']}  TN: {metrics['true_negatives']}")
    print(f"{'='*50}")
    
    print("\nGenerating plots...")
    plot_confusion_matrix(predictions, true_labels)
    plot_roc_curve(true_labels, probabilities)
    plot_det_curve(true_labels, probabilities, RESULTS_DIR / "det_curve.png")
    plot_score_distribution(true_labels, probabilities, RESULTS_DIR / "score_distribution.png")
    
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(RESULTS_DIR / "evaluation_results.csv", index=False)
    print(f"Results saved to {RESULTS_DIR / 'evaluation_results.csv'}")
    
    return metrics


def evaluate_on_custom_files(
    model: nn.Module,
    audio_files: list,
    preprocess_fn,
    device: torch.device,
) -> list:
    """
    Evaluate model on custom audio files.
    
    Returns:
        List of dicts with predictions
    """
    results = []
    
    model.eval()
    
    for audio_file in audio_files:
        spec = preprocess_fn(audio_file)
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            logit = model(spec_tensor)
            prob = torch.sigmoid(logit).item()
            pred = 1 if prob > INFERENCE_THRESHOLD else 0
        
        label = "AI-GENERATED" if pred == 1 else "REAL"
        confidence = abs(0.5 - prob) * 200
        
        results.append({
            'file': audio_file,
            'label': label,
            'probability': prob,
            'confidence': confidence,
            'prediction': pred,
        })
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate VoiceShield model")
    parser.add_argument('--model', type=str, default=str(BEST_MODEL_PATH), help='Model path')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device')
    args = parser.parse_args()
    
    run_full_evaluation(
        model_path=Path(args.model),
        batch_size=args.batch_size,
        device=args.device,
    )