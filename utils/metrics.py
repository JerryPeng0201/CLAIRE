"""
Medical metrics calculation and evaluation for EKG model
"""

import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_recall_fscore_support


class EKGMetrics:
    """
    Medical metrics calculation for EKG model evaluation
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def compute_metrics(self, eval_predictions, eval_labels):
        """
        Compute medical evaluation metrics
        """
        # This is a placeholder - in practice, you'd need to:
        # 1. Generate text predictions from the model
        # 2. Extract binary predictions from the text
        # 3. Compare with ground truth labels
        
        # For now, return dummy metrics
        return {
            'mace_auc': 0.5,
            'mortality_auc': 0.5,
            'mace_sensitivity': 0.5,
            'mace_specificity': 0.5,
            'mortality_sensitivity': 0.5,
            'mortality_specificity': 0.5
        }


def calculate_medical_metrics(
    true_labels: List[int],
    predictions: List[int],
    task_name: str = "",
    probability_scores: List[float] = None,
    true_labels_for_auc: List[int] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive medical metrics.

    AUC uses ``probability_scores`` when provided (real model/text probabilities).
    Hard binary predictions alone are not used as AUC scores.
    """
    metrics = {}
    prefix = f"{task_name}_" if task_name else ""

    accuracy = accuracy_score(true_labels, predictions)
    metrics[f'{prefix}accuracy'] = accuracy

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0
    )
    metrics[f'{prefix}precision'] = precision
    metrics[f'{prefix}recall'] = recall
    metrics[f'{prefix}f1'] = f1

    try:
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics[f'{prefix}sensitivity'] = sensitivity
        metrics[f'{prefix}specificity'] = specificity

        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        metrics[f'{prefix}ppv'] = ppv
        metrics[f'{prefix}npv'] = npv

    except Exception as e:
        print(f"Error calculating confusion matrix metrics: {e}")
        metrics[f'{prefix}sensitivity'] = 0.0
        metrics[f'{prefix}specificity'] = 0.0
        metrics[f'{prefix}ppv'] = 0.0
        metrics[f'{prefix}npv'] = 0.0

    # AUC from calibrated probability scores only — never hard labels or invented noise
    auc_labels = true_labels_for_auc if true_labels_for_auc is not None else true_labels
    try:
        if (
            probability_scores is not None
            and len(probability_scores) == len(auc_labels)
            and len(probability_scores) > 0
            and len(set(auc_labels)) > 1
        ):
            auc = roc_auc_score(auc_labels, probability_scores)
            metrics[f'{prefix}auc'] = float(auc)
            metrics[f'{prefix}auroc'] = float(auc)
        else:
            metrics[f'{prefix}auc'] = None
            metrics[f'{prefix}auroc'] = None
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        metrics[f'{prefix}auc'] = None
        metrics[f'{prefix}auroc'] = None

    return metrics


def extract_predictions_from_text(text: str) -> tuple:
    """
    Extract MACE and mortality predictions from generated text
    
    Args:
        text: Generated text from the model
        
    Returns:
        Tuple of (mace_prediction, mortality_prediction)
    """
    text_lower = text.lower()
    
    # MACE prediction
    mace_pred = 0
    mace_patterns_positive = [
        "mace prediction: 1", "mace: 1", "mace (1)", "mace): 1",
        "mace prediction: yes", "mace: yes", "mace): yes"
    ]
    mace_patterns_negative = [
        "mace prediction: 0", "mace: 0", "mace (0)", "mace): 0", 
        "mace prediction: no", "mace: no", "mace): no"
    ]
    
    for pattern in mace_patterns_positive:
        if pattern in text_lower:
            mace_pred = 1
            break
    
    if mace_pred == 0:  # Only check negative patterns if not already positive
        for pattern in mace_patterns_negative:
            if pattern in text_lower:
                mace_pred = 0
                break
    
    # Mortality prediction
    mortality_pred = 0
    mortality_patterns_positive = [
        "mortality prediction: 1", "mortality: 1", "mortality (1)", "mortality): 1",
        "mortality prediction: yes", "mortality: yes", "mortality): yes"
    ]
    mortality_patterns_negative = [
        "mortality prediction: 0", "mortality: 0", "mortality (0)", "mortality): 0",
        "mortality prediction: no", "mortality: no", "mortality): no"
    ]
    
    for pattern in mortality_patterns_positive:
        if pattern in text_lower:
            mortality_pred = 1
            break
    
    if mortality_pred == 0:  # Only check negative patterns if not already positive
        for pattern in mortality_patterns_negative:
            if pattern in text_lower:
                mortality_pred = 0
                break
    
    return mace_pred, mortality_pred