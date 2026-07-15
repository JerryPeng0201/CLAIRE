"""
Model evaluation utilities for EKG training
"""

import re
import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from .metrics import extract_predictions_from_text, calculate_medical_metrics
from .plotting import generate_auc_curves_and_data


def extract_probabilities_from_text(
    generated_text: str,
) -> Tuple[Optional[float], Optional[float], int, int]:
    """
    Extract probability scores from generated text for AUC calculation.

    Returns None for a probability when the model does not emit an explicit score.
    Does not synthesize or invent probabilities.
    """
    mace_prob: Optional[float] = None
    mortality_prob: Optional[float] = None

    text = generated_text.lower()

    # Prefer explicit "probability: X" / "probability X" near each prediction block
    explicit_prob_patterns = [
        r'probability\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%',
        r'probability\s*[:=]?\s*(\d+(?:\.\d+)?)',
        r'risk\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%',
        r'confidence\s*[:=]?\s*(\d+(?:\.\d+)?)',
    ]

    def _parse_prob_from_section(section: str) -> Optional[float]:
        for pattern in explicit_prob_patterns:
            matches = re.findall(pattern, section)
            if matches:
                try:
                    prob = float(matches[0])
                    if prob > 1:  # percentage
                        prob = prob / 100.0
                    return max(0.0, min(1.0, prob))
                except (ValueError, IndexError):
                    continue
        return None

    mace_start = text.find("mace prediction")
    if mace_start != -1:
        mace_end = text.find("mortality prediction", mace_start)
        if mace_end == -1:
            mace_end = mace_start + 200
        mace_prob = _parse_prob_from_section(text[mace_start:mace_end])

    mortality_start = text.find("mortality prediction")
    if mortality_start != -1:
        mortality_prob = _parse_prob_from_section(
            text[mortality_start:mortality_start + 200]
        )

    mace_binary, mortality_binary = extract_predictions_from_text(generated_text)
    return mace_prob, mortality_prob, mace_binary, mortality_binary


def _token_ids_for_digit(tokenizer, digit: str) -> List[int]:
    """Resolve tokenizer id(s) for digit strings '0' / '1' (with and without leading space)."""
    candidates = []
    for variant in (digit, f" {digit}"):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            candidates.append(ids[0])
    return candidates


def _positive_class_prob_from_next_token(model, tokenizer, prefix: str, device) -> Optional[float]:
    """
    P(class=1) from next-token logits over {'0','1'}, excluding synthetic scores.
    """
    try:
        inputs = tokenizer(
            prefix, return_tensors="pt", truncation=True, max_length=32768
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]

        id0s = _token_ids_for_digit(tokenizer, "0")
        id1s = _token_ids_for_digit(tokenizer, "1")
        if not id0s or not id1s:
            return None

        # Use the highest-logit variant for each digit class
        logit0 = torch.stack([logits[i] for i in id0s]).max()
        logit1 = torch.stack([logits[i] for i in id1s]).max()
        probs = torch.softmax(torch.stack([logit0, logit1]), dim=0)
        return float(probs[1].item())
    except Exception as e:
        print(f"Error computing next-token class probability: {e}")
        return None


def get_binary_class_probabilities(
    model, tokenizer, prompt: str, mace_pred: int, device
) -> Tuple[Optional[float], Optional[float]]:
    """
    Model-calibrated P(class=1) for MACE and mortality from next-token logits
    after a fixed response scaffold matching the training format.
    """
    mace_prefix = (
        prompt
        + "Based on the clinical assessment:\n\n"
        + "1. **MACE Prediction: "
    )
    mace_prob = _positive_class_prob_from_next_token(model, tokenizer, mace_prefix, device)

    mortality_prefix = (
        prompt
        + "Based on the clinical assessment:\n\n"
        + f"1. **MACE Prediction: {mace_pred}** (0 = No, 1 = Yes)\n\n"
        + "2. **Mortality Prediction: "
    )
    mortality_prob = _positive_class_prob_from_next_token(
        model, tokenizer, mortality_prefix, device
    )
    return mace_prob, mortality_prob


def get_generation_confidence(model, tokenizer, prompt: str, response_text: str, device) -> Optional[float]:
    """
    Mean token probability of the generated response under the model (no added noise).

    Returns None if the score cannot be computed. Kept for diagnostics; not used to
    invent class probabilities for AUC.
    """
    try:
        full_text = prompt + response_text
        inputs = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=32768
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

            prompt_length = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            generated_logits = logits[0, prompt_length - 1:-1]
            generated_labels = inputs["input_ids"][0, prompt_length:]

            token_probs = []
            for i, token_id in enumerate(generated_labels):
                if i < generated_logits.shape[0]:
                    prob = torch.softmax(generated_logits[i], dim=-1)[token_id].item()
                    token_probs.append(prob)

            if token_probs:
                return float(np.mean(token_probs))
    except Exception as e:
        print(f"Error calculating generation confidence: {e}")

    return None


def evaluate_model_on_test_set(model, tokenizer, eval_dataset, device, max_samples: int = 100):
    """
    Comprehensive evaluation on test set with real model probability scores for AUC.
    """
    model.eval()
    predictions = []
    probability_predictions = []
    true_labels = []
    generated_texts = []
    prob_source_counts = {"text": 0, "logits": 0, "missing": 0}

    print(f"Evaluating model on {min(max_samples, len(eval_dataset))} samples...")

    risk_prediction_samples = [
        sample for sample in eval_dataset
        if hasattr(sample, 'get') and sample.get('example_type') == 'risk_prediction'
    ]

    if not risk_prediction_samples:
        sample_indices = np.random.choice(
            len(eval_dataset), size=min(max_samples, len(eval_dataset)), replace=False
        )
        risk_prediction_samples = [eval_dataset[i] for i in sample_indices]

    for i, sample in enumerate(risk_prediction_samples[:max_samples]):
        if i % 20 == 0:
            print(f"Evaluating sample {i+1}/{min(max_samples, len(risk_prediction_samples))}")

        input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)

        assistant_pos = input_text.find('<|Assistant|>')
        if assistant_pos == -1:
            for marker in ['<|assistant|>', 'Assistant:', '<Assistant>']:
                assistant_pos = input_text.find(marker)
                if assistant_pos != -1:
                    break

        if assistant_pos == -1:
            prompt = input_text[:int(len(input_text) * 0.7)]
        else:
            prompt = input_text[:assistant_pos + 13]  # +13 for '<|Assistant|>'

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32768)
        if torch.cuda.is_available():
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        )
        generated_texts.append(generated_text)

        mace_prob, mortality_prob, mace_pred, mortality_pred = extract_probabilities_from_text(
            generated_text
        )

        # Prefer explicit text scores; otherwise use next-token class probabilities.
        # Never synthesize random/beta/noise scores for missing values.
        need_logits = mace_prob is None or mortality_prob is None
        logit_mace, logit_mortality = (None, None)
        if need_logits:
            logit_mace, logit_mortality = get_binary_class_probabilities(
                model, tokenizer, prompt, mace_pred, device
            )

        if mace_prob is not None:
            prob_source_counts["text"] += 1
        else:
            mace_prob = logit_mace
            if mace_prob is not None:
                prob_source_counts["logits"] += 1
            else:
                prob_source_counts["missing"] += 1

        if mortality_prob is not None:
            prob_source_counts["text"] += 1
        else:
            mortality_prob = logit_mortality
            if mortality_prob is not None:
                prob_source_counts["logits"] += 1
            else:
                prob_source_counts["missing"] += 1

        predictions.append((mace_pred, mortality_pred))
        probability_predictions.append((mace_prob, mortality_prob))

        true_labels.append({
            'mace_label': sample['mace_label'].item(),
            'mortality_label': sample['mortality_label'].item()
        })

        if i < 3:
            mace_prob_str = f"{mace_prob:.3f}" if mace_prob is not None else "N/A"
            mort_prob_str = f"{mortality_prob:.3f}" if mortality_prob is not None else "N/A"
            print(f"\n=== Sample {i+1} ===")
            print(f"True MACE: {sample['mace_label'].item()}, True Mortality: {sample['mortality_label'].item()}")
            print(f"Predicted MACE: {mace_pred} (prob: {mace_prob_str}), Predicted Mortality: {mortality_pred} (prob: {mort_prob_str})")
            print(f"Generated (first 200 chars): {generated_text[:200]}...")
            print("---")

    print(
        "Probability sources — "
        f"text: {prob_source_counts['text']}, "
        f"logits: {prob_source_counts['logits']}, "
        f"missing: {prob_source_counts['missing']}"
    )

    return predictions, probability_predictions, true_labels, generated_texts


def _filter_pairs_with_scores(y_true, y_prob):
    """Keep only samples with a real probability score for ROC/AUC."""
    filtered_true = []
    filtered_prob = []
    for label, prob in zip(y_true, y_prob):
        if prob is not None and not (isinstance(prob, float) and np.isnan(prob)):
            filtered_true.append(label)
            filtered_prob.append(float(prob))
    return filtered_true, filtered_prob


def compute_final_metrics_with_auc(predictions, probability_predictions, true_labels, save_dir):
    """
    Compute comprehensive final evaluation metrics and generate separate AUC curves for each task.
    AUC uses only real model/text probabilities — samples without scores are excluded from ROC.
    """
    if len(predictions) == 0:
        return {}

    mace_preds = [p[0] for p in predictions]
    mortality_preds = [p[1] for p in predictions]
    mace_probs = [p[0] for p in probability_predictions]
    mortality_probs = [p[1] for p in probability_predictions]
    mace_true = [label['mace_label'] for label in true_labels]
    mortality_true = [label['mortality_label'] for label in true_labels]

    mace_true_scored, mace_probs_scored = _filter_pairs_with_scores(mace_true, mace_probs)
    mort_true_scored, mort_probs_scored = _filter_pairs_with_scores(mortality_true, mortality_probs)

    mace_metrics = calculate_medical_metrics(
        mace_true, mace_preds, "mace", probability_scores=mace_probs_scored if mace_probs_scored else None,
        true_labels_for_auc=mace_true_scored if mace_probs_scored else None,
    )
    mortality_metrics = calculate_medical_metrics(
        mortality_true, mortality_preds, "mortality",
        probability_scores=mort_probs_scored if mort_probs_scored else None,
        true_labels_for_auc=mort_true_scored if mort_probs_scored else None,
    )

    y_true_list = [mace_true_scored, mort_true_scored]
    y_prob_list = [mace_probs_scored, mort_probs_scored]
    task_names = ['MACE', 'Mortality']

    auc_scores = []
    try:
        if mace_probs_scored and mort_probs_scored:
            auc_scores = generate_auc_curves_and_data(y_true_list, y_prob_list, task_names, save_dir)
            print("Separate MACE and Mortality ROC curves generated successfully!")
            print("Data files (JSON and CSV) created for each task!")
            if len(auc_scores) >= 2:
                mace_metrics['mace_auc_from_probs'] = auc_scores[0]
                mortality_metrics['mortality_auc_from_probs'] = auc_scores[1]
        else:
            print(
                "Skipping ROC plots: missing probability scores for one or both tasks "
                f"(MACE scored n={len(mace_probs_scored)}, Mortality scored n={len(mort_probs_scored)})."
            )
    except Exception as e:
        print(f"Failed to generate AUC curves: {e}")
        auc_scores = []

    overall_metrics = {
        'mean_accuracy': (mace_metrics.get('mace_accuracy', 0) + mortality_metrics.get('mortality_accuracy', 0)) / 2,
        'mean_sensitivity': (mace_metrics.get('mace_sensitivity', 0) + mortality_metrics.get('mortality_sensitivity', 0)) / 2,
        'mean_specificity': (mace_metrics.get('mace_specificity', 0) + mortality_metrics.get('mortality_specificity', 0)) / 2,
        'mean_f1': (mace_metrics.get('mace_f1', 0) + mortality_metrics.get('mortality_f1', 0)) / 2,
        'mean_auc': float(np.mean(auc_scores)) if auc_scores else None,
    }

    distribution_metrics = {
        'total_samples': len(predictions),
        'mace_scored_samples': len(mace_probs_scored),
        'mortality_scored_samples': len(mort_probs_scored),
        'mace_positive_rate': sum(mace_true) / len(mace_true) if mace_true else 0,
        'mortality_positive_rate': sum(mortality_true) / len(mortality_true) if mortality_true else 0,
        'mace_predicted_positive_rate': sum(mace_preds) / len(mace_preds) if mace_preds else 0,
        'mortality_predicted_positive_rate': sum(mortality_preds) / len(mortality_preds) if mortality_preds else 0,
        'mace_mean_predicted_prob': float(np.mean(mace_probs_scored)) if mace_probs_scored else None,
        'mortality_mean_predicted_prob': float(np.mean(mort_probs_scored)) if mort_probs_scored else None,
    }

    final_metrics = {**mace_metrics, **mortality_metrics, **overall_metrics, **distribution_metrics}
    return final_metrics
