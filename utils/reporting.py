"""
Comprehensive evaluation reporting utilities
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any


def save_comprehensive_evaluation_summary(final_metrics, y_true_list, y_prob_list, predictions, probability_predictions, task_names, save_dir):
    """Save a comprehensive summary of all evaluation results"""
    
    # Create comprehensive summary
    evaluation_summary = {
        'evaluation_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(predictions),
            'tasks_evaluated': task_names
        },
        'performance_metrics': {
            'MACE': {
                'accuracy': final_metrics.get('mace_accuracy', 0),
                'sensitivity': final_metrics.get('mace_sensitivity', 0),
                'specificity': final_metrics.get('mace_specificity', 0),
                'precision': final_metrics.get('mace_precision', 0),
                'f1_score': final_metrics.get('mace_f1', 0),
                'auc_score': final_metrics.get('mace_auc_from_probs', final_metrics.get('mace_auc', 0)),
                'positive_rate': final_metrics.get('mace_positive_rate', 0),
                'predicted_positive_rate': final_metrics.get('mace_predicted_positive_rate', 0)
            },
            'Mortality': {
                'accuracy': final_metrics.get('mortality_accuracy', 0),
                'sensitivity': final_metrics.get('mortality_sensitivity', 0),
                'specificity': final_metrics.get('mortality_specificity', 0),
                'precision': final_metrics.get('mortality_precision', 0),
                'f1_score': final_metrics.get('mortality_f1', 0),
                'auc_score': final_metrics.get('mortality_auc_from_probs', final_metrics.get('mortality_auc', 0)),
                'positive_rate': final_metrics.get('mortality_positive_rate', 0),
                'predicted_positive_rate': final_metrics.get('mortality_predicted_positive_rate', 0)
            }
        },
        'overall_performance': {
            'mean_accuracy': final_metrics.get('mean_accuracy', 0),
            'mean_sensitivity': final_metrics.get('mean_sensitivity', 0),
            'mean_specificity': final_metrics.get('mean_specificity', 0),
            'mean_f1': final_metrics.get('mean_f1', 0),
            'mean_auc': final_metrics.get('mean_auc', 0)
        },
        'output_files_generated': {
            'individual_plots': [f'{task.lower()}_roc_curve.png' for task in task_names],
            'individual_pdfs': [f'{task.lower()}_roc_curve.pdf' for task in task_names],
            'combined_plot': 'combined_roc_curves.png',
            'data_files': {
                'MACE': ['mace_roc_data.json', 'mace_roc_curve.csv', 'mace_predictions.csv'],
                'Mortality': ['mortality_roc_data.json', 'mortality_roc_curve.csv', 'mortality_predictions.csv']
            },
            'summary_files': ['evaluation_summary.json', 'roc_comparison_summary.json']
        }
    }
    
    # Save comprehensive summary as JSON
    summary_path = os.path.join(save_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print(f"Comprehensive evaluation summary saved to {summary_path}")
    
    # Also save as CSV for easy viewing
    csv_summary_path = os.path.join(save_dir, 'evaluation_summary.csv')
    
    # Create a flattened CSV-friendly format
    csv_data = []
    for task in task_names:
        task_lower = task.lower()
        row = {
            'Task': task,
            'Accuracy': final_metrics.get(f'{task_lower}_accuracy', 0),
            'Sensitivity': final_metrics.get(f'{task_lower}_sensitivity', 0),
            'Specificity': final_metrics.get(f'{task_lower}_specificity', 0),
            'Precision': final_metrics.get(f'{task_lower}_precision', 0),
            'F1_Score': final_metrics.get(f'{task_lower}_f1', 0),
            'AUC_Score': final_metrics.get(f'{task_lower}_auc_from_probs', final_metrics.get(f'{task_lower}_auc', 0)),
            'Positive_Rate': final_metrics.get(f'{task_lower}_positive_rate', 0),
            'Predicted_Positive_Rate': final_metrics.get(f'{task_lower}_predicted_positive_rate', 0)
        }
        csv_data.append(row)
    
    # Add overall metrics row
    csv_data.append({
        'Task': 'Overall',
        'Accuracy': final_metrics.get('mean_accuracy', 0),
        'Sensitivity': final_metrics.get('mean_sensitivity', 0),
        'Specificity': final_metrics.get('mean_specificity', 0),
        'Precision': (final_metrics.get('mace_precision', 0) + final_metrics.get('mortality_precision', 0)) / 2,
        'F1_Score': final_metrics.get('mean_f1', 0),
        'AUC_Score': final_metrics.get('mean_auc', 0),
        'Positive_Rate': (final_metrics.get('mace_positive_rate', 0) + final_metrics.get('mortality_positive_rate', 0)) / 2,
        'Predicted_Positive_Rate': (final_metrics.get('mace_predicted_positive_rate', 0) + final_metrics.get('mortality_predicted_positive_rate', 0)) / 2
    })
    
    df_summary = pd.DataFrame(csv_data)
    df_summary.to_csv(csv_summary_path, index=False, float_format='%.4f')
    
    print(f"Evaluation summary CSV saved to {csv_summary_path}")
    print_files_generated_summary(task_names)


def print_files_generated_summary(task_names: List[str]):
    """Print a summary of all files generated during evaluation"""
    print("\n" + "="*80)
    print("FILES GENERATED:")
    print("="*80)
    print("Individual ROC Curves:")
    for task in task_names:
        print(f"  • {task.lower()}_roc_curve.png/pdf")
        print(f"  • {task.lower()}_roc_data.json")
        print(f"  • {task.lower()}_roc_curve.csv")
        print(f"  • {task.lower()}_predictions.csv")
    print("\nCombined Plots:")
    print("  • combined_roc_curves.png/pdf")
    print("\nSummary Files:")
    print("  • evaluation_summary.json")
    print("  • evaluation_summary.csv")
    print("  • roc_comparison_summary.json")
    print("="*80)


def print_evaluation_results(final_metrics: Dict[str, Any], title: str = "EVALUATION RESULTS"):
    """Print formatted evaluation results"""
    def _fmt_auc(value):
        return f"{value:.4f}" if value is not None else "N/A"

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    mace_auc = final_metrics.get('mace_auc_from_probs', final_metrics.get('mace_auc'))
    mort_auc = final_metrics.get('mortality_auc_from_probs', final_metrics.get('mortality_auc'))
    print(f"MACE - AUC: {_fmt_auc(mace_auc)}")
    print(f"MACE - Accuracy: {final_metrics.get('mace_accuracy', 0):.4f}, F1: {final_metrics.get('mace_f1', 0):.4f}")
    print(f"MACE - Sensitivity: {final_metrics.get('mace_sensitivity', 0):.4f}, Specificity: {final_metrics.get('mace_specificity', 0):.4f}")
    print(f"Mortality - AUC: {_fmt_auc(mort_auc)}")
    print(f"Mortality - Accuracy: {final_metrics.get('mortality_accuracy', 0):.4f}, F1: {final_metrics.get('mortality_f1', 0):.4f}")
    print(f"Mortality - Sensitivity: {final_metrics.get('mortality_sensitivity', 0):.4f}, Specificity: {final_metrics.get('mortality_specificity', 0):.4f}")
    print(f"Overall - Mean AUC: {_fmt_auc(final_metrics.get('mean_auc'))}")
    if 'mean_f1' in final_metrics:
        print(f"Overall - Mean F1: {final_metrics.get('mean_f1', 0):.4f}")
    print("=" * 50)


def save_final_results(final_metrics: Dict[str, Any], model_info: Dict[str, Any], 
                      training_config: Dict[str, Any], predictions: List, 
                      probability_predictions: List, true_labels: List, 
                      generated_texts: List, save_dir: str, filename: str = "final_results.json"):
    """Save comprehensive final results to JSON file"""
    final_results = {
        'model_info': model_info,
        'training_config': training_config,
        'final_metrics': final_metrics,
        'sample_predictions': predictions[:10],
        'sample_probability_predictions': probability_predictions[:10],
        'sample_true_labels': true_labels[:10],
        'sample_generated_texts': generated_texts[:10]
    }
    
    results_path = os.path.join(save_dir, filename)
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Final results saved to {results_path}")
    
    return results_path


def save_evaluation_only_results(final_metrics: Dict[str, Any], predictions: List,
                                probability_predictions: List, true_labels: List,
                                generated_texts: List, save_dir: str):
    """Save evaluation-only results to JSON file"""
    results = {
        'metrics': final_metrics,
        'predictions': predictions[:10],  # Save first 10 for inspection
        'probability_predictions': probability_predictions[:10],
        'true_labels': true_labels[:10],
        'generated_examples': generated_texts[:10]
    }
    
    results_path = os.path.join(save_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    return results_path