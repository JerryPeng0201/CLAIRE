"""
ROC curve plotting and visualization utilities
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d


def create_individual_roc_plots(y_true_list, y_prob_list, task_names, save_dir, colors):
    """Create separate ROC plots for each task and save data to files"""
    
    # Create individual plots for each task
    for i, (y_true, y_prob, task_name) in enumerate(zip(y_true_list, y_prob_list, task_names)):
        if len(set(y_true)) < 2:
            print(f"Warning: Only one class present for {task_name}, skipping")
            continue
        
        y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

        fpr, tpr, thresholds = roc_curve(y_true, y_prob_arr)
        roc_auc = auc(fpr, tpr)
        
        # Create ultra-smooth curves
        if len(fpr) >= 3:
            fpr_fine = np.linspace(0, 1, 500)
            
            # Fixed code with better error handling
            try:
                # First ensure FPR is strictly increasing
                if len(fpr) > 1:
                    # Remove duplicate FPR values and ensure strict increasing
                    unique_indices = []
                    prev_fpr = -1
                    for i, curr_fpr in enumerate(fpr):
                        if curr_fpr > prev_fpr:
                            unique_indices.append(i)
                            prev_fpr = curr_fpr
                    
                    if len(unique_indices) >= 3:
                        fpr_clean = fpr[unique_indices]
                        tpr_clean = tpr[unique_indices]
                        
                        fpr_fine = np.linspace(0, 1, 500)
                        
                        # Try Akima interpolation
                        try:
                            from scipy.interpolate import Akima1DInterpolator
                            akima_interp = Akima1DInterpolator(fpr_clean, tpr_clean)
                            tpr_smooth = akima_interp(fpr_fine)
                            tpr_smooth = np.clip(tpr_smooth, 0, 1)
                            tpr_smooth = gaussian_filter1d(tpr_smooth, sigma=2.0)
                            
                            # Ensure monotonicity
                            for j in range(1, len(tpr_smooth)):
                                if tpr_smooth[j] < tpr_smooth[j-1]:
                                    tpr_smooth[j] = tpr_smooth[j-1]
                                    
                            fpr_plot = fpr_fine
                            tpr_plot = tpr_smooth
                            
                        except (ImportError, ValueError):
                            # Fallback to linear interpolation
                            tpr_plot = np.interp(fpr_fine, fpr_clean, tpr_clean)
                            fpr_plot = fpr_fine
                    else:
                        # Not enough unique points, use original
                        fpr_plot = fpr
                        tpr_plot = tpr
                else:
                    fpr_plot = fpr
                    tpr_plot = tpr
                    
            except Exception as e:
                print(f"Warning: Interpolation failed for {task_name}: {e}")
                # Use original data as fallback
                fpr_plot = fpr
                tpr_plot = tpr
        else:
            fpr_plot = fpr
            tpr_plot = tpr
        
        # Create individual plot
        plt.figure(figsize=(8, 6))
        
        # Plot random classifier
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='Random Classifier')
        
        # Plot task curve
        plt.plot(fpr_plot, tpr_plot, color=colors[i], linewidth=4,
                label=f'{task_name} (AUROC = {roc_auc:.3f})')
        
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title(f'{task_name} ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # Improve plot aesthetics
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        
        # Save individual plot
        os.makedirs(save_dir, exist_ok=True)
        individual_plot_path = os.path.join(save_dir, f'{task_name.lower()}_roc_curve.png')
        plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(individual_plot_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"{task_name} ROC curve saved to {individual_plot_path}")
        
        # Save curve data to files
        save_roc_data(fpr_plot, tpr_plot, fpr, tpr, y_true, y_prob_arr, task_name, roc_auc, save_dir)


def save_roc_data(fpr_smooth, tpr_smooth, fpr_original, tpr_original, y_true, y_prob, task_name, auc_score, save_dir):
    """Save ROC curve data to JSON and CSV files"""
    
    # Prepare data dictionary
    roc_data = {
        'task_name': task_name,
        'auc_score': float(auc_score),
        'total_samples': len(y_true),
        'positive_samples': int(sum(y_true)),
        'negative_samples': int(len(y_true) - sum(y_true)),
        'positive_rate': float(sum(y_true) / len(y_true)),
        'smooth_curve': {
            'fpr': fpr_smooth.tolist(),
            'tpr': tpr_smooth.tolist(),
            'num_points': len(fpr_smooth)
        },
        'original_curve': {
            'fpr': fpr_original.tolist(),
            'tpr': tpr_original.tolist(),
            'num_points': len(fpr_original)
        },
        'predictions': {
            'y_true': [int(x) for x in y_true],
            'y_prob': [float(x) for x in y_prob]
        }
    }
    
    # Save as JSON
    json_path = os.path.join(save_dir, f'{task_name.lower()}_roc_data.json')
    with open(json_path, 'w') as f:
        json.dump(roc_data, f, indent=2)
    
    print(f"{task_name} ROC data saved to {json_path}")
    
    # Save smooth curve as CSV
    csv_path = os.path.join(save_dir, f'{task_name.lower()}_roc_curve.csv')
    df_curve = pd.DataFrame({
        'False_Positive_Rate': fpr_smooth,
        'True_Positive_Rate': tpr_smooth
    })
    df_curve.to_csv(csv_path, index=False)
    
    print(f"{task_name} smooth curve data saved to {csv_path}")
    
    # Save predictions as CSV
    pred_csv_path = os.path.join(save_dir, f'{task_name.lower()}_predictions.csv')
    df_pred = pd.DataFrame({
        'True_Label': y_true,
        'Predicted_Probability': y_prob,
        'Sample_ID': range(len(y_true))
    })
    df_pred.to_csv(pred_csv_path, index=False)
    
    print(f"{task_name} predictions saved to {pred_csv_path}")


def create_combined_roc_plot(y_true_list, y_prob_list, task_names, save_dir, colors):
    """Create a combined ROC plot with both tasks on the same figure"""
    plt.figure(figsize=(10, 8))
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='Random Classifier')
    
    all_auc_scores = []
    
    # Plot each task
    for i, (y_true, y_prob, task_name) in enumerate(zip(y_true_list, y_prob_list, task_names)):
        if len(set(y_true)) < 2:
            continue
            
        y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

        fpr, tpr, _ = roc_curve(y_true, y_prob_arr)
        roc_auc = auc(fpr, tpr)
        all_auc_scores.append(roc_auc)
        
        # Create smooth curves
        if len(fpr) >= 3:
            fpr_fine = np.linspace(0, 1, 500)
            
            try:
                from scipy.interpolate import Akima1DInterpolator
                akima_interp = Akima1DInterpolator(fpr, tpr)
                tpr_smooth = akima_interp(fpr_fine)
                tpr_smooth = np.clip(tpr_smooth, 0, 1)
                tpr_smooth = gaussian_filter1d(tpr_smooth, sigma=2.0)
                
                # Ensure monotonicity
                for j in range(1, len(tpr_smooth)):
                    if tpr_smooth[j] < tpr_smooth[j-1]:
                        tpr_smooth[j] = tpr_smooth[j-1]
                        
                fpr_plot = fpr_fine
                tpr_plot = tpr_smooth
                
            except ImportError:
                try:
                    f = interpolate.interp1d(fpr, tpr, kind='cubic', bounds_error=False, fill_value=0)
                    tpr_cubic = f(fpr_fine)
                    tpr_smooth = gaussian_filter1d(tpr_cubic, sigma=3.0)
                    
                    for j in range(1, len(tpr_smooth)):
                        if tpr_smooth[j] < tpr_smooth[j-1]:
                            tpr_smooth[j] = tpr_smooth[j-1]
                    tpr_smooth = np.clip(tpr_smooth, 0, 1)
                    
                    fpr_plot = fpr_fine
                    tpr_plot = tpr_smooth
                    
                except:
                    tpr_linear = np.interp(fpr_fine, fpr, tpr)
                    tpr_plot = gaussian_filter1d(tpr_linear, sigma=4.0)
                    fpr_plot = fpr_fine
        else:
            fpr_plot = fpr
            tpr_plot = tpr
        
        linestyle = '-' if i == 0 else '--'
        plt.plot(fpr_plot, tpr_plot, color=colors[i], linewidth=4, linestyle=linestyle,
                label=f'{task_name} (AUROC = {roc_auc:.3f})')
    
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Improve plot aesthetics
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    
    # Save combined plot
    combined_plot_path = os.path.join(save_dir, 'combined_roc_curves.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(combined_plot_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Combined ROC curves saved to {combined_plot_path}")
    
    # Save combined summary
    combined_summary = {
        'comparison_summary': {
            'tasks': task_names,
            'auc_scores': [float(score) for score in all_auc_scores],
            'mean_auc': float(np.mean(all_auc_scores)) if all_auc_scores else 0.0,
            'auc_difference': float(abs(all_auc_scores[0] - all_auc_scores[1])) if len(all_auc_scores) == 2 else 0.0
        }
    }
    
    summary_path = os.path.join(save_dir, 'roc_comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(combined_summary, f, indent=2)
    
    print(f"ROC comparison summary saved to {summary_path}")
    
    return all_auc_scores


def generate_auc_curves_and_data(y_true_list, y_prob_list, task_names, save_dir):
    """Generate separate AUC curves for each task and save data"""
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e']  # Red, Green, Blue, Orange
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create individual ROC plots for each task and save data
    create_individual_roc_plots(y_true_list, y_prob_list, task_names, save_dir, colors)
    
    # Create combined comparison plot
    auc_scores = create_combined_roc_plot(y_true_list, y_prob_list, task_names, save_dir, colors)
    
    return auc_scores