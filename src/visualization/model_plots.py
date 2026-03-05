"""Model performance visualization module."""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, roc_auc_score
)
from sklearn.calibration import calibration_curve
from src.config import REPORTS_DIR, METRICS_PATH
from src.models.predict import predict, predict_proba

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')


def plot_roc_curves_comparison(models: dict, X_test: pd.DataFrame, y_test: pd.Series, 
                                save_path: Path = None) -> None:
    """Plot ROC curves for all models with enhanced styling."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    for i, (name, pipe) in enumerate(models.items()):
        y_proba = predict_proba(pipe, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2.5,
                label=f'{name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
        ax.fill_between(fpr, tpr, alpha=0.1, color=colors[i % len(colors)])
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_precision_recall_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series,
                                  save_path: Path = None) -> None:
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    for i, (name, pipe) in enumerate(models.items()):
        y_proba = predict_proba(pipe, X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, color=colors[i % len(colors)], linewidth=2.5,
                label=f'{name.replace("_", " ").title()} (AUC = {pr_auc:.3f})')
    
    # Baseline (proportion of positive class)
    baseline = y_test.mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, 
               label=f'Baseline ({baseline:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve Comparison', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_confusion_matrices(models: dict, X_test: pd.DataFrame, y_test: pd.Series,
                            save_path: Path = None) -> None:
    """Plot confusion matrices for all models in a grid."""
    n_models = len(models)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (name, pipe) in enumerate(models.items()):
        ax = axes[i]
        y_pred = predict(pipe, X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize for display
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Default', 'Default'],
                    yticklabels=['No Default', 'Default'],
                    annot_kws={'size': 14, 'fontweight': 'bold'})
        
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(f'{name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    
    # Remove empty subplots
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_metrics_comparison(results: dict, save_path: Path = None) -> None:
    """Plot comparison of all metrics across models."""
    # Prepare data
    metrics_df = pd.DataFrame([
        {'model': name, **info['metrics']}
        for name, info in results.items()
    ])
    metrics_df['model'] = metrics_df['model'].str.replace('_', ' ').str.title()
    
    # Melt for plotting
    metrics_long = metrics_df.melt(id_vars=['model'], var_name='metric', value_name='score')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Bar plot
    x = np.arange(len(metrics_df))
    width = 0.15
    metrics = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    for i, metric in enumerate(metrics):
        values = metrics_df[metric].values
        bars = ax.bar(x + i * width, values, width, label=metric.upper(), color=colors[i])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metrics_df['model'])
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_calibration_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series,
                            save_path: Path = None) -> None:
    """Plot calibration curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    for i, (name, pipe) in enumerate(models.items()):
        y_proba = predict_proba(pipe, X_test)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_proba, n_bins=10, strategy='uniform'
        )
        
        ax.plot(mean_predicted_value, fraction_of_positives, 's-',
                color=colors[i % len(colors)], linewidth=2, markersize=8,
                label=f'{name.replace("_", " ").title()}')
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfectly Calibrated')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curves (Reliability Diagram)', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_probability_distributions(models: dict, X_test: pd.DataFrame, y_test: pd.Series,
                                   save_path: Path = None) -> None:
    """Plot predicted probability distributions by actual class."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for i, (name, pipe) in enumerate(models.items()):
        ax = axes[i]
        y_proba = predict_proba(pipe, X_test)
        
        # Split by actual class
        proba_0 = y_proba[y_test == 0]
        proba_1 = y_proba[y_test == 1]
        
        ax.hist(proba_0, bins=30, alpha=0.6, label='No Default (Actual)', 
                color='#2ecc71', density=True)
        ax.hist(proba_1, bins=30, alpha=0.6, label='Default (Actual)', 
                color='#e74c3c', density=True)
        
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold (0.5)')
        ax.set_xlabel('Predicted Probability of Default', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
    
    plt.suptitle('Predicted Probability Distributions by Actual Class', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_model_summary_dashboard(models: dict, results: dict, X_test: pd.DataFrame, 
                                  y_test: pd.Series, save_path: Path = None) -> None:
    """Create a comprehensive model summary dashboard."""
    fig = plt.figure(figsize=(20, 14))
    
    # Create a grid of subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    # 1. ROC Curves (top left, 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    for i, (name, pipe) in enumerate(models.items()):
        y_proba = predict_proba(pipe, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        ax1.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2.5,
                label=f'{name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Metrics Table (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    # Create metrics table
    metrics_data = []
    model_names = []
    for name, info in results.items():
        model_names.append(name.replace('_', ' ').title())
        metrics_data.append([
            f"{info['metrics']['roc_auc']:.3f}",
            f"{info['metrics']['accuracy']:.3f}",
            f"{info['metrics']['precision']:.3f}",
            f"{info['metrics']['recall']:.3f}",
            f"{info['metrics']['f1']:.3f}"
        ])
    
    table = ax2.table(cellText=metrics_data,
                      colLabels=['ROC AUC', 'Accuracy', 'Precision', 'Recall', 'F1'],
                      rowLabels=model_names,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.18] * 5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color the header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#3498db')
            cell.set_text_props(color='white', fontweight='bold')
        if key[1] == -1:
            cell.set_text_props(fontweight='bold')
    
    ax2.set_title('Performance Metrics (CV)', fontsize=14, fontweight='bold', pad=20)
    
    # 3. PR Curves (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    for i, (name, pipe) in enumerate(models.items()):
        y_proba = predict_proba(pipe, X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        ax3.plot(recall, precision, color=colors[i % len(colors)], linewidth=2,
                label=f'{name.replace("_", " ").title()[:10]}... ({pr_auc:.2f})')
    ax3.set_xlabel('Recall', fontsize=10)
    ax3.set_ylabel('Precision', fontsize=10)
    ax3.set_title('PR Curves', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='lower left')
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrices (middle center and right)
    for idx, (name, pipe) in enumerate(list(models.items())[:2]):
        ax = fig.add_subplot(gs[1, idx + 1])
        y_pred = predict(pipe, X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                    annot_kws={'size': 12, 'fontweight': 'bold'})
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_title(f'CM: {name.replace("_", " ").title()[:15]}', fontsize=11, fontweight='bold')
    
    # 5. Probability Distributions (bottom, 3 columns)
    for idx, (name, pipe) in enumerate(models.items()):
        ax = fig.add_subplot(gs[2, idx])
        y_proba = predict_proba(pipe, X_test)
        proba_0 = y_proba[y_test == 0]
        proba_1 = y_proba[y_test == 1]
        ax.hist(proba_0, bins=25, alpha=0.6, label='No Default', color='#2ecc71', density=True)
        ax.hist(proba_1, bins=25, alpha=0.6, label='Default', color='#e74c3c', density=True)
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{name.replace("_", " ").title()[:15]}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
    
    plt.suptitle('Credit Risk Model Performance Dashboard', 
                 fontsize=20, fontweight='bold', y=1.01)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def run_model_visualizations(models: dict, results: dict, X_test: pd.DataFrame, 
                              y_test: pd.Series) -> dict:
    """Run all model performance visualizations and save them."""
    model_dir = REPORTS_DIR / "model_plots"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    plots = {}
    
    print("  Generating ROC curves...")
    plot_roc_curves_comparison(models, X_test, y_test, model_dir / "roc_curves.png")
    plots["roc_curves"] = model_dir / "roc_curves.png"
    
    print("  Generating PR curves...")
    plot_precision_recall_curves(models, X_test, y_test, model_dir / "pr_curves.png")
    plots["pr_curves"] = model_dir / "pr_curves.png"
    
    print("  Generating confusion matrices...")
    plot_confusion_matrices(models, X_test, y_test, model_dir / "confusion_matrices.png")
    plots["confusion_matrices"] = model_dir / "confusion_matrices.png"
    
    print("  Generating metrics comparison...")
    plot_metrics_comparison(results, model_dir / "metrics_comparison.png")
    plots["metrics_comparison"] = model_dir / "metrics_comparison.png"
    
    print("  Generating calibration curves...")
    plot_calibration_curves(models, X_test, y_test, model_dir / "calibration_curves.png")
    plots["calibration_curves"] = model_dir / "calibration_curves.png"
    
    print("  Generating probability distributions...")
    plot_probability_distributions(models, X_test, y_test, model_dir / "probability_distributions.png")
    plots["probability_distributions"] = model_dir / "probability_distributions.png"
    
    print("  Generating model summary dashboard...")
    plot_model_summary_dashboard(models, results, X_test, y_test, model_dir / "model_dashboard.png")
    plots["model_dashboard"] = model_dir / "model_dashboard.png"
    
    print(f"  Model plots saved to {model_dir}")
    return plots
