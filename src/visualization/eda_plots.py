"""Exploratory Data Analysis visualization module."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import REPORTS_DIR, TARGET

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_target_distribution(df: pd.DataFrame, save_path: Path = None) -> None:
    """Plot the distribution of the target variable."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    counts = df[TARGET].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    axes[0].bar(['No Default (0)', 'Default (1)'], counts.values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_title('Default Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, f'{v:,}', ha='center', fontsize=11, fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts.values, labels=['No Default', 'Default'], autopct='%1.1f%%', 
                colors=colors, explode=(0, 0.05), shadow=True, startangle=90,
                textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Default Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame, save_path: Path = None) -> None:
    """Plot distributions of key numeric features."""
    num_cols = ['loan_amnt', 'annual_inc', 'dti', 'fico_range_low', 'revol_util', 'delinq_2yrs']
    num_cols = [c for c in num_cols if c in df.columns]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        ax = axes[i]
        # Plot histogram with KDE
        data = df[col].dropna()
        if col == 'annual_inc':
            data = data[data < data.quantile(0.99)]  # Remove outliers for better viz
        
        ax.hist(data, bins=40, color='#3498db', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.axvline(data.mean(), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {data.mean():,.1f}')
        ax.axvline(data.median(), color='#2ecc71', linestyle='-', linewidth=2, label=f'Median: {data.median():,.1f}')
        ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlabel('')
        ax.tick_params(axis='both', labelsize=9)
    
    # Remove empty subplots if any
    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Distribution of Key Numeric Features', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_categorical_distributions(df: pd.DataFrame, save_path: Path = None) -> None:
    """Plot distributions of categorical features."""
    cat_cols = ['home_ownership', 'purpose', 'application_type']
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for i, col in enumerate(cat_cols):
        ax = axes[i]
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, ax=ax, order=order, hue=col, palette='viridis', legend=False)
        ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        
        # Add counts on bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height()):,}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Distribution of Categorical Features', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path = None) -> None:
    """Plot correlation heatmap for numeric features."""
    numeric_data = df.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax, annot_kws={'size': 8},
                cbar_kws={'label': 'Correlation Coefficient'})
    
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_default_by_features(df: pd.DataFrame, save_path: Path = None) -> None:
    """Plot default rates by different feature categories."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Default rate by FICO band
    ax = axes[0, 0]
    if 'fico_band' in df.columns:
        default_by_fico = df.groupby('fico_band')[TARGET].mean().sort_values(ascending=False)
        bars = ax.bar(range(len(default_by_fico)), default_by_fico.values, 
                      color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(default_by_fico))))
        ax.set_xticks(range(len(default_by_fico)))
        ax.set_xticklabels(default_by_fico.index, rotation=30)
        ax.set_ylabel('Default Rate', fontsize=11)
        ax.set_title('Default Rate by FICO Band', fontsize=12, fontweight='bold')
        for bar, val in zip(bars, default_by_fico.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')
    
    # Default rate by home ownership
    ax = axes[0, 1]
    if 'home_ownership' in df.columns:
        default_by_home = df.groupby('home_ownership')[TARGET].mean().sort_values(ascending=False)
        bars = ax.bar(range(len(default_by_home)), default_by_home.values, 
                      color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(default_by_home))))
        ax.set_xticks(range(len(default_by_home)))
        ax.set_xticklabels(default_by_home.index, rotation=30)
        ax.set_ylabel('Default Rate', fontsize=11)
        ax.set_title('Default Rate by Home Ownership', fontsize=12, fontweight='bold')
        for bar, val in zip(bars, default_by_home.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')
    
    # Default rate by purpose
    ax = axes[1, 0]
    if 'purpose' in df.columns:
        default_by_purpose = df.groupby('purpose')[TARGET].mean().sort_values(ascending=False)
        bars = ax.barh(range(len(default_by_purpose)), default_by_purpose.values,
                       color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(default_by_purpose))))
        ax.set_yticks(range(len(default_by_purpose)))
        ax.set_yticklabels(default_by_purpose.index)
        ax.set_xlabel('Default Rate', fontsize=11)
        ax.set_title('Default Rate by Loan Purpose', fontsize=12, fontweight='bold')
        for bar, val in zip(bars, default_by_purpose.values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1%}', va='center', fontsize=10, fontweight='bold')
    
    # Box plot: loan amount by default status
    ax = axes[1, 1]
    colors = ['#2ecc71', '#e74c3c']
    box = ax.boxplot([df[df[TARGET] == 0]['loan_amnt'], df[df[TARGET] == 1]['loan_amnt']],
                     labels=['No Default', 'Default'], patch_artist=True)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Loan Amount ($)', fontsize=11)
    ax.set_title('Loan Amount Distribution by Default Status', fontsize=12, fontweight='bold')
    
    plt.suptitle('Default Rate Analysis by Feature Categories', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_feature_vs_default(df: pd.DataFrame, save_path: Path = None) -> None:
    """Plot key features vs default status."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    features = ['dti', 'fico_range_low', 'revol_util', 'loan_to_income', 'delinq_2yrs', 'annual_inc']
    features = [f for f in features if f in df.columns]
    
    for i, feat in enumerate(features):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        data_0 = df[df[TARGET] == 0][feat].dropna()
        data_1 = df[df[TARGET] == 1][feat].dropna()
        
        # Handle outliers for annual_inc
        if feat == 'annual_inc':
            data_0 = data_0[data_0 < data_0.quantile(0.99)]
            data_1 = data_1[data_1 < data_1.quantile(0.99)]
        
        ax.hist(data_0, bins=30, alpha=0.6, label='No Default', color='#2ecc71', density=True)
        ax.hist(data_1, bins=30, alpha=0.6, label='Default', color='#e74c3c', density=True)
        ax.set_title(feat.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlabel(feat.replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
    
    plt.suptitle('Feature Distributions by Default Status', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def run_eda_visualizations(df: pd.DataFrame) -> dict:
    """Run all EDA visualizations and save them."""
    eda_dir = REPORTS_DIR / "eda_plots"
    eda_dir.mkdir(parents=True, exist_ok=True)
    
    plots = {}
    
    print("  Generating target distribution plot...")
    plot_target_distribution(df, eda_dir / "target_distribution.png")
    plots["target_distribution"] = eda_dir / "target_distribution.png"
    
    print("  Generating numeric distributions plot...")
    plot_numeric_distributions(df, eda_dir / "numeric_distributions.png")
    plots["numeric_distributions"] = eda_dir / "numeric_distributions.png"
    
    print("  Generating categorical distributions plot...")
    plot_categorical_distributions(df, eda_dir / "categorical_distributions.png")
    plots["categorical_distributions"] = eda_dir / "categorical_distributions.png"
    
    print("  Generating correlation heatmap...")
    plot_correlation_heatmap(df, eda_dir / "correlation_heatmap.png")
    plots["correlation_heatmap"] = eda_dir / "correlation_heatmap.png"
    
    print("  Generating default by features plot...")
    plot_default_by_features(df, eda_dir / "default_by_features.png")
    plots["default_by_features"] = eda_dir / "default_by_features.png"
    
    print("  Generating feature vs default plot...")
    plot_feature_vs_default(df, eda_dir / "feature_vs_default.png")
    plots["feature_vs_default"] = eda_dir / "feature_vs_default.png"
    
    print(f"  EDA plots saved to {eda_dir}")
    return plots
