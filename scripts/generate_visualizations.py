#!/usr/bin/env python3
"""
Comprehensive Visualization Suite for Master Run Results
=========================================================
Generates extensive visualizations for analysis and presentation.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering/results/Lichens_Dataset_1_MasterRun")
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
VIZ_DIR = RESULTS_DIR / "visualizations"

# Create visualization subdirectories
SUBDIRS = [
    "01_accuracy_curves",
    "02_parameter_analysis",
    "03_wavelength_analysis",
    "04_efficiency_analysis",
    "05_heatmaps",
    "06_statistical_comparisons",
    "07_3d_surfaces",
    "08_correlation_analysis",
    "09_ranking_analysis",
    "10_summary_figures",
]

# Colors for consistent styling
COLORS = {
    'pca': '#2ecc71',
    'variance': '#e74c3c',
    'none': '#3498db',
    'max_per_excitation': '#f39c12',
    'variance_norm': '#9b59b6',
    'dim1': '#1abc9c',
    'dim3': '#e67e22',
    'percentile': '#16a085',
    'absolute_range': '#c0392b',
    'medium': '#2980b9',
    'high': '#8e44ad',
}

BASELINE_ACCURACY = 0.8815  # From results


def setup_directories():
    """Create visualization directory structure."""
    VIZ_DIR.mkdir(exist_ok=True)
    for subdir in SUBDIRS:
        (VIZ_DIR / subdir).mkdir(exist_ok=True)
    print(f"Created visualization directories in: {VIZ_DIR}")


def load_data() -> Tuple[pd.DataFrame, Dict]:
    """Load all experiment data."""
    print("Loading data...")

    # Load main results
    df = pd.read_csv(RESULTS_DIR / "results.csv")
    baseline = df[df['config'] == 'BASELINE'].iloc[0].to_dict()
    df = df[df['config'] != 'BASELINE'].copy()

    # Add derived columns
    df['config_key'] = df.apply(lambda r: f"{r['dimension_selection_method']}_dim{int(r['n_important_dimensions'])}_{r['perturbation_method']}_{r['normalization_method']}_{r['magnitude_variant']}", axis=1)
    df['dim_method_short'] = df['dimension_selection_method'].map({'variance': 'var', 'pca': 'pca'})
    df['norm_short'] = df['normalization_method'].map({'variance': 'var', 'max_per_excitation': 'max', 'none': 'none'})

    # Load wavelength data
    wavelengths_data = {}
    for _, row in df.iterrows():
        wl_path = EXPERIMENTS_DIR / row['config'] / "wavelengths.json"
        if wl_path.exists():
            with open(wl_path, 'r') as f:
                wavelengths_data[row['config']] = json.load(f)

    print(f"Loaded {len(df)} experiments with {len(wavelengths_data)} wavelength files")
    return df, wavelengths_data, baseline


# ═══════════════════════════════════════════════════════════════════════════
# 1. ACCURACY CURVES
# ═══════════════════════════════════════════════════════════════════════════

def plot_accuracy_curves_all_configs(df: pd.DataFrame):
    """Plot accuracy vs n_bands for all 48 configurations."""
    print("  Plotting all accuracy curves...")

    fig, ax = plt.subplots(figsize=(16, 10))

    configs = df.groupby('config_key')

    # Color by dimension method
    for config_key, group in configs:
        group = group.sort_values('n_bands_to_select')
        color = COLORS['pca'] if 'pca' in config_key else COLORS['variance']
        alpha = 0.6 if 'none' in config_key else 0.3
        linewidth = 2 if 'none' in config_key else 1

        ax.plot(group['n_bands_to_select'], group['accuracy'],
                color=color, alpha=alpha, linewidth=linewidth)

    # Add baseline
    ax.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=2, label=f'Baseline ({BASELINE_ACCURACY:.2%})')

    # Styling
    ax.set_xlabel('Number of Bands Selected', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Accuracy vs Number of Bands - All 48 Configurations', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 185)
    ax.set_ylim(0.25, 1.0)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color=COLORS['pca'], linewidth=2, label='PCA-based'),
        Line2D([0], [0], color=COLORS['variance'], linewidth=2, label='Variance-based'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Baseline ({BASELINE_ACCURACY:.2%})'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "01_accuracy_curves" / "all_configs_accuracy_curves.png", dpi=150)
    plt.savefig(VIZ_DIR / "01_accuracy_curves" / "all_configs_accuracy_curves.pdf")
    plt.close()


def plot_accuracy_curves_best_configs(df: pd.DataFrame):
    """Plot accuracy curves for top performing configurations."""
    print("  Plotting best configuration curves...")

    fig, ax = plt.subplots(figsize=(14, 9))

    # Find top 10 configurations by max accuracy
    top_configs = df.groupby('config_key')['accuracy'].max().nlargest(10).index.tolist()

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(top_configs)))

    for i, config_key in enumerate(top_configs):
        group = df[df['config_key'] == config_key].sort_values('n_bands_to_select')
        max_acc = group['accuracy'].max()
        best_n = group.loc[group['accuracy'].idxmax(), 'n_bands_to_select']

        ax.plot(group['n_bands_to_select'], group['accuracy'],
                color=colors[i], linewidth=2.5, label=f'{config_key[:30]}... ({max_acc:.2%})')

        # Mark peak
        ax.scatter([best_n], [max_acc], color=colors[i], s=100, zorder=5, edgecolor='white', linewidth=2)

    ax.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel('Number of Bands Selected', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Top 10 Configurations - Accuracy Progression', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 185)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "01_accuracy_curves" / "top10_configs_accuracy_curves.png", dpi=150)
    plt.savefig(VIZ_DIR / "01_accuracy_curves" / "top10_configs_accuracy_curves.pdf")
    plt.close()


def plot_accuracy_envelope(df: pd.DataFrame):
    """Plot accuracy envelope (min/max/mean) across n_bands."""
    print("  Plotting accuracy envelope...")

    fig, ax = plt.subplots(figsize=(8, 5))

    stats_df = df.groupby('n_bands_to_select').agg({
        'accuracy': ['min', 'max', 'mean', 'std']
    }).reset_index()
    stats_df.columns = ['n_bands', 'min', 'max', 'mean', 'std']

    # Fill between min and max
    ax.fill_between(stats_df['n_bands'], stats_df['min'], stats_df['max'],
                    alpha=0.25, color='#4A90D9', label='Range (min-max)')

    # Mean line (black for contrast)
    ax.plot(stats_df['n_bands'], stats_df['mean'],
            color='black', linewidth=2, label='Mean')

    # Best line (darker green, thicker)
    ax.plot(stats_df['n_bands'], stats_df['max'],
            color='#1B7A2B', linewidth=2.5, linestyle='--', label='Best')

    # Baseline
    ax.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=1.5, label=f'Baseline ({BASELINE_ACCURACY:.2%})')

    # Mark optimal point with vertical line from peak to x-axis
    best_idx = stats_df['max'].idxmax()
    best_n = stats_df.loc[best_idx, 'n_bands']
    best_acc = stats_df.loc[best_idx, 'max']

    # Vertical dashed line from peak down to x-axis
    ax.plot([best_n, best_n], [0.20, best_acc], color='#1B7A2B', linestyle=':', linewidth=1.5, zorder=4)

    # Star marker at the peak
    ax.scatter([best_n], [best_acc],
               color='gold', s=250, zorder=6, marker='*', edgecolor='#1B7A2B', linewidth=1.5)

    # Label in the middle of the vertical line
    mid_y = (0.20 + best_acc) / 2
    ax.text(best_n + 5, mid_y, f"Peak: {best_acc:.2%}\nn = {int(best_n)}",
            fontsize=14, fontweight='bold', color='#1B7A2B', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#1B7A2B', alpha=0.9))

    ax.set_xlabel('Number of Bands Selected', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_title('Accuracy Envelope Across All Configurations', fontsize=18, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(0, 185)
    ax.set_ylim(0.20, 0.98)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "01_accuracy_curves" / "accuracy_envelope.png", dpi=300)
    plt.savefig(VIZ_DIR / "01_accuracy_curves" / "accuracy_envelope.pdf")
    plt.close()


def plot_accuracy_by_dim_method(df: pd.DataFrame):
    """Compare PCA vs Variance dimension selection."""
    print("  Plotting accuracy by dimension method...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, method in zip(axes, ['pca', 'variance']):
        subset = df[df['dimension_selection_method'] == method]

        stats_df = subset.groupby('n_bands_to_select').agg({
            'accuracy': ['min', 'max', 'mean']
        }).reset_index()
        stats_df.columns = ['n_bands', 'min', 'max', 'mean']

        color = COLORS[method]
        ax.fill_between(stats_df['n_bands'], stats_df['min'], stats_df['max'],
                        alpha=0.3, color=color)
        ax.plot(stats_df['n_bands'], stats_df['mean'], color=color, linewidth=2, label='Mean')
        ax.plot(stats_df['n_bands'], stats_df['max'], color=color, linewidth=2, linestyle='--', label='Best')

        ax.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=2, alpha=0.7)

        best_acc = stats_df['max'].max()
        best_n = stats_df.loc[stats_df['max'].idxmax(), 'n_bands']

        ax.set_xlabel('Number of Bands', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{method.upper()}-based Selection\nBest: {best_acc:.2%} at n={int(best_n)}',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 185)
        ax.set_ylim(0.25, 1.0)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "01_accuracy_curves" / "accuracy_by_dim_method.png", dpi=150)
    plt.savefig(VIZ_DIR / "01_accuracy_curves" / "accuracy_by_dim_method.pdf")
    plt.close()


def plot_accuracy_by_normalization(df: pd.DataFrame):
    """Compare different normalization methods."""
    print("  Plotting accuracy by normalization...")

    fig, ax = plt.subplots(figsize=(14, 8))

    norm_methods = ['none', 'max_per_excitation', 'variance']
    norm_labels = {'none': 'No Normalization', 'max_per_excitation': 'Max per Excitation', 'variance': 'Variance'}

    for method in norm_methods:
        subset = df[df['normalization_method'] == method]
        stats_df = subset.groupby('n_bands_to_select')['accuracy'].max().reset_index()

        color = COLORS.get(method, COLORS.get(method + '_norm', 'gray'))
        ax.plot(stats_df['n_bands_to_select'], stats_df['accuracy'],
                linewidth=2.5, label=norm_labels[method], color=color)

    ax.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=2, label='Baseline')

    ax.set_xlabel('Number of Bands Selected', fontsize=14)
    ax.set_ylabel('Best Accuracy', fontsize=14)
    ax.set_title('Best Accuracy by Normalization Method', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlim(0, 185)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "01_accuracy_curves" / "accuracy_by_normalization.png", dpi=150)
    plt.savefig(VIZ_DIR / "01_accuracy_curves" / "accuracy_by_normalization.pdf")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 2. PARAMETER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_parameter_importance_boxplots(df: pd.DataFrame):
    """Box plots showing accuracy distribution for each parameter value."""
    print("  Plotting parameter importance box plots...")

    params = [
        ('dimension_selection_method', 'Dimension Selection'),
        ('n_important_dimensions', 'N Important Dimensions'),
        ('perturbation_method', 'Perturbation Method'),
        ('normalization_method', 'Normalization'),
        ('magnitude_variant', 'Magnitude Variant'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for ax, (param, title) in zip(axes[:-1], params):
        df.boxplot(column='accuracy', by=param, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Accuracy', fontsize=12)
        plt.suptitle('')

    # Remove extra subplot
    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "02_parameter_analysis" / "parameter_boxplots.png", dpi=150)
    plt.savefig(VIZ_DIR / "02_parameter_analysis" / "parameter_boxplots.pdf")
    plt.close()


def plot_parameter_violin_plots(df: pd.DataFrame):
    """Violin plots for parameter distributions."""
    print("  Plotting parameter violin plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Dimension selection method
    sns.violinplot(data=df, x='dimension_selection_method', y='accuracy', ax=axes[0, 0],
                   palette=[COLORS['pca'], COLORS['variance']])
    axes[0, 0].set_title('Accuracy by Dimension Selection Method', fontsize=14, fontweight='bold')
    axes[0, 0].axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', alpha=0.7)

    # Normalization
    sns.violinplot(data=df, x='normalization_method', y='accuracy', ax=axes[0, 1])
    axes[0, 1].set_title('Accuracy by Normalization Method', fontsize=14, fontweight='bold')
    axes[0, 1].axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', alpha=0.7)

    # N dimensions
    sns.violinplot(data=df, x='n_important_dimensions', y='accuracy', ax=axes[1, 0])
    axes[1, 0].set_title('Accuracy by N Important Dimensions', fontsize=14, fontweight='bold')
    axes[1, 0].axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', alpha=0.7)

    # Magnitude variant
    sns.violinplot(data=df, x='magnitude_variant', y='accuracy', ax=axes[1, 1],
                   palette=[COLORS['medium'], COLORS['high']])
    axes[1, 1].set_title('Accuracy by Magnitude Variant', fontsize=14, fontweight='bold')
    axes[1, 1].axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "02_parameter_analysis" / "parameter_violin_plots.png", dpi=150)
    plt.savefig(VIZ_DIR / "02_parameter_analysis" / "parameter_violin_plots.pdf")
    plt.close()


def plot_parameter_interaction(df: pd.DataFrame):
    """Plot parameter interactions."""
    print("  Plotting parameter interactions...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Dim method × Normalization
    pivot1 = df.groupby(['dimension_selection_method', 'normalization_method'])['accuracy'].max().unstack()
    pivot1.plot(kind='bar', ax=axes[0, 0], rot=0)
    axes[0, 0].set_title('Best Accuracy: Dim Method × Normalization', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].legend(title='Normalization')

    # Dim method × Perturbation
    pivot2 = df.groupby(['dimension_selection_method', 'perturbation_method'])['accuracy'].max().unstack()
    pivot2.plot(kind='bar', ax=axes[0, 1], rot=0)
    axes[0, 1].set_title('Best Accuracy: Dim Method × Perturbation', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].legend(title='Perturbation')

    # N dims × Normalization
    pivot3 = df.groupby(['n_important_dimensions', 'normalization_method'])['accuracy'].max().unstack()
    pivot3.plot(kind='bar', ax=axes[1, 0], rot=0)
    axes[1, 0].set_title('Best Accuracy: N Dims × Normalization', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].legend(title='Normalization')

    # Magnitude × Normalization
    pivot4 = df.groupby(['magnitude_variant', 'normalization_method'])['accuracy'].max().unstack()
    pivot4.plot(kind='bar', ax=axes[1, 1], rot=0)
    axes[1, 1].set_title('Best Accuracy: Magnitude × Normalization', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].legend(title='Normalization')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "02_parameter_analysis" / "parameter_interactions.png", dpi=150)
    plt.savefig(VIZ_DIR / "02_parameter_analysis" / "parameter_interactions.pdf")
    plt.close()


def plot_parameter_effect_sizes(df: pd.DataFrame):
    """Calculate and plot effect sizes for each parameter."""
    print("  Plotting parameter effect sizes...")

    params = [
        'dimension_selection_method',
        'n_important_dimensions',
        'perturbation_method',
        'normalization_method',
        'magnitude_variant'
    ]

    effects = []
    for param in params:
        groups = df.groupby(param)['accuracy'].mean()
        effect = groups.max() - groups.min()
        effects.append((param, effect, groups.idxmax()))

    effects_df = pd.DataFrame(effects, columns=['Parameter', 'Effect Size', 'Best Value'])
    effects_df = effects_df.sort_values('Effect Size', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(effects_df)))
    bars = ax.barh(effects_df['Parameter'], effects_df['Effect Size'], color=colors)

    # Add labels
    for i, (_, row) in enumerate(effects_df.iterrows()):
        ax.text(row['Effect Size'] + 0.002, i, f"Best: {row['Best Value']}",
                va='center', fontsize=10)

    ax.set_xlabel('Effect Size (Max - Min Mean Accuracy)', fontsize=14)
    ax.set_title('Parameter Effect Sizes on Accuracy', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "02_parameter_analysis" / "parameter_effect_sizes.png", dpi=150)
    plt.savefig(VIZ_DIR / "02_parameter_analysis" / "parameter_effect_sizes.pdf")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 3. WAVELENGTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_wavelength_heatmap(df: pd.DataFrame, wavelengths_data: Dict):
    """Mean wavelength importance heatmap across above-baseline config groups.

    Instead of counting selection frequency (which double-counts due to shared
    rankings within config groups), this computes a normalized importance score:
        importance = 1 - (rank - 1) / max_rank
    for each wavelength, averaged across the 16 above-baseline (PCA-based) config
    groups. Only the complete ranking from the max-band experiment per group is used.
    """
    print("  Plotting wavelength importance heatmap...")

    import re

    # Step 1: Identify above-baseline config groups (PCA-based)
    # A config group is defined by all params except band count
    # Config name format: bands_<N>_<group_key>
    def extract_group_key(config_name):
        """Remove 'bands_<N>_' prefix to get the config group key."""
        match = re.match(r'^bands_\d+_(.+)$', config_name)
        return match.group(1) if match else None

    # Find PCA-based configs that exceed baseline at any band count
    pca_df = df[df['dimension_selection_method'] == 'pca']
    above_baseline = pca_df[pca_df['accuracy'] > BASELINE_ACCURACY]

    if len(above_baseline) == 0:
        print("    WARNING: No above-baseline configs found, using all PCA configs")
        above_baseline = pca_df

    above_baseline_groups = set()
    for _, row in above_baseline.iterrows():
        gk = extract_group_key(row['config'])
        if gk:
            above_baseline_groups.add(gk)

    print(f"    Found {len(above_baseline_groups)} above-baseline config groups")

    # Step 2: For each above-baseline group, get the COMPLETE ranking from the
    # experiment with the maximum n_bands_to_select (e.g., bands_180)
    group_rankings = {}  # group_key -> list of wavelength dicts
    for group_key in above_baseline_groups:
        # Find all configs in this group
        group_configs = {}
        for config_name, wl_data in wavelengths_data.items():
            gk = extract_group_key(config_name)
            if gk == group_key:
                # Extract band count from config name
                match = re.match(r'^bands_(\d+)_', config_name)
                if match:
                    n_bands = int(match.group(1))
                    group_configs[n_bands] = (config_name, wl_data)

        if group_configs:
            # Use the experiment with the most bands (complete ranking)
            max_bands = max(group_configs.keys())
            config_name, wl_data = group_configs[max_bands]
            group_rankings[group_key] = wl_data

    print(f"    Loaded complete rankings for {len(group_rankings)} groups")

    # Step 3: Define the proper emission grid (10nm steps from 420-720)
    emission_grid = list(range(420, 730, 10))  # 420, 430, ..., 720
    excitation_grid = [310, 325, 340, 365, 385, 400, 415, 430]

    # Step 4: Determine valid (ex, em) pairs based on Rayleigh cutoffs
    # 1st order: em >= ex + 40
    # 2nd order: |em - 2*ex| >= 40
    def is_valid_pair(ex, em):
        if em < ex + 40:
            return False
        if abs(em - 2 * ex) < 40:
            return False
        return True

    valid_mask = np.zeros((len(excitation_grid), len(emission_grid)), dtype=bool)
    for i, ex in enumerate(excitation_grid):
        for j, em in enumerate(emission_grid):
            valid_mask[i, j] = is_valid_pair(ex, em)

    # Step 5: Compute mean normalized importance
    importance_matrix = np.zeros((len(excitation_grid), len(emission_grid)))
    count_matrix = np.zeros((len(excitation_grid), len(emission_grid)))

    for group_key, wavelengths in group_rankings.items():
        max_rank = len(wavelengths)
        if max_rank == 0:
            continue

        for wl in wavelengths:
            ex = wl['excitation']
            em = wl['emission']

            # Find closest grid positions
            if ex in excitation_grid:
                i = excitation_grid.index(ex)
            else:
                continue

            # Snap emission to grid using floor (handles 5nm-offset grids like Ex=415nm
            # where emissions are 455,465,... instead of 450,460,... due to a loader bug)
            em_rounded = int(em // 10) * 10
            if em_rounded in emission_grid:
                j = emission_grid.index(em_rounded)
            else:
                continue

            # Normalized importance: rank 1 -> 1.0, rank max -> ~0.0
            importance = 1.0 - (wl['rank'] - 1) / max_rank
            importance_matrix[i, j] += importance
            count_matrix[i, j] += 1

    # Average importance across groups (wavelengths not present get 0)
    n_groups = len(group_rankings)
    if n_groups > 0:
        mean_importance = importance_matrix / n_groups
    else:
        mean_importance = importance_matrix

    # Step 6: Create the heatmap — wide figure spanning both paper columns
    fig, ax = plt.subplots(figsize=(10, 4))

    # Mask invalid cells with NaN for grey display
    display_matrix = mean_importance.copy().astype(float)
    display_matrix[~valid_mask] = np.nan

    # Use pcolormesh (no antialiasing seams between cells unlike imshow)
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color='#D3D3D3')  # Grey for invalid/Rayleigh cells

    masked_matrix = np.ma.masked_invalid(display_matrix)
    im = ax.pcolormesh(masked_matrix, cmap=cmap, vmin=0, vmax=1,
                       edgecolors='face', linewidth=0, rasterized=True)

    # Show EVERY emission tick (10nm steps) for precise wavelength identification
    ax.set_xticks([i + 0.5 for i in range(len(emission_grid))])
    ax.set_xticklabels([str(e) for e in emission_grid], rotation=90, fontsize=12)
    ax.set_yticks([i + 0.5 for i in range(len(excitation_grid))])
    ax.set_yticklabels([int(e) for e in excitation_grid], fontsize=12)

    ax.tick_params(which='minor', length=0)

    ax.set_xlabel('Emission Wavelength (nm)', fontsize=13)
    ax.set_ylabel('Excitation Wavelength (nm)', fontsize=13)
    ax.set_title('Mean Wavelength Importance Across Above-Baseline Configurations',
                 fontsize=15, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label('Mean Importance Score (0 = low, 1 = high)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "03_wavelength_analysis" / "wavelength_heatmap.png", dpi=300)
    plt.savefig(VIZ_DIR / "03_wavelength_analysis" / "wavelength_heatmap.pdf")
    plt.close()


def plot_top_wavelengths_bar(df: pd.DataFrame, wavelengths_data: Dict):
    """Bar chart of most frequently selected wavelengths."""
    print("  Plotting top wavelengths bar chart...")

    wl_counts = defaultdict(lambda: {'count': 0, 'ranks': [], 'excitation': 0, 'emission': 0})

    for config, wavelengths in wavelengths_data.items():
        for wl in wavelengths:
            combo = wl['combination_name']
            wl_counts[combo]['count'] += 1
            wl_counts[combo]['ranks'].append(wl['rank'])
            wl_counts[combo]['excitation'] = wl['excitation']
            wl_counts[combo]['emission'] = wl['emission']

    # Top 30 by count
    top_wl = sorted(wl_counts.items(), key=lambda x: x[1]['count'], reverse=True)[:30]

    fig, ax = plt.subplots(figsize=(16, 10))

    names = [w[0] for w in top_wl]
    counts = [w[1]['count'] for w in top_wl]
    avg_ranks = [np.mean(w[1]['ranks']) for w in top_wl]

    colors = plt.cm.viridis(np.array(avg_ranks) / max(avg_ranks))

    bars = ax.barh(range(len(names)), counts, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel('Selection Count (across all experiments)', fontsize=14)
    ax.set_title('Top 30 Most Frequently Selected Wavelength Combinations', fontsize=16, fontweight='bold')

    # Add colorbar for average rank
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=1, vmax=max(avg_ranks)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Average Rank (darker = selected earlier)', fontsize=12)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "03_wavelength_analysis" / "top_wavelengths_bar.png", dpi=150)
    plt.savefig(VIZ_DIR / "03_wavelength_analysis" / "top_wavelengths_bar.pdf")
    plt.close()


def plot_wavelength_scatter(df: pd.DataFrame, wavelengths_data: Dict):
    """Scatter plot of all selected wavelengths colored by frequency."""
    print("  Plotting wavelength scatter...")

    wl_data = defaultdict(lambda: {'count': 0, 'avg_rank': []})

    for config, wavelengths in wavelengths_data.items():
        for wl in wavelengths:
            key = (wl['excitation'], wl['emission'])
            wl_data[key]['count'] += 1
            wl_data[key]['avg_rank'].append(wl['rank'])

    excitations = [k[0] for k in wl_data.keys()]
    emissions = [k[1] for k in wl_data.keys()]
    counts = [v['count'] for v in wl_data.values()]
    avg_ranks = [np.mean(v['avg_rank']) for v in wl_data.values()]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # By count
    scatter1 = axes[0].scatter(excitations, emissions, c=counts, s=[c*2 for c in counts],
                               cmap='YlOrRd', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Excitation Wavelength (nm)', fontsize=14)
    axes[0].set_ylabel('Emission Wavelength (nm)', fontsize=14)
    axes[0].set_title('Wavelength Selection Frequency\n(size & color = frequency)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0], label='Selection Count')

    # By average rank
    scatter2 = axes[1].scatter(excitations, emissions, c=avg_ranks, s=[c*2 for c in counts],
                               cmap='viridis_r', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Excitation Wavelength (nm)', fontsize=14)
    axes[1].set_ylabel('Emission Wavelength (nm)', fontsize=14)
    axes[1].set_title('Wavelength Selection Ranking\n(color = avg rank, darker = better)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='Average Rank')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "03_wavelength_analysis" / "wavelength_scatter.png", dpi=150)
    plt.savefig(VIZ_DIR / "03_wavelength_analysis" / "wavelength_scatter.pdf")
    plt.close()


def plot_excitation_emission_distributions(df: pd.DataFrame, wavelengths_data: Dict):
    """Distribution of selected excitation and emission wavelengths."""
    print("  Plotting excitation/emission distributions...")

    all_excitations = []
    all_emissions = []

    for config, wavelengths in wavelengths_data.items():
        for wl in wavelengths[:20]:  # Top 20 from each
            all_excitations.append(wl['excitation'])
            all_emissions.append(wl['emission'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Excitation distribution
    axes[0].hist(all_excitations, bins=20, color=COLORS['pca'], edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Excitation Wavelength (nm)', fontsize=14)
    axes[0].set_ylabel('Frequency', fontsize=14)
    axes[0].set_title('Distribution of Selected Excitation Wavelengths', fontsize=14, fontweight='bold')

    # Emission distribution
    axes[1].hist(all_emissions, bins=30, color=COLORS['variance'], edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Emission Wavelength (nm)', fontsize=14)
    axes[1].set_ylabel('Frequency', fontsize=14)
    axes[1].set_title('Distribution of Selected Emission Wavelengths', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "03_wavelength_analysis" / "excitation_emission_distributions.png", dpi=150)
    plt.savefig(VIZ_DIR / "03_wavelength_analysis" / "excitation_emission_distributions.pdf")
    plt.close()


def plot_top_wavelengths_by_config(df: pd.DataFrame, wavelengths_data: Dict):
    """Compare top wavelengths between best PCA and best variance configs."""
    print("  Plotting top wavelengths by config type...")

    # Get best PCA and variance configs
    pca_df = df[df['dimension_selection_method'] == 'pca']
    var_df = df[df['dimension_selection_method'] == 'variance']

    best_pca = pca_df.loc[pca_df['accuracy'].idxmax(), 'config']
    best_var = var_df.loc[var_df['accuracy'].idxmax(), 'config']

    pca_wl = wavelengths_data.get(best_pca, [])[:15]
    var_wl = wavelengths_data.get(best_var, [])[:15]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # PCA wavelengths
    if pca_wl:
        names = [w['combination_name'] for w in pca_wl]
        scores = [w['influence_score'] for w in pca_wl]
        axes[0].barh(range(len(names)), scores, color=COLORS['pca'], edgecolor='white')
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Influence Score', fontsize=12)
        axes[0].set_title(f'Top 15 Wavelengths (Best PCA Config)\nAcc: {pca_df["accuracy"].max():.2%}',
                         fontsize=12, fontweight='bold')

    # Variance wavelengths
    if var_wl:
        names = [w['combination_name'] for w in var_wl]
        scores = [w['influence_score'] for w in var_wl]
        axes[1].barh(range(len(names)), scores, color=COLORS['variance'], edgecolor='white')
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Influence Score', fontsize=12)
        axes[1].set_title(f'Top 15 Wavelengths (Best Variance Config)\nAcc: {var_df["accuracy"].max():.2%}',
                         fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "03_wavelength_analysis" / "top_wavelengths_by_config.png", dpi=150)
    plt.savefig(VIZ_DIR / "03_wavelength_analysis" / "top_wavelengths_by_config.pdf")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 4. EFFICIENCY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_pareto_frontier(df: pd.DataFrame):
    """Plot Pareto frontier: accuracy vs n_bands."""
    print("  Plotting Pareto frontier...")

    # Get best accuracy for each n_bands
    best_per_n = df.groupby('n_bands_to_select')['accuracy'].max().reset_index()

    # Find Pareto optimal points
    pareto_points = []
    max_acc = 0
    for _, row in best_per_n.sort_values('n_bands_to_select').iterrows():
        if row['accuracy'] > max_acc:
            pareto_points.append((row['n_bands_to_select'], row['accuracy']))
            max_acc = row['accuracy']

    fig, ax = plt.subplots(figsize=(14, 9))

    # All points
    ax.scatter(df['n_bands_to_select'], df['accuracy'], alpha=0.1, s=20, color='gray', label='All experiments')

    # Best per n_bands
    ax.scatter(best_per_n['n_bands_to_select'], best_per_n['accuracy'],
               s=50, color='steelblue', alpha=0.8, label='Best per n_bands')

    # Pareto frontier
    pareto_n = [p[0] for p in pareto_points]
    pareto_acc = [p[1] for p in pareto_points]
    ax.plot(pareto_n, pareto_acc, 'r-', linewidth=3, label='Pareto Frontier')
    ax.scatter(pareto_n, pareto_acc, s=150, color='red', zorder=5, edgecolor='white', linewidth=2)

    # Baseline
    ax.axhline(y=BASELINE_ACCURACY, color='orange', linestyle='--', linewidth=2,
               label=f'Baseline ({BASELINE_ACCURACY:.2%}, 192 bands)')

    # Annotations for key Pareto points
    for n, acc in pareto_points[::3]:  # Every 3rd point
        ax.annotate(f'n={int(n)}\n{acc:.2%}', xy=(n, acc),
                   xytext=(n+5, acc-0.03), fontsize=9)

    ax.set_xlabel('Number of Bands Selected', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Pareto Frontier: Accuracy vs Band Reduction', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(0, 185)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "04_efficiency_analysis" / "pareto_frontier.png", dpi=150)
    plt.savefig(VIZ_DIR / "04_efficiency_analysis" / "pareto_frontier.pdf")
    plt.close()


def plot_accuracy_vs_reduction(df: pd.DataFrame):
    """Plot accuracy vs reduction percentage."""
    print("  Plotting accuracy vs reduction...")

    fig, ax = plt.subplots(figsize=(14, 9))

    # Best per reduction percentage
    stats = df.groupby('reduction_pct').agg({
        'accuracy': ['max', 'mean'],
        'n_bands_to_select': 'first'
    }).reset_index()
    stats.columns = ['reduction_pct', 'max_acc', 'mean_acc', 'n_bands']

    ax.fill_between(stats['reduction_pct'], stats['mean_acc'], stats['max_acc'],
                    alpha=0.3, color='steelblue')
    ax.plot(stats['reduction_pct'], stats['max_acc'], 'b-', linewidth=2, label='Best Accuracy')
    ax.plot(stats['reduction_pct'], stats['mean_acc'], 'b--', linewidth=2, label='Mean Accuracy')

    ax.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=2, label='Baseline (0% reduction)')

    # Mark key points
    key_reductions = [50, 75, 90, 95]
    for red in key_reductions:
        row = stats[stats['reduction_pct'] >= red].iloc[0] if len(stats[stats['reduction_pct'] >= red]) > 0 else None
        if row is not None:
            ax.scatter([row['reduction_pct']], [row['max_acc']], s=100, zorder=5)
            ax.annotate(f"{row['max_acc']:.2%}\n({int(row['n_bands'])} bands)",
                       xy=(row['reduction_pct'], row['max_acc']),
                       xytext=(row['reduction_pct']-5, row['max_acc']+0.03),
                       fontsize=10, ha='center')

    ax.set_xlabel('Band Reduction (%)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Accuracy vs Spectral Band Reduction', fontsize=16, fontweight='bold')
    ax.legend(loc='lower left', fontsize=12)
    ax.invert_xaxis()  # Higher reduction on right

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "04_efficiency_analysis" / "accuracy_vs_reduction.png", dpi=150)
    plt.savefig(VIZ_DIR / "04_efficiency_analysis" / "accuracy_vs_reduction.pdf")
    plt.close()


def plot_marginal_improvement(df: pd.DataFrame):
    """Plot marginal improvement per additional band."""
    print("  Plotting marginal improvement...")

    # Get best accuracy per n_bands
    best_per_n = df.groupby('n_bands_to_select')['accuracy'].max().sort_index()

    # Calculate marginal improvement
    marginal = best_per_n.diff()

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    # Cumulative accuracy
    axes[0].plot(best_per_n.index, best_per_n.values, 'b-', linewidth=2)
    axes[0].fill_between(best_per_n.index, 0, best_per_n.values, alpha=0.3)
    axes[0].axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=2)
    axes[0].set_ylabel('Best Accuracy', fontsize=14)
    axes[0].set_title('Cumulative Best Accuracy', fontsize=14, fontweight='bold')

    # Marginal improvement
    colors = ['green' if x > 0 else 'red' for x in marginal.values]
    axes[1].bar(marginal.index, marginal.values, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='black', linewidth=1)
    axes[1].set_xlabel('Number of Bands', fontsize=14)
    axes[1].set_ylabel('Marginal Improvement', fontsize=14)
    axes[1].set_title('Marginal Improvement per Additional Band', fontsize=14, fontweight='bold')

    # Add smoothed trend
    smoothed = gaussian_filter1d(marginal.fillna(0).values, sigma=3)
    axes[1].plot(marginal.index, smoothed, 'b-', linewidth=2, label='Smoothed trend')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "04_efficiency_analysis" / "marginal_improvement.png", dpi=150)
    plt.savefig(VIZ_DIR / "04_efficiency_analysis" / "marginal_improvement.pdf")
    plt.close()


def plot_efficiency_score(df: pd.DataFrame):
    """Plot efficiency score (accuracy per band)."""
    print("  Plotting efficiency score...")

    df_copy = df.copy()
    df_copy['efficiency'] = df_copy['accuracy'] / df_copy['n_bands_to_select']

    # Best efficiency per n_bands
    best_eff = df_copy.groupby('n_bands_to_select').agg({
        'efficiency': 'max',
        'accuracy': 'max'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(best_eff['n_bands_to_select'], best_eff['efficiency'], 'g-', linewidth=2)
    ax.fill_between(best_eff['n_bands_to_select'], 0, best_eff['efficiency'], alpha=0.3, color='green')

    # Mark optimal efficiency point
    best_idx = best_eff['efficiency'].idxmax()
    ax.scatter([best_eff.loc[best_idx, 'n_bands_to_select']], [best_eff.loc[best_idx, 'efficiency']],
               s=200, color='red', zorder=5, marker='*')
    ax.annotate(f"Best efficiency\nn={int(best_eff.loc[best_idx, 'n_bands_to_select'])}, acc={best_eff.loc[best_idx, 'accuracy']:.2%}",
               xy=(best_eff.loc[best_idx, 'n_bands_to_select'], best_eff.loc[best_idx, 'efficiency']),
               xytext=(best_eff.loc[best_idx, 'n_bands_to_select'] + 20, best_eff.loc[best_idx, 'efficiency']),
               fontsize=12, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('Number of Bands', fontsize=14)
    ax.set_ylabel('Efficiency Score (Accuracy / n_bands)', fontsize=14)
    ax.set_title('Efficiency Score: Accuracy per Selected Band', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "04_efficiency_analysis" / "efficiency_score.png", dpi=150)
    plt.savefig(VIZ_DIR / "04_efficiency_analysis" / "efficiency_score.pdf")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 5. HEATMAPS
# ═══════════════════════════════════════════════════════════════════════════

def plot_config_heatmap(df: pd.DataFrame):
    """Heatmap of accuracy: config vs n_bands."""
    print("  Plotting configuration heatmap...")

    # Pivot table
    pivot = df.pivot_table(values='accuracy', index='config_key', columns='n_bands_to_select')

    # Sort by max accuracy
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(24, 14))

    sns.heatmap(pivot, cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Accuracy'},
                vmin=0.3, vmax=1.0)

    ax.set_xlabel('Number of Bands Selected', fontsize=14)
    ax.set_ylabel('Configuration', fontsize=14)
    ax.set_title('Accuracy Heatmap: Configuration × Number of Bands', fontsize=16, fontweight='bold')

    # Reduce x-tick density
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::5])
    ax.set_xticklabels([int(pivot.columns[int(t)]) for t in xticks[::5] if int(t) < len(pivot.columns)], rotation=45)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "05_heatmaps" / "config_accuracy_heatmap.png", dpi=150)
    plt.savefig(VIZ_DIR / "05_heatmaps" / "config_accuracy_heatmap.pdf")
    plt.close()


def plot_parameter_heatmaps(df: pd.DataFrame):
    """Heatmaps for each parameter combination."""
    print("  Plotting parameter heatmaps...")

    # Dim method × Normalization at different n_bands
    n_values = [10, 30, 50, 80, 100, 150]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for ax, n in zip(axes, n_values):
        subset = df[df['n_bands_to_select'] == n]
        pivot = subset.pivot_table(values='accuracy',
                                   index='dimension_selection_method',
                                   columns='normalization_method',
                                   aggfunc='max')

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                   vmin=0.5, vmax=1.0, cbar=False)
        ax.set_title(f'n_bands = {n}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.suptitle('Best Accuracy: Dimension Method × Normalization\n(at different n_bands values)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "05_heatmaps" / "parameter_heatmaps_by_nbands.png", dpi=150)
    plt.savefig(VIZ_DIR / "05_heatmaps" / "parameter_heatmaps_by_nbands.pdf")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 6. STATISTICAL COMPARISONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_metric_distributions(df: pd.DataFrame):
    """Distribution of all metrics."""
    print("  Plotting metric distributions...")

    metrics = ['accuracy', 'f1', 'kappa', 'ari', 'nmi']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for ax, metric in zip(axes[:-1], metrics):
        sns.histplot(data=df, x=metric, hue='dimension_selection_method',
                    kde=True, ax=ax, palette=[COLORS['pca'], COLORS['variance']])
        ax.set_title(f'{metric.upper()} Distribution', fontsize=12, fontweight='bold')
        ax.axvline(x=df[metric].mean(), color='black', linestyle='--', label='Mean')

    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "06_statistical_comparisons" / "metric_distributions.png", dpi=150)
    plt.savefig(VIZ_DIR / "06_statistical_comparisons" / "metric_distributions.pdf")
    plt.close()


def plot_paired_comparisons(df: pd.DataFrame):
    """Paired comparisons between parameter values."""
    print("  Plotting paired comparisons...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # PCA vs Variance (paired by other params)
    pca_data = df[df['dimension_selection_method'] == 'pca'].set_index(
        ['n_bands_to_select', 'n_important_dimensions', 'perturbation_method',
         'normalization_method', 'magnitude_variant'])['accuracy']
    var_data = df[df['dimension_selection_method'] == 'variance'].set_index(
        ['n_bands_to_select', 'n_important_dimensions', 'perturbation_method',
         'normalization_method', 'magnitude_variant'])['accuracy']

    common_idx = pca_data.index.intersection(var_data.index)
    axes[0, 0].scatter(var_data.loc[common_idx], pca_data.loc[common_idx], alpha=0.5, s=20)
    axes[0, 0].plot([0.3, 1], [0.3, 1], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('Variance-based Accuracy', fontsize=12)
    axes[0, 0].set_ylabel('PCA-based Accuracy', fontsize=12)
    axes[0, 0].set_title('PCA vs Variance (paired comparison)', fontsize=12, fontweight='bold')

    # Count wins
    pca_wins = (pca_data.loc[common_idx] > var_data.loc[common_idx]).sum()
    var_wins = (pca_data.loc[common_idx] < var_data.loc[common_idx]).sum()
    axes[0, 0].text(0.35, 0.9, f'PCA wins: {pca_wins}\nVariance wins: {var_wins}', fontsize=11)

    # None vs Max normalization
    none_data = df[df['normalization_method'] == 'none'].set_index(
        ['n_bands_to_select', 'dimension_selection_method', 'n_important_dimensions',
         'perturbation_method', 'magnitude_variant'])['accuracy']
    max_data = df[df['normalization_method'] == 'max_per_excitation'].set_index(
        ['n_bands_to_select', 'dimension_selection_method', 'n_important_dimensions',
         'perturbation_method', 'magnitude_variant'])['accuracy']

    common_idx = none_data.index.intersection(max_data.index)
    axes[0, 1].scatter(max_data.loc[common_idx], none_data.loc[common_idx], alpha=0.5, s=20)
    axes[0, 1].plot([0.3, 1], [0.3, 1], 'r--', linewidth=2)
    axes[0, 1].set_xlabel('Max Normalization Accuracy', fontsize=12)
    axes[0, 1].set_ylabel('No Normalization Accuracy', fontsize=12)
    axes[0, 1].set_title('None vs Max Normalization (paired)', fontsize=12, fontweight='bold')

    # Dim 1 vs Dim 3
    dim1_data = df[df['n_important_dimensions'] == 1].set_index(
        ['n_bands_to_select', 'dimension_selection_method', 'perturbation_method',
         'normalization_method', 'magnitude_variant'])['accuracy']
    dim3_data = df[df['n_important_dimensions'] == 3].set_index(
        ['n_bands_to_select', 'dimension_selection_method', 'perturbation_method',
         'normalization_method', 'magnitude_variant'])['accuracy']

    common_idx = dim1_data.index.intersection(dim3_data.index)
    axes[1, 0].scatter(dim3_data.loc[common_idx], dim1_data.loc[common_idx], alpha=0.5, s=20)
    axes[1, 0].plot([0.3, 1], [0.3, 1], 'r--', linewidth=2)
    axes[1, 0].set_xlabel('3 Dimensions Accuracy', fontsize=12)
    axes[1, 0].set_ylabel('1 Dimension Accuracy', fontsize=12)
    axes[1, 0].set_title('1 vs 3 Important Dimensions (paired)', fontsize=12, fontweight='bold')

    # Medium vs High magnitude
    med_data = df[df['magnitude_variant'] == 'medium'].set_index(
        ['n_bands_to_select', 'dimension_selection_method', 'n_important_dimensions',
         'perturbation_method', 'normalization_method'])['accuracy']
    high_data = df[df['magnitude_variant'] == 'high'].set_index(
        ['n_bands_to_select', 'dimension_selection_method', 'n_important_dimensions',
         'perturbation_method', 'normalization_method'])['accuracy']

    common_idx = med_data.index.intersection(high_data.index)
    axes[1, 1].scatter(high_data.loc[common_idx], med_data.loc[common_idx], alpha=0.5, s=20)
    axes[1, 1].plot([0.3, 1], [0.3, 1], 'r--', linewidth=2)
    axes[1, 1].set_xlabel('High Magnitude Accuracy', fontsize=12)
    axes[1, 1].set_ylabel('Medium Magnitude Accuracy', fontsize=12)
    axes[1, 1].set_title('Medium vs High Magnitude (paired)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "06_statistical_comparisons" / "paired_comparisons.png", dpi=150)
    plt.savefig(VIZ_DIR / "06_statistical_comparisons" / "paired_comparisons.pdf")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 7. 3D SURFACES
# ═══════════════════════════════════════════════════════════════════════════

def plot_3d_surface(df: pd.DataFrame):
    """3D surface plots."""
    print("  Plotting 3D surfaces...")

    from mpl_toolkits.mplot3d import Axes3D

    # For PCA only
    pca_df = df[df['dimension_selection_method'] == 'pca'].copy()

    # Map normalization to numeric
    norm_map = {'none': 0, 'max_per_excitation': 1, 'variance': 2}
    pca_df['norm_num'] = pca_df['normalization_method'].map(norm_map)

    # Best accuracy per (n_bands, normalization)
    pivot = pca_df.pivot_table(values='accuracy', index='n_bands_to_select',
                               columns='norm_num', aggfunc='max')

    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

    ax.set_xlabel('Normalization\n(0=none, 1=max, 2=var)', fontsize=12)
    ax.set_ylabel('Number of Bands', fontsize=12)
    ax.set_zlabel('Accuracy', fontsize=12)
    ax.set_title('3D Surface: Accuracy vs n_bands vs Normalization (PCA)', fontsize=14, fontweight='bold')

    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "07_3d_surfaces" / "accuracy_surface_pca.png", dpi=150)
    plt.close()


def plot_3d_scatter(df: pd.DataFrame):
    """3D scatter plot of all experiments."""
    print("  Plotting 3D scatter...")

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Map categorical to numeric
    df_plot = df.copy()
    df_plot['dim_num'] = (df_plot['dimension_selection_method'] == 'pca').astype(int)
    norm_map = {'none': 0, 'max_per_excitation': 1, 'variance': 2}
    df_plot['norm_num'] = df_plot['normalization_method'].map(norm_map)

    scatter = ax.scatter(df_plot['n_bands_to_select'], df_plot['norm_num'], df_plot['accuracy'],
                        c=df_plot['dim_num'], cmap='coolwarm', alpha=0.5, s=20)

    ax.set_xlabel('Number of Bands', fontsize=12)
    ax.set_ylabel('Normalization', fontsize=12)
    ax.set_zlabel('Accuracy', fontsize=12)
    ax.set_title('3D Scatter: All Experiments\n(color: blue=variance, red=pca)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "07_3d_surfaces" / "accuracy_scatter_3d.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 8. CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_metric_correlations(df: pd.DataFrame):
    """Correlation matrix of all metrics."""
    print("  Plotting metric correlations...")

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'kappa', 'ari', 'nmi']
    corr = df[metrics].corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0, ax=ax, square=True, linewidths=1,
                cbar_kws={'label': 'Correlation'})

    ax.set_title('Correlation Matrix of Performance Metrics', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "08_correlation_analysis" / "metric_correlations.png", dpi=150)
    plt.savefig(VIZ_DIR / "08_correlation_analysis" / "metric_correlations.pdf")
    plt.close()


def plot_accuracy_vs_other_metrics(df: pd.DataFrame):
    """Scatter plots: accuracy vs other metrics."""
    print("  Plotting accuracy vs other metrics...")

    other_metrics = ['f1', 'kappa', 'ari', 'nmi']

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    for ax, metric in zip(axes, other_metrics):
        scatter = ax.scatter(df[metric], df['accuracy'],
                            c=df['n_bands_to_select'], cmap='viridis',
                            alpha=0.5, s=20)

        # Fit line
        z = np.polyfit(df[metric], df['accuracy'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[metric].min(), df[metric].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2)

        corr = df['accuracy'].corr(df[metric])
        ax.set_xlabel(metric.upper(), fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Accuracy vs {metric.upper()} (r={corr:.3f})', fontsize=12, fontweight='bold')

    plt.colorbar(scatter, ax=axes, label='n_bands', shrink=0.8)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "08_correlation_analysis" / "accuracy_vs_metrics.png", dpi=150)
    plt.savefig(VIZ_DIR / "08_correlation_analysis" / "accuracy_vs_metrics.pdf")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 9. RANKING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_config_rankings(df: pd.DataFrame):
    """How configuration rankings change with n_bands."""
    print("  Plotting configuration rankings...")

    # Get rankings at different n_bands
    n_values = [10, 20, 30, 50, 80, 100, 150]

    rankings = {}
    for n in n_values:
        subset = df[df['n_bands_to_select'] == n]
        ranking = subset.groupby('config_key')['accuracy'].max().rank(ascending=False)
        rankings[n] = ranking

    ranking_df = pd.DataFrame(rankings)

    # Top 10 configs
    final_ranking = ranking_df[n_values[-1]].sort_values()
    top_configs = final_ranking.head(10).index.tolist()

    fig, ax = plt.subplots(figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_configs)))

    for i, config in enumerate(top_configs):
        if config in ranking_df.index:
            ranks = ranking_df.loc[config, n_values]
            ax.plot(n_values, ranks, 'o-', color=colors[i], linewidth=2, markersize=8,
                   label=config[:25] + '...')

    ax.set_xlabel('Number of Bands', fontsize=14)
    ax.set_ylabel('Rank (1 = best)', fontsize=14)
    ax.set_title('Configuration Ranking Evolution', fontsize=16, fontweight='bold')
    ax.invert_yaxis()  # Rank 1 at top
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_xticks(n_values)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "09_ranking_analysis" / "config_ranking_evolution.png", dpi=150, bbox_inches='tight')
    plt.savefig(VIZ_DIR / "09_ranking_analysis" / "config_ranking_evolution.pdf", bbox_inches='tight')
    plt.close()


def plot_best_config_per_nbands(df: pd.DataFrame):
    """Which config is best at each n_bands."""
    print("  Plotting best config per n_bands...")

    best_configs = df.loc[df.groupby('n_bands_to_select')['accuracy'].idxmax()]

    # Count wins per config type
    dim_wins = best_configs['dimension_selection_method'].value_counts()
    norm_wins = best_configs['normalization_method'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Dimension method wins
    axes[0].pie(dim_wins.values, labels=dim_wins.index, autopct='%1.1f%%',
               colors=[COLORS['pca'], COLORS['variance']], explode=[0.05]*len(dim_wins))
    axes[0].set_title('Best Config Distribution:\nDimension Selection Method', fontsize=12, fontweight='bold')

    # Normalization wins
    colors_norm = [COLORS.get(n, 'gray') for n in norm_wins.index]
    axes[1].pie(norm_wins.values, labels=norm_wins.index, autopct='%1.1f%%',
               colors=colors_norm, explode=[0.05]*len(norm_wins))
    axes[1].set_title('Best Config Distribution:\nNormalization Method', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "09_ranking_analysis" / "best_config_distribution.png", dpi=150)
    plt.savefig(VIZ_DIR / "09_ranking_analysis" / "best_config_distribution.pdf")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# 10. SUMMARY FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def plot_executive_summary(df: pd.DataFrame, baseline: Dict):
    """Executive summary figure for presentations."""
    print("  Plotting executive summary...")

    fig = plt.figure(figsize=(20, 16))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Main accuracy curve (large)
    ax1 = fig.add_subplot(gs[0, :2])
    best_per_n = df.groupby('n_bands_to_select')['accuracy'].max().reset_index()
    ax1.plot(best_per_n['n_bands_to_select'], best_per_n['accuracy'], 'b-', linewidth=3)
    ax1.fill_between(best_per_n['n_bands_to_select'], BASELINE_ACCURACY, best_per_n['accuracy'],
                     where=best_per_n['accuracy'] > BASELINE_ACCURACY, alpha=0.3, color='green')
    ax1.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=2,
               label=f'Baseline: {BASELINE_ACCURACY:.2%} (192 bands)')
    ax1.scatter([80], [df['accuracy'].max()], s=200, color='gold', marker='*', zorder=5,
               edgecolor='black', linewidth=2)
    ax1.set_xlabel('Number of Bands', fontsize=14)
    ax1.set_ylabel('Best Accuracy', fontsize=14)
    ax1.set_title('Band Selection Performance: Best Accuracy vs Number of Bands', fontsize=16, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=12)
    ax1.set_xlim(0, 185)

    # 2. Key metrics box (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    best_row = df.loc[df['accuracy'].idxmax()]
    text = f"""
KEY RESULTS
━━━━━━━━━━━━━━━━━━━━━━
Best Accuracy: {best_row['accuracy']:.2%}
Best F1 Score: {best_row['f1']:.4f}
Best Kappa: {best_row['kappa']:.4f}

Optimal n_bands: {int(best_row['n_bands_to_select'])}
Band Reduction: {best_row['reduction_pct']:.1f}%

vs Baseline:
  +{(best_row['accuracy'] - BASELINE_ACCURACY)*100:.2f}pp accuracy
  -{192 - int(best_row['n_bands_to_select'])} fewer bands
━━━━━━━━━━━━━━━━━━━━━━
    """
    ax2.text(0.1, 0.5, text, fontsize=14, family='monospace', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # 3. Parameter comparison
    ax3 = fig.add_subplot(gs[1, 0])
    param_means = df.groupby('dimension_selection_method')['accuracy'].agg(['mean', 'max'])
    bars = ax3.bar(['PCA', 'Variance'], param_means['max'],
                   color=[COLORS['pca'], COLORS['variance']], edgecolor='black')
    ax3.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=2)
    ax3.set_ylabel('Best Accuracy', fontsize=12)
    ax3.set_title('Dimension Method Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylim(0.8, 1.0)

    # 4. Normalization comparison
    ax4 = fig.add_subplot(gs[1, 1])
    norm_means = df.groupby('normalization_method')['accuracy'].max()
    bars = ax4.bar(norm_means.index, norm_means.values, edgecolor='black')
    ax4.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=2)
    ax4.set_ylabel('Best Accuracy', fontsize=12)
    ax4.set_title('Normalization Comparison', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0.8, 1.0)

    # 5. Accuracy distribution
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(df['accuracy'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax5.axvline(x=df['accuracy'].max(), color='gold', linewidth=3, label=f'Max: {df["accuracy"].max():.2%}')
    ax5.axvline(x=df['accuracy'].mean(), color='green', linewidth=2, linestyle='--',
               label=f'Mean: {df["accuracy"].mean():.2%}')
    ax5.axvline(x=BASELINE_ACCURACY, color='red', linewidth=2, linestyle='--')
    ax5.set_xlabel('Accuracy', fontsize=12)
    ax5.set_ylabel('Count', fontsize=12)
    ax5.set_title('Accuracy Distribution (n=3072)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)

    # 6. Efficiency curve (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    best_per_n = df.groupby('n_bands_to_select')['accuracy'].max()
    reduction = (1 - df.groupby('n_bands_to_select')['n_bands_to_select'].first() / 192) * 100
    ax6.scatter(reduction, best_per_n.values, c=best_per_n.index, cmap='viridis', s=50)
    ax6.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('Band Reduction (%)', fontsize=12)
    ax6.set_ylabel('Best Accuracy', fontsize=12)
    ax6.set_title('Accuracy vs Band Reduction', fontsize=12, fontweight='bold')

    # 7. Top 5 configs table
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')
    top5 = df.nlargest(5, 'accuracy')[['config', 'accuracy', 'n_bands_to_select', 'f1']]
    top5['accuracy'] = top5['accuracy'].apply(lambda x: f'{x:.2%}')
    top5['f1'] = top5['f1'].apply(lambda x: f'{x:.4f}')
    top5['n_bands_to_select'] = top5['n_bands_to_select'].astype(int)
    top5.columns = ['Configuration', 'Accuracy', 'N Bands', 'F1 Score']

    table = ax7.table(cellText=top5.values, colLabels=top5.columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax7.set_title('Top 5 Configurations', fontsize=14, fontweight='bold', y=0.9)

    plt.suptitle('SPECTRAL BAND SELECTION: MASTER RUN RESULTS', fontsize=20, fontweight='bold', y=0.98)

    plt.savefig(VIZ_DIR / "10_summary_figures" / "executive_summary.png", dpi=150, bbox_inches='tight')
    plt.savefig(VIZ_DIR / "10_summary_figures" / "executive_summary.pdf", bbox_inches='tight')
    plt.close()


def plot_paper_figure(df: pd.DataFrame):
    """Publication-ready figure."""
    print("  Plotting publication figure...")

    # Set publication style
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Accuracy vs n_bands with envelope
    ax = axes[0, 0]
    stats_df = df.groupby('n_bands_to_select').agg({
        'accuracy': ['min', 'max', 'mean']
    }).reset_index()
    stats_df.columns = ['n_bands', 'min', 'max', 'mean']

    ax.fill_between(stats_df['n_bands'], stats_df['min'], stats_df['max'],
                    alpha=0.3, color='steelblue', label='Range')
    ax.plot(stats_df['n_bands'], stats_df['max'], 'b-', linewidth=2, label='Best')
    ax.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax.set_xlabel('Number of Selected Bands')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('(a) Accuracy vs. Band Selection')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 185)

    # (b) PCA vs Variance
    ax = axes[0, 1]
    for method, color in [('pca', COLORS['pca']), ('variance', COLORS['variance'])]:
        subset = df[df['dimension_selection_method'] == method]
        best = subset.groupby('n_bands_to_select')['accuracy'].max()
        ax.plot(best.index, best.values, color=color, linewidth=2,
               label=method.upper())
    ax.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Number of Selected Bands')
    ax.set_ylabel('Best Accuracy')
    ax.set_title('(b) Dimension Selection Method')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 185)

    # (c) Normalization comparison
    ax = axes[1, 0]
    norm_labels = {'none': 'None', 'max_per_excitation': 'Max/Excitation', 'variance': 'Variance'}
    for method in ['none', 'max_per_excitation', 'variance']:
        subset = df[df['normalization_method'] == method]
        best = subset.groupby('n_bands_to_select')['accuracy'].max()
        ax.plot(best.index, best.values, linewidth=2, label=norm_labels[method])
    ax.axhline(y=BASELINE_ACCURACY, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Number of Selected Bands')
    ax.set_ylabel('Best Accuracy')
    ax.set_title('(c) Normalization Method')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 185)

    # (d) Parameter effect sizes
    ax = axes[1, 1]
    params = ['dimension_selection_method', 'normalization_method',
              'perturbation_method', 'magnitude_variant', 'n_important_dimensions']
    param_labels = ['Dim. Selection', 'Normalization', 'Perturbation', 'Magnitude', 'N Dimensions']

    effects = []
    for param in params:
        groups = df.groupby(param)['accuracy'].mean()
        effects.append(groups.max() - groups.min())

    bars = ax.barh(param_labels, effects, color='steelblue', edgecolor='black')
    ax.set_xlabel('Effect Size (Max - Min Mean Accuracy)')
    ax.set_title('(d) Parameter Effect Sizes')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "10_summary_figures" / "publication_figure.png", dpi=300, bbox_inches='tight')
    plt.savefig(VIZ_DIR / "10_summary_figures" / "publication_figure.pdf", bbox_inches='tight')
    plt.close()


def generate_readme():
    """Generate README for visualizations folder."""
    readme_content = """# Master Run Visualizations

## Directory Structure

### 01_accuracy_curves/
- `all_configs_accuracy_curves.png/pdf` - All 48 configurations' accuracy vs n_bands
- `top10_configs_accuracy_curves.png/pdf` - Best 10 configurations highlighted
- `accuracy_envelope.png/pdf` - Min/max/mean accuracy envelope
- `accuracy_by_dim_method.png/pdf` - PCA vs Variance comparison
- `accuracy_by_normalization.png/pdf` - Normalization method comparison

### 02_parameter_analysis/
- `parameter_boxplots.png/pdf` - Box plots for each parameter
- `parameter_violin_plots.png/pdf` - Violin plots showing distributions
- `parameter_interactions.png/pdf` - Parameter interaction effects
- `parameter_effect_sizes.png/pdf` - Effect size ranking

### 03_wavelength_analysis/
- `wavelength_heatmap.png/pdf` - Excitation vs Emission selection frequency
- `top_wavelengths_bar.png/pdf` - Top 30 most selected wavelength combinations
- `wavelength_scatter.png/pdf` - Scatter plot colored by frequency and rank
- `excitation_emission_distributions.png/pdf` - Histograms of Ex/Em values
- `top_wavelengths_by_config.png/pdf` - Compare PCA vs Variance top selections

### 04_efficiency_analysis/
- `pareto_frontier.png/pdf` - Pareto optimal accuracy-bands trade-off
- `accuracy_vs_reduction.png/pdf` - Accuracy vs band reduction percentage
- `marginal_improvement.png/pdf` - Marginal gain per additional band
- `efficiency_score.png/pdf` - Accuracy per band efficiency metric

### 05_heatmaps/
- `config_accuracy_heatmap.png/pdf` - Full config × n_bands heatmap
- `parameter_heatmaps_by_nbands.png/pdf` - Parameter interactions at different n_bands

### 06_statistical_comparisons/
- `metric_distributions.png/pdf` - Distribution of all performance metrics
- `paired_comparisons.png/pdf` - Paired parameter comparisons

### 07_3d_surfaces/
- `accuracy_surface_pca.png` - 3D surface plot
- `accuracy_scatter_3d.png` - 3D scatter of all experiments

### 08_correlation_analysis/
- `metric_correlations.png/pdf` - Correlation matrix of metrics
- `accuracy_vs_metrics.png/pdf` - Accuracy vs other metrics scatter

### 09_ranking_analysis/
- `config_ranking_evolution.png/pdf` - How rankings change with n_bands
- `best_config_distribution.png/pdf` - Distribution of "best" config parameters

### 10_summary_figures/
- `executive_summary.png/pdf` - **KEY FIGURE** - Complete results overview
- `publication_figure.png/pdf` - Publication-ready 4-panel figure

## Key Findings

1. **Best Accuracy**: 95.23% achieved with PCA, 80 bands, no normalization
2. **Baseline**: 88.15% with all 192 bands
3. **Improvement**: +7.08 percentage points with 58% fewer bands
4. **Optimal Range**: 50-100 bands provides best accuracy
5. **PCA dominates**: ~8% better than variance-based selection
6. **No normalization wins**: Surprisingly, raw values perform best

## Usage in Presentations

For **executive presentations**: Use `10_summary_figures/executive_summary.png`
For **academic papers**: Use `10_summary_figures/publication_figure.pdf`
For **detailed analysis**: Browse category folders

Generated by: analyze_master_run.py
"""

    with open(VIZ_DIR / "README.md", 'w') as f:
        f.write(readme_content)

    print("  Generated README.md")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*70)

    # Setup
    setup_directories()

    # Load data
    df, wavelengths_data, baseline = load_data()

    # Generate all visualizations
    print("\n1. Accuracy Curves...")
    plot_accuracy_curves_all_configs(df)
    plot_accuracy_curves_best_configs(df)
    plot_accuracy_envelope(df)
    plot_accuracy_by_dim_method(df)
    plot_accuracy_by_normalization(df)

    print("\n2. Parameter Analysis...")
    plot_parameter_importance_boxplots(df)
    plot_parameter_violin_plots(df)
    plot_parameter_interaction(df)
    plot_parameter_effect_sizes(df)

    print("\n3. Wavelength Analysis...")
    plot_wavelength_heatmap(df, wavelengths_data)
    plot_top_wavelengths_bar(df, wavelengths_data)
    plot_wavelength_scatter(df, wavelengths_data)
    plot_excitation_emission_distributions(df, wavelengths_data)
    plot_top_wavelengths_by_config(df, wavelengths_data)

    print("\n4. Efficiency Analysis...")
    plot_pareto_frontier(df)
    plot_accuracy_vs_reduction(df)
    plot_marginal_improvement(df)
    plot_efficiency_score(df)

    print("\n5. Heatmaps...")
    plot_config_heatmap(df)
    plot_parameter_heatmaps(df)

    print("\n6. Statistical Comparisons...")
    plot_metric_distributions(df)
    plot_paired_comparisons(df)

    print("\n7. 3D Surfaces...")
    plot_3d_surface(df)
    plot_3d_scatter(df)

    print("\n8. Correlation Analysis...")
    plot_metric_correlations(df)
    plot_accuracy_vs_other_metrics(df)

    print("\n9. Ranking Analysis...")
    plot_config_rankings(df)
    plot_best_config_per_nbands(df)

    print("\n10. Summary Figures...")
    plot_executive_summary(df, baseline)
    plot_paper_figure(df)
    generate_readme()

    print("\n" + "="*70)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*70)
    print(f"\nAll visualizations saved to: {VIZ_DIR}")
    print(f"Total figures: ~35 PNG + PDF pairs")
    print(f"\nKey figures for presentations:")
    print(f"  📊 {VIZ_DIR}/10_summary_figures/executive_summary.png")
    print(f"  📄 {VIZ_DIR}/10_summary_figures/publication_figure.pdf")


if __name__ == "__main__":
    main()
