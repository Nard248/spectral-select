#!/usr/bin/env python3
"""
Collagen Full Analysis — Statistics, Excel Export, and Publication Figures
===========================================================================
Aggregates ALL experiment results from the Collagen Acetic Acid dataset and
produces:
  1. Comprehensive Excel workbook with multiple sheets
  2. Publication-quality figures matching the paper style
  3. Best combination summary

Uses data from:
  - results/Collagen_Classifier_Comparison/ (classifier comparison)
  - results/Collagen_Acetic_Acid_Normalized/ (main pipeline run)
  - results/Collagen_Acetic_Acid_LowBands/  (low band sweep)
  - results/Collagen_Hyperparam_Tuning/     (hyperparameter tuning)
"""

import json, warnings, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "Collagen_Full_Analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

COLORS = {
    'KNN_k5': '#3498db', 'KNN_k11': '#2980b9', 'KNN_k11_dist': '#1a5276',
    'SVM_rbf': '#e74c3c', 'SVM_linear': '#c0392b',
    'RF_100': '#27ae60', 'RF_300': '#1e8449',
    'GBM': '#f39c12', 'LDA': '#9b59b6', 'MLP': '#e67e22',
}


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_results():
    """Load and merge all result CSVs."""
    dfs = {}

    # 1. Classifier comparison (the main one)
    clf_dirs = sorted(glob.glob(str(
        PROJECT_ROOT / "results" / "Collagen_Classifier_Comparison" / "*")))
    if clf_dirs:
        latest = Path(clf_dirs[-1])
        csv = latest / "classifier_comparison.csv"
        if csv.exists():
            dfs['classifier'] = pd.read_csv(csv)
            print(f"  Classifier comparison: {len(dfs['classifier'])} rows from {latest.name}")

    # 2. Main pipeline (normalized)
    for name in ["Collagen_Acetic_Acid_Normalized", "Collagen_Acetic_Acid_LowBands"]:
        csv = PROJECT_ROOT / "results" / name / "results.csv"
        if csv.exists():
            dfs[name] = pd.read_csv(csv)
            print(f"  {name}: {len(dfs[name])} rows")

    # 3. Hyperparameter tuning
    hp_dirs = sorted(glob.glob(str(
        PROJECT_ROOT / "results" / "Collagen_Hyperparam_Tuning" / "*" / "hyperparameter_results.csv")))
    hp_dfs = []
    for hp_csv in hp_dirs:
        hp_dfs.append(pd.read_csv(hp_csv))
    if hp_dfs:
        dfs['hyperparams'] = pd.concat(hp_dfs, ignore_index=True)
        print(f"  Hyperparameter tuning: {len(dfs['hyperparams'])} rows from {len(hp_dfs)} runs")

    return dfs


# ═══════════════════════════════════════════════════════════════════════════
# EXCEL EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def create_excel(dfs):
    """Create comprehensive Excel workbook."""
    xlsx_path = OUTPUT_DIR / "collagen_analysis.xlsx"
    print(f"\nCreating Excel: {xlsx_path}")

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:

        # Sheet 1: Summary — best per classifier per band count (original ROIs)
        if 'classifier' in dfs:
            clf = dfs['classifier']
            for roi_label in clf['roi'].unique():
                rd = clf[clf['roi'] == roi_label]
                bl = rd[rd['band_selection'] == 'all'][['classifier', 'accuracy', 'f1', 'kappa']].copy()
                bl.columns = ['classifier', 'baseline_acc', 'baseline_f1', 'baseline_kappa']

                sel = rd[rd['band_selection'] == 'autoencoder']
                pivot = sel.pivot_table(index='classifier', columns='n_bands',
                                        values='accuracy', aggfunc='first')
                merged = bl.merge(pivot, on='classifier')
                merged = merged.sort_values('baseline_acc', ascending=False)

                sheet = f"Summary_{roi_label[:8]}"
                merged.to_excel(writer, sheet_name=sheet, index=False)
                print(f"  Sheet: {sheet}")

        # Sheet 2: All classifier results
        if 'classifier' in dfs:
            dfs['classifier'].to_excel(writer, sheet_name='All_Classifier_Results', index=False)
            print("  Sheet: All_Classifier_Results")

        # Sheet 3: Main pipeline results
        for name in ["Collagen_Acetic_Acid_Normalized", "Collagen_Acetic_Acid_LowBands"]:
            if name in dfs:
                sheet = name.replace("Collagen_Acetic_Acid_", "Pipeline_")
                dfs[name].sort_values('accuracy', ascending=False).to_excel(
                    writer, sheet_name=sheet[:31], index=False)
                print(f"  Sheet: {sheet[:31]}")

        # Sheet 4: Best combinations across all experiments
        best_rows = []
        if 'classifier' in dfs:
            clf = dfs['classifier']
            for roi in clf['roi'].unique():
                rd = clf[(clf['roi'] == roi) & (clf['band_selection'] == 'autoencoder')]
                for nb in sorted(rd['n_bands'].unique()):
                    nbd = rd[rd['n_bands'] == nb]
                    best = nbd.loc[nbd['accuracy'].idxmax()]
                    bl_acc = clf[(clf['roi'] == roi) & (clf['classifier'] == best['classifier'])
                                 & (clf['band_selection'] == 'all')]['accuracy'].values
                    bl_v = bl_acc[0] if len(bl_acc) > 0 else np.nan
                    best_rows.append({
                        'roi': roi, 'n_bands': nb,
                        'best_classifier': best['classifier'],
                        'accuracy': best['accuracy'], 'f1': best['f1'],
                        'kappa': best['kappa'],
                        'classifier_baseline': bl_v,
                        'gap_to_own_baseline': best['accuracy'] - bl_v,
                        'reduction_pct': (1 - nb / 149) * 100,
                    })

        if best_rows:
            pd.DataFrame(best_rows).to_excel(writer, sheet_name='Best_Combinations', index=False)
            print("  Sheet: Best_Combinations")

        # Sheet 5: Hyperparameter tuning summary
        if 'hyperparams' in dfs:
            hp = dfs['hyperparams']
            hp_non_bl = hp[~hp['config'].str.contains('BASELINE', na=False)]
            if len(hp_non_bl) > 0 and 'axis' in hp_non_bl.columns:
                hp_best = hp_non_bl.loc[hp_non_bl.groupby(['axis', 'n_bands'])['accuracy'].idxmax()]
                hp_best = hp_best.sort_values(['axis', 'n_bands'])
                hp_best.to_excel(writer, sheet_name='Hyperparam_Best', index=False)
                print("  Sheet: Hyperparam_Best")

    print(f"  Saved: {xlsx_path}")
    return xlsx_path


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def fig1_accuracy_envelope(dfs):
    """Accuracy vs n_bands envelope — matching paper Figure 6 style."""
    if 'classifier' not in dfs:
        return

    clf = dfs['classifier']

    for roi_label in clf['roi'].unique():
        rd = clf[(clf['roi'] == roi_label) & (clf['band_selection'] == 'autoencoder')]
        if len(rd) == 0:
            continue

        # Stats per band count across all classifiers
        stats = rd.groupby('n_bands').agg(
            min_acc=('accuracy', 'min'),
            max_acc=('accuracy', 'max'),
            mean_acc=('accuracy', 'mean'),
        ).reset_index().sort_values('n_bands')

        # KNN-only stats
        knn_data = rd[rd['classifier'] == 'KNN_k5']
        knn_stats = knn_data.groupby('n_bands')['accuracy'].first().reset_index()

        # Get baselines
        bl = clf[(clf['roi'] == roi_label) & (clf['band_selection'] == 'all')]
        knn_bl = bl[bl['classifier'] == 'KNN_k5']['accuracy'].values[0]
        best_bl_clf = bl.loc[bl['accuracy'].idxmax()]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Envelope (all classifiers)
        ax.fill_between(stats['n_bands'], stats['min_acc'], stats['max_acc'],
                        alpha=0.2, color='#4A90D9', label='Range (all classifiers)')
        ax.plot(stats['n_bands'], stats['max_acc'],
                color='#1B7A2B', linewidth=2.5, linestyle='--', label='Best classifier')
        ax.plot(stats['n_bands'], stats['mean_acc'],
                color='black', linewidth=2, label='Mean (all classifiers)')

        # KNN line
        if len(knn_stats) > 0:
            ax.plot(knn_stats['n_bands'], knn_stats['accuracy'],
                    color='#3498db', linewidth=2, marker='o', markersize=5,
                    label='KNN (k=5)')

        # Baselines
        ax.axhline(y=knn_bl, color='#3498db', linestyle=':',
                   linewidth=1.5, alpha=0.7,
                   label=f'KNN baseline ({knn_bl:.1%})')
        ax.axhline(y=best_bl_clf['accuracy'], color='red', linestyle='--',
                   linewidth=1.5,
                   label=f'{best_bl_clf["classifier"]} baseline ({best_bl_clf["accuracy"]:.1%})')

        # Mark peak
        best_idx = stats['max_acc'].idxmax()
        best_n = stats.loc[best_idx, 'n_bands']
        best_acc = stats.loc[best_idx, 'max_acc']
        ax.scatter([best_n], [best_acc], color='gold', s=250, zorder=6,
                   marker='*', edgecolor='#1B7A2B', linewidth=1.5)

        ax.set_xlabel('Number of Selected Bands', fontsize=13)
        ax.set_ylabel('Classification Accuracy', fontsize=13)
        ax.set_ylim(0.50, 1.0)
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f"accuracy_envelope_{roi_label}.pdf"
        fig.savefig(FIG_DIR / fname, dpi=300, bbox_inches='tight')
        fig.savefig(FIG_DIR / fname.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure: {fname}")


def fig2_classifier_comparison_bars(dfs):
    """Grouped bar chart: classifiers at key band counts."""
    if 'classifier' not in dfs:
        return

    clf = dfs['classifier']
    key_bands = [5, 10, 20, 50]

    for roi_label in clf['roi'].unique():
        rd = clf[clf['roi'] == roi_label]
        bl = rd[rd['band_selection'] == 'all']
        sel = rd[rd['band_selection'] == 'autoencoder']

        classifiers = sorted(bl['classifier'].unique(),
                            key=lambda c: -bl[bl['classifier']==c]['accuracy'].values[0])

        fig, axes = plt.subplots(1, len(key_bands), figsize=(4*len(key_bands), 6),
                                 sharey=True)
        if len(key_bands) == 1:
            axes = [axes]

        for ax, nb in zip(axes, key_bands):
            nbd = sel[sel['n_bands'] == nb]
            accs = []
            colors = []
            labels = []
            for c in classifiers:
                row = nbd[nbd['classifier'] == c]
                if len(row) > 0:
                    accs.append(row['accuracy'].values[0])
                    colors.append(COLORS.get(c, '#95a5a6'))
                    labels.append(c)

            bars = ax.barh(range(len(labels)), accs, color=colors, edgecolor='white',
                          height=0.7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels if ax == axes[0] else [], fontsize=9)
            ax.set_xlabel('Accuracy', fontsize=10)
            ax.set_title(f'{nb} bands', fontsize=12, fontweight='bold')
            ax.set_xlim(0.50, 0.90)
            ax.axvline(x=bl[bl['classifier']=='KNN_k5']['accuracy'].values[0],
                       color='#3498db', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.grid(True, axis='x', alpha=0.3)

        fig.tight_layout()
        fname = f"classifier_bars_{roi_label}.pdf"
        fig.savefig(FIG_DIR / fname, dpi=300, bbox_inches='tight')
        fig.savefig(FIG_DIR / fname.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure: {fname}")


def fig3_per_classifier_curves(dfs):
    """Line plot: accuracy vs bands for each classifier (like paper Fig 6)."""
    if 'classifier' not in dfs:
        return

    clf = dfs['classifier']

    for roi_label in clf['roi'].unique():
        rd = clf[clf['roi'] == roi_label]
        bl = rd[rd['band_selection'] == 'all']
        sel = rd[rd['band_selection'] == 'autoencoder']

        fig, ax = plt.subplots(figsize=(10, 6))

        for clf_name in sorted(sel['classifier'].unique()):
            clf_data = sel[sel['classifier'] == clf_name].sort_values('n_bands')
            bl_acc = bl[bl['classifier'] == clf_name]['accuracy'].values[0]

            color = COLORS.get(clf_name, '#95a5a6')

            # Curve
            ax.plot(clf_data['n_bands'], clf_data['accuracy'],
                    color=color, linewidth=2, marker='o', markersize=4,
                    label=f'{clf_name} (bl={bl_acc:.1%})')

            # Baseline as horizontal line
            ax.axhline(y=bl_acc, color=color, linestyle=':', linewidth=1, alpha=0.4)

        ax.set_xlabel('Number of Selected Bands', fontsize=13)
        ax.set_ylabel('Classification Accuracy', fontsize=13)
        ax.set_ylim(0.50, 1.0)
        ax.legend(loc='lower right', fontsize=8, ncol=2, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f"classifier_curves_{roi_label}.pdf"
        fig.savefig(FIG_DIR / fname, dpi=300, bbox_inches='tight')
        fig.savefig(FIG_DIR / fname.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure: {fname}")


def fig4_gap_to_baseline_heatmap(dfs):
    """Heatmap: gap to own baseline for each classifier x band count."""
    if 'classifier' not in dfs:
        return

    clf = dfs['classifier']

    for roi_label in clf['roi'].unique():
        rd = clf[clf['roi'] == roi_label]
        bl = rd[rd['band_selection'] == 'all']
        sel = rd[rd['band_selection'] == 'autoencoder']

        # Build matrix: classifier x n_bands -> gap to own baseline
        classifiers = sorted(bl['classifier'].unique(),
                            key=lambda c: -bl[bl['classifier']==c]['accuracy'].values[0])
        bands = sorted(sel['n_bands'].unique())

        matrix = np.full((len(classifiers), len(bands)), np.nan)
        for i, c in enumerate(classifiers):
            bl_acc = bl[bl['classifier'] == c]['accuracy'].values[0]
            for j, nb in enumerate(bands):
                row = sel[(sel['classifier'] == c) & (sel['n_bands'] == nb)]
                if len(row) > 0:
                    matrix[i, j] = row['accuracy'].values[0] - bl_acc

        fig, ax = plt.subplots(figsize=(12, 6))
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))

        sns.heatmap(matrix * 100, annot=True, fmt='.1f', cmap=cmap,
                    center=0, vmin=-vmax*100, vmax=vmax*100,
                    xticklabels=[str(b) for b in bands],
                    yticklabels=classifiers,
                    cbar_kws={'label': 'Gap to Own Baseline (% points)'},
                    ax=ax, linewidths=0.5)

        ax.set_xlabel('Number of Selected Bands', fontsize=12)
        ax.set_ylabel('Classifier', fontsize=12)

        fig.tight_layout()
        fname = f"gap_heatmap_{roi_label}.pdf"
        fig.savefig(FIG_DIR / fname, dpi=300, bbox_inches='tight')
        fig.savefig(FIG_DIR / fname.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure: {fname}")


def fig5_roi_comparison(dfs):
    """Compare original vs expanded ROIs for each classifier."""
    if 'classifier' not in dfs:
        return
    clf = dfs['classifier']
    if clf['roi'].nunique() < 2:
        return

    bl = clf[clf['band_selection'] == 'all']

    fig, ax = plt.subplots(figsize=(10, 6))
    classifiers = sorted(bl['classifier'].unique())

    x = np.arange(len(classifiers))
    width = 0.35

    for i, roi_label in enumerate(sorted(clf['roi'].unique())):
        rd = bl[bl['roi'] == roi_label]
        accs = [rd[rd['classifier']==c]['accuracy'].values[0]
                if len(rd[rd['classifier']==c]) > 0 else 0
                for c in classifiers]
        label = 'Original ROIs (3 boxes)' if 'orig' in roi_label else 'Expanded ROIs (9 boxes)'
        ax.bar(x + i*width - width/2, accs, width, label=label,
               edgecolor='white', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Baseline Accuracy (all 149 bands)', fontsize=12)
    ax.set_ylim(0.70, 1.0)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fname = "roi_comparison_baselines.pdf"
    fig.savefig(FIG_DIR / fname, dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / fname.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure: {fname}")


def fig6_band_reduction_efficiency(dfs):
    """Efficiency plot: accuracy vs band reduction percentage."""
    if 'classifier' not in dfs:
        return

    clf = dfs['classifier']
    highlight_clfs = ['KNN_k5', 'SVM_rbf', 'LDA', 'MLP']

    for roi_label in clf['roi'].unique():
        rd = clf[clf['roi'] == roi_label]
        bl = rd[rd['band_selection'] == 'all']
        sel = rd[rd['band_selection'] == 'autoencoder']

        fig, ax = plt.subplots(figsize=(10, 6))

        for clf_name in highlight_clfs:
            clf_data = sel[sel['classifier'] == clf_name].sort_values('n_bands')
            if len(clf_data) == 0:
                continue
            bl_acc = bl[bl['classifier'] == clf_name]['accuracy'].values[0]
            reduction = (1 - clf_data['n_bands'] / 149) * 100
            relative_acc = clf_data['accuracy'] / bl_acc * 100

            color = COLORS.get(clf_name, '#95a5a6')
            ax.plot(reduction, relative_acc, color=color, linewidth=2.5,
                    marker='o', markersize=5,
                    label=f'{clf_name} (bl={bl_acc:.1%})')

        ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                   label='Own baseline')
        ax.set_xlabel('Band Reduction (%)', fontsize=13)
        ax.set_ylabel('Accuracy Relative to Own Baseline (%)', fontsize=13)
        ax.set_xlim(50, 100)
        ax.set_ylim(60, 105)
        ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f"efficiency_{roi_label}.pdf"
        fig.savefig(FIG_DIR / fname, dpi=300, bbox_inches='tight')
        fig.savefig(FIG_DIR / fname.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure: {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# BEST COMBINATIONS REPORT
# ═══════════════════════════════════════════════════════════════════════════

def print_best_combinations(dfs):
    """Print and save the definitive best combinations."""
    if 'classifier' not in dfs:
        return

    clf = dfs['classifier']
    report_lines = []

    def p(line=""):
        print(line)
        report_lines.append(line)

    p("=" * 90)
    p("BEST COMBINATIONS — COLLAGEN ACETIC ACID DATASET")
    p("=" * 90)

    for roi_label in clf['roi'].unique():
        rd = clf[clf['roi'] == roi_label]
        bl = rd[rd['band_selection'] == 'all']
        sel = rd[rd['band_selection'] == 'autoencoder']

        p(f"\n{'─'*90}")
        p(f"ROI: {roi_label}")
        p(f"{'─'*90}")

        # Best baseline
        best_bl = bl.loc[bl['accuracy'].idxmax()]
        knn_bl = bl[bl['classifier'] == 'KNN_k5']['accuracy'].values[0]

        p(f"\n  Best baseline: {best_bl['classifier']} = {best_bl['accuracy']:.1%}")
        p(f"  KNN baseline:  KNN_k5 = {knn_bl:.1%}")

        # For each band count: best overall, and best that matches/exceeds KNN baseline
        p(f"\n  {'Bands':>5s} | {'Best Classifier':>15s} | {'Accuracy':>8s} | "
          f"{'vs KNN bl':>9s} | {'vs own bl':>9s} | {'Reduction':>9s} | Note")
        p(f"  {'─'*80}")

        for nb in sorted(sel['n_bands'].unique()):
            nbd = sel[sel['n_bands'] == nb]
            best = nbd.loc[nbd['accuracy'].idxmax()]
            own_bl = bl[bl['classifier'] == best['classifier']]['accuracy'].values[0]

            delta_knn = best['accuracy'] - knn_bl
            delta_own = best['accuracy'] - own_bl
            reduction = (1 - nb / 149) * 100

            note = ""
            if delta_knn >= 0:
                note = "EXCEEDS KNN baseline"
            elif delta_knn >= -0.02:
                note = "~close to KNN baseline"
            if delta_own >= 0:
                note += " | MATCHES own baseline"

            p(f"  {nb:5d} | {best['classifier']:>15s} | {best['accuracy']:>7.1%} | "
              f"{delta_knn:>+8.1%} | {delta_own:>+8.1%} | {reduction:>8.1f}% | {note}")

    # Save report
    report_path = OUTPUT_DIR / "best_combinations.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    p(f"\nSaved: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("Collagen Full Analysis")
    print("=" * 70)

    # Load
    print("\nLoading results...")
    dfs = load_all_results()

    # Excel
    create_excel(dfs)

    # Figures
    print("\nGenerating figures...")
    fig1_accuracy_envelope(dfs)
    fig2_classifier_comparison_bars(dfs)
    fig3_per_classifier_curves(dfs)
    fig4_gap_to_baseline_heatmap(dfs)
    fig5_roi_comparison(dfs)
    fig6_band_reduction_efficiency(dfs)

    # Best combinations
    print()
    print_best_combinations(dfs)

    print(f"\n{'='*70}")
    print(f"ALL DONE")
    print(f"  Excel:   {OUTPUT_DIR / 'collagen_analysis.xlsx'}")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Report:  {OUTPUT_DIR / 'best_combinations.txt'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
