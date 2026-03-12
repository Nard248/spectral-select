#!/usr/bin/env python3
"""
Extract Top 10 Wavelength Combinations per Configuration Group
==============================================================
Multiple analysis approaches:
1. Top 10 from n=10 experiments (exact, per config)
2. Consensus wavelengths (most frequently selected across all configs)
3. PCA vs Variance comparison
4. Performance-weighted ranking
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering/results/Lichens_Dataset_1_MasterRun")
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
OUTPUT_DIR = RESULTS_DIR / "wavelength_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load all experiment data."""
    df = pd.read_csv(RESULTS_DIR / "results.csv")
    df = df[df['config'] != 'BASELINE'].copy()

    # Add config key
    df['config_key'] = df.apply(
        lambda r: f"{r['dimension_selection_method']}_dim{int(r['n_important_dimensions'])}_{r['perturbation_method']}_{r['normalization_method']}_{r['magnitude_variant']}",
        axis=1
    )

    return df


def load_wavelengths(config_name: str) -> list:
    """Load wavelengths from experiment folder."""
    wl_path = EXPERIMENTS_DIR / config_name / "wavelengths.json"
    if wl_path.exists():
        with open(wl_path, 'r') as f:
            return json.load(f)
    return []


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 1: Top 10 from n=10 experiments (per configuration group)
# ═══════════════════════════════════════════════════════════════════════════

def extract_top10_per_config(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract top 10 wavelengths from n=10 experiments for each config group.
    Also includes performance metrics.
    """
    print("Extracting top 10 wavelengths per configuration group...")

    # Filter to n=10 experiments
    n10_df = df[df['n_bands_to_select'] == 10].copy()

    results = []

    for config_key in n10_df['config_key'].unique():
        config_row = n10_df[n10_df['config_key'] == config_key].iloc[0]
        wavelengths = load_wavelengths(config_row['config'])

        # Get best accuracy for this config (at any n_bands)
        best_acc = df[df['config_key'] == config_key]['accuracy'].max()
        best_n = df[df['config_key'] == config_key].loc[
            df[df['config_key'] == config_key]['accuracy'].idxmax(), 'n_bands_to_select'
        ]

        for i, wl in enumerate(wavelengths[:10], 1):
            results.append({
                'config_group': config_key,
                'rank': i,
                'excitation_nm': wl['excitation'],
                'emission_nm': wl['emission'],
                'combination': wl['combination_name'],
                'influence_score': wl['influence_score'],
                'accuracy_at_n10': config_row['accuracy'],
                'f1_at_n10': config_row['f1'],
                'best_accuracy': best_acc,
                'optimal_n_bands': int(best_n),
                # Parse config components
                'dim_method': config_row['dimension_selection_method'],
                'n_dims': int(config_row['n_important_dimensions']),
                'perturbation': config_row['perturbation_method'],
                'normalization': config_row['normalization_method'],
                'magnitude': config_row['magnitude_variant'],
            })

    result_df = pd.DataFrame(results)
    return result_df


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 2: Consensus wavelengths (most frequently in top 10 across configs)
# ═══════════════════════════════════════════════════════════════════════════

def extract_consensus_wavelengths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find wavelengths that appear most frequently in top 10 across all configs.
    """
    print("Extracting consensus wavelengths...")

    n10_df = df[df['n_bands_to_select'] == 10].copy()

    # Count appearances and collect ranks
    wl_stats = defaultdict(lambda: {
        'count': 0,
        'ranks': [],
        'configs': [],
        'excitation': 0,
        'emission': 0
    })

    for _, row in n10_df.iterrows():
        wavelengths = load_wavelengths(row['config'])
        for wl in wavelengths[:10]:
            combo = wl['combination_name']
            wl_stats[combo]['count'] += 1
            wl_stats[combo]['ranks'].append(wl['rank'])
            wl_stats[combo]['configs'].append(row['config_key'])
            wl_stats[combo]['excitation'] = wl['excitation']
            wl_stats[combo]['emission'] = wl['emission']

    results = []
    for combo, stats in wl_stats.items():
        results.append({
            'combination': combo,
            'excitation_nm': stats['excitation'],
            'emission_nm': stats['emission'],
            'appearance_count': stats['count'],
            'appearance_pct': stats['count'] / len(n10_df) * 100,
            'avg_rank': np.mean(stats['ranks']),
            'min_rank': min(stats['ranks']),
            'max_rank': max(stats['ranks']),
            'n_unique_configs': len(set(stats['configs'])),
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('appearance_count', ascending=False)

    return result_df


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 3: PCA vs Variance wavelength comparison
# ═══════════════════════════════════════════════════════════════════════════

def compare_pca_vs_variance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare which wavelengths are selected by PCA vs Variance methods.
    """
    print("Comparing PCA vs Variance wavelength selections...")

    n10_df = df[df['n_bands_to_select'] == 10].copy()

    pca_wl = defaultdict(lambda: {'count': 0, 'ranks': []})
    var_wl = defaultdict(lambda: {'count': 0, 'ranks': []})

    for _, row in n10_df.iterrows():
        wavelengths = load_wavelengths(row['config'])
        target = pca_wl if row['dimension_selection_method'] == 'pca' else var_wl

        for wl in wavelengths[:10]:
            combo = wl['combination_name']
            target[combo]['count'] += 1
            target[combo]['ranks'].append(wl['rank'])
            target[combo]['excitation'] = wl['excitation']
            target[combo]['emission'] = wl['emission']

    # Build comparison
    all_combos = set(pca_wl.keys()) | set(var_wl.keys())

    results = []
    for combo in all_combos:
        pca_data = pca_wl.get(combo, {'count': 0, 'ranks': [], 'excitation': 0, 'emission': 0})
        var_data = var_wl.get(combo, {'count': 0, 'ranks': [], 'excitation': 0, 'emission': 0})

        ex = pca_data.get('excitation', 0) or var_data.get('excitation', 0)
        em = pca_data.get('emission', 0) or var_data.get('emission', 0)

        results.append({
            'combination': combo,
            'excitation_nm': ex,
            'emission_nm': em,
            'pca_count': pca_data['count'],
            'pca_avg_rank': np.mean(pca_data['ranks']) if pca_data['ranks'] else None,
            'variance_count': var_data['count'],
            'variance_avg_rank': np.mean(var_data['ranks']) if var_data['ranks'] else None,
            'count_diff': pca_data['count'] - var_data['count'],
            'pca_only': pca_data['count'] > 0 and var_data['count'] == 0,
            'variance_only': var_data['count'] > 0 and pca_data['count'] == 0,
            'both_methods': pca_data['count'] > 0 and var_data['count'] > 0,
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('count_diff', ascending=False)

    return result_df


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 4: Best performing configs' top 10
# ═══════════════════════════════════════════════════════════════════════════

def extract_top10_from_best_configs(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Extract top 10 wavelengths from the N best-performing configurations.
    """
    print(f"Extracting wavelengths from top {top_n} performing configurations...")

    # Get top N configs by accuracy
    best_configs = df.nlargest(top_n, 'accuracy')

    results = []
    for _, row in best_configs.iterrows():
        wavelengths = load_wavelengths(row['config'])

        for i, wl in enumerate(wavelengths[:10], 1):
            results.append({
                'config': row['config'],
                'config_accuracy': row['accuracy'],
                'config_n_bands': int(row['n_bands_to_select']),
                'rank': i,
                'excitation_nm': wl['excitation'],
                'emission_nm': wl['emission'],
                'combination': wl['combination_name'],
            })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 5: Simplified summary - best top 10 per method
# ═══════════════════════════════════════════════════════════════════════════

def create_simplified_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple summary: top 10 wavelengths from the single best config
    for each dimension selection method.
    """
    print("Creating simplified summary...")

    results = []

    for method in ['pca', 'variance']:
        subset = df[df['dimension_selection_method'] == method]
        best_row = subset.loc[subset['accuracy'].idxmax()]
        wavelengths = load_wavelengths(best_row['config'])

        for i, wl in enumerate(wavelengths[:10], 1):
            results.append({
                'method': method.upper(),
                'rank': i,
                'excitation_nm': wl['excitation'],
                'emission_nm': wl['emission'],
                'combination': wl['combination_name'],
                'config': best_row['config'],
                'accuracy': best_row['accuracy'],
                'n_bands_selected': int(best_row['n_bands_to_select']),
            })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Generate all outputs
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("TOP 10 WAVELENGTH EXTRACTION")
    print("="*70)

    # Load data
    df = load_data()
    print(f"Loaded {len(df)} experiments across {df['config_key'].nunique()} configurations")

    # ─────────────────────────────────────────────────────────────────────
    # Generate all analyses
    # ─────────────────────────────────────────────────────────────────────

    # 1. Top 10 per config group
    top10_per_config = extract_top10_per_config(df)

    # 2. Consensus wavelengths
    consensus = extract_consensus_wavelengths(df)

    # 3. PCA vs Variance comparison
    pca_vs_var = compare_pca_vs_variance(df)

    # 4. Top 10 from best configs
    top10_best = extract_top10_from_best_configs(df, top_n=10)

    # 5. Simplified summary
    simplified = create_simplified_summary(df)

    # ─────────────────────────────────────────────────────────────────────
    # Save to Excel with multiple sheets
    # ─────────────────────────────────────────────────────────────────────

    excel_path = OUTPUT_DIR / "top10_wavelengths_analysis.xlsx"

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Simplified summary (most useful for presentations)
        simplified.to_excel(writer, sheet_name='Summary_Best_Methods', index=False)

        # Sheet 2: Consensus wavelengths
        consensus.to_excel(writer, sheet_name='Consensus_All_Configs', index=False)

        # Sheet 3: PCA vs Variance comparison
        pca_vs_var.to_excel(writer, sheet_name='PCA_vs_Variance', index=False)

        # Sheet 4: Top 10 per config (detailed)
        top10_per_config.to_excel(writer, sheet_name='Top10_Per_Config', index=False)

        # Sheet 5: From best performing configs
        top10_best.to_excel(writer, sheet_name='Top10_Best_Configs', index=False)

        # Sheet 6: Pivot view - configs as rows, ranks as columns
        pivot_view = top10_per_config.pivot(
            index='config_group',
            columns='rank',
            values='combination'
        )
        pivot_view.columns = [f'Rank_{i}' for i in pivot_view.columns]
        pivot_view.to_excel(writer, sheet_name='Pivot_View')

    print(f"\n✓ Saved Excel to: {excel_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Also save individual CSVs
    # ─────────────────────────────────────────────────────────────────────

    top10_per_config.to_csv(OUTPUT_DIR / "top10_per_config.csv", index=False)
    consensus.to_csv(OUTPUT_DIR / "consensus_wavelengths.csv", index=False)
    pca_vs_var.to_csv(OUTPUT_DIR / "pca_vs_variance_comparison.csv", index=False)
    simplified.to_csv(OUTPUT_DIR / "simplified_summary.csv", index=False)

    print(f"✓ Saved CSVs to: {OUTPUT_DIR}")

    # ─────────────────────────────────────────────────────────────────────
    # Print key insights
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    # Most consistently selected wavelengths
    print("\n📊 TOP 10 MOST CONSISTENTLY SELECTED WAVELENGTHS:")
    print("   (appear in top 10 across the most configuration groups)")
    print("-"*60)
    for i, row in consensus.head(10).iterrows():
        print(f"   {row['combination']:20s} | {row['appearance_count']:2.0f} configs ({row['appearance_pct']:5.1f}%) | avg rank: {row['avg_rank']:.1f}")

    # PCA-specific vs Variance-specific
    pca_only = pca_vs_var[pca_vs_var['pca_only'] == True]
    var_only = pca_vs_var[pca_vs_var['variance_only'] == True]
    both = pca_vs_var[pca_vs_var['both_methods'] == True]

    print(f"\n📊 PCA vs VARIANCE WAVELENGTH SELECTION:")
    print("-"*60)
    print(f"   Wavelengths selected by BOTH methods: {len(both)}")
    print(f"   Wavelengths selected by PCA only:     {len(pca_only)}")
    print(f"   Wavelengths selected by Variance only: {len(var_only)}")

    # Best overall
    print(f"\n📊 TOP 10 FROM BEST PCA CONFIGURATION (95.23% accuracy):")
    print("-"*60)
    pca_best = simplified[simplified['method'] == 'PCA']
    for _, row in pca_best.iterrows():
        print(f"   {row['rank']:2d}. Ex{int(row['excitation_nm'])} / Em{int(row['emission_nm'])}")

    print(f"\n📊 TOP 10 FROM BEST VARIANCE CONFIGURATION (86.95% accuracy):")
    print("-"*60)
    var_best = simplified[simplified['method'] == 'VARIANCE']
    for _, row in var_best.iterrows():
        print(f"   {row['rank']:2d}. Ex{int(row['excitation_nm'])} / Em{int(row['emission_nm'])}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
