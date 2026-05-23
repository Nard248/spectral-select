#!/usr/bin/env python3
"""
Export Wavelength Combinations — Pepsin Collagen
==================================================
Exports all wavelength selections from the 432 Pepsin pipeline configurations
into a structured Excel workbook.

Output structure:
  results/Pepsin_Wavelength_Exports/pepsin_wavelengths.xlsx

Workbook sheets (Option A — comprehensive):
  - All_Wavelengths:    long-format, every (config, rank, ex, em) tuple
  - Config_Summary:     one row per config with accuracy + config params
  - Frequency_Analysis: which (ex, em) pairs are picked most often

Workbook sheets (Option C — curated):
  - Top20_Overview:     top 20 configs by accuracy, side-by-side wavelength lists
  - Best_Per_Band:      the best config at each band count with full ranking
  - Baseline_Reference: baseline metrics for comparison
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / "results" / "Collagen_Pepsin_Normalized"
EXPERIMENTS_DIR = PIPELINE_DIR / "experiments"
RESULTS_CSV = PIPELINE_DIR / "results.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "Pepsin_Wavelength_Exports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_wavelengths():
    """Load every wavelengths.json into a long-format DataFrame."""
    rows = []
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        wl_file = exp_dir / "wavelengths.json"
        if not wl_file.exists():
            continue
        with open(wl_file) as f:
            wls = json.load(f)
        for w in wls:
            rows.append({
                'config': exp_dir.name,
                'rank': w['rank'],
                'excitation_nm': w['excitation'],
                'emission_nm': w['emission'],
                'combination': w['combination_name'],
                'influence_score': w['influence_score'],
            })
    return pd.DataFrame(rows)


def load_results():
    """Load the pipeline accuracy results."""
    df = pd.read_csv(RESULTS_CSV)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# FREQUENCY ANALYSIS — USER-CONTRIBUTED WEIGHTING
# ═══════════════════════════════════════════════════════════════════════════

def compute_wavelength_frequency(all_wl_df: pd.DataFrame,
                                  results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute how often each (excitation, emission) pair is selected across
    all 432 configurations, weighted by rank AND by the configuration's
    accuracy.

    Rationale (accuracy-weighted rank):
      - Pairs selected by HIGH-accuracy configurations contribute more
      - Pairs selected at TOP ranks contribute more than low ranks
      - This identifies pairs that are "important for good results"
        rather than just "popular across all configs"

    Formula per (ex, em) pair across configs that include it:
        weight(config, rank) = accuracy(config) * (1 - (rank - 1) / n_bands_in_config)
        importance_score(ex, em) = sum of weights across all configs that selected it

    Args:
        all_wl_df: long-format DataFrame with columns
            [config, rank, excitation_nm, emission_nm, influence_score]
        results_df: accuracy per config with at least [config, accuracy, n_features]

    Returns:
        DataFrame with one row per (excitation, emission) pair containing:
            - excitation_nm, emission_nm
            - n_configs_selected: raw count of configs including this pair
            - raw_count_pct: percentage of all 432 configs that selected it
            - mean_rank: mean rank across configs that selected it
            - best_rank: minimum rank achieved
            - mean_influence: mean influence score
            - importance_score: accuracy-weighted rank sum (the main metric)
            - appeared_in_top10_configs: count of times this pair appeared in the top 10 configs
    """
    # ── Attach accuracy and config n_bands to each wavelength row ──
    config_meta = results_df.set_index('config')[['accuracy', 'n_features']].to_dict('index')

    # Build the weighted rank for each (config, pair)
    working = all_wl_df.copy()
    working['accuracy'] = working['config'].map(lambda c: config_meta.get(c, {}).get('accuracy', np.nan))
    working['n_bands_in_config'] = working['config'].map(lambda c: config_meta.get(c, {}).get('n_features', np.nan))

    # Drop any rows where we couldn't find the config (defensive)
    working = working.dropna(subset=['accuracy', 'n_bands_in_config'])

    # rank_weight = 1 at rank=1, decays to ~0 at the last rank
    working['rank_weight'] = 1.0 - (working['rank'] - 1) / working['n_bands_in_config']
    working['weighted_score'] = working['accuracy'] * working['rank_weight']

    # ── Identify "top 10" configs by accuracy for the bonus column ──
    top10_configs = set(results_df[results_df['config'] != 'BASELINE']
                        .nlargest(10, 'accuracy')['config'].values)
    working['in_top10'] = working['config'].isin(top10_configs).astype(int)

    # ── Aggregate per (ex, em) pair ──
    total_configs = all_wl_df['config'].nunique()
    agg = working.groupby(['excitation_nm', 'emission_nm']).agg(
        n_configs_selected=('config', 'nunique'),
        mean_rank=('rank', 'mean'),
        best_rank=('rank', 'min'),
        mean_influence=('influence_score', 'mean'),
        importance_score=('weighted_score', 'sum'),
        appeared_in_top10_configs=('in_top10', 'sum'),
    ).reset_index()

    agg['raw_count_pct'] = agg['n_configs_selected'] / total_configs * 100

    # Reorder columns for clarity
    agg = agg[[
        'excitation_nm', 'emission_nm',
        'n_configs_selected', 'raw_count_pct',
        'mean_rank', 'best_rank', 'mean_influence',
        'importance_score', 'appeared_in_top10_configs',
    ]]

    # Sort by importance_score descending (most important first)
    agg = agg.sort_values('importance_score', ascending=False).reset_index(drop=True)

    return agg


# ═══════════════════════════════════════════════════════════════════════════
# SHEET BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def build_config_summary(all_wl_df, results_df):
    """One row per config with accuracy and metadata."""
    cfg_params = results_df[results_df['config'] != 'BASELINE'][[
        'config', 'n_features', 'accuracy', 'f1', 'kappa',
        'dimension_selection_method', 'n_important_dimensions',
        'perturbation_method', 'normalization_method', 'magnitude_variant',
    ]].copy()
    cfg_params['reduction_pct'] = (1 - cfg_params['n_features'] / 158) * 100
    cfg_params = cfg_params.sort_values('accuracy', ascending=False).reset_index(drop=True)
    cfg_params.insert(0, 'rank_by_accuracy', range(1, len(cfg_params) + 1))
    return cfg_params


def build_top20_overview(all_wl_df, results_df):
    """Side-by-side wavelength lists for the top 20 configs by accuracy."""
    top20 = (results_df[results_df['config'] != 'BASELINE']
             .nlargest(20, 'accuracy')[['config', 'accuracy', 'n_features']])

    # For each top config, get its sorted wavelength list
    tables = []
    for _, row in top20.iterrows():
        cfg = row['config']
        sub = all_wl_df[all_wl_df['config'] == cfg].sort_values('rank')
        sub = sub[['rank', 'excitation_nm', 'emission_nm', 'influence_score']].reset_index(drop=True)
        sub.columns = pd.MultiIndex.from_product(
            [[f"{cfg} (acc={row['accuracy']:.2%}, n={int(row['n_features'])})"],
             sub.columns]
        )
        tables.append(sub)

    # Concat side-by-side, padding with NaN for shorter configs
    if tables:
        return pd.concat(tables, axis=1)
    return pd.DataFrame()


def build_best_per_band(all_wl_df, results_df):
    """The best config at each band count with its full wavelength list."""
    non_bl = results_df[results_df['config'] != 'BASELINE']
    best = non_bl.loc[non_bl.groupby('n_features')['accuracy'].idxmax()]
    best = best.sort_values('n_features')

    rows = []
    for _, r in best.iterrows():
        cfg = r['config']
        wls = all_wl_df[all_wl_df['config'] == cfg].sort_values('rank')
        for _, w in wls.iterrows():
            rows.append({
                'n_bands_group': int(r['n_features']),
                'config_accuracy': r['accuracy'],
                'config': cfg,
                'rank': int(w['rank']),
                'excitation_nm': w['excitation_nm'],
                'emission_nm': w['emission_nm'],
                'influence_score': w['influence_score'],
            })
    return pd.DataFrame(rows)


def build_baseline_reference(results_df):
    bl = results_df[results_df['config'] == 'BASELINE']
    return bl[['config', 'n_features', 'accuracy', 'f1', 'kappa']].copy()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PEPSIN — WAVELENGTH COMBINATIONS EXPORT")
    print("=" * 70)

    print("\n[1/6] Loading wavelength JSONs...")
    all_wl = load_all_wavelengths()
    print(f"  Loaded {all_wl['config'].nunique()} configs, {len(all_wl):,} total wavelength rows")

    print("\n[2/6] Loading results CSV...")
    results = load_results()
    print(f"  Loaded {len(results)} result rows")

    print("\n[3/6] Building config summary...")
    cfg_summary = build_config_summary(all_wl, results)
    print(f"  {len(cfg_summary)} configs summarized")

    print("\n[4/6] Computing frequency analysis (accuracy-weighted rank)...")
    freq = compute_wavelength_frequency(all_wl, results)
    print(f"  {len(freq)} unique (ex, em) pairs analyzed")
    print(f"  Top 5 pairs by importance_score:")
    for _, r in freq.head(5).iterrows():
        print(f"    Ex={r['excitation_nm']:.0f} Em={r['emission_nm']:.0f}  "
              f"score={r['importance_score']:.2f}  "
              f"n_configs={int(r['n_configs_selected'])}")

    print("\n[5/6] Building curated views (Option C)...")
    top20 = build_top20_overview(all_wl, results)
    best_per_band = build_best_per_band(all_wl, results)
    baseline = build_baseline_reference(results)
    print(f"  Top 20 overview: {top20.shape if hasattr(top20, 'shape') else 'empty'}")
    print(f"  Best per band: {len(best_per_band)} rows")

    print("\n[6/6] Writing Excel workbook...")
    xlsx_path = OUTPUT_DIR / "pepsin_wavelengths.xlsx"
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        # Option A
        all_wl.sort_values(['config', 'rank']).to_excel(
            writer, sheet_name='All_Wavelengths', index=False)
        cfg_summary.to_excel(writer, sheet_name='Config_Summary', index=False)
        freq.to_excel(writer, sheet_name='Frequency_Analysis', index=False)

        # Option C
        if not top20.empty:
            top20.to_excel(writer, sheet_name='Top20_Overview')
        best_per_band.to_excel(writer, sheet_name='Best_Per_Band', index=False)
        baseline.to_excel(writer, sheet_name='Baseline_Reference', index=False)

    size_kb = xlsx_path.stat().st_size / 1024
    print(f"  Saved: {xlsx_path} ({size_kb:.0f} KB)")

    # Also export a few CSVs for users who prefer raw data
    all_wl.sort_values(['config', 'rank']).to_csv(
        OUTPUT_DIR / "all_wavelengths.csv", index=False)
    freq.to_csv(OUTPUT_DIR / "frequency_analysis.csv", index=False)
    cfg_summary.to_csv(OUTPUT_DIR / "config_summary.csv", index=False)
    print(f"  Also saved: all_wavelengths.csv, frequency_analysis.csv, config_summary.csv")

    print(f"\n{'=' * 70}")
    print(f"DONE")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
