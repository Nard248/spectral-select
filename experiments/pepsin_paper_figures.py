#!/usr/bin/env python3
"""
Pepsin Dataset — Publication Figures & Report
===============================================
Generates paper-quality figures matching the Lichens paper style, adapted
for the Pepsin Collagen dataset. Includes multi-classifier validation.

Output: results/Pepsin_Paper_Figures/

Figures generated (NO embedded titles — LaTeX captions handle that):
  1. Accuracy envelope (KNN) — matching paper Figure 6
  2. Wavelength importance heatmap — matching paper Figure 9
  3. Multi-classifier accuracy curves — new for validation
  4. Gap-to-baseline heatmap — new for validation
  5. Band reduction efficiency — relative accuracy vs compression
  6. Best configuration summary (text + LaTeX table)

This project uses pickle for scientific hyperspectral data (.pkl format).
"""

import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "Pepsin_Paper_Figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_CSV = PROJECT_ROOT / "results" / "Collagen_Pepsin_Normalized" / "results.csv"
EXPERIMENTS_DIR = PROJECT_ROOT / "results" / "Collagen_Pepsin_Normalized" / "experiments"
CLF_DIR = sorted((PROJECT_ROOT / "results" / "Pepsin_Classifier_Comparison").iterdir())[-1]
CLF_CSV = CLF_DIR / "classifier_comparison.csv"

CLASSIFIER_COLORS = {
    'KNN_k5': '#3498db', 'KNN_k11': '#2980b9', 'KNN_k11_dist': '#1a5276',
    'SVM_rbf': '#e74c3c', 'SVM_linear': '#c0392b',
    'RF_100': '#27ae60', 'RF_300': '#1e8449',
    'GBM': '#f39c12', 'LDA': '#9b59b6', 'MLP': '#e67e22',
}
CLASSIFIER_LABELS = {
    'KNN_k5': 'KNN (k=5)', 'KNN_k11': 'KNN (k=11)', 'KNN_k11_dist': 'KNN (k=11, dist)',
    'SVM_rbf': 'SVM (RBF)', 'SVM_linear': 'SVM (Linear)',
    'RF_100': 'RF (100)', 'RF_300': 'RF (300)',
    'GBM': 'GBM', 'LDA': 'LDA', 'MLP': 'MLP',
}

generated = []

def save_fig(fig, name, dpi=300):
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f"{name}.{ext}", dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    generated.append(name)
    print(f"  Saved: {name}.png/.pdf")


def fig_accuracy_envelope():
    print("\n[1] Accuracy envelope (KNN pipeline)...")
    df = pd.read_csv(PIPELINE_CSV)
    bl = df[df['config'] == 'BASELINE'].iloc[0]
    bl_acc = bl['accuracy']
    df = df[df['config'] != 'BASELINE']
    stats = df.groupby('n_features').agg(
        min_acc=('accuracy','min'), max_acc=('accuracy','max'), mean_acc=('accuracy','mean')
    ).reset_index().sort_values('n_features')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(stats['n_features'], stats['min_acc'], stats['max_acc'],
                    alpha=0.25, color='#4A90D9', label='Range (min\u2013max)')
    ax.plot(stats['n_features'], stats['mean_acc'], color='black', linewidth=2, label='Mean')
    ax.plot(stats['n_features'], stats['max_acc'],
            color='#1B7A2B', linewidth=2.5, linestyle='--', label='Best')
    ax.axhline(y=bl_acc, color='red', linestyle='--', linewidth=1.5,
               label=f'Baseline ({bl_acc:.2%})')
    bi = stats['max_acc'].idxmax()
    bn, ba = stats.loc[bi, 'n_features'], stats.loc[bi, 'max_acc']
    ax.plot([bn, bn], [0.40, ba], color='#1B7A2B', linestyle=':', linewidth=1.5, zorder=4)
    ax.scatter([bn], [ba], color='gold', s=250, zorder=6, marker='*',
               edgecolor='#1B7A2B', linewidth=1.5)
    ax.text(bn+3, (0.40+ba)/2, f"Peak: {ba:.2%}\nn = {int(bn)}",
            fontsize=14, fontweight='bold', color='#1B7A2B', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#1B7A2B', alpha=0.9))
    ax.set_xlabel('Number of Bands Selected', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(0, 160); ax.set_ylim(0.40, 0.92)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "accuracy_envelope_knn")


def fig_wavelength_heatmap():
    print("\n[2] Wavelength importance heatmap...")
    df = pd.read_csv(PIPELINE_CSV)
    bl_acc = df[df['config']=='BASELINE']['accuracy'].values[0]
    df = df[df['config']!='BASELINE']
    wl_data = {}
    for ed in sorted(EXPERIMENTS_DIR.iterdir()):
        if not ed.is_dir(): continue
        wf = ed / "wavelengths.json"
        if wf.exists():
            with open(wf) as f: wl_data[ed.name] = json.load(f)
    if not wl_data:
        print("  SKIP: No wavelength data"); return
    above = df[df['accuracy'] > bl_acc]
    if len(above) < 5: above = df.nlargest(20, 'accuracy')
    above_cfgs = set(above['config'].values)
    em_grid = list(range(420, 730, 10))
    ex_grid = [310, 325, 340, 365, 385, 400]
    imp = np.zeros((len(ex_grid), len(em_grid))); cnt = 0
    for cn in above_cfgs:
        if cn in wl_data:
            wls = wl_data[cn]; mx = max(len(wls), 1)
            for w in wls:
                ex, em = w['excitation'], w['emission']
                if ex in ex_grid:
                    i = ex_grid.index(ex)
                    emr = int(round(em/10)*10)
                    if emr in em_grid:
                        j = em_grid.index(emr)
                        imp[i,j] += 1.0 - (w['rank']-1)/mx
            cnt += 1
    if cnt > 0: imp /= cnt
    valid = np.ones_like(imp, dtype=bool)
    for i, ex in enumerate(ex_grid):
        for j, em in enumerate(em_grid):
            if em < ex+30: valid[i,j] = False
    disp = imp.copy().astype(float); disp[~valid] = np.nan
    fig, ax = plt.subplots(figsize=(10, 3.5))
    cmap = plt.cm.inferno.copy(); cmap.set_bad(color='#D3D3D3')
    masked = np.ma.masked_invalid(disp)
    im = ax.pcolormesh(masked, cmap=cmap, vmin=0, vmax=max(0.01, np.nanmax(disp)),
                       edgecolors='face', linewidth=0, rasterized=True)
    ax.set_xticks([i+0.5 for i in range(0,len(em_grid),2)])
    ax.set_xticklabels([str(e) for e in em_grid[::2]], rotation=90, fontsize=11)
    ax.set_yticks([i+0.5 for i in range(len(ex_grid))])
    ax.set_yticklabels([int(e) for e in ex_grid], fontsize=11)
    ax.set_xlabel('Emission Wavelength (nm)', fontsize=13)
    ax.set_ylabel('Excitation (nm)', fontsize=13)
    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label('Mean Importance Score', fontsize=11)
    fig.tight_layout()
    save_fig(fig, "wavelength_heatmap")


def fig_classifier_curves():
    print("\n[3] Multi-classifier accuracy curves...")
    clf = pd.read_csv(CLF_CSV)
    rd = clf[clf['roi']=='original_small']
    bl = rd[rd['band_selection']=='all']
    sel = rd[rd['band_selection']=='autoencoder']
    highlight = ['KNN_k5', 'LDA', 'SVM_rbf', 'MLP', 'RF_300', 'GBM']
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cn in highlight:
        cd = sel[sel['classifier']==cn].sort_values('n_bands')
        ba = bl[bl['classifier']==cn]['accuracy'].values[0]
        c = CLASSIFIER_COLORS.get(cn, '#95a5a6')
        lb = CLASSIFIER_LABELS.get(cn, cn)
        all_n = list(cd['n_bands'])+[158]; all_a = list(cd['accuracy'])+[ba]
        ax.plot(all_n, all_a, color=c, linewidth=2.5, marker='o', markersize=5,
                label=f'{lb} (bl={ba:.1%})', zorder=3)
    knn_bl = bl[bl['classifier']=='KNN_k5']['accuracy'].values[0]
    ax.axhline(y=knn_bl, color='#3498db', linestyle=':', linewidth=1.5, alpha=0.6,
               label=f'KNN baseline ({knn_bl:.1%})')
    ax.set_xlabel('Number of Selected Bands', fontsize=14)
    ax.set_ylabel('Classification Accuracy', fontsize=14)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(0, 165); ax.set_ylim(0.55, 0.98)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "classifier_curves")


def fig_gap_heatmap():
    print("\n[4] Gap-to-baseline heatmap...")
    clf = pd.read_csv(CLF_CSV)
    rd = clf[clf['roi']=='original_small']
    bl = rd[rd['band_selection']=='all']
    sel = rd[rd['band_selection']=='autoencoder']
    clfs = sorted(bl['classifier'].unique(),
                  key=lambda c: -bl[bl['classifier']==c]['accuracy'].values[0])
    bands = sorted(sel['n_bands'].unique())
    mat = np.full((len(clfs), len(bands)), np.nan)
    for i, c in enumerate(clfs):
        ba = bl[bl['classifier']==c]['accuracy'].values[0]
        for j, nb in enumerate(bands):
            r = sel[(sel['classifier']==c)&(sel['n_bands']==nb)]
            if len(r) > 0: mat[i,j] = r['accuracy'].values[0] - ba
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    vm = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    ly = [CLASSIFIER_LABELS.get(c, c) for c in clfs]
    sns.heatmap(mat*100, annot=True, fmt='.1f', cmap=cmap, center=0,
                vmin=-vm*100, vmax=vm*100,
                xticklabels=[str(b) for b in bands], yticklabels=ly,
                cbar_kws={'label': 'Gap to Own Baseline (pp)'},
                ax=ax, linewidths=0.5)
    ax.set_xlabel('Number of Selected Bands', fontsize=13)
    ax.set_ylabel('', fontsize=13)
    ax.tick_params(axis='y', labelsize=10)
    fig.tight_layout()
    save_fig(fig, "gap_heatmap")


def fig_efficiency():
    print("\n[5] Band reduction efficiency...")
    clf = pd.read_csv(CLF_CSV)
    rd = clf[clf['roi']=='original_small']
    bl = rd[rd['band_selection']=='all']
    sel = rd[rd['band_selection']=='autoencoder']
    highlight = ['KNN_k5', 'LDA', 'SVM_rbf', 'MLP']
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cn in highlight:
        cd = sel[sel['classifier']==cn].sort_values('n_bands')
        ba = bl[bl['classifier']==cn]['accuracy'].values[0]
        c = CLASSIFIER_COLORS.get(cn, '#95a5a6')
        lb = CLASSIFIER_LABELS.get(cn, cn)
        red = (1 - cd['n_bands']/158)*100
        rel = cd['accuracy']/ba*100
        ax.plot(red, rel, color=c, linewidth=2.5, marker='o', markersize=6,
                label=f'{lb} (bl={ba:.1%})')
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Own baseline (100%)')
    ax.set_xlabel('Band Reduction (%)', fontsize=14)
    ax.set_ylabel('Accuracy Relative to Own Baseline (%)', fontsize=14)
    ax.set_xlim(50, 100); ax.set_ylim(60, 105)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "efficiency")


def generate_summary():
    print("\n[6] Summary report...")
    clf = pd.read_csv(CLF_CSV)
    rd = clf[clf['roi']=='original_small']
    bl = rd[rd['band_selection']=='all']
    sel = rd[rd['band_selection']=='autoencoder']
    pipe = pd.read_csv(PIPELINE_CSV)
    pbl = pipe[pipe['config']=='BASELINE'].iloc[0]
    psel = pipe[pipe['config']!='BASELINE']
    knn_bl = bl[bl['classifier']=='KNN_k5']['accuracy'].values[0]

    L = []
    L.append("="*90)
    L.append("PEPSIN COLLAGEN DATASET — SUMMARY REPORT")
    L.append("="*90)
    L.append(f"\nDataset: Pepsin Collagen, 3 crosslinker concentrations")
    L.append(f"Excitations: 6 (310-400 nm), Total bands: 158, Valid pixels: 39,970")
    L.append(f"ROI: Original small (top row, 5,934 training pixels)")

    L.append(f"\n{'='*90}")
    L.append("KNN PIPELINE (432 configurations)")
    L.append(f"{'='*90}")
    L.append(f"Baseline: Acc={pbl['accuracy']:.2%}  Kappa={pbl['kappa']:.4f}")
    pbest = psel.loc[psel.groupby('n_features')['accuracy'].idxmax()].sort_values('n_features')
    for _, r in pbest.iterrows():
        d = r['accuracy'] - pbl['accuracy']
        red = (1-r['n_features']/pbl['n_features'])*100
        mk = " **EXCEEDS**" if d > 0 else ""
        L.append(f"  {int(r['n_features']):3d} bands ({red:4.1f}% red.): "
                 f"Acc={r['accuracy']:.2%} ({d:+.2%}){mk}  | {r['config']}")

    L.append(f"\n{'='*90}")
    L.append("MULTI-CLASSIFIER BASELINES (all 158 bands)")
    L.append(f"{'='*90}")
    for _, r in bl.sort_values('accuracy', ascending=False).iterrows():
        L.append(f"  {CLASSIFIER_LABELS.get(r['classifier'],r['classifier']):20s}: "
                 f"Acc={r['accuracy']:.2%}  Kappa={r['kappa']:.4f}")

    L.append(f"\n{'='*90}")
    L.append(f"BEST PER BAND COUNT (vs KNN baseline = {knn_bl:.2%})")
    L.append(f"{'='*90}")
    for nb in sorted(sel['n_bands'].unique()):
        nbd = sel[sel['n_bands']==nb]
        best = nbd.loc[nbd['accuracy'].idxmax()]
        d = best['accuracy'] - knn_bl
        mk = " **EXCEEDS KNN BASELINE**" if d > 0 else ""
        L.append(f"  {int(nb):3d} bands: {CLASSIFIER_LABELS.get(best['classifier'],best['classifier']):20s} "
                 f"Acc={best['accuracy']:.2%} ({d:+.2%}){mk}")

    L.append(f"\n{'='*90}")
    L.append("FAIR COMPARISON: EACH CLASSIFIER vs OWN BASELINE")
    L.append(f"{'='*90}")
    for cn in sorted(bl['classifier'].unique(),
                     key=lambda c: -bl[bl['classifier']==c]['accuracy'].values[0]):
        ba = bl[bl['classifier']==cn]['accuracy'].values[0]
        cs = sel[sel['classifier']==cn]
        if len(cs) > 0:
            bs = cs.loc[cs['accuracy'].idxmax()]
            gap = bs['accuracy'] - ba
            status = "MATCHES" if gap >= -0.005 else f"{gap:+.1%} gap"
            L.append(f"  {CLASSIFIER_LABELS.get(cn,cn):20s}: bl={ba:.2%} | "
                     f"best={bs['accuracy']:.2%} at {int(bs['n_bands'])}b | {status}")

    report = '\n'.join(L)
    print(report)
    p = OUTPUT_DIR / "pepsin_summary.txt"
    with open(p, 'w') as f: f.write(report)
    print(f"\n  Saved: {p}")


def main():
    print("="*70)
    print("PEPSIN COLLAGEN — PAPER FIGURES (NO TITLES)")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)
    fig_accuracy_envelope()
    fig_wavelength_heatmap()
    fig_classifier_curves()
    fig_gap_heatmap()
    fig_efficiency()
    generate_summary()
    print(f"\n{'='*70}")
    print(f"DONE — {len(generated)} figures generated:")
    for n in generated: print(f"  {n}.png/.pdf")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
