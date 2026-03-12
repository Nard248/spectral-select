#!/usr/bin/env python3
"""
Regenerate robustness histogram for the paper.

Uses the existing 10,000 random combination results and plots with
the updated learned selection accuracy from the master run KNN evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
RESULTS_CSV = (
    Path(__file__).parent.parent
    / "archive" / "wavelength_analysis" / "Results"
    / "robustness" / "robustness_13bands_results.csv"
)
OUTPUT_PATH = (
    Path(__file__).parent.parent
    / "Paper Source" / "paper" / "figures" / "robustness_histogram.png"
)

# Updated autoencoder accuracy from master run KNN evaluation (13 bands)
LEARNED_ACCURACY = 0.9016


def main():
    print("Loading random combination results...")
    df = pd.read_csv(RESULTS_CSV)
    accuracies = df['accuracy'].values
    print(f"  Loaded {len(accuracies)} random combinations")

    mean_acc = accuracies.mean()
    median_acc = np.median(accuracies)
    max_acc = accuracies.max()

    print(f"  Random mean:   {mean_acc:.4f}")
    print(f"  Random median: {median_acc:.4f}")
    print(f"  Random max:    {max_acc:.4f}")
    print(f"  Learned:       {LEARNED_ACCURACY:.4f}")

    # Create figure sized for IEEE single-column display (~3.35" wide)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Histogram of random accuracies
    ax.hist(accuracies, bins=50, color='steelblue', edgecolor='black',
            alpha=0.7, label='Random selections', zorder=2)

    # Mean and median lines
    ax.axvline(mean_acc, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_acc:.1%}', zorder=3)
    ax.axvline(median_acc, color='darkorange', linestyle='--', linewidth=2,
               label=f'Median: {median_acc:.1%}', zorder=3)

    # Learned selection (prominent)
    ax.axvline(LEARNED_ACCURACY, color='purple', linestyle='-', linewidth=2.5,
               label=f'Learned: {LEARNED_ACCURACY:.1%}', zorder=4)

    # Annotation arrow pointing to the learned line
    ax.annotate(f'{LEARNED_ACCURACY:.1%}',
                xy=(LEARNED_ACCURACY, ax.get_ylim()[1] * 0.05),
                xytext=(LEARNED_ACCURACY - 0.08, ax.get_ylim()[1] * 0.4),
                fontsize=14, fontweight='bold', color='purple',
                arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))

    ax.set_xlabel('Classification Accuracy', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.set_title('Random vs. Learned 13-Band Selection\n(10,000 random combinations)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper center')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
