"""Adapter that surfaces existing AE-perturb selections as a baseline-API
method. Different datasets cache selections in different formats:

* Drop Data: CSV with column 'selected_bands' (string list).
* Lichens MasterRun: per-config directory with `wavelengths.json` (list of
  {excitation, emission, rank} records).
* Pepsin: TBD.

The adapter dispatches on `dataset` keyword.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Drop Data: read selected_bands string from sweep CSV
# ---------------------------------------------------------------------------
def _drop_data_selection(K: int) -> list[tuple[int, int]]:
    csv = ROOT / "results" / "Drop_Data_Cropped_Sweep" / "full_cr" / "ae_perturb_results.csv"
    df = pd.read_csv(csv)
    df_k = df[df["n_bands"] == K]
    if df_k.empty:
        raise KeyError(
            f"No cached AE-perturb selection for K={K} on Drop Data full_cr "
            f"(available K: {sorted(set(df['n_bands']))})"
        )
    sel_str = df_k.iloc[0]["selected_bands"]
    pairs = []
    for m in re.finditer(r"(\d+)\s*/\s*(\d+)", sel_str):
        pairs.append((int(m.group(1)), int(m.group(2))))
    return pairs


# ---------------------------------------------------------------------------
# Lichens MasterRun: read wavelengths.json from the optimal-config experiment dir
# ---------------------------------------------------------------------------
# Optimal config per Phase 0 audit: PCA + k=3 + max_per_excitation
# Folder naming convention: bands_{K}_{dim}_dim_{k_imp}_{perc/abso}_mag_{medium/high}_{var/max/non}
LICHENS_DIR = ROOT / "results" / "Lichens_Dataset_1_MasterRun" / "experiments"
LICHENS_CONFIG_OPTIMAL = "pca_dim_3_perc_mag_medium_max"


def _lichens_selection(K: int) -> list[tuple[int, int]]:
    folder = LICHENS_DIR / f"bands_{K}_{LICHENS_CONFIG_OPTIMAL}"
    if not folder.exists():
        # Try alternative K-keyed dirs available
        avail = sorted(p.name for p in LICHENS_DIR.glob(f"bands_*_{LICHENS_CONFIG_OPTIMAL}"))
        raise KeyError(
            f"No Lichens AE-perturb selection at K={K} for config {LICHENS_CONFIG_OPTIMAL}.\n"
            f"Available: {avail[:10]}{'...' if len(avail)>10 else ''}"
        )
    wj = folder / "wavelengths.json"
    with open(wj) as f:
        data = json.load(f)
    pairs = []
    for rec in data:
        pairs.append((int(rec["excitation"]), int(rec["emission"])))
    return pairs[:K]


# ---------------------------------------------------------------------------
PEPSIN_DIR = ROOT / "results" / "Collagen_Pepsin_Normalized" / "experiments"
# Best Pepsin config per the existing sweep: variance + k=1 + max_per_excitation
PEPSIN_CONFIG_OPTIMAL = "var_dim_1_perc_mag_medium_max"


def _pepsin_selection(K: int) -> list[tuple[int, int]]:
    folder = PEPSIN_DIR / f"bands_{K}_{PEPSIN_CONFIG_OPTIMAL}"
    if not folder.exists():
        avail = sorted(p.name for p in PEPSIN_DIR.glob(f"bands_*_{PEPSIN_CONFIG_OPTIMAL}"))
        raise KeyError(
            f"No Pepsin AE-perturb selection at K={K} for config {PEPSIN_CONFIG_OPTIMAL}.\n"
            f"Available: {avail[:10]}{'...' if len(avail)>10 else ''}"
        )
    wj = folder / "wavelengths.json"
    with open(wj) as f:
        data = json.load(f)
    return [(int(rec["excitation"]), int(rec["emission"])) for rec in data][:K]


DATASET_DISPATCH = {
    "drop_data_full_cr": _drop_data_selection,
    "lichens": _lichens_selection,
    "pepsin": _pepsin_selection,
}


def select_ae_perturb(features: np.ndarray, K: int, *, seed: int = 0,
                     dataset: str = "drop_data_full_cr",
                     band_catalog: list | None = None, **_):
    """Return cached AE-perturb selection for the given K."""
    if dataset not in DATASET_DISPATCH:
        raise NotImplementedError(
            f"AE-perturb cached adapter has no dispatch for dataset={dataset!r}. "
            f"Known: {list(DATASET_DISPATCH)}"
        )
    if band_catalog is None:
        raise ValueError("band_catalog required to map ex/em pairs to indices")
    pairs = DATASET_DISPATCH[dataset](K)
    pair_to_idx = {(int(ex), int(em)): i for i, (ex, em) in enumerate(band_catalog)}
    indices = []
    for key in pairs:
        if key in pair_to_idx:
            indices.append(pair_to_idx[key])
    if not indices:
        raise RuntimeError(
            f"None of {pairs[:5]} (..) found in band_catalog. "
            f"Catalog sample: {band_catalog[:5]}"
        )
    return np.asarray(indices, dtype=int)
