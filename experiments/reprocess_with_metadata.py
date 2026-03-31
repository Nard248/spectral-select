#!/usr/bin/env python3
"""
Reprocess Collagen Data with Exposure & Lamp Power Normalization
=================================================================
Takes existing processed .pkl files (with mask/class info already set up)
and applies proper exposure time and lamp power normalization using the
newly available metadata.

Normalization approach (same as Lichens_2):
  1. Exposure normalization:
     normalized = raw_value * (reference_exposure / this_exposure)
     reference = max exposure across excitations
     Effect: shorter exposures get amplified to match longest exposure

  2. Lamp power normalization:
     normalized = value * (reference_power / this_power)
     reference = min power across excitations
     Effect: high-power excitations get scaled down to match weakest lamp

Note: This project uses pickle for scientific hyperspectral data serialization,
which is required by the SpectraData pipeline. All pickle files are generated
and consumed locally within the project.

Usage:
    python experiments/reprocess_with_metadata.py
"""

import json
import pickle  # required for SpectraData .pkl format
import copy
import shutil
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# METADATA (from 7th HSI Experiment description)
# =============================================================================

EXCITATIONS = [310.0, 325.0, 340.0, 365.0, 385.0, 400.0]

METADATA = {
    "Collagen_Acetic_Acid": {
        "sample": "Acetic Acid Collagen (without pepsin)",
        "exposure_times_ms": {
            310.0: 821.65,
            325.0: 838.45,
            340.0: 194.71,
            365.0: 14.81,
            385.0: 10.30,
            400.0: 32.82,
        },
        "lamp_power_uW": {
            310.0: 131.2,
            325.0: 113.1,
            340.0: 326.0,
            365.0: 3.754,
            385.0: 6.293,
            400.0: 0.695,
        },
        "processed_dir": "Sponges Acid Group 1",
        "output_dir": "Collagen_Acetic_Acid",
    },
    "Collagen_Pepsin": {
        "sample": "Pepsin Collagen (with pepsin)",
        "exposure_times_ms": {
            310.0: 1407.42,
            325.0: 1318.72,
            340.0: 358.85,
            365.0: 20.15,
            385.0: 14.16,
            400.0: 156.46,
        },
        "lamp_power_uW": {
            310.0: 232.2,
            325.0: 205.1,
            340.0: 590.0,
            365.0: 6.360,
            385.0: 11.02,
            400.0: 1.265,
        },
        "processed_dir": None,
        "output_dir": "Collagen_Pepsin",
    },
}


def normalize_exposure(data, exposure_times):
    """Apply exposure time normalization (reference = max exposure)."""
    result = copy.deepcopy(data)
    ref_exposure = max(exposure_times.values())

    print(f"  Exposure normalization (reference = {ref_exposure:.2f} ms):")
    for ex_key in result['data']:
        ex_nm = float(ex_key)
        if ex_nm in exposure_times:
            exp_time = exposure_times[ex_nm]
            factor = ref_exposure / exp_time
            result['data'][ex_key]['cube'] = (
                result['data'][ex_key]['cube'].astype(np.float64) * factor
            )
            print(f"    Ex {ex_nm}nm: {exp_time:.2f}ms -> x{factor:.2f}")

    if 'metadata' not in result:
        result['metadata'] = {}
    result['metadata']['exposure_normalization'] = {
        'reference_type': 'max',
        'reference_exposure': ref_exposure,
        'original_exposures': {str(k): v for k, v in exposure_times.items()},
    }
    return result


def normalize_lamp_power(data, lamp_powers):
    """Apply lamp power normalization (reference = min power)."""
    result = copy.deepcopy(data)
    ref_power = min(lamp_powers.values())

    print(f"  Lamp power normalization (reference = {ref_power:.4f} uW):")
    for ex_key in result['data']:
        ex_nm = float(ex_key)
        if ex_nm in lamp_powers:
            power = lamp_powers[ex_nm]
            factor = ref_power / power
            result['data'][ex_key]['cube'] = (
                result['data'][ex_key]['cube'].astype(np.float64) * factor
            )
            print(f"    Ex {ex_nm}nm: {power:.3f}uW -> x{factor:.4f}")

    if 'metadata' not in result:
        result['metadata'] = {}
    result['metadata']['lamp_power_normalization'] = {
        'reference_type': 'min',
        'reference_power': ref_power,
        'lamp_powers': {str(k): v for k, v in lamp_powers.items()},
    }
    return result


def reprocess_dataset(dataset_name, meta):
    """Reprocess a single dataset with metadata normalization."""
    processed_dir = meta.get('processed_dir')
    if processed_dir is None:
        print(f"\n  Skipping {dataset_name}: no existing processed data")
        return

    src_dir = PROJECT_ROOT / "Data" / "processed" / processed_dir
    out_dir = PROJECT_ROOT / "Data" / "processed" / meta['output_dir']
    src_pkl = src_dir / "spectra_masked.pkl"

    if not src_pkl.exists():
        print(f"\n  ERROR: {src_pkl} not found")
        return

    print(f"\n{'='*70}")
    print(f"REPROCESSING: {dataset_name}")
    print(f"  Source: {src_dir}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    # Load existing data
    print("\n  Loading existing processed data...")
    with open(src_pkl, 'rb') as f:
        raw_data = pickle.load(f)

    # Pre-normalization stats
    print("\n  Pre-normalization intensity stats:")
    for ex_key in sorted(raw_data['data'].keys(), key=float):
        cube = raw_data['data'][ex_key]['cube']
        valid = cube[~np.isnan(cube)]
        print(f"    Ex {ex_key}nm: mean={valid.mean():.1f}, max={valid.max():.1f}")

    # Apply normalizations
    print()
    exp_data = normalize_exposure(raw_data, meta['exposure_times_ms'])
    print()
    full_norm_data = normalize_lamp_power(exp_data, meta['lamp_power_uW'])
    exp_only_data = copy.deepcopy(exp_data)

    # Post-normalization stats
    print("\n  Post-normalization (exposure+power) intensity stats:")
    for ex_key in sorted(full_norm_data['data'].keys(), key=float):
        cube = full_norm_data['data'][ex_key]['cube']
        valid = cube[~np.isnan(cube)]
        print(f"    Ex {ex_key}nm: mean={valid.mean():.1f}, max={valid.max():.1f}")

    print("\n  Post-normalization (exposure only) intensity stats:")
    for ex_key in sorted(exp_only_data['data'].keys(), key=float):
        cube = exp_only_data['data'][ex_key]['cube']
        valid = cube[~np.isnan(cube)]
        print(f"    Ex {ex_key}nm: mean={valid.mean():.1f}, max={valid.max():.1f}")

    # Create output directory and save
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in ['class_mask.png', 'roi_regions.json']:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, out_dir / fname)
            print(f"  Copied {fname}")

    with open(out_dir / "spectra_masked.pkl", 'wb') as f:
        pickle.dump(full_norm_data, f)
    print(f"  Saved: spectra_masked.pkl (exposure + power normalized)")

    with open(out_dir / "spectra_masked_exposure_only.pkl", 'wb') as f:
        pickle.dump(exp_only_data, f)
    print(f"  Saved: spectra_masked_exposure_only.pkl")

    out_raw = out_dir / "spectra_masked_raw.pkl"
    shutil.copy2(src_pkl, out_raw)
    print(f"  Saved: spectra_masked_raw.pkl (original)")

    print(f"\n  Done! Output: {out_dir}")


def main():
    print("Collagen Data Reprocessing with Metadata Normalization")
    print("=" * 70)

    for name, meta in METADATA.items():
        reprocess_dataset(name, meta)

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
    print("\nOutput files per dataset:")
    print("  spectra_masked.pkl              <- exposure + power normalized (use this)")
    print("  spectra_masked_exposure_only.pkl <- exposure only")
    print("  spectra_masked_raw.pkl          <- original un-normalized")
    print("  class_mask.png, roi_regions.json <- copied from source")
    print("\nRun the pipeline:")
    print('  PYTORCH_ENABLE_MPS_FALLBACK=1 python experiments/run_master_experiment.py \\')
    print('    --data-dir "Data/processed/Collagen_Acetic_Acid" \\')
    print('    --output "results/Collagen_Acetic_Acid_Normalized" \\')
    print('    --retrain --n-bands 5,10,20,30,50 --n-dims 1,3')


if __name__ == "__main__":
    main()
