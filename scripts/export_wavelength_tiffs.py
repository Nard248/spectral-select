#!/usr/bin/env python3
"""Export selected wavelength slices as ImageJ-compatible multi-page TIFF files.

Extracts the exact 3-band and 9-band wavelength selections from the paper
(Table IV) and saves them as multi-page TIFF stacks openable in ImageJ and
Nuance Software.

3-band config: bands_3_pca_dim_1_perc_mag_medium_max (PCA, 1-dim, 3 bands)
9-band config: bands_9_pca_dim_3_abso_mag_medium_max (PCA, 3-dim, 9 bands)
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spectral_select.types import SpectraData

# --- Configuration ---
DATA_PKL = PROJECT_ROOT / "data" / "processed" / "Lichens Dataset 1" / "spectra_unmasked.pkl"
RESULTS_DIR = PROJECT_ROOT / "results" / "Lichens_Dataset_1_MasterRun" / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "data" / "exported_tiffs"

CONFIGS = {
    "3_bands": "bands_3_pca_dim_1_perc_mag_medium_max",
    "9_bands": "bands_9_pca_dim_3_abso_mag_medium_max",
}


def load_wavelengths(config_name: str) -> list[dict]:
    """Load wavelength selection from experiment results."""
    wl_path = RESULTS_DIR / config_name / "wavelengths.json"
    with open(wl_path) as f:
        return json.load(f)


def extract_slices(spectra: SpectraData, wavelengths: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Extract 2D image slices for each selected (excitation, emission) pair.

    Returns:
        stack: 3D array of shape [n_bands, height, width] (ImageJ stack order)
        labels: Human-readable label for each slice
    """
    slices = []
    labels = []

    for wl in sorted(wavelengths, key=lambda w: w["rank"]):
        ex = wl["excitation"]
        em = wl["emission"]
        rank = wl["rank"]

        # Find the excitation data
        ex_data = spectra.excitations.get(ex)
        if ex_data is None:
            # Try float matching
            for key in spectra.excitations:
                if abs(key - ex) < 1.0:
                    ex_data = spectra.excitations[key]
                    break

        if ex_data is None:
            print(f"  WARNING: Excitation {ex} nm not found in data, skipping")
            continue

        # Find emission band index
        em_wavelengths = ex_data.emission_wavelengths
        band_idx = None

        # Exact match first
        for i, em_wl in enumerate(em_wavelengths):
            if abs(em_wl - em) < 1.0:
                band_idx = i
                break

        # If no exact match, try floor-snapped match (for the 415nm grid offset case)
        if band_idx is None:
            em_snapped = int(em // 10) * 10
            for i, em_wl in enumerate(em_wavelengths):
                em_wl_snapped = int(em_wl // 10) * 10
                if em_snapped == em_wl_snapped:
                    band_idx = i
                    break

        if band_idx is None:
            print(f"  WARNING: Emission {em} nm not found for Ex={ex} nm")
            print(f"    Available: {em_wavelengths[:5]}...{em_wavelengths[-3:]}")
            continue

        # Extract the 2D slice [height, width]
        img_slice = ex_data.cube[:, :, band_idx]
        slices.append(img_slice)
        labels.append(f"Rank{rank}_Ex{int(ex)}_Em{int(em)}")
        print(f"  Rank {rank}: Ex={int(ex)} nm, Em={int(em)} nm -> band_idx={band_idx}, "
              f"shape={img_slice.shape}, range=[{np.nanmin(img_slice):.4f}, {np.nanmax(img_slice):.4f}]")

    stack = np.stack(slices, axis=0)  # [n_bands, height, width]
    return stack, labels


def save_imagej_tiff(stack: np.ndarray, labels: list[str], output_path: Path):
    """Save as ImageJ-compatible multi-page TIFF.

    Uses tifffile for maximum compatibility with ImageJ and Nuance.
    """
    try:
        import tifffile
    except ImportError:
        print("ERROR: tifffile package required. Install with: pip install tifffile")
        sys.exit(1)

    # Replace NaN (background pixels) with 0 for ImageJ/Nuance compatibility
    stack = np.nan_to_num(stack, nan=0.0)

    # Ensure float32 for compatibility (ImageJ handles this well)
    if stack.dtype == np.float64:
        stack = stack.astype(np.float32)

    metadata = {"Labels": labels}

    # Write as ImageJ hyperstack
    tifffile.imwrite(
        str(output_path),
        stack,
        imagej=True,
        metadata=metadata,
        photometric="minisblack",
    )

    n_slices = stack.shape[0]
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path}")
    print(f"  Size: {file_size:.1f} MB, Slices: {n_slices}, Shape: {stack.shape}")
    print(f"  Labels: {labels}")


def save_individual_tiffs(stack: np.ndarray, labels: list[str], output_dir: Path):
    """Save each slice as a separate TIFF file."""
    import tifffile

    stack = np.nan_to_num(stack, nan=0.0)
    if stack.dtype == np.float64:
        stack = stack.astype(np.float32)

    for i, label in enumerate(labels):
        out_path = output_dir / f"{label}.tif"
        tifffile.imwrite(str(out_path), stack[i], imagej=True, photometric="minisblack")

    print(f"  Individual TIFFs saved to: {output_dir}/")
    for label in labels:
        print(f"    {label}.tif")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPORTING WAVELENGTH SELECTION TIFF FILES")
    print("=" * 60)

    # Load the processed hyperspectral data (project's standard pickle format)
    print(f"\nLoading SpectraData from {DATA_PKL.name}...")
    spectra = SpectraData.from_pickle(DATA_PKL)
    print(f"  Excitations: {sorted(spectra.excitations.keys())}")
    for ex_nm, ex_data in sorted(spectra.excitations.items()):
        print(f"    Ex={int(ex_nm)} nm: cube shape={ex_data.cube.shape}, "
              f"emissions=[{ex_data.emission_wavelengths[0]:.0f}..{ex_data.emission_wavelengths[-1]:.0f}] nm")

    for label, config_name in CONFIGS.items():
        print(f"\n{'─' * 60}")
        print(f"Config: {config_name} ({label})")
        print(f"{'─' * 60}")

        # Load wavelength selection
        wavelengths = load_wavelengths(config_name)
        print(f"  Selected {len(wavelengths)} wavelength pairs")

        # Extract slices
        stack, slice_labels = extract_slices(spectra, wavelengths)

        # Save as multi-page stack
        output_path = OUTPUT_DIR / f"selected_{label}_{config_name}.tif"
        save_imagej_tiff(stack, slice_labels, output_path)

        # Save individual slices in a subfolder
        individual_dir = OUTPUT_DIR / f"selected_{label}_individual"
        individual_dir.mkdir(parents=True, exist_ok=True)
        save_individual_tiffs(stack, slice_labels, individual_dir)

    print(f"\n{'=' * 60}")
    print(f"EXPORT COMPLETE — files in: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
