#!/usr/bin/env python3
"""
Drop Data — AE smoke test
==========================
Trains the existing spectral_select Analyzer on ONE preprocessing variant
with minimal epochs, just to confirm the pipeline works end-to-end on the
Drop Data shape, before the full sweep.

Usage: .venv/bin/python experiments/drop_data_smoke_test.py [variant]
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spectral_select import Analyzer, Config, SpectraData


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "dark_norm"
    pkl = PROJECT_ROOT / "Data" / "processed" / "Drop Data" / variant / "spectra_data.pkl"
    print(f"[smoke] loading {pkl}")
    data = SpectraData.from_pickle(pkl)
    print(f"[smoke] excitations: {data.excitation_wavelengths}")
    print(f"[smoke] spatial: {data.spatial_shape}")
    print(f"[smoke] mask present: {data.mask is not None}")
    if data.mask is not None:
        print(f"[smoke] mask valid pixels: {int(data.mask.sum())} / {data.mask.size}")

    cfg = Config(
        sample_name=f"DropData_smoke_{variant}",
        n_bands_to_select=10,
        device="mps",
        training_epochs=3,
        patch_size=16,
        patch_stride=8,
        n_baseline_patches=20,
        n_important_dimensions=10,
        save_visualizations=False,
        save_tiff_layers=False,
        output_dir=PROJECT_ROOT / "results" / "Drop_Data_Smoke" / variant,
    )
    print(f"[smoke] config: {cfg!r}")

    analyzer = Analyzer(cfg)
    print("[smoke] fitting...")
    analyzer.fit(data)
    bands = analyzer.get_wavelengths()
    print(f"[smoke] selected {len(bands)} bands:")
    for b in bands:
        print(f"   {b}")


if __name__ == "__main__":
    main()
