"""Generate a demo synthetic ME-HSI dataset + ground truth, then report where it went."""
from __future__ import annotations

import argparse
from pathlib import Path

from spectraforge.acquisition import AcquisitionConfig
from spectraforge.artifacts import ArtifactConfig
from spectraforge.forward import render
from spectraforge.library import load_builtin_library
from spectraforge.material import Material
from spectraforge.scene import Scene


def build_demo():
    """A 3-excitation, 4-fluorophore painted scene with realistic artifacts."""
    lib = load_builtin_library()
    scene = Scene(64, 64)
    scene.paint_rect(Material("collagen_patch", {"collagen": 1.0, "NADH": 0.3}), 5, 40, 5, 40)
    scene.paint_circle(Material("fad_spot", {"FAD": 1.0}), cy=45, cx=45, radius=12)
    scene.paint_rect(Material("egfp_stripe", {"EGFP": 0.8}), 50, 60, 5, 60)
    acq = AcquisitionConfig(
        excitations=[340.0, 450.0, 488.0], em_min=360, em_max=700, em_step=5,
        exposure={340.0: 2.0}, power={488.0: 0.8},
    )
    artifacts = ArtifactConfig(
        rayleigh_strength=0.15, rayleigh_fwhm=12, photon_scale=400.0, read_sigma=0.005,
    )
    return render(scene, lib, acq, artifacts=artifacts, seed=42, sample_name="spectraforge_demo")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a SpectraForge demo dataset.")
    ap.add_argument("-o", "--out", default="spectraforge_demo_out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spectra, gt = build_demo()
    spectra.to_pickle(out / "spectra_unmasked.pkl")
    gt.save(out)
    print(f"Wrote {spectra.n_excitations} excitations to {out / 'spectra_unmasked.pkl'}")
    print(f"Ground truth (concentration maps + clean cubes) in {out / 'groundtruth.npz'}")
    print("Load it with: spectral-select-gui  (Step 1 -> open this folder), or via SpectraData.from_pickle")


if __name__ == "__main__":
    main()
