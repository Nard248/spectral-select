"""E9 step 1: labelled synthetic dataset (dataset.py): 14 captures of three tissue types under a collagen wedge whose
start/end thickness are randomised (LHS): t0 in [0, 0.3] mm, t1 in [1.0, 1.5] mm; 4 excitations x 31 bands = 124
channels, 'fast' mode (preview resolution 87 x 64 px, 15 % MC photons).  Resumable; written to
~/.cache/spectralforge/e9_dataset (about 100 MB, not committed).
    .venv/bin/python publications/spectralforge_studio/experiments/E9_make_dataset.py [--quick] [--ext]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from E9_scene import build_e9  # noqa: E402
from spectraforge.studio.dataset import generate_dataset  # noqa: E402
from spectraforge.studio.study_schema import DatasetSpec, ParameterSpec  # noqa: E402

EXC = [340.0, 365.0, 400.0, 430.0]
EXC_B = [310.0, 325.0, 415.0]     # extension (same design/seeds) so that the E5-optimal channels (310-430 nm lines) are available
OUT = Path.home() / ".cache" / "spectralforge" / "e9_dataset"
OUT_B = Path.home() / ".cache" / "spectralforge" / "e9_dataset_b"
OUT_QUICK = Path.home() / ".cache" / "spectralforge" / "e9_dataset_quick"


def spec(quick=False, ext=False):
    T = "/objects/{}/shape/layers/2/thickness/"
    ps = [ParameterSpec(pointer=T.format(0) + "thickness_mm", label="t0_mm", low=0.0, high=0.3,
                        also_set=[T.format(1) + "thickness_mm", T.format(2) + "thickness_mm"]),
          ParameterSpec(pointer=T.format(0) + "end_thickness_mm", label="t1_mm", low=1.0, high=1.5,
                        also_set=[T.format(1) + "end_thickness_mm", T.format(2) + "end_thickness_mm"])]
    return DatasetSpec(name="E9_tissue_wedge", base_experiment=build_e9(excitations=EXC_B if ext else EXC), parameters=ps,
                       n=2 if quick else 14, design="lhs", seed=9, mode="fast", label_scheme="object",
                       maps=[f"layer_thickness:cover_{i}" for i in range(3)])


if __name__ == "__main__":
    q = "--quick" in sys.argv
    ext = "--ext" in sys.argv
    t = time.time()
    idx = generate_dataset(spec(q, ext), OUT_QUICK if q else (OUT_B if ext else OUT), progress=lambda f, m: print(f"{m} [{time.time()-t:.0f}s]", flush=True))
    print("done", idx["n"], idx["wall_time_s"])
