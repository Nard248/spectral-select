"""Rebuild the tabulated chromophore spectra in ``src/spectraforge/studio/data`` from omlc.org.

One-off maintenance script (network needed).  Run: ``python scripts/build_tissue_data.py``.
Each output CSV starts with ``#`` header lines citing the source, units and processing.
"""
from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src/spectraforge/studio/data"


def get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("latin-1")


def numeric_rows(text: str, ncol: int) -> np.ndarray:
    rows = []
    for ln in text.splitlines():
        p = ln.split()
        if len(p) >= ncol and re.fullmatch(r"[-+0-9.eE]+", p[0]):
            try:
                rows.append([float(x) for x in p[:ncol]])
            except ValueError:
                pass
    return np.array(rows)


def write(name: str, header: list[str], cols: str, arr: np.ndarray, fmt: str) -> None:
    with open(OUT / name, "w") as f:
        for h in header:
            f.write(f"# {h}\n")
        f.write(cols + "\n")
        for r in arr:
            f.write(",".join(fmt.format(v) for v in r) + "\n")
    print(name, arr.shape, arr[0, 0], arr[-1, 0])


def rebin(arr: np.ndarray, step: float, lo: float, hi: float) -> np.ndarray:
    """Mean-bin a densely sampled noisy spectrum onto ``step`` nm centres."""
    c = np.arange(lo, hi + 1e-9, step)
    out = np.empty_like(c)
    for i, x in enumerate(c):
        m = np.abs(arr[:, 0] - x) <= step / 2
        out[i] = arr[m, 1].mean() if m.any() else np.nan
    ok = np.isfinite(out)
    return np.column_stack([c[ok], out[ok]])


def main() -> None:
    hb = numeric_rows(get("https://omlc.org/spectra/hemoglobin/summary.html").split("Hb02\tHb")[1], 3)
    write("hemoglobin_prahl.csv",
          ["Molar extinction coefficient of oxy- and deoxy-hemoglobin in water, cm^-1/(mol/L).",
           "Compiled by S. Prahl from W. B. Gratzer (Med. Res. Council Labs, London) and N. Kollias (Wellman Labs).",
           "Source: https://omlc.org/spectra/hemoglobin/summary.html  (fetched verbatim, 250-1000 nm).",
           "mu_a [1/cm] = ln(10) * eps * c[g/L] / 64500 g/mol  (Hb tetramer)."],
          "wavelength_nm,hbo2_cm-1_per_M,hb_cm-1_per_M", hb, "{:.6g}")

    seg = numeric_rows(get("https://omlc.org/spectra/water/data/segelstein81.txt"), 2)
    pope = numeric_rows(get("https://omlc.org/spectra/water/data/pope97.txt"), 2)
    seg = seg[(seg[:, 0] >= 250) & (seg[:, 0] <= 1000)]
    use_seg = seg[(seg[:, 0] < 380) | (seg[:, 0] > pope[-1, 0])]
    water = np.vstack([use_seg[use_seg[:, 0] < 380], pope, use_seg[use_seg[:, 0] > pope[-1, 0]]])
    write("water_pure.csv",
          ["Absorption coefficient of pure water, 1/cm.",
           "380-727.5 nm: Pope & Fry 1997, Appl. Opt. 36, 8710 (doi:10.1364/AO.36.008710), integrating-cavity",
           "  measurement (omlc.org/spectra/water/data/pope97.txt).",
           "250-375 nm and 729-1000 nm: Segelstein 1981 (Univ. Missouri-Kansas City MS thesis), compiled on omlc.org",
           "  (water/data/segelstein81.txt), itself built from Hale & Querry 1973, Appl. Opt. 12, 555 and others.",
           "  NOTE: below 380 nm the compiled values are an upper bound (Segelstein exceeds Pope & Fry by ~5x at 380 nm);",
           "  pure-water absorption there is < 1e-3 /cm and negligible next to the other tissue absorbers."],
          "wavelength_nm,water_cm-1", water, "{:.7g}")

    fat = numeric_rows(get("https://omlc.org/spectra/fat/fat.txt"), 2)
    fat[:, 1] /= 100.0   # 1/m -> 1/cm
    write("lipid_vanveen.csv",
          ["Absorption coefficient of pure mammalian (pig) fat, 1/cm (source file in 1/m; divided by 100).",
           "R. L. P. van Veen, H. J. C. M. Sterenborg, A. Pifferi, A. Torricelli, R. Cubeddu, 'Determination of",
           "  visible near-IR absorption coefficients of mammalian fat using time- and spatially resolved diffuse",
           "  reflectance and transmission spectroscopy', J. Biomed. Opt. 10(5), 054004 (2005), doi:10.1117/1.2085149.",
           "Source: https://omlc.org/spectra/fat/fat.txt  (429-1098 nm, 1 nm).  Outside the range held constant."],
          "wavelength_nm,lipid_cm-1", fat, "{:.6g}")

    for code, name, desc, hi in (("119", "bilirubin", "Bilirubin in chloroform", 700),
                                 ("041", "beta_carotene", "beta-Carotene in hexane", 700)):
        d = numeric_rows(get(f"https://omlc.org/spectra/PhotochemCAD/data/{code}-abs.txt"), 2)
        d = rebin(d, 1.0, np.ceil(d[0, 0]), min(hi, d[-1, 0]))
        d[:, 1] = np.maximum(d[:, 1], 0.0)
        write(f"{name}_photochemcad.csv",
              [f"Molar extinction coefficient of {desc}, cm^-1/(mol/L) (PhotochemCAD entry {code}).",
               "H. Du, R. A. Fuh, J. Li, A. Corkan, J. S. Lindsey, Photochem. Photobiol. 68, 141 (1998);",
               "  J. M. Dixon, M. Taniguchi, J. S. Lindsey, Photochem. Photobiol. 81, 212 (2005); omlc.org/spectra/PhotochemCAD.",
               "Processing: 0.25 nm raw data mean-binned to 1 nm, negative noise clipped to 0.",
               "  Solvent spectra: tissue-bound pigment peaks are red/blue-shifted by up to ~10-20 nm (not modelled)."],
              "wavelength_nm,eps_cm-1_per_M", d, "{:.6g}")


if __name__ == "__main__":
    main()
